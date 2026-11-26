# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ultra-low-latency host/runtime hooks for Mistral Large 4 on SM107 (VR200).

Declared on :class:`ModelingV2MistralLarge4Nvfp4Sm107Tp4` as ``runtime_hooks``
and consumed once by ``PyExecutor`` and ``DecoderRunner``. Compatible with
TP4, TP8, and TP16 deployments using pure Python and Triton kernels.

Enables strictly lossless host/runtime optimizations:

- NUMA-local exclusive executor thread pinning (avoiding CPU 0 and evicting
  other process threads from the reserved core and its SMT siblings), with
  periodic thread refreshes;
- 200 us Python thread switch interval and cyclic GC disabled during the
  overlap serving loop, restored cleanly on shutdown;
- Overlap loop executed directly on ``execution_stream`` with side-stream
  iteration and per-request timing events;
- Pure-Triton ``StepInputPack`` (``parallel_scatter``, ``in_graph``,
  ``early_copy``), ``H2DSkipCache`` with 2-D ``changed_span`` staging,
  single-launch ``SteadyMtpPrepare``, single-pass ``on_update_kv_lens``, and
  single-launch Triton ``fused_spec_offsets_op``.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm.logger import logger

__all__ = [
    "GIL_SWITCH_US",
    "PIN_REFRESH_ITERS",
    "ExecutorThreadPin",
    "Ml4UllExecutorEvents",
    "fused_spec_offsets_op",
    "gpu_numa_node",
    "make_ml4_runtime_hooks",
    "numa_node_cpus",
    "physical_gpu_index",
    "pin_executor_cpu",
    "smt_siblings",
]

_LOG = "ML4 ULL runtime"

# Python thread switch interval while serving (interpreter default is 5000 us).
GIL_SWITCH_US = 200.0
# Iterations between scans that move newly spawned threads off the executor CPU.
PIN_REFRESH_ITERS = 512

_LOCAL_RANK_ENV_VARS = (
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "SLURM_LOCALID",
    "LOCAL_RANK",
    "PMI_LOCAL_RANK",
)


def _parse_cpu_list(text: str) -> set[int]:
    """Parse a Linux sysfs CPU list (e.g. ``"0-87,176-263"`` or ``"4,68"``) into a set."""
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            cpus.update(range(int(start_s), int(end_s) + 1))
        else:
            cpus.add(int(part))
    return cpus


def smt_siblings(cpu: int) -> set[int]:
    """Return ``cpu`` and its SMT siblings (just ``{cpu}`` when sysfs does not report them)."""
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(path) as f:
            text = f.read().strip()
        return _parse_cpu_list(text) | {cpu}
    except (OSError, ValueError):
        return {cpu}


def gpu_numa_node(device: Optional[int] = None) -> Optional[int]:
    """Return the host NUMA node owning ``device`` (or the current CUDA device), or None.

    Reads ``/sys/bus/pci/devices/<bdf>/numa_node`` using the CUDA device's PCI
    domain/bus/device tuple, falling back to ``CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID``
    when sysfs is unavailable.
    """
    try:
        if not torch.cuda.is_available():
            return None
        dev = torch.cuda.current_device() if device is None else int(device)
        props = torch.cuda.get_device_properties(dev)
    except (RuntimeError, ValueError):
        return None
    domain = getattr(props, "pci_domain_id", None)
    bus = getattr(props, "pci_bus_id", None)
    slot = getattr(props, "pci_device_id", None)
    if domain is not None and bus is not None and slot is not None:
        for domain_fmt in (f"{int(domain):04x}", f"{int(domain):08x}"):
            bdf = f"{domain_fmt}:{int(bus):02x}:{int(slot):02x}.0"
            path = f"/sys/bus/pci/devices/{bdf}/numa_node"
            try:
                with open(path) as f:
                    node = int(f.read().strip())
                if node >= 0:
                    return node
            except (OSError, ValueError):
                continue
    try:
        from cuda.bindings import driver

        err, cu_dev = driver.cuDeviceGet(dev)
        if err == driver.CUresult.CUDA_SUCCESS:
            err, host_numa = driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cu_dev
            )
            if err == driver.CUresult.CUDA_SUCCESS and int(host_numa) >= 0:
                return int(host_numa)
    except (ImportError, AttributeError, RuntimeError, ValueError):
        pass
    return None


def numa_node_cpus(numa_node: int) -> Optional[set[int]]:
    """Return the CPU set for ``numa_node`` from sysfs ``cpulist``, or None when unavailable."""
    if numa_node < 0:
        return None
    path = f"/sys/devices/system/node/node{int(numa_node)}/cpulist"
    try:
        with open(path) as f:
            cpus = _parse_cpu_list(f.read().strip())
    except (OSError, ValueError):
        return None
    return cpus or None


def physical_gpu_index(fallback_local_rank: Optional[int] = None) -> Optional[int]:
    """Return the node-local GPU index when it can be determined."""
    try:
        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
    except RuntimeError:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return device
    devices = visible.split(",")
    if device < len(devices):
        identifier = devices[device].strip()
        if identifier.isdecimal():
            return int(identifier)
    # UUID masks (common on Slurm nodes) hide the numeric ordinal; fall back to local rank.
    for env_var in _LOCAL_RANK_ENV_VARS:
        raw = os.environ.get(env_var)
        if raw is not None and raw.strip().isdecimal():
            return int(raw.strip())
    if fallback_local_rank is not None and fallback_local_rank >= 0:
        return int(fallback_local_rank)
    return None


def pin_executor_cpu(
    available: set[int],
    gpu_index: Optional[int] = None,
    *,
    fallback_local_rank: Optional[int] = None,
    numa_cpus: Optional[set[int]] = None,
) -> Optional[int]:
    """Pick the ``gpu_index``-th non-CPU-0 physical core from ``available`` on the GPU's NUMA node.

    When ``numa_cpus`` is provided (or resolved dynamically from :func:`gpu_numa_node` and
    :func:`numa_node_cpus`), selection is restricted to ``available & numa_cpus`` so ranks
    stay socket-local to their GPU and host KV cache across TP4, TP8, and TP16 topologies.
    Falls back to ``available`` only when the intersection is empty. A physical core is
    represented by the lowest CPU ID in its SMT sibling group; the core containing CPU 0
    (housekeeping interrupts) is skipped when other cores are available.
    """
    if not available:
        return None
    if gpu_index is None:
        gpu_index = physical_gpu_index(fallback_local_rank=fallback_local_rank)
        if gpu_index is None:
            return None
    if numa_cpus is None:
        node = gpu_numa_node()
        if node is not None:
            numa_cpus = numa_node_cpus(node)
    scoped = (available & numa_cpus) if numa_cpus else set()
    cpus = sorted(scoped if scoped else available)
    cores: list[int] = []
    seen: set[int] = set()
    for cpu in cpus:
        if cpu in seen:
            continue
        siblings = smt_siblings(cpu)
        seen |= siblings
        if 0 not in siblings:
            cores.append(cpu)
    if not cores:
        return cpus[0]
    return cores[gpu_index % len(cores)]


class ExecutorThreadPin:
    """Reserves one NUMA-local physical CPU core exclusively for the executor loop thread.

    The calling thread's affinity is set to ``{cpu}``. Every other thread in the
    process has ``cpu`` and its SMT siblings removed from its affinity mask when
    that leaves at least one allowed CPU. Threads created later inherit their
    creator's mask, so :meth:`maybe_refresh` periodically rescans ``/proc/self/task``.
    """

    def __init__(
        self,
        cpu: int,
        refresh_iters: int = PIN_REFRESH_ITERS,
        *,
        numa_node: Optional[int] = None,
    ) -> None:
        self.cpu = cpu
        self.numa_node = numa_node
        self.tid = threading.get_native_id()
        self._reserved_cpus = smt_siblings(cpu)
        self.refresh_iters = max(1, int(refresh_iters))
        self._known: set[int] = set()

    @classmethod
    def create(
        cls,
        *,
        gpu_index: Optional[int] = None,
        fallback_local_rank: Optional[int] = None,
        refresh_iters: int = PIN_REFRESH_ITERS,
        numa_node: Optional[int] = None,
    ) -> Optional["ExecutorThreadPin"]:
        """Pin the calling thread; return None when no CPU can be chosen or OS refuses."""
        try:
            available = os.sched_getaffinity(0)
            resolved_numa = gpu_numa_node() if numa_node is None else numa_node
            node_cpus = numa_node_cpus(resolved_numa) if resolved_numa is not None else None
            if node_cpus is not None and not (available & node_cpus):
                logger.warning(
                    f"{_LOG}: available CPUs do not intersect GPU NUMA node {resolved_numa}; "
                    f"falling back to unrestricted affinity mask"
                )
            cpu = pin_executor_cpu(
                available,
                gpu_index=gpu_index,
                fallback_local_rank=fallback_local_rank,
                numa_cpus=node_cpus if node_cpus is not None else set(),
            )
            if cpu is None:
                return None
            pin = cls(cpu, refresh_iters=refresh_iters, numa_node=resolved_numa)
            os.sched_setaffinity(0, {cpu})
            pin.refresh()
        except (OSError, ValueError) as error:
            logger.warning(f"{_LOG}: executor thread not pinned ({error})")
            return None
        numa_desc = f"NUMA node {resolved_numa}, " if resolved_numa is not None else ""
        logger.info(
            f"{_LOG}: executor thread {pin.tid} pinned to CPU {cpu} "
            f"({numa_desc}reserved SMT group {sorted(pin._reserved_cpus)})"
        )
        return pin

    def refresh(self) -> int:
        """Move newly observed process threads off the reserved core; return count moved."""
        moved = 0
        try:
            tids = [int(t) for t in os.listdir("/proc/self/task")]
        except OSError:
            return 0
        for tid in tids:
            if tid == self.tid or tid in self._known:
                continue
            try:
                current = os.sched_getaffinity(tid)
                remaining = current - self._reserved_cpus
                if remaining and remaining != current:
                    os.sched_setaffinity(tid, remaining)
                    moved += 1
            except OSError:
                # Thread exited or its affinity is not ours to change.
                pass
            self._known.add(tid)
        self._known.intersection_update(tids)
        return moved

    def maybe_refresh(self, iteration: int) -> None:
        if iteration % self.refresh_iters == 0:
            self.refresh()


@triton.jit(do_not_specialize=["n_pos", "n_kv"])
def _fused_spec_offsets_kernel(
    positions_ptr,
    pos_offsets_ptr,
    kv_lens_ptr,
    kv_offsets_ptr,
    n_pos,
    n_kv,
    BLOCK: tl.constexpr,
):
    """Apply overlap-scheduler position and KV-length offsets in a single kernel launch."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos_mask = offs < n_pos
    pos = tl.load(positions_ptr + offs, mask=pos_mask, other=0)
    pos_off = tl.load(pos_offsets_ptr + offs, mask=pos_mask, other=0)
    tl.store(positions_ptr + offs, pos + pos_off, mask=pos_mask)

    kv_mask = offs < n_kv
    kv = tl.load(kv_lens_ptr + offs, mask=kv_mask, other=0)
    kv_off = tl.load(kv_offsets_ptr + offs, mask=kv_mask, other=0)
    tl.store(kv_lens_ptr + offs, kv + kv_off, mask=kv_mask)


def fused_spec_offsets_op(
    positions: torch.Tensor,
    position_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    kv_lens_offsets: torch.Tensor,
) -> bool:
    """In-place ``positions += position_offsets`` and ``kv_lens += kv_lens_offsets`` in one launch.

    Returns ``False`` without writing anything if shapes, devices, contiguity, or dtypes
    do not match the fused fast path, so the caller falls back to the two stock adds.
    """
    if not (
        positions.is_cuda
        and position_offsets.is_cuda
        and kv_lens.is_cuda
        and kv_lens_offsets.is_cuda
        and positions.ndim == 1
        and position_offsets.ndim == 1
        and kv_lens.ndim == 1
        and kv_lens_offsets.ndim == 1
        and positions.is_contiguous()
        and position_offsets.is_contiguous()
        and kv_lens.is_contiguous()
        and kv_lens_offsets.is_contiguous()
        and positions.dtype == position_offsets.dtype
        and kv_lens.dtype == kv_lens_offsets.dtype
        and positions.dtype in (torch.int32, torch.int64)
        and kv_lens.dtype in (torch.int32, torch.int64)
        and positions.numel() == position_offsets.numel()
        and kv_lens.numel() == kv_lens_offsets.numel()
    ):
        return False
    n_pos = positions.numel()
    n_kv = kv_lens.numel()
    n_max = max(n_pos, n_kv)
    if n_max == 0:
        return True
    block = 256
    _fused_spec_offsets_kernel[(triton.cdiv(n_max, block),)](
        positions,
        position_offsets,
        kv_lens,
        kv_lens_offsets,
        n_pos,
        n_kv,
        BLOCK=block,
        num_warps=4,
    )
    return True


class Ml4UllExecutorEvents:
    """Executor lifecycle events for ML4 low-latency serving.

    Pins the executor thread to an exclusive NUMA-local physical core, lowers the
    Python GIL switch interval, disables periodic cyclic GC during serving, and
    restores interpreter state on clean shutdown.
    """

    def __init__(
        self,
        *,
        pin_refresh_iters: int = PIN_REFRESH_ITERS,
        gil_switch_us: float = GIL_SWITCH_US,
        local_rank: Optional[int] = None,
    ) -> None:
        self.pin_refresh_iters = max(1, int(pin_refresh_iters))
        self.gil_switch_us = float(gil_switch_us)
        self.local_rank = local_rank
        self.pin: Optional[ExecutorThreadPin] = None
        self._prev_switch_interval: Optional[float] = None
        self._gc_was_enabled: bool = False

    def after_warmup(self) -> None:
        return

    def before_overlap_loop(self) -> None:
        self.pin = ExecutorThreadPin.create(
            fallback_local_rank=self.local_rank,
            refresh_iters=self.pin_refresh_iters,
        )
        self._prev_switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(self.gil_switch_us / 1_000_000)
        logger.info(f"{_LOG}: thread switch interval {sys.getswitchinterval() * 1e6:.0f} us")
        self._gc_was_enabled = gc.isenabled()
        if self._gc_was_enabled:
            gc.collect()
            gc.disable()
            logger.info(f"{_LOG}: Python cyclic GC disabled for overlap serving loop")

    def on_iteration(self, iteration: int) -> None:
        if self.pin is not None:
            self.pin.maybe_refresh(iteration)

    def at_shutdown(self) -> None:
        if self._prev_switch_interval is not None:
            sys.setswitchinterval(self._prev_switch_interval)
            self._prev_switch_interval = None
        if self._gc_was_enabled:
            gc.enable()
            self._gc_was_enabled = False


def make_ml4_runtime_hooks(mapping: object = None) -> RuntimeHooks:
    """Build ML4 SM107 low-latency :class:`RuntimeHooks` across TP4, TP8, and TP16 topologies."""
    local_rank = getattr(mapping, "local_rank", None)
    events = Ml4UllExecutorEvents(
        local_rank=local_rank if isinstance(local_rank, int) else None,
    )
    return RuntimeHooks(
        pack_step_inputs=True,
        pack_parallel_scatter=True,
        pack_in_graph=True,
        pack_early_copy=True,
        executor_events=events,
        skip_unchanged_h2d=True,
        h2d_changed_span=True,
        incremental_block_offsets=True,
        steady_dsa_prepare=True,
        kv_lens_update_once=True,
        spec_offsets_op=fused_spec_offsets_op,
        loop_on_execution_stream=True,
        steady_spec_prepare=True,
        pack_spec_rng_uploads=True,
        iteration_timing_one_event=True,
        perf_metrics_timing_side_stream=True,
    )
