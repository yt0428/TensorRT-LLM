# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NUMA-local pinned host memory allocation helpers for multi-socket GPU servers.

On multi-socket systems (such as 4-GPU / 2-NUMA VR200 Grace/Vera nodes), pinned host
buffers allocated before executor thread pinning can land on a remote NUMA node, adding
cross-socket C2C latency on every host-to-device copy or zero-copy kernel read.

This module resolves the host NUMA node owning a CUDA device from PCI sysfs (falling back
to ``CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID``) and scopes both thread CPU affinity and Linux
``set_mempolicy`` (``MPOL_BIND`` with ``MPOL_PREFERRED`` fallback) around pinned host
tensor allocation so the physical pages are faulted and DMA-pinned locally.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import platform
from typing import Iterator, Optional

import torch

from tensorrt_llm._utils import prefer_pinned

__all__ = [
    "gpu_numa_node",
    "numa_local_pin_context",
    "numa_node_cpus",
    "numa_pinned_empty",
    "numa_pinned_zeros",
    "pointer_numa_node",
]

# Linux set_mempolicy / get_mempolicy constants (from <linux/mempolicy.h>).
_MPOL_DEFAULT = 0
_MPOL_PREFERRED = 1
_MPOL_BIND = 2
_MPOL_F_NODE = 1 << 0
_MPOL_F_ADDR = 1 << 1

_SYS_SET_MEMPOLICY_BY_ARCH: dict[str, int] = {
    "aarch64": 237,
    "arm64": 237,
    "x86_64": 238,
    "amd64": 238,
}
_SYS_GET_MEMPOLICY_BY_ARCH: dict[str, int] = {
    "aarch64": 236,
    "arm64": 236,
    "x86_64": 239,
    "amd64": 239,
}

_LIBC: Optional[ctypes.CDLL] = None
_LIBC_TRIED = False


def _get_libc() -> Optional[ctypes.CDLL]:
    global _LIBC, _LIBC_TRIED
    if not _LIBC_TRIED:
        _LIBC_TRIED = True
        try:
            _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
        except OSError:
            _LIBC = None
    return _LIBC


def _parse_cpu_list(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            cpus.update(range(int(lo_s), int(hi_s) + 1))
        else:
            cpus.add(int(part))
    return cpus


def gpu_numa_node(device: Optional[int | torch.device] = None) -> Optional[int]:
    """Return the host NUMA node owning ``device`` (or the current CUDA device), or None."""
    try:
        if not torch.cuda.is_available():
            return None
        if device is None:
            dev = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            if device.type != "cuda":
                return None
            dev = torch.cuda.current_device() if device.index is None else int(device.index)
        else:
            dev = int(device)
        props = torch.cuda.get_device_properties(dev)
    except (RuntimeError, ValueError):
        return None
    domain = getattr(props, "pci_domain_id", None)
    bus = getattr(props, "pci_bus_id", None)
    slot = getattr(props, "pci_device_id", None)
    if domain is not None and bus is not None and slot is not None:
        bdf = f"{int(domain):04x}:{int(bus):02x}:{int(slot):02x}.0"
        path = f"/sys/bus/pci/devices/{bdf}/numa_node"
        try:
            with open(path) as f:
                node = int(f.read().strip())
            if node >= 0:
                return node
        except (OSError, ValueError):
            pass
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


def _set_mempolicy_node(numa_node: Optional[int]) -> bool:
    """Bind current thread's allocation policy to ``numa_node`` (or restore ``MPOL_DEFAULT``)."""
    sysno = _SYS_SET_MEMPOLICY_BY_ARCH.get(platform.machine().lower())
    if sysno is None:
        return False
    libc = _get_libc()
    if libc is None:
        return False
    try:
        if numa_node is None:
            rc = libc.syscall(
                ctypes.c_long(sysno),
                ctypes.c_int(_MPOL_DEFAULT),
                ctypes.c_void_p(0),
                ctypes.c_ulong(0),
            )
            return int(rc) == 0
        if numa_node < 0 or numa_node >= 64:
            return False
        mask = ctypes.c_ulong(1 << int(numa_node))
        maxnode = ctypes.c_ulong(64)
        for mode in (_MPOL_BIND, _MPOL_PREFERRED):
            rc = libc.syscall(
                ctypes.c_long(sysno),
                ctypes.c_int(mode),
                ctypes.byref(mask),
                maxnode,
            )
            if int(rc) == 0:
                return True
    except (OSError, ValueError):
        return False
    return False


def pointer_numa_node(ptr: int) -> Optional[int]:
    """Query the physical NUMA node backing ``ptr`` via ``get_mempolicy``, or None."""
    if ptr <= 0:
        return None
    sysno = _SYS_GET_MEMPOLICY_BY_ARCH.get(platform.machine().lower())
    if sysno is None:
        return None
    libc = _get_libc()
    if libc is None:
        return None
    try:
        node = ctypes.c_int(-1)
        rc = libc.syscall(
            ctypes.c_long(sysno),
            ctypes.byref(node),
            ctypes.c_void_p(0),
            ctypes.c_ulong(0),
            ctypes.c_void_p(ptr),
            ctypes.c_ulong(_MPOL_F_NODE | _MPOL_F_ADDR),
        )
        if int(rc) == 0 and int(node.value) >= 0:
            return int(node.value)
    except (OSError, ValueError):
        return None
    return None


@contextlib.contextmanager
def numa_local_pin_context(device: Optional[int | torch.device] = None) -> Iterator[Optional[int]]:
    """Temporarily bind thread CPU affinity & mempolicy to ``device``'s local NUMA node."""
    node = gpu_numa_node(device)
    if node is None:
        yield None
        return
    prev_affinity: Optional[set[int]] = None
    mempolicy_set = False
    try:
        node_cpus = numa_node_cpus(node)
        if node_cpus:
            try:
                avail = os.sched_getaffinity(0)
                scoped = avail & node_cpus
                if scoped and scoped != avail:
                    prev_affinity = avail
                    os.sched_setaffinity(0, scoped)
            except OSError:
                prev_affinity = None
        mempolicy_set = _set_mempolicy_node(node)
        yield node
    finally:
        if mempolicy_set:
            _set_mempolicy_node(None)
        if prev_affinity is not None:
            try:
                os.sched_setaffinity(0, prev_affinity)
            except OSError:
                pass


def numa_pinned_zeros(
    *shape: int,
    dtype: torch.dtype,
    device: Optional[int | torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    """Allocate a zero-initialized host tensor pinned on ``device``'s local NUMA node."""
    pin = prefer_pinned() if pin_memory is None else bool(pin_memory)
    if not pin:
        return torch.zeros(*shape, dtype=dtype, device="cpu", pin_memory=False)
    with numa_local_pin_context(device):
        tensor = torch.zeros(*shape, dtype=dtype, device="cpu", pin_memory=prefer_pinned())
        # Touch pages while thread affinity and mempolicy are bound locally.
        tensor.zero_()
    return tensor


def numa_pinned_empty(
    *shape: int,
    dtype: torch.dtype,
    device: Optional[int | torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    """Allocate an uninitialized host tensor pinned on ``device``'s local NUMA node."""
    pin = prefer_pinned() if pin_memory is None else bool(pin_memory)
    if not pin:
        return torch.empty(*shape, dtype=dtype, device="cpu", pin_memory=False)
    with numa_local_pin_context(device):
        tensor = torch.empty(*shape, dtype=dtype, device="cpu", pin_memory=prefer_pinned())
        # First-touch zero-fill faults pages onto the bound local NUMA node before return.
        tensor.zero_()
    return tensor
