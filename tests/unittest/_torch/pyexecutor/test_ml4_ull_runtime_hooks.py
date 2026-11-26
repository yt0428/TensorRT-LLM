# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for Mistral Large 4 SM107 ULL runtime hooks (TP4/TP8/TP16)."""

import gc
import os
import sys
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4 import (
    runtime_hooks as ml4_hooks,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

pytestmark = pytest.mark.cpu_only


def test_smt_siblings_parses_sysfs_formats(monkeypatch):
    def _mock_open_factory(payload: str):
        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return payload

        return lambda _path: _Ctx()

    monkeypatch.setattr("builtins.open", _mock_open_factory("4,68\n"))
    assert ml4_hooks.smt_siblings(4) == {4, 68}

    monkeypatch.setattr("builtins.open", _mock_open_factory("8-11\n"))
    assert ml4_hooks.smt_siblings(9) == {8, 9, 10, 11}

    def _raise_oserror(_path):
        raise OSError("missing sysfs")

    monkeypatch.setattr("builtins.open", _raise_oserror)
    assert ml4_hooks.smt_siblings(5) == {5}


def test_physical_gpu_index_numeric_unset_and_uuid_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert ml4_hooks.physical_gpu_index() == 1

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert ml4_hooks.physical_gpu_index() == 5

    # UUID mask on Slurm node falls back to OMPI_COMM_WORLD_LOCAL_RANK / SLURM_LOCALID / mapping.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa-bbbb-cccc")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    for var in ("OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "LOCAL_RANK", "PMI_LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)
    assert ml4_hooks.physical_gpu_index(fallback_local_rank=3) == 3

    monkeypatch.setenv("SLURM_LOCALID", "2")
    assert ml4_hooks.physical_gpu_index(fallback_local_rank=3) == 2

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ml4_hooks.physical_gpu_index() is None


def test_pin_executor_cpu_skips_cpu0_and_wraps_across_ranks(monkeypatch):
    # 4 physical cores (0..3) with SMT siblings (64..67). Core 0/64 must be skipped.
    monkeypatch.setattr(ml4_hooks, "gpu_numa_node", lambda *_a, **_kw: None)
    monkeypatch.setattr(ml4_hooks, "smt_siblings", lambda cpu: {cpu % 64, (cpu % 64) + 64})
    available = {0, 1, 2, 3, 64, 65, 66, 67}
    assert ml4_hooks.pin_executor_cpu(available, gpu_index=0) == 1
    assert ml4_hooks.pin_executor_cpu(available, gpu_index=1) == 2
    assert ml4_hooks.pin_executor_cpu(available, gpu_index=2) == 3
    assert ml4_hooks.pin_executor_cpu(available, gpu_index=3) == 1  # wraps over non-0 cores

    # Single-core fallback when only CPU 0's core is available.
    assert ml4_hooks.pin_executor_cpu({0, 64}, gpu_index=0) == 0
    assert ml4_hooks.pin_executor_cpu(set(), gpu_index=0) is None


def test_gpu_numa_node_and_numa_node_cpus_parse_sysfs(monkeypatch):
    files = {
        "/sys/bus/pci/devices/0002:81:00.0/numa_node": "0\n",
        "/sys/bus/pci/devices/0012:81:00.0/numa_node": "1\n",
        "/sys/bus/pci/devices/0022:81:00.0/numa_node": "-1\n",
        "/sys/devices/system/node/node0/cpulist": "0-87,176-263\n",
        "/sys/devices/system/node/node1/cpulist": "88-175,264-351\n",
    }

    class _Ctx:
        def __init__(self, payload: str) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> str:
            return self._payload

    def _fake_open(path, *_args, **_kwargs):
        if path in files:
            return _Ctx(files[path])
        raise OSError(f"missing {path}")

    props_by_dev = {
        0: types.SimpleNamespace(pci_domain_id=0x0002, pci_bus_id=0x81, pci_device_id=0x00),
        2: types.SimpleNamespace(pci_domain_id=0x0012, pci_bus_id=0x81, pci_device_id=0x00),
        3: types.SimpleNamespace(pci_domain_id=0x0022, pci_bus_id=0x81, pci_device_id=0x00),
    }
    monkeypatch.setattr("builtins.open", _fake_open)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: props_by_dev[d])

    assert ml4_hooks.gpu_numa_node(0) == 0
    assert ml4_hooks.gpu_numa_node() == 1
    assert ml4_hooks.gpu_numa_node(3) is None

    numa0 = ml4_hooks.numa_node_cpus(0)
    numa1 = ml4_hooks.numa_node_cpus(1)
    assert numa0 == set(range(0, 88)) | set(range(176, 264))
    assert numa1 == set(range(88, 176)) | set(range(264, 352))
    assert ml4_hooks.numa_node_cpus(-1) is None
    assert ml4_hooks.numa_node_cpus(99) is None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ml4_hooks.gpu_numa_node() is None


def test_pin_executor_cpu_is_numa_aware_for_vr200_4gpu_2numa(monkeypatch):
    # VR200 topology: 176 physical cores (88/NUMA) x 2 SMT threads = 352 logical CPUs.
    # GPU 0/1 live on NUMA 0 (0..87, 176..263); GPU 2/3 live on NUMA 1 (88..175, 264..351).
    numa0 = set(range(0, 88)) | set(range(176, 264))
    numa1 = set(range(88, 176)) | set(range(264, 352))
    all_cpus = numa0 | numa1
    gpu_to_numa = {0: 0, 1: 0, 2: 1, 3: 1}
    numa_to_cpus = {0: numa0, 1: numa1}

    monkeypatch.setattr(ml4_hooks, "smt_siblings", lambda cpu: {cpu % 176, (cpu % 176) + 176})
    monkeypatch.setattr(ml4_hooks, "numa_node_cpus", lambda n: numa_to_cpus.get(n))

    pinned: dict[int, int] = {}
    for gpu in range(4):
        monkeypatch.setattr(ml4_hooks, "gpu_numa_node", lambda _d=None, g=gpu: gpu_to_numa[g])
        cpu = ml4_hooks.pin_executor_cpu(all_cpus, gpu_index=gpu)
        assert cpu is not None
        pinned[gpu] = cpu

    # GPU 0/1 pinned to distinct non-0 physical cores on NUMA 0; GPU 2/3 strictly on NUMA 1.
    assert pinned[0] in range(1, 88)
    assert pinned[1] in range(1, 88)
    assert pinned[0] != pinned[1]
    assert pinned[2] in range(88, 176)
    assert pinned[3] in range(88, 176)
    assert pinned[2] != pinned[3]
    assert len(set(pinned.values())) == 4

    # When external affinity mask does not intersect the GPU's NUMA node, fall back safely.
    monkeypatch.setattr(ml4_hooks, "gpu_numa_node", lambda _d=None: 1)
    assert ml4_hooks.pin_executor_cpu({0, 1, 2, 176, 177, 178}, gpu_index=2) in {1, 2}


def test_executor_thread_pin_moves_other_threads_off_reserved_smt_group(monkeypatch):
    monkeypatch.setattr(ml4_hooks, "gpu_numa_node", lambda *_a, **_kw: None)
    monkeypatch.setattr(ml4_hooks, "smt_siblings", lambda cpu: {cpu, cpu + 64})
    affinities = {
        0: {0, 1, 2, 3, 64, 65, 66, 67},
        100: {0, 1, 2, 3, 64, 65, 66, 67},  # self tid
        101: {0, 1, 2, 3, 64, 65, 66, 67},  # sibling thread -> loses {2, 66}
        102: {2, 66},  # would become empty -> untouched
    }
    monkeypatch.setattr(ml4_hooks.threading, "get_native_id", lambda: 100)
    monkeypatch.setattr(os, "sched_getaffinity", lambda tid: set(affinities[tid]))

    def _set_aff(tid: int, mask: set[int]) -> None:
        affinities[tid] = set(mask)

    monkeypatch.setattr(os, "sched_setaffinity", _set_aff)
    monkeypatch.setattr(os, "listdir", lambda _p: ["100", "101", "102"])

    pin = ml4_hooks.ExecutorThreadPin.create(gpu_index=1, refresh_iters=4)
    assert pin is not None
    assert pin.cpu == 2
    assert pin.numa_node is None
    assert affinities[0] == {2}
    assert affinities[101] == {0, 1, 3, 64, 65, 67}
    assert affinities[102] == {2, 66}

    # Newly spawned thread 103 is moved on the refresh cadence.
    affinities[103] = {0, 2, 66}
    monkeypatch.setattr(os, "listdir", lambda _p: ["100", "101", "102", "103"])
    pin.maybe_refresh(1)
    assert affinities[103] == {0, 2, 66}
    pin.maybe_refresh(4)
    assert affinities[103] == {0}


def test_executor_thread_pin_create_respects_numa_node(monkeypatch):
    numa0 = set(range(0, 88)) | set(range(176, 264))
    numa1 = set(range(88, 176)) | set(range(264, 352))
    all_cpus = numa0 | numa1
    affinities = {0: set(all_cpus), 200: set(all_cpus), 201: set(all_cpus)}

    monkeypatch.setattr(ml4_hooks, "gpu_numa_node", lambda *_a, **_kw: 1)
    monkeypatch.setattr(ml4_hooks, "numa_node_cpus", lambda n: numa1 if n == 1 else numa0)
    monkeypatch.setattr(ml4_hooks, "smt_siblings", lambda cpu: {cpu % 176, (cpu % 176) + 176})
    monkeypatch.setattr(ml4_hooks.threading, "get_native_id", lambda: 200)
    monkeypatch.setattr(os, "sched_getaffinity", lambda tid: set(affinities[tid]))
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda tid, mask: affinities.__setitem__(tid, set(mask))
    )
    monkeypatch.setattr(os, "listdir", lambda _p: ["200", "201"])

    pin = ml4_hooks.ExecutorThreadPin.create(gpu_index=2, refresh_iters=8)
    assert pin is not None
    assert pin.numa_node == 1
    assert pin.cpu == 90
    assert affinities[0] == {90}
    assert 90 not in affinities[201] and 266 not in affinities[201]
    assert len(affinities[201]) == len(all_cpus) - 2


def test_ml4_executor_events_sets_and_restores_gil_and_gc(monkeypatch):
    orig_interval = sys.getswitchinterval()
    orig_gc = gc.isenabled()
    if not orig_gc:
        gc.enable()

    pin_created = []
    fake_pin = types.SimpleNamespace(
        refreshes=[], maybe_refresh=lambda it: fake_pin.refreshes.append(it)
    )
    monkeypatch.setattr(
        ml4_hooks.ExecutorThreadPin,
        "create",
        classmethod(lambda cls, **kw: (pin_created.append(kw), fake_pin)[1]),
    )

    events = ml4_hooks.Ml4UllExecutorEvents(
        pin_refresh_iters=128,
        gil_switch_us=200.0,
        local_rank=2,
    )
    try:
        events.after_warmup()
        events.before_overlap_loop()
        assert pin_created == [{"fallback_local_rank": 2, "refresh_iters": 128}]
        assert abs(sys.getswitchinterval() - 200e-6) < 1e-9
        assert gc.isenabled() is False

        events.on_iteration(0)
        events.on_iteration(128)
        assert fake_pin.refreshes == [0, 128]

        events.at_shutdown()
        assert abs(sys.getswitchinterval() - orig_interval) < 1e-9
        assert gc.isenabled() is True
    finally:
        sys.setswitchinterval(orig_interval)
        if orig_gc:
            gc.enable()
        else:
            gc.disable()


def test_fused_spec_offsets_op_declines_cpu_or_mismatched_tensors():
    pos = torch.zeros(4, dtype=torch.int32)
    pos_off = torch.ones(4, dtype=torch.int32)
    kv = torch.zeros(1, dtype=torch.int32)
    kv_off = torch.ones(1, dtype=torch.int32)
    assert ml4_hooks.fused_spec_offsets_op(pos, pos_off, kv, kv_off) is False


@pytest.mark.parametrize("tp_size", [4, 8, 16])
def test_make_ml4_runtime_hooks_enables_all_low_latency_hooks(tp_size):
    mapping = types.SimpleNamespace(tp_size=tp_size, world_size=tp_size, local_rank=tp_size % 4)
    hooks = ml4_hooks.make_ml4_runtime_hooks(mapping)
    assert isinstance(hooks, RuntimeHooks)
    assert hooks.pack_step_inputs is True
    assert hooks.pack_parallel_scatter is True
    assert hooks.pack_in_graph is True
    assert hooks.pack_early_copy is True
    assert hooks.steady_spec_prepare is True
    assert hooks.kv_lens_update_once is True
    assert hooks.spec_offsets_op is ml4_hooks.fused_spec_offsets_op
    assert hooks.skip_unchanged_h2d is True
    assert hooks.h2d_changed_span is True
    assert hooks.loop_on_execution_stream is True
    assert hooks.iteration_timing_one_event is True
    assert hooks.perf_metrics_timing_side_stream is True
    assert isinstance(hooks.executor_events, ml4_hooks.Ml4UllExecutorEvents)
    assert hooks.executor_events.local_rank == tp_size % 4
