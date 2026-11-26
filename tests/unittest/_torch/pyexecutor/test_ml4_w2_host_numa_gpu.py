# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU unit tests for NUMA-local pinned allocation and incremental block-offset updates
(including packed in-graph StepInputPack replay across page boundaries and request switches)."""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import h2d_cache, numa_pinned, step_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


def test_numa_pinned_zeros_lands_on_gpu_local_numa_node() -> None:
    expected = numa_pinned.gpu_numa_node()
    buf = numa_pinned.numa_pinned_zeros(1, 16, 2, 24532, dtype=torch.int32)
    assert buf.is_pinned()
    assert int(buf.sum().item()) == 0
    if expected is not None:
        actual = numa_pinned.pointer_numa_node(buf.data_ptr())
        if actual is not None:
            assert actual == expected, f"expected pinned buffer on NUMA {expected}, got {actual}"


def test_incremental_block_offsets_packed_in_graph_replay_page_crossings_and_switches() -> None:
    dev = torch.device("cuda")
    bad = -1
    scales = [72]
    kv_offs = [5]
    num_blocks = 24532

    host_src = numa_pinned.numa_pinned_zeros(1, 8, 2, num_blocks, dtype=torch.int32)
    host_src.fill_(bad)
    host_src[0, 0, 0, :2048] = torch.arange(10, 2058, dtype=torch.int32)
    host_src[0, 3, 0, :128] = torch.arange(500, 628, dtype=torch.int32)

    dst = torch.zeros((1, 2, 2, num_blocks), dtype=torch.int32, device=dev)
    dst_ref = torch.zeros_like(dst)
    full_calls = [0]

    def _ref_compute(slots: list[int], target: torch.Tensor) -> None:
        for s, sl in enumerate(slots):
            raw = host_src[0, sl, 0, :].to(dev)
            target[0, s, 0, :].copy_(torch.where(raw == bad, 0, raw * scales[0]))
            target[0, s, 1, :].copy_(torch.where(raw == bad, 0, raw * scales[0] + kv_offs[0]))

    def _fallback(slots: list[int]) -> None:
        full_calls[0] += 1
        _ref_compute(slots, dst)

    pack = step_inputs.StepInputPack(dev, parallel_scatter=True, in_graph=True, early_copy=True)
    cache = h2d_cache.H2DSkipCache(changed_span=True, incremental_block_offsets=True)

    # Capture a tiny graph whose prologue scatters packed incremental updates and reads dst.
    probe = torch.zeros(4, dtype=torch.int32, device=dev)
    graph = torch.cuda.CUDAGraph()
    with h2d_cache.active(cache), step_inputs.active(pack):
        with torch.cuda.graph(graph):
            step_inputs.capture_prologue()
            probe[0] = dst[0, 0, 0, 2047]
            probe[1] = dst[0, 0, 0, 2048]
            probe[2] = dst[0, 0, 1, 2048]
            probe[3] = dst[0, 0, 0, 5]

        # Step 0: initial full copy for slot [0].
        step_inputs.begin()
        m0 = cache.copy_block_offsets(
            dst, host_src, [0], scales, kv_offs, bad, lambda: _fallback([0]), manager_id=("m", 0)
        )
        step_inputs.flush(in_graph=True)
        graph.replay()
        pack.on_graph_replay()
        torch.cuda.synchronize()
        _ref_compute([0], dst_ref)
        assert m0 == "full" and full_calls[0] == 1
        assert torch.equal(dst[:, :1], dst_ref[:, :1])

        # Step 1: steady unchanged decode -> skip, 0 fallback calls.
        step_inputs.begin()
        m1 = cache.copy_block_offsets(
            dst, host_src, [0], scales, kv_offs, bad, lambda: _fallback([0]), manager_id=("m", 0)
        )
        step_inputs.flush(in_graph=True)
        graph.replay()
        pack.on_graph_replay()
        torch.cuda.synchronize()
        assert m1 == "skip" and full_calls[0] == 1
        assert torch.equal(dst[:, :1], dst_ref[:, :1])

        # Step 2: page boundary crossing (+1 new block at col 2048) -> packed in-graph incremental!
        host_src[0, 0, 0, 2048] = 777
        step_inputs.begin()
        m2 = cache.copy_block_offsets(
            dst, host_src, [0], scales, kv_offs, bad, lambda: _fallback([0]), manager_id=("m", 0)
        )
        step_inputs.flush(in_graph=True)
        graph.replay()
        pack.on_graph_replay()
        torch.cuda.synchronize()
        _ref_compute([0], dst_ref)
        assert m2 == "incremental" and full_calls[0] == 1
        assert torch.equal(dst[:, :1], dst_ref[:, :1])
        assert int(probe[1].item()) == 777 * 72
        assert int(probe[2].item()) == 777 * 72 + 5

        # Step 3: eviction (col 5 -> BAD_PAGE_INDEX) -> packed in-graph incremental sets 0!
        host_src[0, 0, 0, 5] = bad
        step_inputs.begin()
        m3 = cache.copy_block_offsets(
            dst, host_src, [0], scales, kv_offs, bad, lambda: _fallback([0]), manager_id=("m", 0)
        )
        step_inputs.flush(in_graph=True)
        graph.replay()
        pack.on_graph_replay()
        torch.cuda.synchronize()
        _ref_compute([0], dst_ref)
        assert m3 == "incremental" and full_calls[0] == 1
        assert torch.equal(dst[:, :1], dst_ref[:, :1])
        assert int(probe[3].item()) == 0

        # Step 4: request switch to slot [3] -> full fallback.
        step_inputs.begin()
        m4 = cache.copy_block_offsets(
            dst, host_src, [3], scales, kv_offs, bad, lambda: _fallback([3]), manager_id=("m", 0)
        )
        step_inputs.flush(in_graph=True)
        graph.replay()
        pack.on_graph_replay()
        torch.cuda.synchronize()
        _ref_compute([3], dst_ref)
        assert m4 == "full" and full_calls[0] == 2
        assert torch.equal(dst[:, :1], dst_ref[:, :1])
