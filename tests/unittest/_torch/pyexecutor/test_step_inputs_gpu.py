# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU unit tests for pure-Triton ``step_inputs``, ``h2d_cache``, ``steady_spec_prepare``, and
ML4 ``fused_spec_offsets_op`` on CUDA / SM107."""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4 import (
    runtime_hooks as ml4_hooks,
)
from tensorrt_llm._torch.pyexecutor import h2d_cache, step_inputs
from tensorrt_llm._torch.pyexecutor.h2d_cache import (
    H2DSkipCache,
    copy_h2d_if_changed,
    copy_h2d_pool_shared,
    fill_tail_if_changed,
    fill_tail_pool_shared,
)
from tensorrt_llm._torch.pyexecutor.steady_spec_prepare import SteadyMtpPrepare, stage_steady_inputs
from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _host(values, dtype=torch.int32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype).pin_memory()


def _zeros(*shape, dtype=torch.int32) -> torch.Tensor:
    return torch.zeros(*shape, dtype=dtype, device="cuda")


def test_step_input_pack_serial_triton_preserves_overlapping_order():
    """Overlapping H2D and D2D read-after-write fall back to pure-Triton serial scatter."""
    pack = StepInputPack(_device(), parallel_scatter=True)
    buf = _zeros(8)
    with step_inputs.active(pack):
        step_inputs.begin()
        # Overlapping H2D writes: second write must win over [2:6].
        step_inputs.copy_(buf[:6], _host([1, 2, 3, 4, 5, 6]))
        step_inputs.copy_(buf[2:6], _host([20, 30, 40, 50]))
        # D2D RAW chain: buf[6:8] <- buf[4:6] after H2D, then buf[0:2] <- buf[6:8].
        step_inputs.copy_(buf[6:8], buf[4:6])
        step_inputs.copy_(buf[0:2], buf[6:8])
        step_inputs.flush()
    torch.cuda.synchronize()
    assert buf.tolist() == [40, 50, 20, 30, 40, 50, 40, 50]


def test_step_input_pack_parallel_triton_h2d_feeds_d2d():
    """Disjoint H2D copies execute before disjoint D2D copies in parallel Triton scatter."""
    pack = StepInputPack(_device(), parallel_scatter=True)
    a = _zeros(4)
    b = _zeros(4)
    c = _zeros(4)
    with step_inputs.active(pack):
        step_inputs.begin()
        step_inputs.copy_(a, _host([10, 20, 30, 40]))
        step_inputs.copy_(b, _host([5, 6, 7, 8]))
        step_inputs.copy_(c, a)
        step_inputs.flush()
    torch.cuda.synchronize()
    assert a.tolist() == [10, 20, 30, 40]
    assert b.tolist() == [5, 6, 7, 8]
    assert c.tolist() == [10, 20, 30, 40]


@pytest.mark.parametrize("early_copy", [False, True], ids=["in_graph", "in_graph_early_copy"])
def test_step_input_pack_in_graph_cuda_graph_capture_and_replay(early_copy):
    """Capture a real CUDA Graph with StepInputPack's pure-Triton prologue and replay across
    packed steps, empty steps, and step_epoch increments."""
    pack = StepInputPack(
        _device(),
        parallel_scatter=True,
        in_graph=True,
        early_copy=early_copy,
        step_epoch=True,
    )
    assert pack.in_graph is True
    assert pack.early_copy is early_copy
    assert pack.epoch is not None

    x = _zeros(4)
    y = _zeros(4)
    out = _zeros(4)

    # Warmup eager before capture.
    torch.cuda.synchronize()
    epoch_before = int(pack.epoch.item())

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with step_inputs.active(pack):
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            captured = step_inputs.capture_prologue()
            assert captured is pack
            torch.add(x, y, out=out)
    torch.cuda.current_stream().wait_stream(stream)

    with step_inputs.active(pack):
        # Step 1: pack both H2D and D2D (y copied from x) into graph window.
        step_inputs.begin()
        step_inputs.copy_(x, _host([1, 2, 3, 4]))
        step_inputs.copy_(y, x)
        step_inputs.flush(in_graph=True)
        graph.replay()
        pack.on_graph_replay()
        torch.cuda.synchronize()
        assert x.tolist() == [1, 2, 3, 4]
        assert y.tolist() == [1, 2, 3, 4]
        assert out.tolist() == [2, 4, 6, 8]
        assert int(pack.epoch.item()) == epoch_before + 1

        # Step 2: update only x via H2D.
        step_inputs.begin()
        step_inputs.copy_(x, _host([10, 20, 30, 40]))
        step_inputs.flush(in_graph=True)
        graph.replay()
        pack.on_graph_replay()
        torch.cuda.synchronize()
        assert out.tolist() == [11, 22, 33, 44]
        assert int(pack.epoch.item()) == epoch_before + 2

        # Step 3: empty pack quiesces header so stale descriptors do not overwrite x.
        x.fill_(100)
        step_inputs.begin()
        step_inputs.flush(in_graph=True)
        graph.replay()
        pack.on_graph_replay()
        torch.cuda.synchronize()
        assert x.tolist() == [100, 100, 100, 100]
        assert out.tolist() == [101, 102, 103, 104]
        assert int(pack.epoch.item()) == epoch_before + 3


def test_h2d_skip_cache_gpu_changed_span_with_pack():
    """Verify H2DSkipCache(changed_span=True) stages only changed row spans into StepInputPack."""
    pack = StepInputPack(_device(), parallel_scatter=True)
    cache = H2DSkipCache(changed_span=True)
    dst = _zeros(4, 64)
    src = torch.arange(256, dtype=torch.int32).reshape(4, 64).pin_memory()

    with h2d_cache.active(cache), step_inputs.active(pack):
        step_inputs.begin()
        assert copy_h2d_if_changed(dst, src, pack=True) is True
        step_inputs.flush()
        torch.cuda.synchronize()
        assert torch.equal(dst.cpu(), src)

        # Identical upload skipped.
        step_inputs.begin()
        assert copy_h2d_if_changed(dst, src, pack=True) is False
        assert pack.h2d == []
        step_inputs.flush()

        # Mutate 1 element in row 2 -> stages exactly 4 bytes.
        src2 = src.clone().pin_memory()
        src2[2, 19] = 99999
        step_inputs.begin()
        assert copy_h2d_if_changed(dst, src2, pack=True) is True
        assert [nbytes for _, _, nbytes in pack.h2d] == [4]
        step_inputs.flush()
        torch.cuda.synchronize()
        assert torch.equal(dst.cpu(), src2)

        # Pool-shared write always issues and drops shadow.
        assert copy_h2d_pool_shared(dst[:1], _host([[7] * 64])) is True
        fill_tail_if_changed(dst, 2, -1)
        fill_tail_pool_shared(dst, 3, -2)
        torch.cuda.synchronize()
        assert dst[0, 0].item() == 7
        assert dst[2, 0].item() == -1
        assert dst[3, 0].item() == -2


@pytest.mark.parametrize("b,t", [(1, 4), (2, 4)], ids=["ml4_c1_dspark3", "ml4_c2_dspark3"])
def test_steady_spec_prepare_triton_kernel_gpu(b, t):
    """Verify Triton _steady_stage_kernel matches full prepare_inputs for ML4 DSpark (T=4)."""
    slots = [3, 1][:b]
    bases = [1024, 4096][:b]
    positions = [p for base in bases for p in range(base, base + t)]
    n = b * t
    d = t - 1

    gen = torch.Generator().manual_seed(42 + b)
    new_tokens = torch.randint(0, 204800, (t, 8, 1), generator=gen, dtype=torch.int32).cuda()
    next_drafts = torch.randint(0, 204800, (8, d), generator=gen, dtype=torch.int32).cuda()
    new_lens = torch.randint(1, t + 1, (8,), generator=gen, dtype=torch.int32).cuda()
    dev = types.SimpleNamespace(
        new_tokens=new_tokens,
        next_draft_tokens=next_drafts,
        new_tokens_lens=new_lens,
    )

    slab = _host(slots + positions)
    i32 = dict(dtype=torch.int32, device="cuda")
    input_ids = torch.full((16,), -1, **i32)
    draft_tokens = torch.full((16,), -1, **i32)
    prev_batch_idx = torch.full((16,), -1, **i32)
    prev_pos_idx = torch.full((16,), -1, **i32)
    prev_pos_offsets = torch.full((16,), -1, **i32)
    prev_kv_offsets = torch.full((8,), -1, **i32)
    pos_buf = torch.full((16,), -1, **i32)
    gather_ids = torch.full((16,), -1, **i32)
    static_ids = torch.full((16,), -1, **i32)
    static_pos = torch.full((1, 16), -1, **i32)

    stage_steady_inputs(
        slab,
        b,
        t,
        dev,
        input_ids=input_ids,
        draft_tokens=draft_tokens,
        prev_batch_idx=prev_batch_idx,
        prev_pos_idx=prev_pos_idx,
        prev_pos_offsets=prev_pos_offsets,
        prev_kv_offsets=prev_kv_offsets,
        positions=pos_buf,
        mrope_positions=None,
        gather_ids=gather_ids,
        static_input_ids=static_ids,
        static_position_ids=static_pos,
    )
    torch.cuda.synchronize()

    slots_cu = torch.tensor(slots, dtype=torch.int32, device="cuda")
    exp_ids = new_tokens.transpose(0, 1)[slots_cu, :t].flatten()
    exp_drafts = next_drafts[slots_cu, :d].flatten()
    exp_pos_idx = torch.tensor(
        [s for s in slots for _ in range(t)], dtype=torch.int32, device="cuda"
    )

    assert torch.equal(input_ids[:n], exp_ids)
    assert torch.equal(static_ids[:n], exp_ids)
    assert torch.equal(draft_tokens[: b * d], exp_drafts)
    assert torch.equal(prev_batch_idx[:b], slots_cu)
    assert torch.equal(prev_pos_idx[:n], exp_pos_idx)
    assert torch.equal(prev_pos_offsets[:n], new_lens[exp_pos_idx])
    assert torch.equal(prev_kv_offsets[:b], new_lens[slots_cu] - t)
    assert pos_buf[:n].tolist() == positions
    assert static_pos[0, :n].tolist() == positions
    assert gather_ids[:n].tolist() == list(range(n))

    # Exercise SteadyMtpPrepare warmup constructor on GPU.
    prep = SteadyMtpPrepare(
        max_batch_size=8,
        max_num_tokens=32,
        max_tokens_per_step=4,
        use_mrope=False,
        has_static_inputs=True,
        stage_rng=False,
    )
    torch.cuda.synchronize()
    assert prep.block == 32


@pytest.mark.parametrize("n_pos,n_kv", [(4, 1), (32, 8), (300, 75)])
def test_fused_spec_offsets_op_gpu_matches_reference(n_pos, n_kv):
    """Verify ML4 single-launch Triton fused_spec_offsets_op matches PyTorch in-place adds."""
    gen = torch.Generator(device="cuda").manual_seed(123 + n_pos)
    pos = torch.randint(0, 100000, (n_pos,), generator=gen, dtype=torch.int32, device="cuda")
    pos_off = torch.randint(1, 5, (n_pos,), generator=gen, dtype=torch.int32, device="cuda")
    kv = torch.randint(10, 100000, (n_kv,), generator=gen, dtype=torch.int32, device="cuda")
    kv_off = torch.randint(-3, 1, (n_kv,), generator=gen, dtype=torch.int32, device="cuda")

    exp_pos = pos + pos_off
    exp_kv = kv + kv_off
    assert ml4_hooks.fused_spec_offsets_op(pos, pos_off, kv, kv_off) is True
    torch.cuda.synchronize()
    assert torch.equal(pos, exp_pos)
    assert torch.equal(kv, exp_kv)
