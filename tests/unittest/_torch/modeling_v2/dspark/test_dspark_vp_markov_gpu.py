# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 GPU tests for DSpark ops: split query rope tail, fused embed_draft_block,
vocab-parallel softcap argmax/sampling/RNN-Markov kernels, and spec_tail_kernels CUDA graph replay."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.dspark import draft_ops
from tensorrt_llm._torch.speculative import spec_tail_kernels as stk
from tensorrt_llm._torch.speculative import vocab_parallel as vp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires VR200/SM107",
)

VOCAB = 204800
SOFTCAP = 30.0
HIDDEN = 6144
MARKOV_RANK = 512
LAT = draft_ops.LAT
PE = draft_ops.PE
ROPE_ROWS = 65536


def _rope_table() -> torch.Tensor:
    inv = 1.0 / 10000 ** (torch.arange(0, PE, 2, dtype=torch.float32) / PE)
    angle = torch.arange(ROPE_ROWS, dtype=torch.float32)[:, None] * inv[None, :]
    cos = angle.cos().to(torch.bfloat16).float()
    sin = angle.sin().to(torch.bfloat16).float()
    return torch.cat((cos, sin), dim=1).cuda().contiguous()


@pytest.mark.parametrize(
    "cell",
    draft_ops.triton_query_rope_tail.CELLS,
    ids=[c.why for c in draft_ops.triton_query_rope_tail.CELLS],
)
def test_query_rope_tail_cells_sm107(cell) -> None:
    rows, heads = cell.spec["rows"], cell.spec["heads"]
    nope = 128
    gen = torch.Generator(device="cuda").manual_seed(cell.seed_base())
    q = torch.randn(rows, heads * (nope + PE), device="cuda", generator=gen).to(torch.bfloat16)
    query = torch.randn(rows // 3, 3, heads, LAT + PE, device="cuda", generator=gen).to(
        torch.bfloat16
    )
    positions = torch.randint(0, ROPE_ROWS, (rows,), device="cuda", generator=gen)
    table = _rope_table()

    expected = query.clone()
    draft_ops.triton_query_rope_tail.reference(q, positions, table, expected, nope=nope)
    actual = query.clone()
    draft_ops.triton_query_rope_tail(q, positions, table, actual, nope=nope)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("batch", [1, 4, 16])
def test_embed_draft_block_graph_replay_sm107(batch: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(55 + batch)
    embed = torch.randn(VOCAB, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
    masks = torch.randn(2, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
    anchors = torch.randint(0, VOCAB, (batch,), device="cuda", generator=gen, dtype=torch.int64)
    if batch > 1:
        anchors[0] = -1
        anchors[-1] = VOCAB

    draft_ops.triton_embed_draft_block(anchors, embed, masks, 3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = draft_ops.triton_embed_draft_block(anchors, embed, masks, 3)

    for seed in range(4):
        fresh = torch.randint(
            0,
            VOCAB,
            (batch,),
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        if batch > 1:
            fresh[0] = -1
        anchors.copy_(fresh)
        graph.replay()
        ref = draft_ops.triton_embed_draft_block.reference(anchors, embed, masks, 3)
        assert torch.equal(out, ref)


@pytest.mark.parametrize("tp_size", [4, 8, 16])
@pytest.mark.parametrize("rows", [1, 4, 16])
def test_vocab_parallel_softcap_argmax_cuda_sm107(tp_size: int, rows: int) -> None:
    """Bit-identical softcap argmax on full ML4 vocab=204800 across simulated TP4/TP8/TP16."""
    shard = VOCAB // tp_size
    gen = torch.Generator(device="cuda").manual_seed(1000 + tp_size * 10 + rows)
    full = (torch.randn(rows, VOCAB, device="cuda", generator=gen) * 8.0).to(torch.bfloat16)
    # Plant a saturation tie (300 vs 400 -> both 30.0) across ranks on row 0.
    full[0, :] = 0.0
    full[0, 123] = 300.0
    full[0, VOCAB - 123] = 400.0

    has_sc, inv_cap, cap = vp._softcap_params(SOFTCAP)
    bufs = []
    n_slots = 0
    for r in range(tp_size):
        ex = vp._PairExchange(None, tp_size, r)
        layout = ex.layout(rows, shard, full.device)
        local = full[:, r * shard : (r + 1) * shard].contiguous()
        vp._local_argmax_kernel[(rows, layout.splits)](
            local,
            local.stride(0),
            shard,
            r * shard,
            layout.chunk,
            inv_cap,
            cap,
            layout.buffer,
            layout.buffer.stride(0),
            layout.slot0,
            layout.zero_lo,
            layout.zero_hi,
            HAS_SOFTCAP=has_sc,
            USE_LIBDEVICE=True,
            ZERO_WIDTH=layout.zero_width,
            BLOCK_V=vp._LOCAL_BLOCK_V,
            num_warps=8,
            enable_fp_fusion=False,
        )
        bufs.append(layout.buffer)
        n_slots = tp_size * layout.splits

    summed = torch.stack(bufs, dim=0).sum(dim=0)
    out = torch.empty((rows,), dtype=torch.int64, device="cuda")
    import triton

    vp._reduce_pairs_kernel[(triton.cdiv(rows, vp._REDUCE_BLOCK_ROWS),)](
        summed,
        summed.stride(0),
        rows,
        n_slots,
        out,
        BLOCK_ROWS=vp._REDUCE_BLOCK_ROWS,
        BLOCK_SLOTS=triton.next_power_of_2(n_slots),
        num_warps=4,
    )
    expected = torch.argmax(torch.tanh(full.float() / SOFTCAP) * SOFTCAP, dim=-1)
    assert torch.equal(out, expected)
    assert int(out[0]) == 123


@pytest.mark.parametrize("tp_size", [4, 8, 16])
def test_vocab_parallel_rnn_markov_cuda_sm107(tp_size: int) -> None:
    """Single-GPU TP-simulated ML4 full-vocab (204800, rank=512) VP RNN Markov bitwise check."""
    import triton

    B, K = 2, 3
    shard = VOCAB // tp_size
    gen = torch.Generator(device="cuda").manual_seed(2000 + tp_size)
    raw_block = (torch.randn(B, K, VOCAB, device="cuda", generator=gen) * 3.0).to(torch.bfloat16)
    anchors = torch.tensor([42, -1], device="cuda", dtype=torch.int64)
    hidden = (torch.randn(B, K, HIDDEN, device="cuda", generator=gen) * 0.2).to(torch.bfloat16)
    emb_w = (torch.randn(VOCAB, MARKOV_RANK, device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    joint_w = (
        torch.randn(3 * MARKOV_RANK, HIDDEN + 2 * MARKOV_RANK, device="cuda", generator=gen) * 0.05
    ).to(torch.bfloat16)
    joint_b = (torch.randn(3 * MARKOV_RANK, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    out_w = (torch.randn(VOCAB, MARKOV_RANK, device="cuda", generator=gen) * 0.1).to(torch.bfloat16)

    # Unsharded reference on full vocab.
    base_ref = torch.tanh(raw_block.float() / SOFTCAP) * SOFTCAP
    prev = anchors.clone()
    state = None
    ref_tokens = torch.empty((B, K), dtype=torch.int32, device="cuda")
    ref_refined = torch.empty_like(base_ref)
    for i in range(K):
        jin = draft_ops.triton_markov_input(prev, emb_w, state, hidden[:, i])
        joint = F.linear(jin, joint_w, joint_b)
        state, emitted = draft_ops.triton_markov_gate(joint, state)
        bias = F.linear(emitted, out_w)
        prev = draft_ops.triton_markov_logits(base_ref[:, i], bias, ref_refined[:, i])
        ref_tokens[:, i] = prev.to(torch.int32)

    # Multi-rank simulated TP4/TP8/TP16 VP RNN Markov using the actual Triton kernels.
    has_sc, inv_cap, cap = vp._softcap_params(SOFTCAP)
    rank_states = [None] * tp_size
    rank_tokens = [torch.empty((B, K), dtype=torch.int32, device="cuda") for _ in range(tp_size)]
    rank_joint_in = [hidden.new_empty((B, 2 * MARKOV_RANK + HIDDEN)) for _ in range(tp_size)]
    rank_layouts = [
        vp._PairExchange(None, tp_size, r).layout(B, shard, hidden.device) for r in range(tp_size)
    ]
    pairs = anchors
    n_slots = 1
    for step in range(K + 1):
        build_input = step < K
        step_hid = hidden[:, step] if build_input else hidden[:, 0]
        round_bufs = []
        for r in range(tp_size):
            vp._rnn_markov_reduce_input_kernel[(B,)](
                pairs,
                pairs.stride(0) if step > 0 else 0,
                n_slots,
                anchors,
                emb_w,
                VOCAB,
                rank_states[r] if rank_states[r] is not None else rank_joint_in[r],
                step_hid,
                step_hid.stride(0),
                rank_joint_in[r],
                rank_tokens[r],
                rank_tokens[r].stride(0),
                step,
                RANK=MARKOV_RANK,
                HIDDEN=HIDDEN,
                HID_BLOCK=2048,
                FROM_ANCHOR=step == 0,
                HAS_STATE=rank_states[r] is not None,
                BUILD_INPUT=build_input,
                BLOCK_SLOTS=triton.next_power_of_2(n_slots),
                BLOCK_R=MARKOV_RANK,
                num_warps=4,
            )
            if not build_input:
                continue
            joint = F.linear(rank_joint_in[r], joint_w, joint_b)
            rank_states[r], emitted = draft_ops.triton_markov_gate(joint, rank_states[r])
            local_bias = F.linear(emitted, out_w[r * shard : (r + 1) * shard].contiguous())
            step_logits = raw_block[:, step, r * shard : (r + 1) * shard].contiguous()
            layout = rank_layouts[r]
            vp._markov_add_local_argmax_kernel[(B, layout.splits)](
                step_logits,
                step_logits.stride(0),
                shard,
                r * shard,
                layout.chunk,
                local_bias,
                local_bias.stride(0),
                step_logits,
                step_logits.stride(0),
                inv_cap,
                cap,
                layout.buffer,
                layout.buffer.stride(0),
                layout.slot0,
                layout.zero_lo,
                layout.zero_hi,
                HAS_SOFTCAP=has_sc,
                USE_LIBDEVICE=True,
                STORE_BIASED=False,
                ZERO_WIDTH=layout.zero_width,
                BLOCK_V=vp._LOCAL_BLOCK_V,
                num_warps=8,
                enable_fp_fusion=False,
            )
            round_bufs.append(layout.buffer)
        if build_input:
            pairs = torch.stack(round_bufs, dim=0).sum(dim=0)
            n_slots = tp_size * rank_layouts[0].splits

    for r in range(tp_size):
        assert torch.equal(rank_tokens[r], ref_tokens)


def test_spec_tail_kernels_graph_replay_sm107() -> None:
    """strict_accept + prepare_next_new_tokens capture & replay cleanly on SM107."""
    B, K = 4, 3
    target = torch.randint(0, 1000, (B * (K + 1),), device="cuda", dtype=torch.int64)
    draft = torch.randint(0, 1000, (B, K), device="cuda", dtype=torch.int32)
    next_draft = torch.randint(0, 1000, (B, K), device="cuda", dtype=torch.int32)

    stk.strict_accept(target, draft, 0, B)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        acc, nacc = stk.strict_accept(target, draft, 0, B)
        nxt = stk.prepare_next_new_tokens(acc, next_draft, nacc, B)

    for seed in range(5):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        target.copy_(torch.randint(0, 1000, target.shape, device="cuda", generator=gen))
        draft.copy_(torch.randint(0, 1000, draft.shape, device="cuda", generator=gen).int())
        next_draft.copy_(
            torch.randint(0, 1000, next_draft.shape, device="cuda", generator=gen).int()
        )
        graph.replay()
        exp_acc, exp_nacc = stk.strict_accept(target, draft, 0, B)
        exp_nxt = stk.prepare_next_new_tokens(exp_acc, next_draft, exp_nacc, B)
        assert torch.equal(acc, exp_acc)
        assert torch.equal(nacc, exp_nacc)
        assert torch.equal(nxt, exp_nxt)
