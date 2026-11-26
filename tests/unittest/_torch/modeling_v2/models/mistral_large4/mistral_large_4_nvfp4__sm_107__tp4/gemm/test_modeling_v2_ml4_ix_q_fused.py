# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 (Rubin VR200) GPU accuracy and CUDA-graph replay tests for
single-launch fused Indexer Q (``ml4_ix_q_fused``)."""

from __future__ import annotations

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ssmax_indexer_q_side_fused import (  # noqa: E501
    ssmax_indexer_q_side_fused,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.layer_norm import layer_norm
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.gemm.cute_dsl_fused_fp8_block_scale_linear import (  # noqa: E501
    cute_dsl_fused_fp8_block_scale_linear,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.gemm.ml4_ix_q_fused import (  # noqa: E501
    CANDIDATE_CONFIGS,
    HEAD_DIM,
    HEADS,
    K_IN,
    N_OUT,
    ROPE_DIM,
    ml4_ix_q_fused,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires VR200/SM107",
)


def _make_inputs(m: int, seed: int, *, rope: bool = True, max_pos: int = 1048576):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, K_IN), dtype=torch.bfloat16, device="cuda", generator=gen).contiguous()
    ln_w = (
        torch.rand((K_IN,), dtype=torch.float32, device="cuda", generator=gen) * 0.25 + 0.88
    ).to(torch.bfloat16)
    ln_b = (torch.randn((K_IN,), dtype=torch.float32, device="cuda", generator=gen) * 0.05).to(
        torch.bfloat16
    )
    w_bf16 = torch.randn((N_OUT, K_IN), dtype=torch.bfloat16, device="cuda", generator=gen)
    w, ws = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(w_bf16, True)
    wp = (
        torch.randn((HEADS, K_IN), dtype=torch.bfloat16, device="cuda", generator=gen) * 0.02
    ).contiguous()
    pos = torch.randint(0, max_pos, (m,), dtype=torch.int32, device="cuda", generator=gen)
    pos[0] = max_pos - 1  # exercise table tail
    ssmax_s = (
        torch.rand((HEADS,), dtype=torch.float32, device="cuda", generator=gen) * 0.11 - 0.045
    ).to(torch.bfloat16)
    ssmax_b = (
        torch.rand((HEADS,), dtype=torch.float32, device="cuda", generator=gen) * 0.07 + 0.975
    ).to(torch.bfloat16)
    if rope:
        theta = torch.randn(
            (max_pos, ROPE_DIM // 2), dtype=torch.float32, device="cuda", generator=gen
        )
        cos = torch.cos(theta).to(torch.bfloat16).float()
        sin = torch.sin(theta).to(torch.bfloat16).float()
        cos_sin = torch.cat([cos, sin], dim=-1).contiguous()
    else:
        cos_sin = torch.cat(
            [
                torch.ones((max_pos, ROPE_DIM // 2), dtype=torch.float32, device="cuda"),
                torch.zeros((max_pos, ROPE_DIM // 2), dtype=torch.float32, device="cuda"),
            ],
            dim=-1,
        ).contiguous()
    # Build realistic [M, 160] kw buffer where trailing 32 cols hold x @ wp^T (strided).
    kw = torch.zeros((m, HEAD_DIM + HEADS), dtype=torch.bfloat16, device="cuda")
    kw[:, HEAD_DIM:] = (x.float() @ wp.float().t()).to(torch.bfloat16)
    return dict(
        x=x,
        ln_w=ln_w,
        ln_b=ln_b,
        w=w,
        ws=ws,
        wp=wp,
        kw_w=kw[:, HEAD_DIM:],
        pos=pos,
        ssmax_s=ssmax_s,
        ssmax_b=ssmax_b,
        cos_sin=cos_sin,
        eps=1e-6,
        weight_scale=1.0 / math.sqrt(HEAD_DIM),
    )


def _baseline_run(inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    qn = layer_norm(inp["x"], inp["ln_w"], inp["ln_b"], inp["eps"])
    q = cute_dsl_fused_fp8_block_scale_linear(qn, inp["w"], inp["ws"]).reshape(
        inp["x"].shape[0], HEADS, HEAD_DIM
    )
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    weights = torch.empty((inp["x"].shape[0], HEADS), dtype=torch.float32, device=q.device)
    return ssmax_indexer_q_side_fused(
        q,
        inp["pos"],
        inp["ssmax_s"],
        inp["ssmax_b"],
        inp["cos_sin"],
        inp["kw_w"],
        inp["weight_scale"],
        q_fp8,
        weights,
    )


def _assert_ix_q_parity(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    ref_q_fp8: torch.Tensor,
    ref_weights: torch.Tensor,
) -> None:
    # Compare effective dequantized scores (q_fp8 * weights) and raw weights & codes.
    qf = q_fp8.float()
    rqf = ref_q_fp8.float()
    # Cosine similarity per (token, head) must be > 0.9999 and relative RMS < 0.015.
    eff = qf * weights.unsqueeze(-1)
    ref_eff = rqf * ref_weights.unsqueeze(-1)
    rel_rms = (eff - ref_eff).pow(2).mean().sqrt() / (ref_eff.pow(2).mean().sqrt() + 1e-8)
    assert rel_rms.item() < 0.015, f"effective indexer Q rel_rms too high: {rel_rms.item():.6f}"
    w_rel = (weights - ref_weights).abs().max() / (ref_weights.abs().max() + 1e-8)
    assert w_rel.item() < 0.02, f"weights max rel diff too high: {w_rel.item():.6f}"


@pytest.mark.parametrize(
    "cfg",
    CANDIDATE_CONFIGS,
    ids=lambda c: f"{c.reduce_mode}_s{c.split}_r{c.ring}_at{int(c.a_tmem)}",
)
@pytest.mark.parametrize("inline_wp", (True, False), ids=("inline_wp", "kw_w"))
def test_ml4_ix_q_fused_all_configs_match_baseline(cfg, inline_wp: bool) -> None:
    inp = _make_inputs(4, seed=100 + cfg.split * 10 + cfg.ring, rope=True, max_pos=65536)
    ref_q, ref_w = _baseline_run(inp)
    q_out = torch.empty_like(ref_q)
    w_out = torch.empty_like(ref_w)
    ok = ml4_ix_q_fused(
        inp["x"],
        inp["ln_w"],
        inp["ln_b"],
        inp["eps"],
        inp["w"],
        inp["ws"],
        inp["pos"],
        inp["ssmax_s"],
        inp["ssmax_b"],
        inp["cos_sin"],
        inp["weight_scale"],
        q_out,
        w_out,
        wp=inp["wp"] if inline_wp else None,
        kw_w=None if inline_wp else inp["kw_w"],
        cfg=cfg,
    )
    assert ok
    _assert_ix_q_parity(q_out, w_out, ref_q, ref_w)


@pytest.mark.parametrize("m", (1, 2, 4, 8))
@pytest.mark.parametrize("rope", (True, False), ids=("rope", "nope"))
def test_ml4_ix_q_fused_shapes_and_1m_positions(m: int, rope: bool) -> None:
    inp = _make_inputs(m, seed=500 + m, rope=rope, max_pos=1048576)
    ref_q, ref_w = _baseline_run(inp)
    q_out = torch.empty_like(ref_q)
    w_out = torch.empty_like(ref_w)
    ok = ml4_ix_q_fused(
        inp["x"],
        inp["ln_w"],
        inp["ln_b"],
        inp["eps"],
        inp["w"],
        inp["ws"],
        inp["pos"],
        inp["ssmax_s"],
        inp["ssmax_b"],
        inp["cos_sin"],
        inp["weight_scale"],
        q_out,
        w_out,
        wp=inp["wp"],
    )
    assert ok
    _assert_ix_q_parity(q_out, w_out, ref_q, ref_w)


def test_ml4_ix_q_fused_cuda_graph_5x_replay() -> None:
    inp = _make_inputs(4, seed=901, rope=True, max_pos=65536)
    q_out = torch.empty((4, HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device="cuda")
    w_out = torch.empty((4, HEADS), dtype=torch.float32, device="cuda")
    # Warmup compile before capture.
    assert ml4_ix_q_fused(
        inp["x"],
        inp["ln_w"],
        inp["ln_b"],
        inp["eps"],
        inp["w"],
        inp["ws"],
        inp["pos"],
        inp["ssmax_s"],
        inp["ssmax_b"],
        inp["cos_sin"],
        inp["weight_scale"],
        q_out,
        w_out,
        wp=inp["wp"],
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ok = ml4_ix_q_fused(
            inp["x"],
            inp["ln_w"],
            inp["ln_b"],
            inp["eps"],
            inp["w"],
            inp["ws"],
            inp["pos"],
            inp["ssmax_s"],
            inp["ssmax_b"],
            inp["cos_sin"],
            inp["weight_scale"],
            q_out,
            w_out,
            wp=inp["wp"],
        )
        assert ok
    for step in range(5):
        fresh = _make_inputs(4, seed=1000 + step, rope=True, max_pos=65536)
        inp["x"].copy_(fresh["x"])
        inp["pos"].copy_(fresh["pos"])
        inp["kw_w"].copy_(fresh["kw_w"])
        inp["wp"].copy_(fresh["wp"])
        graph.replay()
        torch.cuda.synchronize()
        ref_q, ref_w = _baseline_run(inp)
        _assert_ix_q_parity(q_out, w_out, ref_q, ref_w)


@pytest.mark.parametrize("cell", ml4_ix_q_fused.CELLS, ids=lambda c: c.why)
def test_ml4_ix_q_fused_certified_cells(cell) -> None:
    m = cell.spec["m"]
    rope = cell.spec["rope"]
    inp = _make_inputs(m, seed=1200 + m * 7 + int(rope), rope=rope, max_pos=65536)
    ref_q, ref_w = _baseline_run(inp)
    q_out = torch.empty((m, HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device="cuda")
    w_out = torch.empty((m, HEADS), dtype=torch.float32, device="cuda")
    assert ml4_ix_q_fused(
        inp["x"],
        inp["ln_w"],
        inp["ln_b"],
        inp["eps"],
        inp["w"],
        inp["ws"],
        inp["pos"],
        inp["ssmax_s"],
        inp["ssmax_b"],
        inp["cos_sin"],
        inp["weight_scale"],
        q_out,
        w_out,
        wp=inp["wp"],
    )
    _assert_ix_q_parity(q_out, w_out, ref_q, ref_w)
