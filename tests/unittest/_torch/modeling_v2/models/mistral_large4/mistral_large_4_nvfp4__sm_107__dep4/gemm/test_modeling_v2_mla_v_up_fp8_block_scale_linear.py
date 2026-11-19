# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VR200 test for the sm_107 dep4 target's mla_v_up_fp8_block_scale_linear entry: its quantized
operands against the separate `bmm_out` and `fp8_quantize_1x128_packed_ue8m0`, and its output
against the reference."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cute_dsl_fp8_block_scale_linear import (
    prepare_weight_scale,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__dep4.ops.gemm.mla_v_up_fp8_block_scale_linear import (  # noqa: E501
    HEADS,
    HIDDEN,
    KV_LORA,
    V_DIM,
    v_up_fp8_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__dep4.ops.gemm.mla_v_up_fp8_block_scale_linear import (  # noqa: E501
    mla_v_up_fp8_block_scale_linear as op,
)
from tensorrt_llm.quantization.utils.fp8_utils import get_col_major_tma_aligned_packed_tensor

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires a VR200/SM107 device",
)

#: The share of codes, and of scale bytes, that may differ from the separate ops': the
#: per-head products accumulate in another order than cuBLAS's batched GEMM, so an output on a
#: bf16 rounding boundary rounds the other way now and then. Measured on VR200 over the cells'
#: inputs: see the test's assertion messages.
MAX_CODE_MISMATCH = 1e-3
MAX_SCALE_MISMATCH = 1e-3


def _inputs(rows: int, seed: int):
    """FlashMLA-shaped attention output `[rows, 64, 512]` (FlashMLA's output, its live heads),
    `wkv_b`'s V half as the target views it (`[64 * 128, 512]` transposed per head to
    `[64, 512, 128]`), and an FP8 wo `[6144, 8192]` under power-of-two K128 block scales."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    lat = (torch.randn(rows, HEADS, KV_LORA, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    v_half = (torch.randn(HEADS * V_DIM, KV_LORA, generator=g, device="cuda") * 0.05).to(
        torch.bfloat16
    )
    v_b_t = torch.transpose(v_half.reshape(HEADS, V_DIM, KV_LORA), 1, 2)
    k = HEADS * V_DIM
    weight = torch.randn(HIDDEN, k, generator=g, device="cuda").to(torch.float8_e4m3fn)
    exponent = torch.randint(-6, 4, (HIDDEN // 128, k // 128), device="cuda", generator=g)
    block_scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    packed = get_col_major_tma_aligned_packed_tensor(block_scale.repeat_interleave(128, dim=0))
    return lat, v_b_t, weight, prepare_weight_scale(packed, k)


def _logical_scale(scale: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """uint8 `[rows, k / 32]` K32 scale slots out of an R128c4 buffer."""
    padded = (rows + 127) // 128 * 128
    return (
        scale[: padded * k // 32]
        .reshape(padded // 128, k // 128, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(padded, k // 32)
    )


def _check_rows(rows: int, seed: int) -> None:
    lat, v_b_t, weight, weight_scale = _inputs(rows, seed)
    codes, scale = v_up_fp8_quantize(lat, v_b_t)
    attn = torch.empty(rows, HEADS * V_DIM, dtype=torch.bfloat16, device="cuda")
    bmm_out(
        torch.transpose(lat, 0, 1),
        v_b_t,
        torch.transpose(attn.reshape(rows, HEADS, V_DIM), 0, 1),
    )
    ref_codes, ref_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
        attn, use_r128c4_layout=True
    )
    k = HEADS * V_DIM
    ours, theirs = _logical_scale(scale, rows, k), _logical_scale(ref_scale, rows, k)
    scale_mismatch = (ours[:rows] != theirs[:rows]).float().mean().item()
    assert scale_mismatch <= MAX_SCALE_MISMATCH, f"rows {rows}: {scale_mismatch:.2e} scales differ"
    assert not bool(ours[rows:].any()), "padding rows hold a scale"
    # Under the same scale a code differs by at most one e4m3 step: the ordering of an e4m3
    # value's bits (sign apart) is its magnitude's.
    same_scale = (ours[:rows] == theirs[:rows]).repeat_interleave(32, dim=1)
    a = codes.view(torch.uint8).to(torch.int32)
    b = ref_codes.view(torch.uint8).to(torch.int32)
    differ = (a != b) & same_scale
    code_mismatch = differ.float().mean().item()
    assert code_mismatch <= MAX_CODE_MISMATCH, f"rows {rows}: {code_mismatch:.2e} codes differ"
    step = torch.where((a & 0x80) == (b & 0x80), (a & 0x7F) - (b & 0x7F), (a & 0x7F) + (b & 0x7F))
    assert bool((step[differ].abs() <= 1).all()), f"rows {rows}: a code moved past one step"
    out = op(lat, v_b_t, weight, weight_scale)
    # FP8 products are exact in fp32; accumulation order can straddle a bf16 rounding boundary.
    # The FP8 linear's band: four elements, two RMS.
    assert_within_ulp(out, op.reference(lat, v_b_t, weight, weight_scale), 4, 2)


def _check_graph(rows: int, seed: int) -> None:
    """Capture at `rows` after one warm-up call, replay on new data, require the eager result."""
    static = _inputs(rows, seed)
    fresh = _inputs(rows, seed + 1)
    v_up_fp8_quantize(*static[:2])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        quantized = v_up_fp8_quantize(*static[:2])
    torch.cuda.current_stream().wait_stream(stream)
    for i in (0, 1):
        static[i].copy_(fresh[i])
    graph.replay()
    expected = v_up_fp8_quantize(*fresh[:2])
    torch.cuda.synchronize()
    assert torch.equal(quantized[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(quantized[1], expected[1])


@pytest.mark.parametrize("cell", op.CELLS, ids=[cell.why[:48] for cell in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest fixture
    certifying(op)
    spec = cell.spec
    seed = cell.seed_base()
    if "sweep_rows" in spec:
        lo, hi = spec["sweep_rows"]
        for rows in range(lo, hi + 1):
            _check_rows(rows, seed + rows)
    for rows in spec.get("graph_rows", ()):
        _check_rows(rows, seed + rows)
        _check_graph(rows, seed + 1000 + rows)
