# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VR200 test for the layer_norm_fp8_block_scale_linear catalog entry: its quantized operands
against the separate `layer_norm` and `fp8_quantize_1x128_packed_ue8m0`, byte for byte, and its
output against the reference."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cute_dsl_fp8_block_scale_linear import (
    prepare_weight_scale,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.layer_norm_fp8_block_scale_linear import (
    layer_norm_fp8_block_scale_linear as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.layer_norm_fp8_block_scale_linear import (
    layer_norm_fp8_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.layer_norm import layer_norm
from tensorrt_llm.quantization.utils.fp8_utils import get_col_major_tma_aligned_packed_tensor

__extra_import_path__ = [".."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires a VR200/SM107 device",
)


def _inputs(spec, rows: int, seed: int):
    """Rows shaped like attention-normed hidden states, a quarter each at 1e-6, 1e-4, 1e-2 and 1
    times unit variance around a per-row offset (so the norm's centering and eps both matter),
    the indexer q_input_norm's affine ranges (weights about 0.89..1.14, biases within 0.09),
    and an FP8 weight under power-of-two K128 block scales."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, n_out = spec["n"], spec["n_out"]
    spread = torch.tensor([1e-3, 1e-2, 1e-1, 1.0], device="cuda").repeat((rows + 3) // 4)[:rows]
    offset = torch.randn(rows, 1, generator=g, device="cuda")
    x = (torch.randn(rows, n, generator=g, device="cuda") * spread[:, None] + offset).to(
        torch.bfloat16
    )
    norm_weight = (1.0 + 0.05 * torch.randn(n, generator=g, device="cuda")).clamp(0.89, 1.14)
    norm_bias = (0.03 * torch.randn(n, generator=g, device="cuda")).clamp(-0.09, 0.09)
    weight = torch.randn(n_out, n, generator=g, device="cuda").to(torch.float8_e4m3fn)
    exponent = torch.randint(-6, 4, (n_out // 128, n // 128), device="cuda", generator=g)
    block_scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    packed = get_col_major_tma_aligned_packed_tensor(block_scale.repeat_interleave(128, dim=0))
    return (
        x,
        norm_weight.to(torch.bfloat16),
        norm_bias.to(torch.bfloat16),
        spec["eps"],
        weight,
        prepare_weight_scale(packed, n),
    )


def _logical_scale(scale: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """uint8 `[rows, k / 32]` K32 scale slots out of an R128c4 buffer."""
    padded = (rows + 127) // 128 * 128
    return (
        scale[: padded * k // 32]
        .reshape(padded // 128, k // 128, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(padded, k // 32)[:rows]
    )


def _check_rows(spec, rows: int, seed: int) -> None:
    x, norm_weight, norm_bias, eps, weight, weight_scale = _inputs(spec, rows, seed)
    codes, scale = layer_norm_fp8_quantize(x, norm_weight, norm_bias, eps)
    normed = layer_norm(x, norm_weight, norm_bias, eps)
    ref_codes, ref_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
        normed, use_r128c4_layout=True
    )
    k = x.shape[1]
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8)), (
        f"rows {rows}: codes differ from the separate ops' on "
        f"{int((codes.view(torch.uint8) != ref_codes.view(torch.uint8)).sum())} elements"
    )
    assert torch.equal(_logical_scale(scale, rows, k), _logical_scale(ref_scale, rows, k)), (
        f"rows {rows}: scale bytes differ from the separate ops'"
    )
    padded = (rows + 127) // 128 * 128
    tail = scale.reshape(padded // 128, k // 128, 32, 4, 4).permute(0, 3, 2, 1, 4)
    assert not bool(tail.reshape(padded, k // 32)[rows:].any()), "padding rows hold a scale"
    out = op(x, norm_weight, norm_bias, eps, weight, weight_scale)
    # FP8 products are exact in fp32; accumulation order can straddle a bf16 rounding boundary.
    # The FP8 linear's band: four elements, two RMS.
    assert_within_ulp(out, op.reference(x, norm_weight, norm_bias, eps, weight, weight_scale), 4, 2)


def _check_graph(spec, rows: int, seed: int) -> None:
    """Capture at `rows` after one warm-up call, replay on new data, require the eager result."""
    static = _inputs(spec, rows, seed)
    fresh = _inputs(spec, rows, seed + 1)
    op(*static)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        quantized = layer_norm_fp8_quantize(*static[:4])
    torch.cuda.current_stream().wait_stream(stream)
    for i in (0, 1, 2):
        static[i].copy_(fresh[i])
    graph.replay()
    expected = layer_norm_fp8_quantize(*fresh[:4])
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
            _check_rows(spec, rows, seed + rows)
    for rows in spec.get("graph_rows", ()):
        _check_rows(spec, rows, seed + rows)
        _check_graph(spec, rows, seed + 1000 + rows)
