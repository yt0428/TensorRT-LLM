# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 LayerNorm into a K128 FP8 linear: the norm and the activation's 1x128 quantization in
one Triton launch, then the Rubin MXFP8 GEMM, computing ``layer_norm`` followed by
``cute_dsl_fp8_block_scale_linear``."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cute_dsl_fp8_block_scale_linear import (
    cute_dsl_fp8_block_scale_linear,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.layer_norm import layer_norm

#: The one call site's widths: the Mistral Large 4 DSA indexer's q_input_norm over hidden 6144,
#: feeding its wq_b, 32 heads x 128.
HIDDEN = 6144
WQ_B = 4096
#: Rows of one R128c4 scale tile: the scale buffer covers the rows rounded up to it.
_SF_TILE_ROWS = 128


@triton.jit
def _round_bf16(x):
    """fp32 rounded to bf16 (to nearest, ties to even) and widened back, as `layer_norm`'s bf16
    store rounds its row. An inline `cvt.rn.bf16.f32`: the compiler folds a bare
    `.to(bf16).to(fp32)` pair into nothing."""
    r = tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;", "=h,r", [x], dtype=tl.bfloat16, is_pure=True, pack=1
    )
    return r.to(tl.float32)


@triton.jit
def _rcp_approx(x):
    """`rcp.approx.ftz.f32`, the reciprocal the MXFP8 quantizer scales amax by."""
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _layer_norm_fp8_quantize_kernel(
    x_ptr,  # bf16 [rows, N], unit stride over N
    x_st,
    w_ptr,  # [N]
    b_ptr,  # [N]
    codes_ptr,  # e4m3 [rows, N], contiguous
    scale_ptr,  # int32 words of the uint8 R128c4 scale, [ceil(rows / 128) * 128 * N / 128]
    rows,
    N,  # a runtime value, as `layer_norm`'s kernel takes it
    eps,
    BLOCK_N: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    BLOCKS: tl.constexpr = BLOCK_N // 128
    kb = tl.arange(0, BLOCKS)
    # R128c4 is [rows / 128, N / 128, 32, 4, 4] bytes: a row's K128 exponent fills its four K32
    # slots, one int32 word at row % 32, (row % 128) // 32 of its K128 tile.
    word = (row // 128) * (N // 128) * 128 + kb * 128 + (row % 32) * 4 + (row % 128) // 32
    if USE_PDL:
        gdc_wait()
    if row < rows:
        # `layer_norm`'s kernel, line for line: one program per row over BLOCK_N lanes.
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * x_st + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        xbar = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = tl.reshape(_round_bf16(x_hat * w + b), [BLOCKS, 128])
        # The MXFP8 quantizer's K128 scale: amax floored at 1e-10, times rcp.approx(448), its
        # exponent rounded up (the UE8M0 byte); the codes are the row times the byte's exact
        # inverse power of two, rounded to e4m3.
        amax = tl.maximum(tl.max(tl.abs(y), axis=1), 1.0e-10)
        raw = (amax * _rcp_approx(tl.full([BLOCKS], 448.0, tl.float32))).to(tl.int32, bitcast=True)
        exponent = tl.minimum((raw >> 23) + ((raw & 0x7FFFFF) != 0).to(tl.int32), 254)
        inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
        codes = (y * inverse[:, None]).to(tl.float8e4nv)
        cols2 = tl.reshape(cols, [BLOCKS, 128])
        tl.store(codes_ptr + row * N + cols2, codes, mask=cols2 < N)
        tl.store(scale_ptr + word, exponent * 0x01010101, mask=kb < N // 128)
    else:
        # The tile's padding rows hold zero scale bytes, as the separate quantizer leaves them.
        tl.store(scale_ptr + word, tl.zeros([BLOCKS], tl.int32), mask=kb < N // 128)


def layer_norm_fp8_quantize(
    x: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`fp8_quantize_1x128_packed_ue8m0(layer_norm(x, ...), use_r128c4_layout=True)`: e4m3
    codes `[rows, N]` and the uint8 R128c4 scale (each K128 exponent in its four K32 slots, the
    rows rounded up to 128, the padding rows zero), the two operands the Rubin MXFP8 GEMM
    reads. One launch, one program per row of the padded scale tile."""
    rows, n = x.shape
    codes = torch.empty((rows, n), dtype=torch.float8_e4m3fn, device=x.device)
    padded = (rows + _SF_TILE_ROWS - 1) // _SF_TILE_ROWS * _SF_TILE_ROWS
    scale = torch.empty((padded * n // 32,), dtype=torch.uint8, device=x.device)
    if rows:
        _layer_norm_fp8_quantize_kernel[(padded,)](
            x, x.stride(0), norm_weight, norm_bias, codes, scale.view(torch.int32), rows, n,
            float(eps), BLOCK_N=triton.next_power_of_2(n), USE_PDL=True, num_warps=8,
            launch_pdl=True,
        )  # fmt: skip
    return codes, scale


class LayerNormFp8BlockScaleLinear(OpWrapper):
    """`cute_dsl_fp8_block_scale_linear(layer_norm(x, norm_weight, norm_bias, eps), weight,
    weight_scale)`: LayerNorm over the row (biased variance, fp32 inside, rounded once to bf16),
    the bf16 row quantized per 128 columns to e4m3 under a power-of-two scale, and the FP8
    GEMM against the K128-scaled FP8 weight; bf16 `[rows, N_out]` out.

    `x` is bf16 `[rows, N]` with unit stride over N (any row stride); `norm_weight` and
    `norm_bias` `[N]`; `weight` FP8 `[N_out, N]` and `weight_scale` its R128c4 K32 scale
    (`prepare_weight_scale`). Nothing is mutated.

    Fusion boundary: the norm and the activation quantization share one launch; the GEMM is
    `cute_dsl_mxfp8_gemm_rubin`, the one `cute_dsl_fp8_block_scale_linear` runs. The
    normalized bf16 row is not written: a caller that needs it calls the two entries.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The Mistral Large 4 dep4 decode target's indexer: q_input_norm over hidden 6144 at the
    # checkpoint's indexer_norm_eps 1e-6 into wq_b [4096, 6144], on each of the 44 indexer
    # layers, the rank's own rows: one per request without speculation, DSpark's four per
    # request with it, captured at batches 1..32 (rows 1..128).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep4 decode graphs, rows 1/2/4/8/16/32/64/128: indexer q_input_norm into "
            "wq_b, captured and replayed on new data bitwise equal to eager",
            spec=dict(
                rows=128, n=HIDDEN, n_out=WQ_B, eps=1e-6, graph_rows=(1, 2, 4, 8, 16, 32, 64, 128)
            ),
        ),
        Cell(
            why="dep4 decode, eager rows 1..129: every decode row count, past one scale tile",
            spec=dict(rows=129, n=HIDDEN, n_out=WQ_B, eps=1e-6, sweep_rows=(1, 129)),
        ),
    )

    note = """
    Bits. The quantized operands are the separate ops' bytes: the norm is `layer_norm`'s
    kernel computation (the same block of 8192 lanes at 8 warps, the same expressions), and
    the scale and codes are the MXFP8 quantizer's rule on its rounded row. The test holds the
    codes and the scale buffer bytewise equal to `fp8_quantize_1x128_packed_ue8m0` of
    `layer_norm`'s output, and the GEMM's output within the FP8 linear's ULP band of
    `reference`.

    Cost: one launch replaces two. The grid is the rows rounded up to 128 (the padding
    programs only zero their scale words).

    Not certified: other widths than 6144 (N must be a multiple of 128, and the lanes are the
    next power of two of N); more than 129 rows; NaN or infinite inputs.
    """

    def __call__(
        self,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        eps: float,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        codes, scale = layer_norm_fp8_quantize(x, norm_weight, norm_bias, eps)
        return torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(codes, weight, scale, weight_scale)

    def reference(
        self,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        eps: float,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """The two entries' references in sequence: the fp64 LayerNorm rounded to bf16, then
        the FP8 linear's fp32 product of the quantized operands."""
        normed = layer_norm.reference(x, norm_weight, norm_bias, eps)
        return cute_dsl_fp8_block_scale_linear.reference(normed, weight, weight_scale)

    def is_valid(
        self,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        eps: float,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(-1) == 1, (
            f"x must be 2-D bf16 with unit stride over its last dim; got {x.dtype}, "
            f"{tuple(x.shape)}, strides {tuple(x.stride())}"
        )
        assert x.shape[1] % 128 == 0 and weight.shape[1] == x.shape[1], (
            f"the width must be whole 128-column blocks and the weight's K; got x "
            f"{tuple(x.shape)}, weight {tuple(weight.shape)}"
        )
        assert norm_weight.is_contiguous() and norm_bias.is_contiguous(), (
            "norm_weight and norm_bias are read as dense [N] vectors"
        )


layer_norm_fp8_block_scale_linear = LayerNormFp8BlockScaleLinear()
