# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 absorbed-MLA output projection: the per-head V up-projection and the K128 FP8
quantization of its output in one Triton launch, then the Rubin MXFP8 GEMM of `wo`. Each head's
128 values are one 1x128 quantization block of `wo`'s input, so a head's block is quantized
where it is computed."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cute_dsl_fp8_block_scale_linear import (
    cute_dsl_fp8_block_scale_linear,
)

#: Mistral Large 4 at dep4: 64 heads, the 512-wide latent, v_head_dim 128, wo [6144, 8192].
HEADS = 64
KV_LORA = 512
V_DIM = 128
HIDDEN = 6144
#: Rows of one R128c4 scale tile: the scale buffer covers the rows rounded up to it.
_SF_TILE_ROWS = 128


@triton.jit
def _round_bf16(x):
    """fp32 rounded to bf16 (to nearest, ties to even) and widened back, as the bf16 output of
    the batched GEMM it replaces rounds. An inline `cvt.rn.bf16.f32`: the compiler folds a bare
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
def _v_up_fp8_quantize_kernel(
    lat_ptr,  # bf16 [rows, heads, KV], unit stride over KV
    lat_st_m,
    lat_st_h,
    v_ptr,  # bf16 [heads, KV, V], any strides
    v_st_h,
    v_st_k,
    v_st_n,
    codes_ptr,  # e4m3 [rows, heads * V], contiguous
    scale_ptr,  # int32 words of the uint8 R128c4 scale, [ceil(rows / 128) * 128 * heads]
    rows,
    heads,
    KV: tl.constexpr,
    V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    h = tl.program_id(0)
    m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.arange(0, V)
    live = m < rows
    # R128c4 is [rows / 128, heads, 32, 4, 4] bytes (one K128 block per head): a row's exponent
    # fills its four K32 slots, one int32 word at row % 32, (row % 128) // 32 of head h's tile.
    word = (m // 128) * heads * 128 + h * 128 + (m % 32) * 4 + (m % 128) // 32
    if USE_PDL:
        gdc_wait()
    if tl.program_id(1) * BLOCK_M < rows:
        acc = tl.zeros([BLOCK_M, V], dtype=tl.float32)
        for k0 in range(0, KV, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                lat_ptr + m[:, None].to(tl.int64) * lat_st_m + h * lat_st_h + k[None, :],
                mask=live[:, None],
                other=0.0,
            )
            b = tl.load(v_ptr + h * v_st_h + k[:, None] * v_st_k + n[None, :] * v_st_n)
            acc = tl.dot(a, b, acc)
        y = _round_bf16(acc)
        # The MXFP8 quantizer's K128 scale: amax floored at 1e-10, times rcp.approx(448), its
        # exponent rounded up (the UE8M0 byte); the codes are the row times the byte's exact
        # inverse power of two, rounded to e4m3.
        amax = tl.maximum(tl.max(tl.abs(y), axis=1), 1.0e-10)
        raw = (amax * _rcp_approx(tl.full([BLOCK_M], 448.0, tl.float32))).to(tl.int32, bitcast=True)
        exponent = tl.minimum((raw >> 23) + ((raw & 0x7FFFFF) != 0).to(tl.int32), 254)
        inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
        codes = (y * inverse[:, None]).to(tl.float8e4nv)
        tl.store(
            codes_ptr + m[:, None].to(tl.int64) * (heads * V) + h * V + n[None, :],
            codes,
            mask=live[:, None],
        )
        tl.store(scale_ptr + word, tl.where(live, exponent * 0x01010101, 0))
    else:
        # The tile's padding rows hold zero scale bytes, as the separate quantizer leaves them.
        tl.store(scale_ptr + word, tl.zeros([BLOCK_M], tl.int32))


def _launch_config(rows: int) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_K, num_warps, num_stages) for `rows`. A program streams its head's
    [512, 128] block of V from HBM, so the time is set by the bytes each program keeps in
    flight: 32 KB K-tiles four deep. Up to 64 rows, two programs per head (16 or 32 rows each)
    beat one taller one; past that, 64-row programs with 8 warps and 16 KB tiles five deep.
    Measured on VR200 over every config of 16..64 rows, 32..256 K, 4 or 8 warps, 2..5 stages."""
    if rows <= 32:
        return 16, 128, 4, 4
    if rows <= 64:
        return 32, 128, 4, 4
    return 64, 64, 8, 5


def v_up_fp8_quantize(lat: torch.Tensor, v_b_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`fp8_quantize_1x128_packed_ue8m0(bmm(lat, v_b_t) as [rows, heads * V])` with the R128c4
    layout: e4m3 codes `[rows, heads * V]` and the uint8 R128c4 scale (each head's exponent in
    its four K32 slots, the rows rounded up to 128, the padding rows zero). `lat` is bf16
    `[rows, heads, KV]` (unit stride over KV), `v_b_t` bf16 `[heads, KV, V]`."""
    rows, heads, kv = lat.shape
    v = v_b_t.shape[2]
    codes = torch.empty((rows, heads * v), dtype=torch.float8_e4m3fn, device=lat.device)
    padded = (rows + _SF_TILE_ROWS - 1) // _SF_TILE_ROWS * _SF_TILE_ROWS
    scale = torch.empty((padded * heads * v // 32,), dtype=torch.uint8, device=lat.device)
    if rows:
        block_m, block_k, warps, stages = _launch_config(rows)
        _v_up_fp8_quantize_kernel[(heads, padded // block_m)](
            lat, lat.stride(0), lat.stride(1), v_b_t, v_b_t.stride(0), v_b_t.stride(1),
            v_b_t.stride(2), codes, scale.view(torch.int32), rows, heads, KV=kv, V=v,
            BLOCK_M=block_m, BLOCK_K=block_k, USE_PDL=True, num_warps=warps,
            num_stages=stages, launch_pdl=True,
        )  # fmt: skip
    return codes, scale


class MlaVUpFp8BlockScaleLinear(OpWrapper):
    """`cute_dsl_fp8_block_scale_linear(attn, weight, weight_scale)` of the absorbed MLA's
    output, `attn[t, h * V:(h + 1) * V] = lat[t, h] @ v_b_t[h]` rounded to bf16: the per-head V
    up-projection (fp32 accumulation, one bf16 rounding, as the batched GEMM it replaces), each
    head's row quantized as its own 1x128 block to e4m3 under a power-of-two scale, and the FP8
    GEMM against the K128-scaled FP8 `weight`; bf16 `[rows, N_out]` out.

    `lat` is bf16 `[rows, heads, KV]` with unit stride over KV (FlashMLA's output, its live
    heads); `v_b_t` bf16 `[heads, KV, V]`, any strides (`wkv_b`'s V half, transposed per head);
    `weight` FP8 `[N_out, heads * V]` and `weight_scale` its R128c4 K32 scale
    (`prepare_weight_scale`). Nothing is mutated.

    Fusion boundary: the V up-projection and the activation quantization share one launch; the
    GEMM is `cute_dsl_mxfp8_gemm_rubin`, the one `cute_dsl_fp8_block_scale_linear` runs. The
    bf16 attention output is not written.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The Mistral Large 4 dep4 decode target's attention output into wo [6144, 8192], on each
    # of the 72 layers, the rank's own rows: one per request without speculation, DSpark's four
    # per request with it, captured at batches 1..32 (rows 1..128).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep4 decode graphs, rows 1/2/4/8/16/32/64/128: the 64-head V up-projection "
            "into wo, captured and replayed on new data bitwise equal to eager",
            spec=dict(rows=128, graph_rows=(1, 2, 4, 8, 16, 32, 64, 128)),
        ),
        Cell(
            why="dep4 decode, eager rows 1..129: every decode row count, past one scale tile",
            spec=dict(rows=129, sweep_rows=(1, 129)),
        ),
    )

    note = """
    Bits. The quantization is the MXFP8 quantizer's rule on the rounded per-head output, so
    on a bf16 output equal to the batched GEMM's the codes and scales are its bytes. The V
    up-projection accumulates in another order than cuBLAS's batched GEMM, so an output on a
    bf16 rounding boundary can round the other way: the test bounds the codes that differ
    from `bmm_out` followed by `fp8_quantize_1x128_packed_ue8m0` (a small fraction, each one
    e4m3 step) and holds the GEMM's output within the FP8 linear's ULP band of `reference`.

    Cost: one launch replaces two. The grid is (heads, the rows rounded up to 128 over the
    rows per program); the padding programs only zero their scale words. Measured on VR200
    with V streaming from HBM: 3.5-3.8 us at 4..32 rows, 4.1 at 64, 4.5 at 128, against
    5.1-6.2 us for `bmm_out` and `fp8_quantize_1x128_packed_ue8m0`.

    Not certified: other than 64 heads of a 512-wide latent and V 128; more than 129 rows;
    NaN or infinite inputs.
    """

    def __call__(
        self,
        lat: torch.Tensor,
        v_b_t: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        codes, scale = v_up_fp8_quantize(lat, v_b_t)
        return torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(codes, weight, scale, weight_scale)

    def reference(
        self,
        lat: torch.Tensor,
        v_b_t: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """The per-head product in fp32 rounded once to bf16, then the FP8 linear's
        reference."""
        with true_fp32_matmul():
            attn = torch.bmm(lat.transpose(0, 1).float(), v_b_t.float()).transpose(0, 1)
        attn = attn.to(torch.bfloat16).reshape(lat.shape[0], -1)
        return cute_dsl_fp8_block_scale_linear.reference(attn, weight, weight_scale)

    def is_valid(
        self,
        lat: torch.Tensor,
        v_b_t: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        assert lat.dtype == torch.bfloat16 and lat.dim() == 3 and lat.stride(-1) == 1, (
            f"lat must be 3-D bf16 with unit stride over the latent; got {lat.dtype}, "
            f"{tuple(lat.shape)}, strides {tuple(lat.stride())}"
        )
        assert v_b_t.dtype == torch.bfloat16 and v_b_t.shape[:2] == lat.shape[1:], (
            f"v_b_t must be bf16 [heads, KV, V] matching lat; got {v_b_t.dtype}, "
            f"{tuple(v_b_t.shape)} against lat {tuple(lat.shape)}"
        )
        assert v_b_t.shape[2] == 128 and weight.shape[1] == lat.shape[1] * 128, (
            "each head's output must be one 128-wide quantization block of the weight's K; got "
            f"V {v_b_t.shape[2]}, weight {tuple(weight.shape)}"
        )


mla_v_up_fp8_block_scale_linear = MlaVUpFp8BlockScaleLinear()
