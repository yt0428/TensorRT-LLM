# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mistral Large 4 low-M FP8 GEMV CuTe DSL kernel on tcgen05 (SM107 / Rubin VR200).

Single-launch split-K cluster GEMV (``cluster=(split, 1, 1)``, eliminating second-stage reduce
launches) computing ``y[M, N] = quant_1x128(x[M, K]) @ W[N, K]^T`` with:
* Exact ML4 1x128 activation quantization (amax floored at 1e-10, UE8M0 ceil_pow2(amax / 448),
  matching ``fp8_quantize_1x128_packed_ue8m0``),
* Optional fused LayerNorm prologue (``x_layernorm=True``, 2-pass exact mean/var in FP32 + bf16
  rounding for 44-layer ``ix_q_input_norm + ix_wq_b``),
* Optional fused standard SwiGLU prologue (``x_swiglu=True``, ``silu(gate) * up`` for ``dense0`` /
  ``shared`` down projection),
* Optional Paradigm-B single-kernel die-aware Rubin uGPU placement (``ugpu=True``: dual TMA
  descriptors for die-0/die-1 weight halves + 4-slot lock-free die-local tile claim via ``%smid``),
* Optional direct multicast all-reduce push (``ar_push > 0``, mode 0 two-slot or mode 1 three-slot
  ``LamportWorkspace``, ``world`` in ``{4, 8, 16}``).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

CTA_M = 128
MMA_N = 8
CTA_K = 128
MMA_INST_K = 32
K_BLOCKS = CTA_K // MMA_INST_K
THREADS = 256
SF_ATOM_BYTES = 512
SF_ATOM_U16 = SF_ATOM_BYTES // 2
SF_COLS = 4
W_TILE_BYTES = CTA_M * CTA_K
EVICT_FIRST = 0x12F0000000000000
A_COLS_PER_MMA = MMA_INST_K // 4
_LOG2E = 1.4426950408889634

a_dtype = cutlass.Float8E4M3FN
b_dtype = cutlass.Float8E4M3FN


def num_k_tiles(k_in: int) -> int:
    return k_in // CTA_K


def acc_cols(nacc: int, mma_n: int = MMA_N) -> int:
    return max(16, nacc * mma_n)


def a_col0(nacc: int, my_tiles: int, mma_n: int = MMA_N) -> int:
    return (acc_cols(nacc, mma_n) + (my_tiles + 1) * SF_COLS + 31) // 32 * 32


def tmem_cols(
    ring: int, nacc: int, a_tmem: bool = False, my_tiles: int = 0, mma_n: int = MMA_N
) -> int:
    need = acc_cols(nacc, mma_n) + 2 * ring * SF_COLS
    if a_tmem:
        need = a_col0(nacc, my_tiles, mma_n) + my_tiles * K_BLOCKS * A_COLS_PER_MMA
    cols = 32
    while cols < need:
        cols *= 2
    return cols


# =============================================================================
# Inline-PTX helpers.
# =============================================================================


@dsl_user_op
def _smid(*, loc=None, ip=None):
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(),
            [],
            "mov.u32 $0, %smid;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _atom_add_u32(addr, val, *, loc=None, ip=None):
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(),
            [
                cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
                cutlass.Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_release_u32(addr, val, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "st.release.gpu.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mapa_u32(smem_ptr, peer, *, loc=None, ip=None):
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(),
            [
                smem_ptr.toint(loc=loc, ip=ip).ir_value(),
                cutlass.Int32(peer).ir_value(loc=loc, ip=ip),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_async_v4(dst, w0, w1, w2, w3, mbar, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [
            cutlass.Int32(dst).ir_value(loc=loc, ip=ip),
            cutlass.Int32(w0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(w1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(w2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(w3).ir_value(loc=loc, ip=ip),
            cutlass.Int32(mbar).ir_value(loc=loc, ip=ip),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_v4_f32(smem_ptr, *, loc=None, ip=None):
    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        [smem_ptr.toint(loc=loc, ip=ip).ir_value()],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [
        cutlass.Float32(_llvm.extractvalue(_T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _test_wait_cluster(mbar, parity, *, loc=None, ip=None):
    done = cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(),
            [mbar.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}",
            "=r,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
    return done != cutlass.Int32(0)


@dsl_user_op
def _absmax(acc, v, *, loc=None, ip=None):
    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(),
            [
                cutlass.Float32(acc).ir_value(loc=loc, ip=ip),
                cutlass.Float32(v).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .f32 a;\n\tabs.f32 a, $2;\n\tmax.f32 $0, $1, a;\n\t}",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rcp_approx_ftz(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3x4(v0, v1, v2, v3, *, loc=None, ip=None):
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(),
            [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
            "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}",
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_volatile_s32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.s32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_b32(addr, value, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bf16x2_word(lo, hi, *, loc=None, ip=None):
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b32 w, l, h;\n\t.reg .pred pl, ph;\n\tcvt.rn.bf16x2.f32 w, $1, $2;\n\t"
            "and.b32 l, w, 0xFFFF;\n\tsetp.eq.u32 pl, l, 0x8000;\n\t@pl and.b32 w, w, 0xFFFF0000;\n\t"
            "shr.u32 h, w, 16;\n\tsetp.eq.u32 ph, h, 0x8000;\n\t@ph and.b32 w, w, 0x0000FFFF;\n\t"
            "mov.b32 $0, w;\n\t}",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16_round(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 h, z;\n\tcvt.rn.bf16.f32 h, $1;\n\tmov.b16 z, 0;\n\tmov.b32 $0, {z, h};\n\t}",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _swiglu_std_bf16x2(gate_word, up_word, *, loc=None, ip=None):
    """Standard SwiGLU ``bf16(silu(gate) * up)`` on a packed bf16x2 pair."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(),
            [cutlass.Int32(w).ir_value(loc=loc, ip=ip) for w in (gate_word, up_word)],
            "{\n\t.reg .f32 gl, gh, ul, uh, one, t0, t1, e0, e1, s0, s1, r0, r1;\n\t.reg .b16 hl, hh;\n\t"
            "mov.b32 {hl, hh}, $1;\n\tcvt.f32.bf16 gl, hl;\n\tcvt.f32.bf16 gh, hh;\n\t"
            "mov.b32 {hl, hh}, $2;\n\tcvt.f32.bf16 ul, hl;\n\tcvt.f32.bf16 uh, hh;\n\t"
            "mov.b32 one, 0f3F800000;\n\t"
            "neg.f32 t0, gl;\n\tneg.f32 t1, gh;\n\t"
            "mul.rn.f32 t0, t0, 0f3FB8AA3B;\n\tmul.rn.f32 t1, t1, 0f3FB8AA3B;\n\t"
            "ex2.approx.f32 e0, t0;\n\tex2.approx.f32 e1, t1;\n\t"
            "add.rn.f32 e0, e0, one;\n\tadd.rn.f32 e1, e1, one;\n\t"
            "div.full.f32 s0, one, e0;\n\tdiv.full.f32 s1, one, e1;\n\t"
            "mul.rn.f32 r0, s0, gl;\n\tmul.rn.f32 r1, s1, gh;\n\t"
            "mul.rn.f32 r0, ul, r0;\n\tmul.rn.f32 r1, uh, r1;\n\t"
            "cvt.rn.bf16x2.f32 $0, r1, r0;\n\t}",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _k_tile(rank, i: int, split: int, my_tiles: int, contig: bool):
    if contig:
        return rank * cutlass.Int32(my_tiles) + cutlass.Int32(i)
    return rank + cutlass.Int32(i * split)


@cute.kernel
def ml4_mx_gemv_kernel(
    tma_desc_w0: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_wsf0: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_w1: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_wsf1: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_x: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_xsf: cutlass.GridConstant[cuda.TensorMap],
    y: cutlass.Array,  # bf16 [M * N]
    x_words: cutlass.Array,  # x_bf16: bf16 x [M, K] (or [M, 2K] for x_swiglu) as int32 pairs
    ln_w: cutlass.Array,  # x_layernorm: bf16 [K] as int32 pairs
    ln_b: cutlass.Array,  # x_layernorm: bf16 [K] as int32 pairs
    ln_eps: cutlass.Float32,  # x_layernorm: eps
    sm_table: cutlass.Array,  # ugpu: int32 [num_sms]: (die << 16) | rank_in_die
    ugpu_state: cutlass.Array,  # ugpu: int32 [20]: [launch_ctr, 0, 0, 0, 4 x (c0, c1, done, pad)]
    num_tokens: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    contig: cutlass.Constexpr[bool],
    nacc: cutlass.Constexpr[int],
    a_tmem: cutlass.Constexpr[bool],
    st_async: cutlass.Constexpr[bool],
    x_bf16: cutlass.Constexpr[bool],
    x_layernorm: cutlass.Constexpr[bool],
    x_swiglu: cutlass.Constexpr[bool],
    ugpu: cutlass.Constexpr[bool],
    ar_ws: cutlass.Int64,
    ar_flags: cutlass.Int64,
    ar_rank: cutlass.Int32,
    ar_mode: cutlass.Int32,  # 0 = 2-slot M3 MC, 1 = 3-slot LamportWorkspace
    ar_push: cutlass.Constexpr[int],
    ar_world: cutlass.Constexpr[int],
    ar_maxt: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[int],
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = tx // 32
    lane = tx % 32

    k_tiles = num_k_tiles(k_in)
    my_tiles = k_tiles // split
    n_tiles = n_out // CTA_M
    half_tiles = n_tiles // 2
    rows_per_rank = CTA_M // split
    x_tile_bytes = mma_n * CTA_K
    mb_stride = mma_n + 4

    rank = bx % cutlass.Int32(split)
    cluster_id = bx // cutlass.Int32(split)

    tma_ptr_w0 = tma_desc_w0.get_ptr()
    tma_ptr_wsf0 = tma_desc_wsf0.get_ptr()
    tma_ptr_w1 = tma_desc_w1.get_ptr()
    tma_ptr_wsf1 = tma_desc_wsf1.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()
    tma_ptr_xsf = tma_desc_xsf.get_ptr()

    smem_a = cutlass.Array(
        cutlass.Uint8, ring * W_TILE_BYTES, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_sfa = cutlass.Array(
        cutlass.Uint8, ring * SF_ATOM_BYTES, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        cutlass.Uint8, my_tiles * x_tile_bytes, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_sfb = cutlass.Array(
        cutlass.Uint8, my_tiles * SF_ATOM_BYTES, space=cutlass.AddressSpace.smem, alignment=1024
    )
    w_full = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    mma_done = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    act_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    mailbox = cutlass.Array(
        cutlass.Float32,
        split * rows_per_rank * mb_stride if split > 1 else 4,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    reduce_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    s_ntile = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    s_ln_mean = None
    s_ln_rstd = None
    if cutlass.const_expr(x_layernorm):
        s_ln_mean = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.smem, alignment=16)
        s_ln_rstd = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.smem, alignment=16)

    # Paradigm-B single-kernel die-local tile claim (4-slot lock-free ring, proven bijection).
    if cutlass.const_expr(ugpu):
        if (warp_id == 0) & (lane == 0) & (rank == cutlass.Int32(0)):
            entry = sm_table.load(idx=_smid())
            die_raw = entry >> cutlass.Int32(16)
            die = cutlass.select_(entry >= cutlass.Int32(0), die_raw, cluster_id & cutlass.Int32(1))
            st_base = ugpu_state.data_ptr(0).toint()
            enter_idx = _atom_add_u32(st_base, cutlass.Int32(1))
            slot = (enter_idx // cutlass.Int32(n_tiles)) & cutlass.Int32(3)
            slot_off = cutlass.Int64(4) * (
                cutlass.Int64(4) + cutlass.Int64(slot) * cutlass.Int64(4)
            )
            # Clear slot (slot + 2) & 3 once on the last arrival of this launch.
            if (enter_idx % cutlass.Int32(n_tiles)) == cutlass.Int32(n_tiles - 1):
                clr_slot = (slot + cutlass.Int32(2)) & cutlass.Int32(3)
                clr_off = cutlass.Int64(4) * (
                    cutlass.Int64(4) + cutlass.Int64(clr_slot) * cutlass.Int64(4)
                )
                _st_release_u32(st_base + clr_off, cutlass.Int32(0))
                _st_release_u32(st_base + clr_off + cutlass.Int64(4), cutlass.Int32(0))
            t_own = _atom_add_u32(
                st_base + slot_off + cutlass.Int64(die) * cutlass.Int64(4), cutlass.Int32(1)
            )
            claimed = cutlass.Int32(0)
            if t_own < cutlass.Int32(half_tiles):
                claimed = die * cutlass.Int32(half_tiles) + t_own
            else:
                other = cutlass.Int32(1) - die
                t_other = _atom_add_u32(
                    st_base + slot_off + cutlass.Int64(other) * cutlass.Int64(4), cutlass.Int32(1)
                )
                claimed = other * cutlass.Int32(half_tiles) + t_other
            s_ntile.store(claimed, idx=0)
            if cutlass.const_expr(split > 1):
                for peer in cutlass.range_constexpr(1, split):
                    prims.mapa(s_ntile, peer).store(claimed)

    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w0)
        prims.prefetch_tensormap(tma_ptr_wsf0)
        if cutlass.const_expr(ugpu):
            prims.prefetch_tensormap(tma_ptr_w1)
            prims.prefetch_tensormap(tma_ptr_wsf1)
        if cutlass.const_expr(not x_bf16):
            prims.prefetch_tensormap(tma_ptr_x)
            prims.prefetch_tensormap(tma_ptr_xsf)
        if prims.elect_sync():
            for s in cutlass.range_constexpr(ring):
                prims.mbarrier_init(w_full.subview(s), 1)
                prims.mbarrier_init(mma_done.subview(s), 1)
            prims.mbarrier_init(act_full, 1)
            prims.mbarrier_init(acc_done, 1)
            if cutlass.const_expr(split > 1 and st_async):
                prims.mbarrier_init(reduce_ready, 1)
                prims.mbarrier_arrive_expect_tx(
                    reduce_ready, (split - 1) * rows_per_rank * mma_n * 4
                )
            elif cutlass.const_expr(split > 1):
                prims.mbarrier_init(reduce_ready, (split - 1) * rows_per_rank)
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, tmem_cols(ring, nacc, a_tmem, my_tiles, mma_n))
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    if cutlass.const_expr(split > 1):
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)

    n_tile = s_ntile.load(idx=0) if cutlass.const_expr(ugpu) else cluster_id
    m_offset = n_tile * cutlass.Int32(CTA_M)
    tile_d1 = n_tile >= cutlass.Int32(half_tiles)
    tile_local = cutlass.select_(tile_d1, n_tile - cutlass.Int32(half_tiles), n_tile)
    m_offset_local = tile_local * cutlass.Int32(CTA_M)

    if warp_id == 0:
        if prims.elect_sync():
            for i in cutlass.range_constexpr(ring, my_tiles):
                k_global = _k_tile(rank, i, split, my_tiles, contig)
                if cutlass.const_expr(ugpu):
                    if tile_d1:
                        prims.cp_async_bulk_tensor_prefetch(
                            tma_ptr_w1, [k_global * cutlass.Int32(CTA_K), m_offset_local], []
                        )
                    else:
                        prims.cp_async_bulk_tensor_prefetch(
                            tma_ptr_w0, [k_global * cutlass.Int32(CTA_K), m_offset_local], []
                        )
                else:
                    prims.cp_async_bulk_tensor_prefetch(
                        tma_ptr_w0, [k_global * cutlass.Int32(CTA_K), m_offset], []
                    )
            for i in cutlass.range_constexpr(my_tiles):
                stage = i % ring
                if cutlass.const_expr(i >= ring):
                    while not cute.arch.mbarrier_try_wait(
                        mma_done.subview(stage).data_ptr(), (i // ring - 1) % 2
                    ):
                        pass
                k_global = _k_tile(rank, i, split, my_tiles, contig)
                prims.mbarrier_arrive_expect_tx(w_full.subview(stage), W_TILE_BYTES + SF_ATOM_BYTES)
                if cutlass.const_expr(ugpu):
                    if tile_d1:
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_a.subview(stage * W_TILE_BYTES),
                            tma_ptr_w1,
                            (k_global * cutlass.Int32(CTA_K), m_offset_local),
                            w_full.subview(stage),
                            l2_cache_hint=EVICT_FIRST,
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_sfa.subview(stage * SF_ATOM_BYTES),
                            tma_ptr_wsf1,
                            (cutlass.Int32(0), tile_local * cutlass.Int32(k_tiles) + k_global),
                            w_full.subview(stage),
                            l2_cache_hint=EVICT_FIRST,
                        )
                    else:
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_a.subview(stage * W_TILE_BYTES),
                            tma_ptr_w0,
                            (k_global * cutlass.Int32(CTA_K), m_offset_local),
                            w_full.subview(stage),
                            l2_cache_hint=EVICT_FIRST,
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_sfa.subview(stage * SF_ATOM_BYTES),
                            tma_ptr_wsf0,
                            (cutlass.Int32(0), tile_local * cutlass.Int32(k_tiles) + k_global),
                            w_full.subview(stage),
                            l2_cache_hint=EVICT_FIRST,
                        )
                else:
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(stage * W_TILE_BYTES),
                        tma_ptr_w0,
                        (k_global * cutlass.Int32(CTA_K), m_offset),
                        w_full.subview(stage),
                        l2_cache_hint=EVICT_FIRST,
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_sfa.subview(stage * SF_ATOM_BYTES),
                        tma_ptr_wsf0,
                        (cutlass.Int32(0), n_tile * cutlass.Int32(k_tiles) + k_global),
                        w_full.subview(stage),
                        l2_cache_hint=EVICT_FIRST,
                    )

    elif warp_id == 1:
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        if cutlass.const_expr(trigger_early):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        if cutlass.const_expr(not x_bf16):
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(act_full, my_tiles * (x_tile_bytes + SF_ATOM_BYTES))
                for i in cutlass.range_constexpr(my_tiles):
                    k_global = _k_tile(rank, i, split, my_tiles, contig)
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b.subview(i * x_tile_bytes),
                        tma_ptr_x,
                        (k_global * cutlass.Int32(CTA_K), cutlass.Int32(0)),
                        act_full,
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_sfb.subview(i * SF_ATOM_BYTES),
                        tma_ptr_xsf,
                        (cutlass.Int32(0), k_global),
                        act_full,
                    )

    elif warp_id == 2:
        tmem_raw = tmem_ptr_i32.load()
        base_col = tmem_raw & 0xFFFF
        base_row = tmem_raw >> 16
        idesc = prims.Tcgen05MxInstrDesc.build(
            a_dtype=a_dtype, b_dtype=b_dtype, scale_format=1, n_dim=mma_n, m_dim=CTA_M
        )
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
        if cutlass.const_expr(a_tmem):
            sfa_col0 = base_col + acc_cols(nacc, mma_n)
            sfb_col0 = sfa_col0 + my_tiles * SF_COLS
            a_base = tmem_raw + cutlass.Int32(a_col0(nacc, my_tiles, mma_n))
            for i in cutlass.range_constexpr(my_tiles):
                stage = i % ring
                while not cute.arch.mbarrier_try_wait(
                    w_full.subview(stage).data_ptr(), (i // ring) % 2
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                desc_a = prims.Tcgen05SmemDesc.build(
                    smem_a.subview(stage * W_TILE_BYTES),
                    leading_byte_offset=16,
                    stride_byte_offset=1024,
                    base_offset=0,
                    layout=2,
                )
                desc_sfa = prims.Tcgen05SmemDesc.build(
                    smem_sfa.subview(stage * SF_ATOM_BYTES),
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    base_offset=0,
                    layout=0,
                )
                if prims.elect_sync():
                    for kb in cutlass.range_constexpr(K_BLOCKS):
                        prims.tcgen05_cp(
                            prims.Tcgen05CpShape.SHAPE_128X256B,
                            cutlass.inttoptr(
                                a_base + cutlass.Int32((i * K_BLOCKS + kb) * A_COLS_PER_MMA),
                                6,
                                cutlass.Int32,
                            ),
                            desc_a + ((MMA_INST_K >> 4) * kb),
                            group=prims.CTAGroup.CTA_1,
                        )
                    prims.tcgen05_cp(
                        s2t_shape,
                        cutlass.inttoptr(
                            (base_row << 16) | (sfa_col0 + i * SF_COLS), 6, cutlass.Int32
                        ),
                        desc_sfa,
                        multicast=s2t_multicast,
                    )
                    if cutlass.const_expr(i + ring < my_tiles):
                        prims.tcgen05_commit(mma_done.subview(stage))
            if cutlass.const_expr(x_bf16):
                while not cute.arch.mbarrier_test_wait(act_full.data_ptr(), 0):
                    pass
            else:
                while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                    pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if prims.elect_sync():
                for i in cutlass.range_constexpr(my_tiles):
                    desc_sfb = prims.Tcgen05SmemDesc.build(
                        smem_sfb.subview(i * SF_ATOM_BYTES),
                        leading_byte_offset=16,
                        stride_byte_offset=128,
                        base_offset=0,
                        layout=0,
                    )
                    desc_b = prims.Tcgen05SmemDesc.build(
                        smem_b.subview(i * x_tile_bytes),
                        leading_byte_offset=16,
                        stride_byte_offset=1024,
                        base_offset=0,
                        layout=2,
                    )
                    sfa_tmem = cutlass.inttoptr(
                        (base_row << 16) | (sfa_col0 + i * SF_COLS), 6, cutlass.Int32
                    )
                    sfb_tmem = cutlass.inttoptr((base_row << 16) | sfb_col0, 6, cutlass.Int32)
                    prims.tcgen05_cp(s2t_shape, sfb_tmem, desc_sfb, multicast=s2t_multicast)
                    for kb in cutlass.range_constexpr(K_BLOCKS):
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF8F6F4,
                            prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(tmem_raw, 6, cutlass.Float32),
                            cutlass.inttoptr(
                                a_base + cutlass.Int32((i * K_BLOCKS + kb) * A_COLS_PER_MMA),
                                6,
                                cutlass.Int32,
                            ),
                            desc_b + ((MMA_INST_K >> 4) * kb),
                            idesc.set_sf_ids(a_sf_id=kb, b_sf_id=kb),
                            i > 0 or kb > 0,
                            sfa_tmem,
                            sfb_tmem,
                        )
            if prims.elect_sync():
                prims.tcgen05_commit(acc_done)
        else:
            sfa_col0 = base_col + acc_cols(nacc, mma_n)
            sfb_col0 = sfa_col0 + ring * SF_COLS
            if cutlass.const_expr(x_bf16):
                while not cute.arch.mbarrier_test_wait(act_full.data_ptr(), 0):
                    pass
            else:
                while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                    pass
            for i in cutlass.range_constexpr(my_tiles):
                stage = i % ring
                while not cute.arch.mbarrier_try_wait(
                    w_full.subview(stage).data_ptr(), (i // ring) % 2
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                sfa_tmem = (base_row << 16) | (sfa_col0 + stage * SF_COLS)
                sfb_tmem = (base_row << 16) | (sfb_col0 + stage * SF_COLS)
                desc_sfa = prims.Tcgen05SmemDesc.build(
                    smem_sfa.subview(stage * SF_ATOM_BYTES),
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    base_offset=0,
                    layout=0,
                )
                desc_sfb = prims.Tcgen05SmemDesc.build(
                    smem_sfb.subview(i * SF_ATOM_BYTES),
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    base_offset=0,
                    layout=0,
                )
                desc_a = prims.Tcgen05SmemDesc.build(
                    smem_a.subview(stage * W_TILE_BYTES),
                    leading_byte_offset=16,
                    stride_byte_offset=1024,
                    base_offset=0,
                    layout=2,
                )
                desc_b = prims.Tcgen05SmemDesc.build(
                    smem_b.subview(i * x_tile_bytes),
                    leading_byte_offset=16,
                    stride_byte_offset=1024,
                    base_offset=0,
                    layout=2,
                )
                if prims.elect_sync():
                    prims.tcgen05_cp(
                        s2t_shape,
                        cutlass.inttoptr(sfa_tmem, 6, cutlass.Int32),
                        desc_sfa,
                        multicast=s2t_multicast,
                    )
                    prims.tcgen05_cp(
                        s2t_shape,
                        cutlass.inttoptr(sfb_tmem, 6, cutlass.Int32),
                        desc_sfb,
                        multicast=s2t_multicast,
                    )
                    for kb in cutlass.range_constexpr(K_BLOCKS):
                        inc = (MMA_INST_K >> 4) * kb
                        j = i * K_BLOCKS + kb
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF8F6F4,
                            prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(
                                tmem_raw + cutlass.Int32((j % nacc) * mma_n), 6, cutlass.Float32
                            ),
                            desc_a + inc,
                            desc_b + inc,
                            idesc.set_sf_ids(a_sf_id=kb, b_sf_id=kb),
                            j >= nacc,
                            cutlass.inttoptr(sfa_tmem, 6, cutlass.Int32),
                            cutlass.inttoptr(sfb_tmem, 6, cutlass.Int32),
                        )
                    prims.tcgen05_commit(mma_done.subview(stage))
            if prims.elect_sync():
                prims.tcgen05_commit(acc_done)

    elif warp_id >= 4:
        row = (warp_id % 4) * 32 + lane
        if cutlass.const_expr(x_bf16):
            qz_tid = tx - cutlass.Int32(128)
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            # Optional fused LayerNorm statistics (2-pass exact FP32 mean & variance over full K).
            if cutlass.const_expr(x_layernorm):
                ln_tok = qz_tid // cutlass.Int32(16)
                ln_part = qz_tid % cutlass.Int32(16)
                ln_tok_ld = cutlass.select_(ln_tok < num_tokens, ln_tok, cutlass.Int32(0))
                ln_base = ln_tok_ld * cutlass.Int32(k_in // 2)
                ln_vecs_per_part = k_in // 8 // 16
                # Pass 1: mean
                ln_sum = cutlass.Float32(0.0)
                for r in cutlass.range(ln_vecs_per_part, unroll=1):
                    v4 = x_words.load(
                        idx=ln_base + (ln_part + r * cutlass.Int32(16)) * cutlass.Int32(4),
                        vector_size=4,
                        alignment=16,
                    )
                    for e4 in cutlass.range_constexpr(4):
                        w0 = cutlass.Int32(v4[e4])
                        ln_sum = (
                            ln_sum
                            + (w0 << cutlass.Int32(16)).bitcast(cutlass.Float32)
                            + (w0 & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                        )
                for off in cutlass.range_constexpr(4):
                    ln_sum = ln_sum + cute.arch.shuffle_sync_bfly(ln_sum, offset=1 << off)
                ln_mean = ln_sum / cutlass.Float32(k_in)
                # Pass 2: biased variance = mean((x - mean)^2)
                ln_var = cutlass.Float32(0.0)
                for r in cutlass.range(ln_vecs_per_part, unroll=1):
                    v4 = x_words.load(
                        idx=ln_base + (ln_part + r * cutlass.Int32(16)) * cutlass.Int32(4),
                        vector_size=4,
                        alignment=16,
                    )
                    for e4 in cutlass.range_constexpr(4):
                        w0 = cutlass.Int32(v4[e4])
                        d0 = (w0 << cutlass.Int32(16)).bitcast(cutlass.Float32) - ln_mean
                        d1 = (w0 & cutlass.Int32(-65536)).bitcast(cutlass.Float32) - ln_mean
                        ln_var = ln_var + d0 * d0 + d1 * d1
                for off in cutlass.range_constexpr(4):
                    ln_var = ln_var + cute.arch.shuffle_sync_bfly(ln_var, offset=1 << off)
                if ln_part == cutlass.Int32(0):
                    s_ln_mean.store(ln_mean, idx=ln_tok)
                    s_ln_rstd.store(
                        cute.math.rsqrt(ln_var / cutlass.Float32(k_in) + ln_eps, fastmath=False),
                        idx=ln_tok,
                    )
                prims.barrier_cta_sync(3, thread_count=128)

            # Exact ML4 1x128 activation quantization: consecutive 4 lanes hold qz_kb = 0..3
            # of the same (qz_i, qz_t), so 2 butterfly shuffles compute exact K128 amax.
            qz_b_words = cutlass.Array(
                smem_b.data_ptr(0),
                shape=(my_tiles * x_tile_bytes // 4,),
                dtype=cutlass.Int32,
                alignment=16,
            )
            qz_items = my_tiles * K_BLOCKS * mma_n
            qz_rounds = (qz_items + 127) // 128
            for qz_r in cutlass.range_constexpr(qz_rounds):
                qz_item = qz_tid + cutlass.Int32(qz_r * 128)
                qz_kb = qz_item & cutlass.Int32(3)
                qz_t = (qz_item >> cutlass.Int32(2)) % cutlass.Int32(mma_n)
                qz_i = (qz_item >> cutlass.Int32(2)) // cutlass.Int32(mma_n)
                if cutlass.const_expr(qz_r * 128 + 127 >= qz_items):
                    qz_i = cutlass.select_(
                        qz_item < cutlass.Int32(qz_items), qz_i, cutlass.Int32(0)
                    )
                qz_k_global = rank + qz_i * cutlass.Int32(split)
                if cutlass.const_expr(contig):
                    qz_k_global = rank * cutlass.Int32(my_tiles) + qz_i
                qz_t_ld = cutlass.select_(qz_t < num_tokens, qz_t, cutlass.Int32(0))
                qz_x_base = (
                    qz_t_ld * cutlass.Int32(k_in if x_swiglu else k_in // 2)
                    + qz_k_global * cutlass.Int32(CTA_K // 2)
                    + qz_kb * cutlass.Int32(MMA_INST_K // 2)
                )
                qz_vecs = [
                    x_words.load(
                        idx=qz_x_base + cutlass.Int32(qz_v * 4), vector_size=4, alignment=16
                    )
                    for qz_v in range(4)
                ]
                qz_up_vecs = None
                if cutlass.const_expr(x_swiglu):
                    qz_up_vecs = [
                        x_words.load(
                            idx=qz_x_base + cutlass.Int32(k_in // 2 + qz_v * 4),
                            vector_size=4,
                            alignment=16,
                        )
                        for qz_v in range(4)
                    ]
                qz_lnw = None
                qz_lnb = None
                qz_mean = cutlass.Float32(0.0)
                qz_rstd = cutlass.Float32(1.0)
                if cutlass.const_expr(x_layernorm):
                    qz_col_w = qz_k_global * cutlass.Int32(CTA_K // 2) + qz_kb * cutlass.Int32(
                        MMA_INST_K // 2
                    )
                    qz_lnw = [
                        ln_w.load(
                            idx=qz_col_w + cutlass.Int32(qz_v * 4), vector_size=4, alignment=16
                        )
                        for qz_v in range(4)
                    ]
                    qz_lnb = [
                        ln_b.load(
                            idx=qz_col_w + cutlass.Int32(qz_v * 4), vector_size=4, alignment=16
                        )
                        for qz_v in range(4)
                    ]
                    qz_mean = s_ln_mean.load(idx=qz_t_ld)
                    qz_rstd = s_ln_rstd.load(idx=qz_t_ld)

                qz_vals = []
                qz_amax = cutlass.Float32(0.0)
                for qz_v in cutlass.range_constexpr(4):
                    for qz_e4 in cutlass.range_constexpr(4):
                        qz_word = cutlass.Int32(qz_vecs[qz_v][qz_e4])
                        if cutlass.const_expr(x_swiglu):
                            qz_word = _swiglu_std_bf16x2(
                                qz_word, cutlass.Int32(qz_up_vecs[qz_v][qz_e4])
                            )
                        qz_lo = (qz_word << cutlass.Int32(16)).bitcast(cutlass.Float32)
                        qz_hi = (qz_word & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                        if cutlass.const_expr(x_layernorm):
                            ww = cutlass.Int32(qz_lnw[qz_v][qz_e4])
                            bw = cutlass.Int32(qz_lnb[qz_v][qz_e4])
                            wl = (ww << cutlass.Int32(16)).bitcast(cutlass.Float32)
                            wh = (ww & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                            bl = (bw << cutlass.Int32(16)).bitcast(cutlass.Float32)
                            bh = (bw & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                            qz_lo = _bf16_round((qz_lo - qz_mean) * qz_rstd * wl + bl)
                            qz_hi = _bf16_round((qz_hi - qz_mean) * qz_rstd * wh + bh)
                        qz_vals.append(qz_lo)
                        qz_vals.append(qz_hi)
                        qz_amax = _absmax(_absmax(qz_amax, qz_lo), qz_hi)

                # Reduce amax across the 4 consecutive lanes of this 1x128 block, floor at 1e-10.
                qz_amax = cute.arch.fmax(qz_amax, cute.arch.shuffle_sync_bfly(qz_amax, offset=1))
                qz_amax = cute.arch.fmax(qz_amax, cute.arch.shuffle_sync_bfly(qz_amax, offset=2))
                qz_amax = cute.arch.fmax(qz_amax, cutlass.Float32(1.0e-10))
                qz_bits = (qz_amax * _rcp_approx_ftz(cutlass.Float32(448.0))).bitcast(cutlass.Int32)
                qz_e = ((qz_bits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)) + cutlass.select_(
                    (qz_bits & cutlass.Int32(0x7FFFFF)) != cutlass.Int32(0),
                    cutlass.Int32(1),
                    cutlass.Int32(0),
                )
                qz_e = cutlass.select_(qz_e > cutlass.Int32(254), cutlass.Int32(254), qz_e)
                qz_scale = ((cutlass.Int32(254) - qz_e) << cutlass.Int32(23)).bitcast(
                    cutlass.Float32
                )
                qz_q = [
                    _e4m3x4(
                        qz_vals[4 * qz_j] * qz_scale,
                        qz_vals[4 * qz_j + 1] * qz_scale,
                        qz_vals[4 * qz_j + 2] * qz_scale,
                        qz_vals[4 * qz_j + 3] * qz_scale,
                    )
                    for qz_j in range(8)
                ]
                if qz_item < cutlass.Int32(qz_items):
                    if qz_t < num_tokens:
                        qz_row_w = qz_i * cutlass.Int32(x_tile_bytes // 4) + qz_t * cutlass.Int32(
                            CTA_K // 4
                        )
                        qz_c0 = qz_kb * cutlass.Int32(2)
                        qz_b_words.store(
                            (qz_q[0], qz_q[1], qz_q[2], qz_q[3]),
                            idx=qz_row_w + (qz_c0 ^ (qz_t & cutlass.Int32(7))) * cutlass.Int32(4),
                            alignment=16,
                        )
                        qz_b_words.store(
                            (qz_q[4], qz_q[5], qz_q[6], qz_q[7]),
                            idx=qz_row_w
                            + ((qz_c0 + cutlass.Int32(1)) ^ (qz_t & cutlass.Int32(7)))
                            * cutlass.Int32(4),
                            alignment=16,
                        )
                        smem_sfb.store(
                            cutlass.Uint8(qz_e),
                            idx=qz_i * cutlass.Int32(SF_ATOM_BYTES)
                            + (qz_t % cutlass.Int32(32)) * cutlass.Int32(16)
                            + (qz_t // cutlass.Int32(32)) * cutlass.Int32(4)
                            + qz_kb,
                        )
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            prims.barrier_cta_sync(2, thread_count=128)
            if qz_tid == cutlass.Int32(0):
                prims.mbarrier_arrive(act_full)

        if cutlass.const_expr(st_async):
            while not cute.arch.mbarrier_test_wait(acc_done.data_ptr(), 0):
                pass
        else:
            while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
                pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc_all = []
        ld_col = 0
        for ld_w in (128, 64, 32, 16, 8):
            if cutlass.const_expr((mma_n * nacc - ld_col) >= ld_w):
                piece = prims.tcgen05_ld(
                    "32x32b",
                    cutlass.inttoptr(
                        tmem_ptr_i32.load() + cutlass.Int32(ld_col), 6, cutlass.Float32
                    ),
                    num=ld_w,
                )
                acc_all = acc_all + [piece[c] for c in range(ld_w)]
                ld_col += ld_w
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        acc = []
        for t in cutlass.range_constexpr(mma_n):
            v = cutlass.Float32(acc_all[t])
            for a in cutlass.range_constexpr(1, nacc):
                v = v + cutlass.Float32(acc_all[a * mma_n + t])
            acc.append(v)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)

        n = m_offset + row
        if cutlass.const_expr(split == 1):
            for t in cutlass.range_constexpr(mma_n):
                if cutlass.Int32(t) < num_tokens:
                    y.store(
                        cutlass.BFloat16(cutlass.Float32(acc[t])),
                        idx=cutlass.Int32(t * n_out) + n,
                    )
            if cutlass.const_expr(ar_push > 0):
                prims.griddepcontrol(prims.GridDepAction.WAIT)
                slot_l3 = _ld_volatile_s32(ar_flags + cutlass.Int64(4)) % cutlass.Int32(3)
                ar_par = [
                    _ld_volatile_s32(
                        ar_flags
                        + cutlass.Int64(4)
                        * (cutlass.Int64(t * 8) + cutlass.Int64(n // cutlass.Int32(n_out // 8)))
                    )
                    & cutlass.Int32(1)
                    for t in range(mma_n)
                ]
                for t in cutlass.range_constexpr(mma_n):
                    ar_v = cutlass.Float32(acc[t])
                    ar_hi = cute.arch.shuffle_sync_bfly(ar_v, offset=1)
                    if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                        if cutlass.Int32(t) < num_tokens:
                            off0 = (
                                cutlass.Int64(ar_par[t])
                                * cutlass.Int64(ar_maxt * ar_world * (n_out // 2))
                                + (cutlass.Int64(t * ar_world) + cutlass.Int64(ar_rank))
                                * cutlass.Int64(n_out // 2)
                                + (cutlass.Int64(n) // cutlass.Int64(2))
                            )
                            off1 = (
                                (
                                    cutlass.Int64(slot_l3) * cutlass.Int64(ar_world)
                                    + cutlass.Int64(ar_rank)
                                )
                                * cutlass.Int64(ar_maxt)
                                + cutlass.Int64(t)
                            ) * cutlass.Int64(n_out // 2) + (cutlass.Int64(n) // cutlass.Int64(2))
                            ar_off = cutlass.select_(ar_mode == cutlass.Int32(0), off0, off1)
                            _st_global_b32(
                                ar_ws + ar_off * cutlass.Int64(4), _bf16x2_word(ar_v, ar_hi)
                            )
        else:
            owner = row // cutlass.Int32(rows_per_rank)
            local = row % cutlass.Int32(rows_per_rank)
            if owner != rank:
                base = (rank * cutlass.Int32(rows_per_rank) + local) * cutlass.Int32(mb_stride)
                if cutlass.const_expr(st_async):
                    dst = _mapa_u32(mailbox.subview(base).data_ptr(), owner)
                    mb = _mapa_u32(reduce_ready.data_ptr(), owner)
                    words = [cutlass.Float32(acc[t]).bitcast(cutlass.Int32) for t in range(mma_n)]
                    for g in cutlass.range_constexpr(mma_n // 4):
                        _st_async_v4(
                            dst + cutlass.Int32(16 * g),
                            words[4 * g],
                            words[4 * g + 1],
                            words[4 * g + 2],
                            words[4 * g + 3],
                            mb,
                        )
                else:
                    for t in cutlass.range_constexpr(mma_n):
                        prims.mapa(mailbox.subview(base + cutlass.Int32(t)), owner).store(
                            cutlass.Float32(acc[t])
                        )
                    prims.mbarrier_arrive(
                        prims.mapa(reduce_ready, owner), scope=prims.MemScope.CLUSTER
                    )
            else:
                if cutlass.const_expr(st_async):
                    while not _test_wait_cluster(reduce_ready.data_ptr(), 0):
                        pass
                else:
                    while not prims.mbarrier_try_wait_parity(
                        reduce_ready, 0, scope=prims.MBarrierScope.CLUSTER
                    ):
                        pass
                totals = []
                for g in cutlass.range_constexpr(mma_n // 4):
                    sums = [cutlass.Float32(0.0)] * 4
                    for q in cutlass.range_constexpr(split):
                        parts = _ld_shared_v4_f32(
                            mailbox.subview(
                                (cutlass.Int32(q * rows_per_rank) + local)
                                * cutlass.Int32(mb_stride)
                                + cutlass.Int32(4 * g)
                            ).data_ptr()
                        )
                        sums = [
                            sums[j]
                            + cutlass.select_(
                                rank == cutlass.Int32(q),
                                cutlass.Float32(acc[4 * g + j]),
                                parts[j],
                            )
                            for j in range(4)
                        ]
                    for j in cutlass.range_constexpr(4):
                        totals.append(sums[j])
                        if cutlass.Int32(4 * g + j) < num_tokens:
                            y.store(
                                cutlass.BFloat16(sums[j]),
                                idx=cutlass.Int32((4 * g + j) * n_out) + n,
                            )
                if cutlass.const_expr(ar_push > 0):
                    prims.griddepcontrol(prims.GridDepAction.WAIT)
                    slot_l3 = _ld_volatile_s32(ar_flags + cutlass.Int64(4)) % cutlass.Int32(3)
                    ar_par = [
                        _ld_volatile_s32(
                            ar_flags
                            + cutlass.Int64(4)
                            * (cutlass.Int64(t * 8) + cutlass.Int64(n // cutlass.Int32(n_out // 8)))
                        )
                        & cutlass.Int32(1)
                        for t in range(mma_n)
                    ]
                    for t in cutlass.range_constexpr(mma_n):
                        ar_hi = cute.arch.shuffle_sync_bfly(totals[t], offset=1)
                        if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                            if cutlass.Int32(t) < num_tokens:
                                off0 = (
                                    cutlass.Int64(ar_par[t])
                                    * cutlass.Int64(ar_maxt * ar_world * (n_out // 2))
                                    + (cutlass.Int64(t * ar_world) + cutlass.Int64(ar_rank))
                                    * cutlass.Int64(n_out // 2)
                                    + (cutlass.Int64(n) // cutlass.Int64(2))
                                )
                                off1 = (
                                    (
                                        cutlass.Int64(slot_l3) * cutlass.Int64(ar_world)
                                        + cutlass.Int64(ar_rank)
                                    )
                                    * cutlass.Int64(ar_maxt)
                                    + cutlass.Int64(t)
                                ) * cutlass.Int64(n_out // 2) + (
                                    cutlass.Int64(n) // cutlass.Int64(2)
                                )
                                ar_off = cutlass.select_(ar_mode == cutlass.Int32(0), off0, off1)
                                _st_global_b32(
                                    ar_ws + ar_off * cutlass.Int64(4),
                                    _bf16x2_word(totals[t], ar_hi),
                                )
        prims.barrier_cta_sync(1, thread_count=128)
        if tx == 128:
            if cutlass.const_expr(ar_push == 2):
                prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        if warp_id == 4:
            prims.tcgen05_dealloc(
                cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32),
                tmem_cols(ring, nacc, a_tmem, my_tiles, mma_n),
            )


def _bytes_map(t, cols, rows, box_rows, swizzle):
    return cuda.create_tensor_map_tiled(
        global_address=t.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[cols, rows],
        global_strides=[cols // 16],
        box_dims=[CTA_K, box_rows],
        swizzle=swizzle,
    )


def _atom_map(t, atoms):
    return cuda.create_tensor_map_tiled(
        global_address=t.iterator.toint(),
        dtype=cutlass.Uint16,
        global_dims=[SF_ATOM_U16, atoms],
        global_strides=[SF_ATOM_BYTES // 16],
        box_dims=[SF_ATOM_U16, 1],
        swizzle=cuda.TensorMapSwizzle.none,
    )


@cute.jit
def ml4_mx_gemv_jit(
    w0: cute.Tensor,
    wsf0: cute.Tensor,
    w1: cute.Tensor,
    wsf1: cute.Tensor,
    x: cute.Tensor,
    xsf: cute.Tensor,
    y: cute.Tensor,
    x_words: cute.Tensor,
    ln_w: cute.Tensor,
    ln_b: cute.Tensor,
    ln_eps: cutlass.Float32,
    sm_table: cute.Tensor,
    ugpu_state: cute.Tensor,
    num_tokens: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    contig: cutlass.Constexpr[bool],
    nacc: cutlass.Constexpr[int],
    a_tmem: cutlass.Constexpr[bool],
    st_async: cutlass.Constexpr[bool],
    x_bf16: cutlass.Constexpr[bool],
    x_layernorm: cutlass.Constexpr[bool],
    x_swiglu: cutlass.Constexpr[bool],
    ugpu: cutlass.Constexpr[bool],
    ar_ws: cutlass.Int64,
    ar_flags: cutlass.Int64,
    ar_rank: cutlass.Int32,
    ar_mode: cutlass.Int32,
    ar_push: cutlass.Constexpr[int],
    ar_world: cutlass.Constexpr[int],
    ar_maxt: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    assert not (x_layernorm and x_swiglu), "x_layernorm and x_swiglu are mutually exclusive"
    assert x_bf16 or (not x_layernorm and not x_swiglu), "prologues require x_bf16"
    k_tiles = num_k_tiles(k_in)
    rows_per_desc = n_out // 2 if ugpu else n_out
    tma_desc_w0 = _bytes_map(w0, k_in, rows_per_desc, CTA_M, cuda.TensorMapSwizzle.s128b)
    tma_desc_wsf0 = _atom_map(wsf0, (rows_per_desc // CTA_M) * k_tiles)
    tma_desc_w1 = (
        _bytes_map(w1, k_in, rows_per_desc, CTA_M, cuda.TensorMapSwizzle.s128b)
        if cutlass.const_expr(ugpu)
        else tma_desc_w0
    )
    tma_desc_wsf1 = (
        _atom_map(wsf1, (rows_per_desc // CTA_M) * k_tiles)
        if cutlass.const_expr(ugpu)
        else tma_desc_wsf0
    )
    tma_desc_x = _bytes_map(x, k_in, num_tokens, mma_n, cuda.TensorMapSwizzle.s128b)
    tma_desc_xsf = _atom_map(xsf, k_tiles)
    ml4_mx_gemv_kernel(
        tma_desc_w0,
        tma_desc_wsf0,
        tma_desc_w1,
        tma_desc_wsf1,
        tma_desc_x,
        tma_desc_xsf,
        y,
        x_words,
        ln_w,
        ln_b,
        ln_eps,
        sm_table,
        ugpu_state,
        num_tokens,
        n_out,
        k_in,
        split,
        ring,
        trigger_early,
        contig,
        nacc,
        a_tmem,
        st_async,
        x_bf16,
        x_layernorm,
        x_swiglu,
        ugpu,
        ar_ws,
        ar_flags,
        ar_rank,
        ar_mode,
        ar_push,
        ar_world,
        ar_maxt,
        mma_n,
    ).launch(
        grid=((n_out // CTA_M) * split, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(split, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )
