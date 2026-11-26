# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused 1-launch Indexer Q projection for Mistral Large 4 on SM107 (Rubin VR200).

Fuses the three per-leader-layer indexer Q kernels into a single tcgen05 CuTe DSL launch:

1. ``ix_q_input_norm`` — 1-pass shift-invariant FP32 LayerNorm over ``K = 6144`` with bf16 output
   rounding, parallelized across 32 threads/token (``M <= 4``) or 16 threads/token (``M <= 8``)
   and hidden behind weight TMA/TMEM staging.
2. ``ix_wq_b`` — ``[M, 6144] x [4096, 6144]^T`` FP8_PB_WO block-scaled GEMV (``N = 4096`` =
   ``32 heads x 128 head_dim``, so every ``CTA_M = 128`` N-tile maps 1:1 to one indexer head),
   supporting both cluster DSMEM reduction (``split in {4, 8}``) and cluster-free L2 partial
   reduction (``split in {4, 6, 8}``).
3. ``ssmax_indexer_q_side_fused`` — exact SSMax scaling (``s * log(pos + 1) + b`` with Cody-Waite
   exact ``logf``), interleaved RoPE / NoPE, NaN-propagating head amax, UE8M0 power-of-two FP8
   e4m3 quantization, and head-weight scaling (computing ``x @ wp^T`` inline on the otherwise idle
   warp 3 to decouple from side-stream ``wk_wp``, or reading external ``kw[:, 128:]``).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover - pure-Python host helpers work without torch
    torch = None  # type: ignore[assignment]

__all__ = [
    "CANDIDATE_CONFIGS",
    "DEFAULT_CONFIG",
    "HEADS",
    "HEAD_DIM",
    "K_IN",
    "MAX_TOKENS",
    "N_OUT",
    "ROPE_DIM",
    "IxQFusedConfig",
    "Ml4IxQFused",
    "exact_log_bf16_host",
    "get_config",
    "ml4_ix_q_fused",
    "precompile",
    "serves",
    "smem_bytes",
    "tmem_cols",
]

HEADS = 32
HEAD_DIM = 128
ROPE_DIM = 64
N_OUT = HEADS * HEAD_DIM  # 4096
K_IN = 6144
CTA_M = 128
CTA_K = 128
MMA_INST_K = 32
K_BLOCKS = CTA_K // MMA_INST_K  # 4
K_TILES = K_IN // CTA_K  # 48
N_TILES = N_OUT // CTA_M  # 32
MMA_N = 8
MAX_TOKENS = 8
THREADS = 256
SF_ATOM_BYTES = 512
SF_ATOM_U16 = SF_ATOM_BYTES // 2
SF_COLS = 4
W_TILE_BYTES = CTA_M * CTA_K
A_COLS_PER_MMA = MMA_INST_K // 4
EVICT_FIRST = 0x12F0000000000000
_SMEM_LIMIT = 220 * 1024


@dataclass(frozen=True)
class IxQFusedConfig:
    """Launch geometry for :func:`ml4_ix_q_fused`."""

    reduce_mode: str  # "cluster" or "l2"
    split: int  # K-split factor dividing 48: {4, 8} for cluster, {4, 6, 8} for l2
    ring: int  # TMA weight pipeline stages
    contig: bool = False  # False = strided K-tiles, True = contiguous K-tiles
    a_tmem: bool = True  # Stage W in TMEM concurrently with LN + activation quant
    trigger_early: bool = False  # PDL early trigger (safe for cluster mode)


CANDIDATE_CONFIGS: tuple[IxQFusedConfig, ...] = (
    IxQFusedConfig("cluster", 4, 11, False, True, True),
    IxQFusedConfig("cluster", 4, 8, False, True, True),
    IxQFusedConfig("l2", 6, 4, False, True, False),
    IxQFusedConfig("l2", 6, 6, False, True, False),
    IxQFusedConfig("l2", 6, 8, False, True, False),
    IxQFusedConfig("l2", 6, 4, False, False, False),
    IxQFusedConfig("l2", 8, 3, False, True, False),
    IxQFusedConfig("l2", 8, 3, False, False, False),
)

DEFAULT_CONFIG = IxQFusedConfig("cluster", 4, 11, False, True, True)

_LOCK = threading.Lock()
_COMPILED: dict[tuple, object] = {}
_WORKSPACES: dict[tuple, tuple] = {}
_PLACEHOLDERS: dict[tuple, object] = {}
_ENGAGED: set[tuple] = set()
_DSL_MODULE = None


# =============================================================================
# Pure-Python host planning, SMEM/TMEM sizing, and numerical verification helpers.
# =============================================================================


def _acc_cols(nacc: int = 1, mma_n: int = MMA_N) -> int:
    return max(16, nacc * mma_n)


def _a_col0(my_tiles: int, nacc: int = 1, mma_n: int = MMA_N) -> int:
    return (_acc_cols(nacc, mma_n) + (my_tiles + 1) * SF_COLS + 31) // 32 * 32


def tmem_cols(ring: int, my_tiles: int, a_tmem: bool = True, mma_n: int = MMA_N) -> int:
    """TMEM columns allocated per CTA (power of two)."""
    need = _acc_cols(1, mma_n) + 2 * ring * SF_COLS
    if a_tmem:
        need = _a_col0(my_tiles, 1, mma_n) + my_tiles * K_BLOCKS * A_COLS_PER_MMA
    cols = 32
    while cols < need:
        cols *= 2
    return cols


def smem_bytes(cfg: IxQFusedConfig) -> int:
    """Estimated per-CTA shared memory usage in bytes for ``cfg``."""
    my_tiles = K_TILES // cfg.split
    ring = min(cfg.ring, my_tiles)
    mb = cfg.split * (CTA_M // cfg.split) * (MMA_N + 4) * 4 if cfg.reduce_mode == "cluster" else 0
    return (
        ring * (W_TILE_BYTES + SF_ATOM_BYTES)
        + my_tiles * (MMA_N * CTA_K + SF_ATOM_BYTES)
        + mb
        + 2048
    )


def get_config(cfg: Optional[IxQFusedConfig] = None) -> Optional[IxQFusedConfig]:
    """Validate and resolve a launch config for ``(N=4096, K=6144)``."""
    resolved = cfg if cfg is not None else DEFAULT_CONFIG
    if resolved.reduce_mode not in ("l2", "cluster"):
        return None
    if resolved.split <= 0 or K_TILES % resolved.split != 0:
        return None
    if resolved.reduce_mode == "cluster" and (
        resolved.split not in (4, 8) or CTA_M % resolved.split != 0
    ):
        return None
    if resolved.reduce_mode == "l2" and resolved.split < 1:
        return None
    my_tiles = K_TILES // resolved.split
    ring = min(max(2, resolved.ring), my_tiles)
    trig = resolved.trigger_early and resolved.reduce_mode == "cluster"
    cand = IxQFusedConfig(
        resolved.reduce_mode,
        resolved.split,
        ring,
        resolved.contig,
        resolved.a_tmem,
        trig,
    )
    while cand.ring > 2 and smem_bytes(cand) > _SMEM_LIMIT:
        cand = IxQFusedConfig(
            cand.reduce_mode,
            cand.split,
            cand.ring - 1,
            cand.contig,
            cand.a_tmem,
            cand.trigger_early,
        )
    if smem_bytes(cand) > _SMEM_LIMIT:
        return None
    if cand.a_tmem and tmem_cols(cand.ring, my_tiles, True) > 512:
        cand = IxQFusedConfig(
            cand.reduce_mode,
            cand.split,
            cand.ring,
            cand.contig,
            False,
            cand.trigger_early,
        )
    return cand


def serves(m: int, *, cfg: Optional[IxQFusedConfig] = None) -> bool:
    """Whether :data:`ml4_ix_q_fused` serves ``m`` tokens with the given config."""
    if not (0 < m <= MAX_TOKENS):
        return False
    return get_config(cfg) is not None


def exact_log_bf16_host(pos: int) -> float:
    """Host mirror of the kernel's Cody-Waite FP32 ``log(pos + 1.0)`` helper."""
    import struct

    x = float(pos) + 1.0
    bits = struct.unpack("<i", struct.pack("<f", x))[0]
    adj = bits + 0x004AFB0D
    exp_i = (adj >> 23) - 127
    m_bits = (adj & 0x007FFFFF) + 0x3F3504F3
    m = struct.unpack("<f", struct.pack("<i", m_bits))[0]

    def f32(v: float) -> float:
        return struct.unpack("<f", struct.pack("<f", v))[0]

    u = f32(f32(m - 1.0) / f32(m + 1.0))
    s = f32(u * u)
    poly = f32(1.0 / 9.0)
    poly = f32(f32(s * poly) + f32(1.0 / 7.0))
    poly = f32(f32(s * poly) + f32(1.0 / 5.0))
    poly = f32(f32(s * poly) + f32(1.0 / 3.0))
    ef = f32(float(exp_i))
    ln2_hi = f32(0.693145751953125)
    ln2_lo = f32(1.428606765330187e-6)
    rem = f32(f32(f32(2.0 * u) * f32(s * poly)) + f32(ef * ln2_lo))
    return f32(f32(ef * ln2_hi) + f32(f32(2.0 * u) + rem))


# =============================================================================
# CuTe DSL kernel (lazily built so CPU-only environments can import helpers).
# =============================================================================


def _build_dsl():
    global _DSL_MODULE, cuda_driver, cutlass, cute, cuda
    if _DSL_MODULE is not None:
        return _DSL_MODULE

    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import cutlass.experimental.cuda as cuda
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass.cutlass_dsl import T as _T
    from cutlass.cutlass_dsl import dsl_user_op
    from cutlass.experimental import primitives as prims

    a_dtype = cutlass.Float8E4M3FN
    b_dtype = cutlass.Float8E4M3FN

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
            [
                cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
                cutlass.Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "st.release.gpu.global.u32 [$0], $1;",
            "l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _fence_acq_rel_gpu(*, loc=None, ip=None):
        _llvm.inline_asm(
            None,
            [],
            "fence.acq_rel.gpu;",
            "",
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
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
            "[$0], {$1, $2, $3, $4}, [$5];",
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
            cutlass.Float32(_llvm.extractvalue(_T.f32(), out, [i], loc=loc, ip=ip))
            for i in range(4)
        ]

    @dsl_user_op
    def _test_wait_cluster(mbar, parity, *, loc=None, ip=None):
        done = cutlass.Int32(
            _llvm.inline_asm(
                _T.i32(),
                [
                    mbar.toint(loc=loc, ip=ip).ir_value(),
                    cutlass.Int32(parity).ir_value(loc=loc, ip=ip),
                ],
                "{\n\t.reg .pred p;\n\t"
                "mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\t"
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
    def _fmax_nan(a, b, *, loc=None, ip=None):
        return cutlass.Float32(
            _llvm.inline_asm(
                _T.f32(),
                [
                    cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                ],
                "max.NaN.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _abs_f32(x, *, loc=None, ip=None):
        return cutlass.Float32(
            _llvm.inline_asm(
                _T.f32(),
                [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
                "abs.f32 $0, $1;",
                "=f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _mul_rn(a, b, *, loc=None, ip=None):
        return cutlass.Float32(
            _llvm.inline_asm(
                _T.f32(),
                [
                    cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                ],
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _add_rn(a, b, *, loc=None, ip=None):
        return cutlass.Float32(
            _llvm.inline_asm(
                _T.f32(),
                [
                    cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                ],
                "add.rn.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _fma_rn(a, b, c, *, loc=None, ip=None):
        return cutlass.Float32(
            _llvm.inline_asm(
                _T.f32(),
                [
                    cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(c).ir_value(loc=loc, ip=ip),
                ],
                "fma.rn.f32 $0, $1, $2, $3;",
                "=f,f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _div_rn(a, b, *, loc=None, ip=None):
        return cutlass.Float32(
            _llvm.inline_asm(
                _T.f32(),
                [
                    cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                ],
                "div.rn.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _i32_to_f32(x, *, loc=None, ip=None):
        return cutlass.Float32(
            _llvm.inline_asm(
                _T.f32(),
                [cutlass.Int32(x).ir_value(loc=loc, ip=ip)],
                "cvt.rn.f32.s32 $0, $1;",
                "=f,r",
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
    def _bf16_round(x, *, loc=None, ip=None):
        return cutlass.Float32(
            _llvm.inline_asm(
                _T.f32(),
                [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
                "{\n\t.reg .b16 h, z;\n\tcvt.rn.bf16.f32 h, $1;\n\t"
                "mov.b16 z, 0;\n\tmov.b32 $0, {z, h};\n\t}",
                "=f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _bf16_round_ftz(x, *, loc=None, ip=None):
        return cutlass.Float32(
            _llvm.inline_asm(
                _T.f32(),
                [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
                "{\n\t.reg .b16 h, z;\n\t.reg .b32 bits, exp, sgn;\n\t.reg .pred p;\n\t"
                "cvt.rn.bf16.f32 h, $1;\n\tmov.b16 z, 0;\n\tmov.b32 bits, {z, h};\n\t"
                "and.b32 exp, bits, 0x7F800000;\n\tsetp.eq.u32 p, exp, 0;\n\t"
                "and.b32 sgn, bits, 0x80000000;\n\tselp.b32 bits, sgn, bits, p;\n\t"
                "mov.b32 $0, bits;\n\t}",
                "=f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    def _exact_log_f32(pos_i32):
        x = _add_rn(_i32_to_f32(pos_i32), cutlass.Float32(1.0))
        bits = x.bitcast(cutlass.Int32)
        adj = bits + cutlass.Int32(0x004AFB0D)
        exp_i = (adj >> cutlass.Int32(23)) - cutlass.Int32(127)
        m = ((adj & cutlass.Int32(0x007FFFFF)) + cutlass.Int32(0x3F3504F3)).bitcast(cutlass.Float32)
        u = _div_rn(
            _add_rn(m, cutlass.Float32(-1.0)),
            _add_rn(m, cutlass.Float32(1.0)),
        )
        s = _mul_rn(u, u)
        poly = cutlass.Float32(1.0 / 9.0)
        poly = _add_rn(_mul_rn(s, poly), cutlass.Float32(1.0 / 7.0))
        poly = _add_rn(_mul_rn(s, poly), cutlass.Float32(1.0 / 5.0))
        poly = _add_rn(_mul_rn(s, poly), cutlass.Float32(1.0 / 3.0))
        ef = _i32_to_f32(exp_i)
        two_u = _mul_rn(cutlass.Float32(2.0), u)
        rem = _add_rn(
            _mul_rn(two_u, _mul_rn(s, poly)),
            _mul_rn(ef, cutlass.Float32(1.428606765330187e-6)),
        )
        return _add_rn(
            _mul_rn(ef, cutlass.Float32(0.693145751953125)),
            _add_rn(two_u, rem),
        )

    def _k_tile(rank, i: int, split: int, my_tiles: int, contig: bool):
        if contig:
            return rank * cutlass.Int32(my_tiles) + cutlass.Int32(i)
        return rank + cutlass.Int32(i * split)

    @dsl_user_op
    def _neg_f32(x, *, loc=None, ip=None):
        return cutlass.Float32(
            _llvm.inline_asm(
                _T.f32(),
                [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
                "neg.f32 $0, $1;",
                "=f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    def _ssmax_rope_elem(val_f32, row_i32, pos_i32, ss_scale_f32, cos_sin_arr):
        q_bf16 = _bf16_round(val_f32)
        x_sc = _mul_rn(q_bf16, ss_scale_f32)
        x_r = _bf16_round_ftz(x_sc)
        partner = cute.arch.shuffle_sync_bfly(x_r, offset=1)
        is_rope = row_i32 < cutlass.Int32(ROPE_DIM)
        pair_idx = cutlass.select_(is_rope, row_i32 >> cutlass.Int32(1), cutlass.Int32(0))
        tab_base = cutlass.Int64(pos_i32) * cutlass.Int64(ROPE_DIM)
        c = cos_sin_arr.load(idx=tab_base + cutlass.Int64(pair_idx))
        sn = cos_sin_arr.load(idx=tab_base + cutlass.Int64(32) + cutlass.Int64(pair_idx))
        ps = _mul_rn(partner, sn)
        is_even = (row_i32 & cutlass.Int32(1)) == cutlass.Int32(0)
        signed_ps = cutlass.select_(is_even, _neg_f32(ps), ps)
        y_rope = _bf16_round(_fma_rn(x_r, c, signed_ps))
        y_nope = _bf16_round(x_sc)
        return cutlass.select_(is_rope, y_rope, y_nope)

    def _ue8m0_q_scale_and_inv(amax_f32):
        clamped = _fmax_nan(amax_f32, cutlass.Float32(1.0e-12))
        raw = _mul_rn(clamped, cutlass.Float32(1.0 / 448.0)).bitcast(cutlass.Int32)
        exp_bits = raw & cutlass.Int32(0x7F800000)
        exp_bits = cutlass.Int32(
            cutlass.select_(
                (raw & cutlass.Int32(0x007FFFFF)) != cutlass.Int32(0),
                exp_bits + cutlass.Int32(0x00800000),
                exp_bits,
            )
        )
        q_scale = exp_bits.bitcast(cutlass.Float32)
        inv = cutlass.Int32(cutlass.Int32(0x7F000000) - exp_bits).bitcast(cutlass.Float32)
        return q_scale, inv

    @cute.kernel
    def ml4_ix_q_fused_kernel(
        tma_desc_w0: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_wsf0: cutlass.GridConstant[cuda.TensorMap],
        x_words: cutlass.Array,  # int32 [M * (K/2)]: bf16x2 input x
        ln_w: cutlass.Array,  # int32 [K/2]: bf16x2 LayerNorm weight
        ln_b: cutlass.Array,  # int32 [K/2]: bf16x2 LayerNorm bias
        ln_eps: cutlass.Float32,
        positions: cutlass.Array,  # int32 [M]
        ssmax_s: cutlass.Array,  # int32 [16] (packed bf16x2 pairs)
        ssmax_b: cutlass.Array,  # int32 [16] (packed bf16x2 pairs)
        cos_sin: cutlass.Array,  # fp32 [max_pos * 64]
        wp_words: cutlass.Array,  # int32 [32 * (K/2)] when inline_wp else dummy
        kw_w: cutlass.Array,  # bf16 [M * kw_stride] when not inline_wp
        kw_stride: cutlass.Int32,  # row stride in bf16 elements of kw_w
        weight_scale: cutlass.Float32,
        q_fp8_u32: cutlass.Array,  # int32 [M * 32 * 32]: packed 4xe4m3 output
        weights_out: cutlass.Array,  # fp32 [M * 32]
        l2_partials: cutlass.Array,  # fp32 [32 * split * 8 * 128] when not use_cluster
        l2_wp_partials: cutlass.Array,  # fp32 [32 * split * 8] when not use_cluster and inline_wp
        l2_counters: cutlass.Array,  # int32 [32] when not use_cluster
        num_tokens: cutlass.Int32,
        split: cutlass.Constexpr[int],
        ring: cutlass.Constexpr[int],
        contig: cutlass.Constexpr[bool],
        a_tmem: cutlass.Constexpr[bool],
        use_cluster: cutlass.Constexpr[bool],
        trigger_early: cutlass.Constexpr[bool],
        inline_wp: cutlass.Constexpr[bool],
        max_m: cutlass.Constexpr[int],
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        warp_id = tx // 32
        lane = tx % 32

        my_tiles = K_TILES // split
        rows_per_rank = CTA_M // split if use_cluster else CTA_M
        x_tile_bytes = MMA_N * CTA_K
        mb_stride = MMA_N + 4

        tma_ptr_w0 = tma_desc_w0.get_ptr()
        tma_ptr_wsf0 = tma_desc_wsf0.get_ptr()

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
        wp_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
        mailbox = cutlass.Array(
            cutlass.Float32,
            split * rows_per_rank * mb_stride if (use_cluster and split > 1) else 4,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )
        reduce_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        amax_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        s_is_last = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
        s_ln_mean = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.smem, alignment=16)
        s_ln_rstd = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.smem, alignment=16)
        s_ss_sc = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.smem, alignment=16)
        s_wp = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.smem, alignment=16)
        s_amax = cutlass.Array(
            cutlass.Float32,
            max(split, 4) * 8,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )
        s_wp_cluster = cutlass.Array(
            cutlass.Float32,
            split * 8 if (use_cluster and inline_wp) else 8,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )

        rank = bx % cutlass.Int32(split)
        n_tile = bx // cutlass.Int32(split)
        m_offset = n_tile * cutlass.Int32(CTA_M)

        if warp_id == 0:
            prims.prefetch_tensormap(tma_ptr_w0)
            prims.prefetch_tensormap(tma_ptr_wsf0)
            if prims.elect_sync():
                for s in cutlass.range_constexpr(ring):
                    prims.mbarrier_init(w_full.subview(s), 1)
                    prims.mbarrier_init(mma_done.subview(s), 1)
                prims.mbarrier_init(act_full, 1)
                prims.mbarrier_init(acc_done, 1)
                if cutlass.const_expr(inline_wp):
                    prims.mbarrier_init(wp_done, 1)
                if cutlass.const_expr(use_cluster and split > 1):
                    prims.mbarrier_init(reduce_ready, 1)
                    prims.mbarrier_arrive_expect_tx(
                        reduce_ready, (split - 1) * rows_per_rank * MMA_N * 4
                    )
                    prims.mbarrier_init(amax_ready, split - 1)
        if warp_id == 2:
            prims.tcgen05_alloc(tmem_ptr_i32, tmem_cols(ring, my_tiles, a_tmem, MMA_N))
            prims.tcgen05_relinquish_alloc_permit()
        prims.fence_mbarrier_init()
        if cutlass.const_expr(use_cluster and split > 1):
            prims.barrier_cluster_arrive_relaxed()
            prims.barrier_cluster_wait()
        prims.barrier_cta_sync(0)

        if warp_id == 0:
            if prims.elect_sync():
                for i in cutlass.range_constexpr(ring, my_tiles):
                    k_global = _k_tile(rank, i, split, my_tiles, contig)
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
                    prims.mbarrier_arrive_expect_tx(
                        w_full.subview(stage), W_TILE_BYTES + SF_ATOM_BYTES
                    )
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
                        (cutlass.Int32(0), n_tile * cutlass.Int32(K_TILES) + k_global),
                        w_full.subview(stage),
                        l2_cache_hint=EVICT_FIRST,
                    )

        elif warp_id == 1:
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            if cutlass.const_expr(trigger_early):
                prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)

        elif warp_id == 2:
            tmem_raw = tmem_ptr_i32.load()
            base_col = tmem_raw & 0xFFFF
            base_row = tmem_raw >> 16
            idesc = prims.Tcgen05MxInstrDesc.build(
                a_dtype=a_dtype, b_dtype=b_dtype, scale_format=1, n_dim=MMA_N, m_dim=CTA_M
            )
            s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
            if cutlass.const_expr(a_tmem):
                sfa_col0 = base_col + _acc_cols(1, MMA_N)
                sfb_col0 = sfa_col0 + my_tiles * SF_COLS
                a_base = tmem_raw + cutlass.Int32(_a_col0(my_tiles, 1, MMA_N))
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
                while not cute.arch.mbarrier_test_wait(act_full.data_ptr(), 0):
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
                sfa_col0 = base_col + _acc_cols(1, MMA_N)
                sfb_col0 = sfa_col0 + ring * SF_COLS
                while not cute.arch.mbarrier_test_wait(act_full.data_ptr(), 0):
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
                                cutlass.inttoptr(tmem_raw, 6, cutlass.Float32),
                                desc_a + inc,
                                desc_b + inc,
                                idesc.set_sf_ids(a_sf_id=kb, b_sf_id=kb),
                                j > 0,
                                cutlass.inttoptr(sfa_tmem, 6, cutlass.Int32),
                                cutlass.inttoptr(sfb_tmem, 6, cutlass.Int32),
                            )
                        prims.tcgen05_commit(mma_done.subview(stage))
                if prims.elect_sync():
                    prims.tcgen05_commit(acc_done)

        elif warp_id == 3:
            # Inline head-weight projection dot(x[t, k_slice], wp[n_tile, k_slice]) on idle warp 3.
            if cutlass.const_expr(inline_wp):
                prims.griddepcontrol(prims.GridDepAction.WAIT)
                wp_acc = [cutlass.Float32(0.0) for _ in range(max_m)]
                wp_row_base = n_tile * cutlass.Int32(K_IN // 2)
                for i in cutlass.range_constexpr(my_tiles):
                    k_global = _k_tile(rank, i, split, my_tiles, contig)
                    k_word_off = k_global * cutlass.Int32(CTA_K // 2) + lane * cutlass.Int32(2)
                    w_v2 = wp_words.load(idx=wp_row_base + k_word_off, vector_size=2, alignment=8)
                    w0_i = cutlass.Int32(w_v2[0])
                    w1_i = cutlass.Int32(w_v2[1])
                    wf0 = (w0_i << cutlass.Int32(16)).bitcast(cutlass.Float32)
                    wf1 = (w0_i & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                    wf2 = (w1_i << cutlass.Int32(16)).bitcast(cutlass.Float32)
                    wf3 = (w1_i & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                    for t in cutlass.range_constexpr(max_m):
                        t_ld = cutlass.select_(
                            cutlass.Int32(t) < num_tokens, cutlass.Int32(t), cutlass.Int32(0)
                        )
                        x_v2 = x_words.load(
                            idx=t_ld * cutlass.Int32(K_IN // 2) + k_word_off,
                            vector_size=2,
                            alignment=8,
                        )
                        x0_i = cutlass.Int32(x_v2[0])
                        x1_i = cutlass.Int32(x_v2[1])
                        xf0 = (x0_i << cutlass.Int32(16)).bitcast(cutlass.Float32)
                        xf1 = (x0_i & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                        xf2 = (x1_i << cutlass.Int32(16)).bitcast(cutlass.Float32)
                        xf3 = (x1_i & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                        wp_acc[t] = wp_acc[t] + xf0 * wf0 + xf1 * wf1 + xf2 * wf2 + xf3 * wf3
                for t in cutlass.range_constexpr(max_m):
                    v = wp_acc[t]
                    for off in cutlass.range_constexpr(5):
                        v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << off)
                    if lane == 0:
                        s_wp.store(v, idx=t)
                if lane == 0:
                    prims.mbarrier_arrive(wp_done)

        elif warp_id >= 4:
            row = (warp_id - 4) * 32 + lane
            qz_tid = tx - cutlass.Int32(128)
            prims.griddepcontrol(prims.GridDepAction.WAIT)

            # Single-pass shift-invariant FP32 LayerNorm (shifted by x[t, 0]).
            if cutlass.const_expr(max_m <= 4):
                ln_tok = qz_tid // cutlass.Int32(32)
                ln_part = qz_tid % cutlass.Int32(32)
                ln_tok_ld = cutlass.select_(ln_tok < num_tokens, ln_tok, cutlass.Int32(0))
                ln_base = ln_tok_ld * cutlass.Int32(K_IN // 2)
                ln_vecs_per_part = K_IN // 8 // 32  # 24
                x0_w = cutlass.Int32(x_words.load(idx=ln_base))
                ln_c = (x0_w << cutlass.Int32(16)).bitcast(cutlass.Float32)
                ln_s1 = cutlass.Float32(0.0)
                ln_s2 = cutlass.Float32(0.0)
                for r in cutlass.range(ln_vecs_per_part, unroll=4):
                    v4 = x_words.load(
                        idx=ln_base + (ln_part + r * cutlass.Int32(32)) * cutlass.Int32(4),
                        vector_size=4,
                        alignment=16,
                    )
                    for e4 in cutlass.range_constexpr(4):
                        w0 = cutlass.Int32(v4[e4])
                        d0 = (w0 << cutlass.Int32(16)).bitcast(cutlass.Float32) - ln_c
                        d1 = (w0 & cutlass.Int32(-65536)).bitcast(cutlass.Float32) - ln_c
                        ln_s1 = ln_s1 + d0 + d1
                        ln_s2 = ln_s2 + d0 * d0 + d1 * d1
                for off in cutlass.range_constexpr(5):
                    ln_s1 = ln_s1 + cute.arch.shuffle_sync_bfly(ln_s1, offset=1 << off)
                    ln_s2 = ln_s2 + cute.arch.shuffle_sync_bfly(ln_s2, offset=1 << off)
                if ln_part == cutlass.Int32(0):
                    mean_d = ln_s1 / cutlass.Float32(K_IN)
                    ln_mean = ln_c + mean_d
                    ln_var = cute.arch.fmax(
                        ln_s2 / cutlass.Float32(K_IN) - mean_d * mean_d,
                        cutlass.Float32(0.0),
                    )
                    s_ln_mean.store(ln_mean, idx=ln_tok)
                    s_ln_rstd.store(
                        cute.math.rsqrt(ln_var + ln_eps, fastmath=False),
                        idx=ln_tok,
                    )
            else:
                ln_tok = qz_tid // cutlass.Int32(16)
                ln_part = qz_tid % cutlass.Int32(16)
                ln_tok_ld = cutlass.select_(ln_tok < num_tokens, ln_tok, cutlass.Int32(0))
                ln_base = ln_tok_ld * cutlass.Int32(K_IN // 2)
                ln_vecs_per_part = K_IN // 8 // 16  # 48
                x0_w = cutlass.Int32(x_words.load(idx=ln_base))
                ln_c = (x0_w << cutlass.Int32(16)).bitcast(cutlass.Float32)
                ln_s1 = cutlass.Float32(0.0)
                ln_s2 = cutlass.Float32(0.0)
                for r in cutlass.range(ln_vecs_per_part, unroll=4):
                    v4 = x_words.load(
                        idx=ln_base + (ln_part + r * cutlass.Int32(16)) * cutlass.Int32(4),
                        vector_size=4,
                        alignment=16,
                    )
                    for e4 in cutlass.range_constexpr(4):
                        w0 = cutlass.Int32(v4[e4])
                        d0 = (w0 << cutlass.Int32(16)).bitcast(cutlass.Float32) - ln_c
                        d1 = (w0 & cutlass.Int32(-65536)).bitcast(cutlass.Float32) - ln_c
                        ln_s1 = ln_s1 + d0 + d1
                        ln_s2 = ln_s2 + d0 * d0 + d1 * d1
                for off in cutlass.range_constexpr(4):
                    ln_s1 = ln_s1 + cute.arch.shuffle_sync_bfly(ln_s1, offset=1 << off)
                    ln_s2 = ln_s2 + cute.arch.shuffle_sync_bfly(ln_s2, offset=1 << off)
                if ln_part == cutlass.Int32(0):
                    mean_d = ln_s1 / cutlass.Float32(K_IN)
                    ln_mean = ln_c + mean_d
                    ln_var = cute.arch.fmax(
                        ln_s2 / cutlass.Float32(K_IN) - mean_d * mean_d,
                        cutlass.Float32(0.0),
                    )
                    s_ln_mean.store(ln_mean, idx=ln_tok)
                    s_ln_rstd.store(
                        cute.math.rsqrt(ln_var + ln_eps, fastmath=False),
                        idx=ln_tok,
                    )
            prims.barrier_cta_sync(3, thread_count=128)

            # Exact ML4 1x128 FP8 activation quantization into smem_b / smem_sfb.
            qz_b_words = cutlass.Array(
                smem_b.data_ptr(0),
                shape=(my_tiles * x_tile_bytes // 4,),
                dtype=cutlass.Int32,
                alignment=16,
            )
            qz_items = my_tiles * K_BLOCKS * MMA_N
            qz_rounds = (qz_items + 127) // 128
            for qz_r in cutlass.range_constexpr(qz_rounds):
                qz_item = qz_tid + cutlass.Int32(qz_r * 128)
                qz_kb = qz_item & cutlass.Int32(3)
                qz_t = (qz_item >> cutlass.Int32(2)) % cutlass.Int32(MMA_N)
                qz_i = (qz_item >> cutlass.Int32(2)) // cutlass.Int32(MMA_N)
                if cutlass.const_expr(qz_r * 128 + 127 >= qz_items):
                    qz_i = cutlass.select_(
                        qz_item < cutlass.Int32(qz_items), qz_i, cutlass.Int32(0)
                    )
                qz_k_global = rank + qz_i * cutlass.Int32(split)
                if cutlass.const_expr(contig):
                    qz_k_global = rank * cutlass.Int32(my_tiles) + qz_i
                qz_t_ld = cutlass.select_(qz_t < num_tokens, qz_t, cutlass.Int32(0))
                qz_col_w = qz_k_global * cutlass.Int32(CTA_K // 2) + qz_kb * cutlass.Int32(
                    MMA_INST_K // 2
                )
                qz_x_base = qz_t_ld * cutlass.Int32(K_IN // 2) + qz_col_w
                qz_vecs = [
                    x_words.load(
                        idx=qz_x_base + cutlass.Int32(qz_v * 4), vector_size=4, alignment=16
                    )
                    for qz_v in range(4)
                ]
                qz_vals = []
                qz_amax = cutlass.Float32(0.0)
                qz_mean = s_ln_mean.load(idx=qz_t_ld)
                qz_rstd = s_ln_rstd.load(idx=qz_t_ld)
                for qz_v in cutlass.range_constexpr(4):
                    lnw_v = ln_w.load(
                        idx=qz_col_w + cutlass.Int32(qz_v * 4), vector_size=4, alignment=16
                    )
                    lnb_v = ln_b.load(
                        idx=qz_col_w + cutlass.Int32(qz_v * 4), vector_size=4, alignment=16
                    )
                    for qz_e4 in cutlass.range_constexpr(4):
                        qz_word = cutlass.Int32(qz_vecs[qz_v][qz_e4])
                        ww = cutlass.Int32(lnw_v[qz_e4])
                        bw = cutlass.Int32(lnb_v[qz_e4])
                        qz_lo = (qz_word << cutlass.Int32(16)).bitcast(cutlass.Float32)
                        qz_hi = (qz_word & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                        wl = (ww << cutlass.Int32(16)).bitcast(cutlass.Float32)
                        wh = (ww & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                        bl = (bw << cutlass.Int32(16)).bitcast(cutlass.Float32)
                        bh = (bw & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                        qz_lo = _bf16_round((qz_lo - qz_mean) * qz_rstd * wl + bl)
                        qz_hi = _bf16_round((qz_hi - qz_mean) * qz_rstd * wh + bh)
                        qz_vals.append(qz_lo)
                        qz_vals.append(qz_hi)
                        qz_amax = _absmax(_absmax(qz_amax, qz_lo), qz_hi)

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

            # Precompute per-token SSMax scale once (4/8 threads run _exact_log_f32 into SMEM).
            if qz_tid < cutlass.Int32(max_m):
                s_pair = ssmax_s.load(idx=n_tile >> cutlass.Int32(1))
                b_pair = ssmax_b.load(idx=n_tile >> cutlass.Int32(1))
                head_even = (n_tile & cutlass.Int32(1)) == cutlass.Int32(0)
                s_h = cutlass.select_(
                    head_even,
                    (s_pair << cutlass.Int32(16)).bitcast(cutlass.Float32),
                    (s_pair & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
                )
                b_h = cutlass.select_(
                    head_even,
                    (b_pair << cutlass.Int32(16)).bitcast(cutlass.Float32),
                    (b_pair & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
                )
                t_pre = cutlass.select_(qz_tid < num_tokens, qz_tid, cutlass.Int32(0))
                lp_pre = _bf16_round(_exact_log_f32(positions.load(idx=t_pre)))
                s_ss_sc.store(
                    _bf16_round(_add_rn(_bf16_round(_mul_rn(lp_pre, s_h)), b_h)), idx=qz_tid
                )
            prims.barrier_cta_sync(7, thread_count=128)

            # Wait for MMA completion and load FP32 partial accumulators from TMEM.
            while not cute.arch.mbarrier_test_wait(acc_done.data_ptr(), 0):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            piece = prims.tcgen05_ld(
                "32x32b",
                cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32),
                num=MMA_N,
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            acc = [cutlass.Float32(piece[t]) for t in range(MMA_N)]
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)

            if cutlass.const_expr(use_cluster):
                owner = row // cutlass.Int32(rows_per_rank)
                local = row % cutlass.Int32(rows_per_rank)
                if owner != rank:
                    base = (rank * cutlass.Int32(rows_per_rank) + local) * cutlass.Int32(mb_stride)
                    dst = _mapa_u32(mailbox.subview(base).data_ptr(), owner)
                    mb = _mapa_u32(reduce_ready.data_ptr(), owner)
                    words = [acc[t].bitcast(cutlass.Int32) for t in range(MMA_N)]
                    for g in cutlass.range_constexpr(MMA_N // 4):
                        _st_async_v4(
                            dst + cutlass.Int32(16 * g),
                            words[4 * g],
                            words[4 * g + 1],
                            words[4 * g + 2],
                            words[4 * g + 3],
                            mb,
                        )
                elif cutlass.const_expr(split == 4):
                    # Fast warp-uniform path (1 full warp = 32 lanes owns rows_per_rank=32):
                    # zero mailbox re-write, zero extra CTA barriers, 3/4 warps skip epilogue.
                    while not _test_wait_cluster(reduce_ready.data_ptr(), 0):
                        pass
                    totals = []
                    for g in cutlass.range_constexpr(max_m // 4):
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
                                    acc[4 * g + j],
                                    parts[j],
                                )
                                for j in range(4)
                            ]
                        for j in cutlass.range_constexpr(4):
                            totals.append(sums[j])
                    y_vals = []
                    local_amax = []
                    for t in cutlass.range_constexpr(max_m):
                        t_ld = cutlass.select_(
                            cutlass.Int32(t) < num_tokens, cutlass.Int32(t), cutlass.Int32(0)
                        )
                        pos_t = positions.load(idx=t_ld)
                        ss_sc = s_ss_sc.load(idx=cutlass.Int32(t))
                        y_t = _ssmax_rope_elem(totals[t], row, pos_t, ss_sc, cos_sin)
                        y_vals.append(y_t)
                        am = _abs_f32(y_t)
                        for off in cutlass.range_constexpr(5):
                            am = _fmax_nan(am, cute.arch.shuffle_sync_bfly(am, offset=1 << off))
                        local_amax.append(am)
                    if local == cutlass.Int32(0):
                        if cutlass.const_expr(inline_wp):
                            while not cute.arch.mbarrier_test_wait(wp_done.data_ptr(), 0):
                                pass
                        for t in cutlass.range_constexpr(max_m):
                            s_amax.store(
                                local_amax[t], idx=rank * cutlass.Int32(8) + cutlass.Int32(t)
                            )
                            if cutlass.const_expr(inline_wp):
                                wp_v = s_wp.load(idx=t)
                                if rank == cutlass.Int32(0):
                                    s_wp_cluster.store(wp_v, idx=cutlass.Int32(t))
                                else:
                                    prims.mapa(
                                        s_wp_cluster.subview(
                                            rank * cutlass.Int32(8) + cutlass.Int32(t)
                                        ),
                                        0,
                                    ).store(wp_v)
                        for peer in cutlass.range_constexpr(split):
                            if rank != cutlass.Int32(peer):
                                for t in cutlass.range_constexpr(max_m):
                                    prims.mapa(
                                        s_amax.subview(rank * cutlass.Int32(8) + cutlass.Int32(t)),
                                        peer,
                                    ).store(local_amax[t])
                                prims.mbarrier_arrive(
                                    prims.mapa(amax_ready, peer), scope=prims.MemScope.CLUSTER
                                )
                        while not prims.mbarrier_try_wait_parity(
                            amax_ready, 0, scope=prims.MBarrierScope.CLUSTER
                        ):
                            pass
                    cute.arch.sync_warp()
                    for t in cutlass.range_constexpr(max_m):
                        head_am = s_amax.load(idx=cutlass.Int32(t))
                        for q in cutlass.range_constexpr(1, split):
                            head_am = _fmax_nan(head_am, s_amax.load(idx=cutlass.Int32(q * 8 + t)))
                        q_scale, inv = _ue8m0_q_scale_and_inv(head_am)
                        q0 = _mul_rn(y_vals[t], inv)
                        q1 = cute.arch.shuffle_sync_bfly(q0, offset=1)
                        q2 = cute.arch.shuffle_sync_bfly(q0, offset=2)
                        q3 = cute.arch.shuffle_sync_bfly(q1, offset=2)
                        packed = _e4m3x4(q0, q1, q2, q3)
                        if ((lane & cutlass.Int32(3)) == cutlass.Int32(0)) & (
                            cutlass.Int32(t) < num_tokens
                        ):
                            out_idx = (cutlass.Int32(t * HEADS) + n_tile) * cutlass.Int32(
                                HEAD_DIM // 4
                            ) + (row >> cutlass.Int32(2))
                            q_fp8_u32.store(packed, idx=out_idx)
                        if (
                            (rank == cutlass.Int32(0))
                            & (local == cutlass.Int32(0))
                            & (cutlass.Int32(t) < num_tokens)
                        ):
                            if cutlass.const_expr(inline_wp):
                                w_sum = cutlass.Float32(0.0)
                                for q in cutlass.range_constexpr(split):
                                    w_sum = w_sum + s_wp_cluster.load(idx=cutlass.Int32(q * 8 + t))
                                w_raw = _bf16_round(w_sum)
                            else:
                                w_raw = cutlass.Float32(
                                    kw_w.load(idx=cutlass.Int32(t) * kw_stride + n_tile)
                                )
                            weights_out.store(
                                _mul_rn(_mul_rn(w_raw, q_scale), weight_scale),
                                idx=cutlass.Int32(t * HEADS) + n_tile,
                            )
                else:
                    while not _test_wait_cluster(reduce_ready.data_ptr(), 0):
                        pass
                    for g in cutlass.range_constexpr(max_m // 4):
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
                                    acc[4 * g + j],
                                    parts[j],
                                )
                                for j in range(4)
                            ]
                        for j in cutlass.range_constexpr(4):
                            mailbox.store(
                                sums[j],
                                idx=local * cutlass.Int32(mb_stride) + cutlass.Int32(4 * g + j),
                            )
                if cutlass.const_expr(split != 4):
                    prims.barrier_cta_sync(2, thread_count=128)
                    y_vals = []
                    local_amax = []
                    shfl_steps = 5 if rows_per_rank == 32 else 4
                    for t in cutlass.range_constexpr(max_m):
                        t_ld = cutlass.select_(
                            cutlass.Int32(t) < num_tokens, cutlass.Int32(t), cutlass.Int32(0)
                        )
                        tot_t = cutlass.select_(
                            owner == rank,
                            mailbox.load(idx=local * cutlass.Int32(mb_stride) + cutlass.Int32(t)),
                            cutlass.Float32(0.0),
                        )
                        pos_t = positions.load(idx=t_ld)
                        ss_sc = s_ss_sc.load(idx=cutlass.Int32(t))
                        y_t = _ssmax_rope_elem(tot_t, row, pos_t, ss_sc, cos_sin)
                        y_vals.append(y_t)
                        am = _abs_f32(y_t)
                        for off in cutlass.range_constexpr(shfl_steps):
                            am = _fmax_nan(am, cute.arch.shuffle_sync_bfly(am, offset=1 << off))
                        local_amax.append(am)

                    if (owner == rank) & (local == cutlass.Int32(0)):
                        if cutlass.const_expr(inline_wp):
                            while not cute.arch.mbarrier_test_wait(wp_done.data_ptr(), 0):
                                pass
                        for t in cutlass.range_constexpr(max_m):
                            s_amax.store(
                                local_amax[t], idx=rank * cutlass.Int32(8) + cutlass.Int32(t)
                            )
                            if cutlass.const_expr(inline_wp):
                                wp_v = s_wp.load(idx=t)
                                if rank == cutlass.Int32(0):
                                    s_wp_cluster.store(wp_v, idx=cutlass.Int32(t))
                                else:
                                    prims.mapa(
                                        s_wp_cluster.subview(
                                            rank * cutlass.Int32(8) + cutlass.Int32(t)
                                        ),
                                        0,
                                    ).store(wp_v)
                        for peer in cutlass.range_constexpr(split):
                            if rank != cutlass.Int32(peer):
                                for t in cutlass.range_constexpr(max_m):
                                    prims.mapa(
                                        s_amax.subview(rank * cutlass.Int32(8) + cutlass.Int32(t)),
                                        peer,
                                    ).store(local_amax[t])
                                prims.mbarrier_arrive(
                                    prims.mapa(amax_ready, peer), scope=prims.MemScope.CLUSTER
                                )
                        while not prims.mbarrier_try_wait_parity(
                            amax_ready, 0, scope=prims.MBarrierScope.CLUSTER
                        ):
                            pass
                    prims.barrier_cta_sync(4, thread_count=128)

                    for t in cutlass.range_constexpr(max_m):
                        head_am = s_amax.load(idx=cutlass.Int32(t))
                        for q in cutlass.range_constexpr(1, split):
                            head_am = _fmax_nan(head_am, s_amax.load(idx=cutlass.Int32(q * 8 + t)))
                        q_scale, inv = _ue8m0_q_scale_and_inv(head_am)
                        q0 = _mul_rn(y_vals[t], inv)
                        q1 = cute.arch.shuffle_sync_bfly(q0, offset=1)
                        q2 = cute.arch.shuffle_sync_bfly(q0, offset=2)
                        q3 = cute.arch.shuffle_sync_bfly(q1, offset=2)
                        packed = _e4m3x4(q0, q1, q2, q3)
                        if (
                            (owner == rank)
                            & ((lane & cutlass.Int32(3)) == cutlass.Int32(0))
                            & (cutlass.Int32(t) < num_tokens)
                        ):
                            out_idx = (cutlass.Int32(t * HEADS) + n_tile) * cutlass.Int32(
                                HEAD_DIM // 4
                            ) + (row >> cutlass.Int32(2))
                            q_fp8_u32.store(packed, idx=out_idx)
                        if (
                            (rank == cutlass.Int32(0))
                            & (owner == cutlass.Int32(0))
                            & (local == cutlass.Int32(0))
                            & (cutlass.Int32(t) < num_tokens)
                        ):
                            if cutlass.const_expr(inline_wp):
                                w_sum = cutlass.Float32(0.0)
                                for q in cutlass.range_constexpr(split):
                                    w_sum = w_sum + s_wp_cluster.load(idx=cutlass.Int32(q * 8 + t))
                                w_raw = _bf16_round(w_sum)
                            else:
                                w_raw = cutlass.Float32(
                                    kw_w.load(idx=cutlass.Int32(t) * kw_stride + n_tile)
                                )
                            weights_out.store(
                                _mul_rn(_mul_rn(w_raw, q_scale), weight_scale),
                                idx=cutlass.Int32(t * HEADS) + n_tile,
                            )
            else:
                # L2 partial-sum reduction (cluster=(1,1,1), supports split=6 -> 192 CTAs = 96/die).
                p_base = (n_tile * cutlass.Int32(split) + rank) * cutlass.Int32(8 * CTA_M)
                for t in cutlass.range_constexpr(max_m):
                    if cutlass.Int32(t) < num_tokens:
                        l2_partials.store(acc[t], idx=p_base + cutlass.Int32(t * CTA_M) + row)
                prims.barrier_cta_sync(4, thread_count=128)
                if qz_tid == cutlass.Int32(0):
                    if cutlass.const_expr(inline_wp):
                        while not cute.arch.mbarrier_test_wait(wp_done.data_ptr(), 0):
                            pass
                        wp_base = (n_tile * cutlass.Int32(split) + rank) * cutlass.Int32(8)
                        for t in cutlass.range_constexpr(max_m):
                            if cutlass.Int32(t) < num_tokens:
                                l2_wp_partials.store(
                                    s_wp.load(idx=t), idx=wp_base + cutlass.Int32(t)
                                )
                    _fence_acq_rel_gpu()
                    ctr_addr = l2_counters.data_ptr(0).toint() + cutlass.Int64(
                        n_tile
                    ) * cutlass.Int64(4)
                    old = _atom_add_u32(ctr_addr, cutlass.Int32(1))
                    s_is_last.store(
                        cutlass.select_(
                            old == cutlass.Int32(split - 1), cutlass.Int32(1), cutlass.Int32(0)
                        ),
                        idx=0,
                    )
                prims.barrier_cta_sync(5, thread_count=128)

                if s_is_last.load(idx=0) != cutlass.Int32(0):
                    head_p_base = n_tile * cutlass.Int32(split * 8 * CTA_M)
                    y_vals = []
                    for t in cutlass.range_constexpr(max_m):
                        t_ld = cutlass.select_(
                            cutlass.Int32(t) < num_tokens, cutlass.Int32(t), cutlass.Int32(0)
                        )
                        tot = cutlass.Float32(0.0)
                        for q in cutlass.range_constexpr(split):
                            tot = tot + l2_partials.load(
                                idx=head_p_base
                                + cutlass.Int32(q * 8 * CTA_M)
                                + t_ld * cutlass.Int32(CTA_M)
                                + row
                            )
                        pos_t = positions.load(idx=t_ld)
                        ss_sc = s_ss_sc.load(idx=cutlass.Int32(t))
                        y_t = _ssmax_rope_elem(tot, row, pos_t, ss_sc, cos_sin)
                        y_vals.append(y_t)
                        am = _abs_f32(y_t)
                        for off in cutlass.range_constexpr(5):
                            am = _fmax_nan(am, cute.arch.shuffle_sync_bfly(am, offset=1 << off))
                        if lane == 0:
                            s_amax.store(
                                am, idx=(warp_id - 4) * cutlass.Int32(8) + cutlass.Int32(t)
                            )
                    prims.barrier_cta_sync(6, thread_count=128)

                    for t in cutlass.range_constexpr(max_m):
                        head_am = _fmax_nan(
                            _fmax_nan(
                                s_amax.load(idx=cutlass.Int32(t)),
                                s_amax.load(idx=cutlass.Int32(8 + t)),
                            ),
                            _fmax_nan(
                                s_amax.load(idx=cutlass.Int32(16 + t)),
                                s_amax.load(idx=cutlass.Int32(24 + t)),
                            ),
                        )
                        q_scale, inv = _ue8m0_q_scale_and_inv(head_am)
                        q0 = _mul_rn(y_vals[t], inv)
                        q1 = cute.arch.shuffle_sync_bfly(q0, offset=1)
                        q2 = cute.arch.shuffle_sync_bfly(q0, offset=2)
                        q3 = cute.arch.shuffle_sync_bfly(q1, offset=2)
                        packed = _e4m3x4(q0, q1, q2, q3)
                        if ((lane & cutlass.Int32(3)) == cutlass.Int32(0)) & (
                            cutlass.Int32(t) < num_tokens
                        ):
                            out_idx = (cutlass.Int32(t * HEADS) + n_tile) * cutlass.Int32(
                                HEAD_DIM // 4
                            ) + (row >> cutlass.Int32(2))
                            q_fp8_u32.store(packed, idx=out_idx)
                        if (qz_tid == cutlass.Int32(0)) & (cutlass.Int32(t) < num_tokens):
                            if cutlass.const_expr(inline_wp):
                                w_sum = cutlass.Float32(0.0)
                                wp_h_base = n_tile * cutlass.Int32(split * 8)
                                for q in cutlass.range_constexpr(split):
                                    w_sum = w_sum + l2_wp_partials.load(
                                        idx=wp_h_base + cutlass.Int32(q * 8 + t)
                                    )
                                w_raw = _bf16_round(w_sum)
                            else:
                                w_raw = cutlass.Float32(
                                    kw_w.load(idx=cutlass.Int32(t) * kw_stride + n_tile)
                                )
                            weights_out.store(
                                _mul_rn(_mul_rn(w_raw, q_scale), weight_scale),
                                idx=cutlass.Int32(t * HEADS) + n_tile,
                            )
                    prims.barrier_cta_sync(7, thread_count=128)
                    if qz_tid == cutlass.Int32(0):
                        ctr_addr = l2_counters.data_ptr(0).toint() + cutlass.Int64(
                            n_tile
                        ) * cutlass.Int64(4)
                        _st_release_u32(ctr_addr, cutlass.Int32(0))

            prims.barrier_cta_sync(1, thread_count=128)
            if warp_id == 4:
                prims.tcgen05_dealloc(
                    cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32),
                    tmem_cols(ring, my_tiles, a_tmem, MMA_N),
                )

        if cutlass.const_expr(use_cluster and split > 1):
            prims.barrier_cluster_arrive_relaxed()
            prims.barrier_cluster_wait()

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
    def ml4_ix_q_fused_jit(
        w0: cute.Tensor,
        wsf0: cute.Tensor,
        x_words: cute.Tensor,
        ln_w: cute.Tensor,
        ln_b: cute.Tensor,
        ln_eps: cutlass.Float32,
        positions: cute.Tensor,
        ssmax_s: cute.Tensor,
        ssmax_b: cute.Tensor,
        cos_sin: cute.Tensor,
        wp_words: cute.Tensor,
        kw_w: cute.Tensor,
        kw_stride: cutlass.Int32,
        weight_scale: cutlass.Float32,
        q_fp8_u32: cute.Tensor,
        weights_out: cute.Tensor,
        l2_partials: cute.Tensor,
        l2_wp_partials: cute.Tensor,
        l2_counters: cute.Tensor,
        num_tokens: cutlass.Int32,
        split: cutlass.Constexpr[int],
        ring: cutlass.Constexpr[int],
        contig: cutlass.Constexpr[bool],
        a_tmem: cutlass.Constexpr[bool],
        use_cluster: cutlass.Constexpr[bool],
        trigger_early: cutlass.Constexpr[bool],
        inline_wp: cutlass.Constexpr[bool],
        max_m: cutlass.Constexpr[int],
        use_pdl: cutlass.Constexpr[bool],
        stream: cuda_driver.CUstream,
    ) -> None:
        tma_desc_w0 = _bytes_map(w0, K_IN, N_OUT, CTA_M, cuda.TensorMapSwizzle.s128b)
        tma_desc_wsf0 = _atom_map(wsf0, N_TILES * K_TILES)
        cluster_dim = (split, 1, 1) if cutlass.const_expr(use_cluster) else (1, 1, 1)
        ml4_ix_q_fused_kernel(
            tma_desc_w0,
            tma_desc_wsf0,
            x_words,
            ln_w,
            ln_b,
            ln_eps,
            positions,
            ssmax_s,
            ssmax_b,
            cos_sin,
            wp_words,
            kw_w,
            kw_stride,
            weight_scale,
            q_fp8_u32,
            weights_out,
            l2_partials,
            l2_wp_partials,
            l2_counters,
            num_tokens,
            split,
            ring,
            contig,
            a_tmem,
            use_cluster,
            trigger_early,
            inline_wp,
            max_m,
        ).launch(
            grid=(N_TILES * split, 1, 1),
            block=(THREADS, 1, 1),
            cluster=cluster_dim,
            stream=stream,
            use_pdl=use_pdl,
        )

    _DSL_MODULE = ml4_ix_q_fused_jit
    return _DSL_MODULE


# =============================================================================
# Runtime dispatch, workspace management, and OpWrapper.
# =============================================================================


def _arg(t: "torch.Tensor", align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _placeholder_i32(device) -> "torch.Tensor":
    key = ("i32", device)
    t = _PLACEHOLDERS.get(key)
    if t is None:
        t = _PLACEHOLDERS[key] = torch.zeros(16, dtype=torch.int32, device=device)
    return t  # type: ignore[return-value]


def _placeholder_bf16(device) -> "torch.Tensor":
    key = ("bf16", device)
    t = _PLACEHOLDERS.get(key)
    if t is None:
        t = _PLACEHOLDERS[key] = torch.zeros(64, dtype=torch.bfloat16, device=device)
    return t  # type: ignore[return-value]


def _placeholder_f32(device) -> "torch.Tensor":
    key = ("f32", device)
    t = _PLACEHOLDERS.get(key)
    if t is None:
        t = _PLACEHOLDERS[key] = torch.zeros(16, dtype=torch.float32, device=device)
    return t  # type: ignore[return-value]


def _get_l2_workspace(device, split: int) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    key = (device, split)
    ws = _WORKSPACES.get(key)
    if ws is None:
        with _LOCK:
            ws = _WORKSPACES.get(key)
            if ws is None:
                partials = torch.empty(
                    N_TILES * split * 8 * CTA_M, dtype=torch.float32, device=device
                )
                wp_partials = torch.empty(N_TILES * split * 8, dtype=torch.float32, device=device)
                counters = torch.zeros(N_TILES, dtype=torch.int32, device=device)
                ws = _WORKSPACES[key] = (partials, wp_partials, counters)
    return ws  # type: ignore[return-value]


def _get_enable_pdl() -> bool:
    try:
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        return bool(get_env_enable_pdl())
    except ImportError:
        return True


def _run(
    x: "torch.Tensor",
    norm_weight: "torch.Tensor",
    norm_bias: "torch.Tensor",
    eps: float,
    weight: "torch.Tensor",
    weight_scale: "torch.Tensor",
    positions: "torch.Tensor",
    ssmax_s: "torch.Tensor",
    ssmax_b: "torch.Tensor",
    cos_sin: "torch.Tensor",
    head_weight_scale: float,
    q_fp8_out: "torch.Tensor",
    weights_out: "torch.Tensor",
    *,
    wp: Optional["torch.Tensor"] = None,
    kw_w: Optional["torch.Tensor"] = None,
    cfg: Optional[IxQFusedConfig] = None,
    use_pdl: Optional[bool] = None,
) -> bool:
    if torch is None:
        return False
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.dim() != 2
        or not x.is_contiguous()
        or x.data_ptr() % 16 != 0
        or x.shape[1] != K_IN
    ):
        return False
    m = x.shape[0]
    resolved = get_config(cfg)
    if resolved is None or not (0 < m <= MAX_TOKENS):
        return False
    if (
        norm_weight.dtype != torch.bfloat16
        or norm_bias.dtype != torch.bfloat16
        or not norm_weight.is_contiguous()
        or not norm_bias.is_contiguous()
        or norm_weight.numel() != K_IN
        or norm_bias.numel() != K_IN
    ):
        return False
    if (
        positions.dtype != torch.int32
        or positions.shape != (m,)
        or not positions.is_contiguous()
        or ssmax_s.dtype != torch.bfloat16
        or ssmax_b.dtype != torch.bfloat16
        or ssmax_s.shape != (HEADS,)
        or ssmax_b.shape != (HEADS,)
        or not ssmax_s.is_contiguous()
        or not ssmax_b.is_contiguous()
        or cos_sin.dtype != torch.float32
        or cos_sin.dim() != 2
        or cos_sin.shape[1] != ROPE_DIM
        or not cos_sin.is_contiguous()
        or q_fp8_out.dtype != torch.float8_e4m3fn
        or tuple(q_fp8_out.shape) != (m, HEADS, HEAD_DIM)
        or not q_fp8_out.is_contiguous()
        or q_fp8_out.data_ptr() % 4 != 0
        or weights_out.dtype != torch.float32
        or tuple(weights_out.shape) != (m, HEADS)
        or not weights_out.is_contiguous()
    ):
        return False

    inline_wp = wp is not None
    if inline_wp:
        if (
            wp.dtype != torch.bfloat16
            or tuple(wp.shape) != (HEADS, K_IN)
            or not wp.is_contiguous()
            or wp.data_ptr() % 8 != 0
        ):
            return False
    else:
        if (
            kw_w is None
            or kw_w.dtype != torch.bfloat16
            or tuple(kw_w.shape) != (m, HEADS)
            or kw_w.stride(1) != 1
        ):
            return False

    if tuple(weight.shape) != (N_OUT, K_IN) or not weight.is_contiguous():
        return False

    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute

    dev = x.device
    use_cluster = resolved.reduce_mode == "cluster"
    trigger_early = resolved.trigger_early
    max_m = 4 if m <= 4 else 8
    if use_pdl is None:
        use_pdl = _get_enable_pdl()

    key = (
        resolved.reduce_mode,
        resolved.split,
        resolved.ring,
        resolved.contig,
        resolved.a_tmem,
        trigger_early,
        inline_wp,
        max_m,
        use_pdl,
    )

    pi32 = _placeholder_i32(dev)
    pbf16 = _placeholder_bf16(dev)
    pf32 = _placeholder_f32(dev)

    w0 = weight.view(torch.uint8)
    wsf0 = weight_scale.view(torch.uint8).reshape(-1)
    if use_cluster:
        l2_p, l2_wp, l2_ctr = pf32, pf32, pi32
    else:
        l2_p, l2_wp, l2_ctr = _get_l2_workspace(dev, resolved.split)

    wp_arg = wp.view(torch.int32).reshape(-1) if inline_wp else pi32
    if inline_wp:
        kw_bf16_arg = pbf16
        kw_stride_i = 0
    else:
        kw_stride_i = int(kw_w.stride(0))
        flat_elems = (m - 1) * kw_stride_i + HEADS
        kw_bf16_arg = torch.as_strided(kw_w, (flat_elems,), (1,), kw_w.storage_offset())

    tensor_args = (
        _arg(w0),
        _arg(wsf0),
        _arg(x.view(torch.int32).reshape(-1)),
        _arg(norm_weight.view(torch.int32).reshape(-1)),
        _arg(norm_bias.view(torch.int32).reshape(-1)),
        float(eps),
        _arg(positions, align=4),
        _arg(ssmax_s.view(torch.int32), align=4),
        _arg(ssmax_b.view(torch.int32), align=4),
        _arg(cos_sin.reshape(-1), align=4),
        _arg(wp_arg, align=8 if inline_wp else 4),
        _arg(kw_bf16_arg, align=2),
        int(kw_stride_i),
        float(head_weight_scale),
        _arg(q_fp8_out.view(torch.int32).reshape(-1), align=4),
        _arg(weights_out.reshape(-1), align=4),
        _arg(l2_p, align=4),
        _arg(l2_wp, align=4),
        _arg(l2_ctr, align=4),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)

    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        with _LOCK:
            fn = _COMPILED.get(key)
            if fn is None:
                jit_fn = _build_dsl()
                fn = _COMPILED[key] = cute.compile(
                    jit_fn,
                    *tensor_args,
                    m,
                    resolved.split,
                    resolved.ring,
                    resolved.contig,
                    resolved.a_tmem,
                    use_cluster,
                    trigger_early,
                    inline_wp,
                    max_m,
                    use_pdl,
                    stream,
                )
    fn(*tensor_args, m, stream)
    return True


def precompile(
    weight: "torch.Tensor",
    weight_scale: "torch.Tensor",
    m: int,
    *,
    inline_wp: bool = True,
    cfg: Optional[IxQFusedConfig] = None,
    use_pdl: Optional[bool] = None,
) -> bool:
    """Compile the fused indexer Q kernel ahead of CUDA-graph capture."""
    if torch is None or not serves(m, cfg=cfg):
        return False
    dev = weight.device
    x = torch.zeros((m, K_IN), dtype=torch.bfloat16, device=dev)
    ln_w = torch.ones((K_IN,), dtype=torch.bfloat16, device=dev)
    ln_b = torch.zeros((K_IN,), dtype=torch.bfloat16, device=dev)
    pos = torch.zeros((m,), dtype=torch.int32, device=dev)
    s = torch.zeros((HEADS,), dtype=torch.bfloat16, device=dev)
    b = torch.ones((HEADS,), dtype=torch.bfloat16, device=dev)
    cs = torch.zeros((4, ROPE_DIM), dtype=torch.float32, device=dev)
    q_fp8 = torch.empty((m, HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device=dev)
    weights = torch.empty((m, HEADS), dtype=torch.float32, device=dev)
    wp = torch.zeros((HEADS, K_IN), dtype=torch.bfloat16, device=dev) if inline_wp else None
    kw_w = None if inline_wp else torch.zeros((m, HEADS), dtype=torch.bfloat16, device=dev)
    return _run(
        x,
        ln_w,
        ln_b,
        1e-6,
        weight,
        weight_scale,
        pos,
        s,
        b,
        cs,
        1.0 / math.sqrt(HEAD_DIM),
        q_fp8,
        weights,
        wp=wp,
        kw_w=kw_w,
        cfg=cfg,
        use_pdl=use_pdl,
    )


try:
    from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ssmax_indexer_q_side_fused import (  # noqa: E501
        ssmax_indexer_q_side_fused as _ssmax_q_side,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cute_dsl_fp8_block_scale_linear import (  # noqa: E501
        cute_dsl_fp8_block_scale_linear as _fp8_linear,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.layer_norm import (
        layer_norm as _layer_norm,
    )
    from tensorrt_llm.logger import logger
except ImportError:  # pragma: no cover - CPU pure-Python tests without full trtllm runtime
    Arch = None  # type: ignore[assignment]
    Cell = None  # type: ignore[assignment]
    OpWrapper = object  # type: ignore[assignment,misc]
    _ssmax_q_side = None  # type: ignore[assignment]
    _fp8_linear = None  # type: ignore[assignment]
    _layer_norm = None  # type: ignore[assignment]
    logger = None  # type: ignore[assignment]


class Ml4IxQFused(OpWrapper):
    """Single-launch ``LayerNorm + ix_wq_b + SSMax + RoPE + UE8M0 E4M3 + head weights``."""

    if Arch is not None:
        ARCHS = frozenset({Arch.SM_107})
        CELLS = tuple(
            Cell(
                why=f"ML4 SM107 fused indexer Q (LN+wq_b+SSMax+RoPE+E4M3), M={m}, rope={rope}",
                spec=dict(m=m, rope=rope),
            )
            for m in (1, 2, 4, 8)
            for rope in (True, False)
        )

    def __call__(
        self,
        x: "torch.Tensor",
        norm_weight: "torch.Tensor",
        norm_bias: "torch.Tensor",
        eps: float,
        weight: "torch.Tensor",
        weight_scale: "torch.Tensor",
        positions: "torch.Tensor",
        ssmax_s: "torch.Tensor",
        ssmax_b: "torch.Tensor",
        cos_sin: "torch.Tensor",
        head_weight_scale: float,
        q_fp8_out: "torch.Tensor",
        weights_out: "torch.Tensor",
        *,
        wp: Optional["torch.Tensor"] = None,
        kw_w: Optional["torch.Tensor"] = None,
        cfg: Optional[IxQFusedConfig] = None,
        use_pdl: Optional[bool] = None,
    ) -> bool:
        ok = _run(
            x,
            norm_weight,
            norm_bias,
            eps,
            weight,
            weight_scale,
            positions,
            ssmax_s,
            ssmax_b,
            cos_sin,
            head_weight_scale,
            q_fp8_out,
            weights_out,
            wp=wp,
            kw_w=kw_w,
            cfg=cfg,
            use_pdl=use_pdl,
        )
        if ok:
            resolved = get_config(cfg)
            ek = (x.device, x.shape[0], resolved, wp is not None)
            if ek not in _ENGAGED:
                _ENGAGED.add(ek)
                if logger is not None:
                    logger.info(
                        f"[ML4 sm_107] ml4_ix_q_fused engaged: M={x.shape[0]}, "
                        f"cfg={resolved}, inline_wp={wp is not None}"
                    )
        return ok

    def reference(
        self,
        x: "torch.Tensor",
        norm_weight: "torch.Tensor",
        norm_bias: "torch.Tensor",
        eps: float,
        weight: "torch.Tensor",
        weight_scale: "torch.Tensor",
        positions: "torch.Tensor",
        ssmax_s: "torch.Tensor",
        ssmax_b: "torch.Tensor",
        cos_sin: "torch.Tensor",
        head_weight_scale: float,
        q_fp8_out: "torch.Tensor",
        weights_out: "torch.Tensor",
        *,
        wp: Optional["torch.Tensor"] = None,
        kw_w: Optional["torch.Tensor"] = None,
        **kwargs,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        assert _layer_norm is not None and _fp8_linear is not None and _ssmax_q_side is not None
        qn = _layer_norm.reference(x, norm_weight, norm_bias, eps)
        q = _fp8_linear.reference(qn, weight, weight_scale).reshape(x.shape[0], HEADS, HEAD_DIM)
        wq = (x.float() @ wp.float().t()).to(torch.bfloat16) if wp is not None else kw_w
        assert wq is not None
        return _ssmax_q_side.reference(
            q,
            positions,
            ssmax_s,
            ssmax_b,
            cos_sin,
            wq,
            head_weight_scale,
            q_fp8_out,
            weights_out,
        )


ml4_ix_q_fused = Ml4IxQFused()
