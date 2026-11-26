# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mistral Large 4 low-M FP8 GEMV with single-kernel Paradigm-B Rubin uGPU and fused LayerNorm.

Eliminates the 187 second-stage split-K reduce launches of ``_fused_fp8_quant_gemv`` via tcgen05
cluster DSMEM reduction, covers every ML4 TP4/TP8/TP16 FP8 linear projection (``wq``, ``wo``,
``ix_wq_b``, ``dense0``, ``shared``), supports Paradigm-B die-aware uGPU placement via
:class:`LocalityMap`, and optionally fuses ``ix_q_input_norm`` (LayerNorm) into ``ix_wq_b``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover - CPU pure-Python config helpers work without torch
    torch = None  # type: ignore[assignment]

from .locality import LocalityMap, PlacedSplit

__all__ = [
    "MAX_TOKENS",
    "MMA_WIDTHS",
    "ML4_GEMV_CONFIGS",
    "ML4_PROJECTIONS",
    "Ml4MxGemv",
    "Ml4MxGemvLayerNorm",
    "PlacedGemvWeight",
    "get_config",
    "ml4_mx_gemv",
    "ml4_mx_gemv_layernorm",
    "ml4_mx_gemv_swiglu",
    "precompile",
    "prepare_ugpu_gemv_weight",
    "serves",
    "simulate_ugpu_tile_claim",
]

MAX_TOKENS = 48
MMA_WIDTHS = (8, 16, 32, 48)
_SMEM_LIMIT = 220 * 1024
_CONTIG = True
_NACC = 1
_A_TMEM = True
_ST_ASYNC = True

# Every ML4 TP4 / TP8 / TP16 FP8_PB_WO linear projection (N, K).
ML4_PROJECTIONS: dict[str, tuple[int, int]] = {
    # TP16 (world=16, 4 Q heads/rank, I_moe=384, I_dense=1536)
    "tp16_wq": (512, 6144),
    "tp16_wo": (6144, 512),
    "tp16_shared_gu": (768, 6144),
    "tp16_shared_dn": (6144, 384),
    "tp16_dense0_gu": (3072, 6144),
    "tp16_dense0_dn": (6144, 1536),
    # TP8 (world=8, 8 Q heads/rank, I_moe=768, I_dense=3072)
    "tp8_wq": (1024, 6144),
    "tp8_wo": (6144, 1024),
    "tp8_shared_gu": (1536, 6144),
    "tp8_shared_dn": (6144, 768),
    "tp8_dense0_gu": (6144, 6144),
    "tp8_dense0_dn": (6144, 3072),
    # Replicated across TP4/TP8/TP16
    "ix_wq_b": (4096, 6144),
    # TP4 (world=4, 16 Q heads/rank, I_moe=1536, I_dense=6144)
    "tp4_wq": (2048, 6144),
    "tp4_wo": (6144, 2048),
    "tp4_dense0_gu": (12288, 6144),
}

# (N, K) -> (split, ring[, contig, nacc, a_tmem, st_async]) tuned for VR200 (212 SMs).
# All grids <= 192 CTAs (<= 212 SMs) and all cluster sizes split in {1, 2, 4, 8}.
ML4_GEMV_CONFIGS: dict[tuple[int, int], tuple] = {
    # TP16
    (512, 6144): (8, 6),  # 32 CTAs, 6 k-tiles/CTA
    (6144, 512): (2, 2),  # 96 CTAs, 2 k-tiles/CTA
    (768, 6144): (8, 6),  # 48 CTAs, 6 k-tiles/CTA
    (6144, 384): (1, 3),  # 48 CTAs, 3 k-tiles/CTA
    (3072, 6144): (4, 12),  # 96 CTAs, 12 k-tiles/CTA
    (6144, 1536): (2, 6),  # 96 CTAs, 6 k-tiles/CTA
    # TP8
    (1024, 6144): (8, 6),  # 64 CTAs, 6 k-tiles/CTA
    (6144, 1024): (2, 4),  # 96 CTAs, 4 k-tiles/CTA
    (1536, 6144): (8, 6),  # 96 CTAs, 6 k-tiles/CTA
    (6144, 768): (2, 3),  # 96 CTAs, 3 k-tiles/CTA
    (6144, 6144): (4, 12),  # 192 CTAs, 12 k-tiles/CTA (a_tmem fits)
    (6144, 3072): (2, 12),  # 96 CTAs, 12 k-tiles/CTA
    # Replicated indexer wq_b
    (4096, 6144): (4, 12),  # 128 CTAs (64 die-0 + 64 die-1)
    # TP4
    (2048, 6144): (4, 12),  # 64 CTAs, 12 k-tiles/CTA
    (6144, 2048): (2, 8),  # 96 CTAs, 8 k-tiles/CTA
    (12288, 6144): (2, 12),  # 192 CTAs, 24 k-tiles/CTA
}

# Strided k-tile ordering at M <= 8 for low-M SM107 latency.
_STRIDED_SHAPES = frozenset(
    {
        (512, 6144),
        (1024, 6144),
        (1536, 6144),
        (2048, 6144),
        (3072, 6144),
        (4096, 6144),
        (6144, 1024),
        (6144, 1536),
        (6144, 3072),
    }
)
_STRIDED_MAX_ROWS = 8

_LOCK = threading.Lock()
_COMPILED: dict[tuple, object] = {}
_PLACEHOLDERS: dict[tuple, object] = {}
_ENGAGED: set[tuple] = set()


# =============================================================================
# Pure-Python host config & Paradigm-B tile-claim model (CPU-testable).
# =============================================================================


def _mma_n(m: int) -> int:
    return next(w for w in MMA_WIDTHS if m <= w)


def _smem_bytes(ring: int, tiles: int, split: int, mma_n: int) -> int:
    return (
        ring * (128 * 128 + 512)
        + tiles * (mma_n * 128 + 512)
        + split * (128 // split) * (mma_n + 4) * 4
        + 1024
    )


def get_config(n: int, k: int, m: Optional[int] = None) -> Optional[tuple]:
    """Return ``(split, ring[, contig, ...])`` for an ML4 FP8 linear shape, or None."""
    cfg = ML4_GEMV_CONFIGS.get((n, k))
    if cfg is None or n % 128 != 0 or k % 128 != 0 or (k // 128) % cfg[0] != 0 or 128 % cfg[0] != 0:
        return None
    if m is not None and m <= _STRIDED_MAX_ROWS and (n, k) in _STRIDED_SHAPES and len(cfg) == 2:
        cfg = cfg + (0,)
    return cfg


def serves(
    n: int,
    k: int,
    m: int,
    *,
    x_bf16: bool = True,
    x_layernorm: bool = False,
    x_swiglu: bool = False,
    ugpu: bool = False,
    cfg: Optional[tuple] = None,
) -> bool:
    """Whether :data:`ml4_mx_gemv` serves ``[M, K] x [N, K]^T`` under the requested options."""
    resolved = cfg or get_config(n, k, m)
    if resolved is None:
        return False
    split, ring = resolved[0], resolved[1]
    if not (0 < m <= MAX_TOKENS) or 128 % split != 0 or (k // 128) % split != 0:
        return False
    if x_layernorm and x_swiglu:
        return False
    if (x_layernorm or x_swiglu) and not x_bf16:
        return False
    if x_layernorm and (m > 8 or k % 128 != 0):
        return False
    if ugpu and (n // 128) % 2 != 0:
        return False
    tiles = (k // 128) // split
    mma_n = _mma_n(m)
    if 16 < m <= 32 and k < 1024:
        return False
    if x_bf16 and tiles * 4 * mma_n > 8 * 128:
        return False
    return _smem_bytes(min(ring, tiles, 2), tiles, split, mma_n) <= _SMEM_LIMIT


def simulate_ugpu_tile_claim(cluster_dies: list[int], n_tiles: int) -> list[int]:
    """Exact host simulation of the kernel's die-local atomic tile claim (for CPU contract tests).

    Verifies that for ANY hardware cluster-to-die placement of ``n_tiles`` clusters (even skewed
    ``k0 != k1``), the claim outputs a bijection onto ``[0, n_tiles)`` while maximizing die-local
    matches (``min(k0, n_tiles // 2) + min(k1, n_tiles // 2)`` die-local tiles).
    """
    if len(cluster_dies) != n_tiles or n_tiles % 2 != 0:
        raise ValueError(f"expected even n_tiles == len(cluster_dies), got {n_tiles}")
    half = n_tiles // 2
    ctrs = [0, 0]
    claimed = []
    for cid, d in enumerate(cluster_dies):
        die = d if d in (0, 1) else (cid & 1)
        t_own = ctrs[die]
        ctrs[die] += 1
        if t_own < half:
            claimed.append(die * half + t_own)
        else:
            other = 1 - die
            t_other = ctrs[other]
            ctrs[other] += 1
            claimed.append(other * half + t_other)
    return claimed


# =============================================================================
# Placed uGPU GEMV weight bundle and runtime dispatch.
# =============================================================================


@dataclass
class PlacedGemvWeight:
    """Die-placed FP8 weight + R128c4 scales for Paradigm-B single-kernel ``ml4_mx_gemv``."""

    weight_split: PlacedSplit
    scale_split: PlacedSplit
    sm_table: "torch.Tensor"
    ugpu_state: "torch.Tensor"  # int32 [20]: [launch_ctr, pad*3, 4 x (c0, c1, pad, pad)]
    n: int
    k: int


def prepare_ugpu_gemv_weight(
    weight: "torch.Tensor",
    weight_scale: "torch.Tensor",
    locality_map: Optional[LocalityMap] = None,
) -> Optional[PlacedGemvWeight]:
    """Place ``weight [N, K]`` (e4m3) and R128c4 ``weight_scale`` onto die 0 (``[0, N/2)``) and
    die 1 (``[N/2, N)``) for single-kernel Paradigm-B ``ml4_mx_gemv``. Returns None if ``N/2`` is
    not 128-aligned or dual-domain localization is unavailable."""
    if torch is None:
        return None
    if weight.dim() != 2 or weight.dtype != torch.float8_e4m3fn or not weight.is_contiguous():
        return None
    n, k = weight.shape
    if (n // 2) % 128 != 0 or k % 128 != 0:
        return None
    if locality_map is None:
        try:
            locality_map = LocalityMap.get(weight.device)
        except (ImportError, RuntimeError):
            return None
    w_split, ws_split = locality_map.place_n_split(weight, weight_scale, use_slab=True)
    ugpu_state = torch.zeros(20, dtype=torch.int32, device=weight.device)
    return PlacedGemvWeight(
        weight_split=w_split,
        scale_split=ws_split,
        sm_table=locality_map.sm_table,
        ugpu_state=ugpu_state,
        n=n,
        k=k,
    )


def _arg(t: "torch.Tensor", align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _placeholder_words(device) -> "torch.Tensor":
    key = ("w", device)
    t = _PLACEHOLDERS.get(key)
    if t is None:
        t = _PLACEHOLDERS[key] = torch.zeros(16, dtype=torch.int32, device=device)
    return t  # type: ignore[return-value]


def _placeholder_scales(k: int, device) -> "torch.Tensor":
    key = ("sf", k, device)
    t = _PLACEHOLDERS.get(key)
    if t is None:
        t = _PLACEHOLDERS[key] = torch.zeros((k // 128) * 512, dtype=torch.uint8, device=device)
    return t  # type: ignore[return-value]


def _get_enable_pdl() -> bool:
    try:
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        return bool(get_env_enable_pdl())
    except ImportError:
        return True


def _run(
    x: "torch.Tensor",
    weight: "torch.Tensor",
    weight_scale: "torch.Tensor",
    *,
    placed: Optional[PlacedGemvWeight] = None,
    ln_weight: Optional["torch.Tensor"] = None,
    ln_bias: Optional["torch.Tensor"] = None,
    ln_eps: float = 1e-6,
    x_swiglu: bool = False,
    trigger_early: bool = True,
    cfg: Optional[tuple] = None,
    use_pdl: Optional[bool] = None,
    ar: Optional[tuple] = None,
) -> Optional["torch.Tensor"]:
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute

    from . import _ml4_mx_gemv_dsl as dsl

    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous() or x.data_ptr() % 16 != 0:
        return None
    m, k_in = x.shape
    k = k_in // 2 if x_swiglu else k_in
    n = placed.n if placed is not None else weight.shape[0]
    w_k = placed.k if placed is not None else weight.shape[1]
    if w_k != k:
        return None

    x_layernorm = ln_weight is not None
    if x_layernorm:
        if (
            ln_bias is None
            or ln_weight.dtype != torch.bfloat16
            or ln_bias.dtype != torch.bfloat16
            or not ln_weight.is_contiguous()
            or not ln_bias.is_contiguous()
            or ln_weight.numel() != k
            or ln_bias.numel() != k
        ):
            return None

    ugpu = placed is not None
    resolved = cfg or get_config(n, k, m)
    if resolved is None or not serves(
        n, k, m, x_bf16=True, x_layernorm=x_layernorm, x_swiglu=x_swiglu, ugpu=ugpu, cfg=resolved
    ):
        return None

    split, ring = resolved[0], resolved[1]
    contig = bool(resolved[2]) if len(resolved) > 2 else _CONTIG
    nacc = int(resolved[3]) if len(resolved) > 3 else _NACC
    a_tmem = bool(resolved[4]) if len(resolved) > 4 else _A_TMEM
    st_async = bool(resolved[5]) if len(resolved) > 5 else _ST_ASYNC
    if nacc != 1:
        return None

    tiles = (k // 128) // split
    ring = min(ring, tiles)
    mma_n = _mma_n(m)
    while ring > 2 and _smem_bytes(ring, tiles, split, mma_n) > _SMEM_LIMIT:
        ring -= 1
    if a_tmem and dsl.tmem_cols(ring, nacc, True, tiles, mma_n) > 512:
        a_tmem = False

    ar_ws, ar_flags, ar_rank, ar_mode, ar_push, ar_world, ar_maxt = (0, 0, 0, 0, 0, 1, 8)
    if ar is not None and split <= 4 and n % 2 == 0:
        if len(ar) == 7:
            ar_ws, ar_flags, ar_rank, ar_world, ar_maxt, ar_push, ar_mode = ar
        else:
            ar_ws, ar_flags, ar_rank, ar_world, ar_maxt, ar_push = ar
            ar_mode = 0
        if ar_push == 2:
            trigger_early = False

    if use_pdl is None:
        use_pdl = _get_enable_pdl()

    key = (
        n,
        k,
        split,
        ring,
        trigger_early,
        contig,
        nacc,
        a_tmem,
        st_async,
        True,  # x_bf16
        x_layernorm,
        x_swiglu,
        ugpu,
        ar_push,
        ar_world,
        ar_maxt,
        mma_n,
        use_pdl,
    )

    dev = x.device
    y = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    pw = _placeholder_words(dev)
    psf = _placeholder_scales(k, dev)

    if placed is not None:
        w0 = placed.weight_split.half0.view(torch.uint8)
        w1 = placed.weight_split.half1.view(torch.uint8)
        wsf0 = placed.scale_split.half0.view(torch.uint8).reshape(-1)
        wsf1 = placed.scale_split.half1.view(torch.uint8).reshape(-1)
        sm_tbl = placed.sm_table
        ug_st = placed.ugpu_state
    else:
        w0 = w1 = weight.view(torch.uint8)
        wsf0 = wsf1 = weight_scale.view(torch.uint8).reshape(-1)
        sm_tbl = pw
        ug_st = pw

    ln_w_arg = ln_weight.view(torch.int32).reshape(-1) if x_layernorm else pw
    ln_b_arg = ln_bias.view(torch.int32).reshape(-1) if x_layernorm else pw

    tensor_args = (
        _arg(w0),
        _arg(wsf0),
        _arg(w1),
        _arg(wsf1),
        _arg(x.view(torch.uint8)),
        _arg(psf),
        _arg(y.view(-1)),
        _arg(x.view(torch.int32).reshape(-1)),
        _arg(ln_w_arg),
        _arg(ln_b_arg),
        float(ln_eps),
        _arg(sm_tbl, align=4),
        _arg(ug_st, align=4),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)

    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        with _LOCK:
            fn = _COMPILED.get(key)
            if fn is None:
                fn = _COMPILED[key] = cute.compile(
                    dsl.ml4_mx_gemv_jit,
                    *tensor_args,
                    m,
                    n,
                    k,
                    split,
                    ring,
                    trigger_early,
                    contig,
                    nacc,
                    a_tmem,
                    st_async,
                    True,
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
                    use_pdl,
                    stream,
                )
    fn(*tensor_args, m, ar_ws, ar_flags, ar_rank, ar_mode, stream)
    return y


def precompile(
    weight: "torch.Tensor",
    weight_scale: "torch.Tensor",
    m: int,
    *,
    placed: Optional[PlacedGemvWeight] = None,
    x_layernorm: bool = False,
    x_swiglu: bool = False,
    use_pdl: Optional[bool] = None,
) -> bool:
    """Compile the kernel for ``(weight.shape, m, options)`` ahead of CUDA-graph capture."""
    if torch is None:
        return False
    n, k = (placed.n, placed.k) if placed is not None else tuple(weight.shape)
    if not serves(
        n, k, m, x_bf16=True, x_layernorm=x_layernorm, x_swiglu=x_swiglu, ugpu=(placed is not None)
    ):
        return False
    dev = weight.device
    x = torch.zeros((m, 2 * k if x_swiglu else k), dtype=torch.bfloat16, device=dev)
    ln_w = torch.ones((k,), dtype=torch.bfloat16, device=dev) if x_layernorm else None
    ln_b = torch.zeros((k,), dtype=torch.bfloat16, device=dev) if x_layernorm else None
    out = _run(
        x,
        weight,
        weight_scale,
        placed=placed,
        ln_weight=ln_w,
        ln_bias=ln_b,
        x_swiglu=x_swiglu,
        use_pdl=use_pdl,
    )
    return out is not None


def ml4_mx_gemv_swiglu(
    gate_up: "torch.Tensor",
    weight: "torch.Tensor",
    weight_scale: "torch.Tensor",
    *,
    placed: Optional[PlacedGemvWeight] = None,
    trigger_early: bool = True,
    cfg: Optional[tuple] = None,
    use_pdl: Optional[bool] = None,
    ar: Optional[tuple] = None,
) -> Optional["torch.Tensor"]:
    """Single-launch ``ml4_mx_gemv(silu_and_mul(gate_up), weight, weight_scale)`` (standard SwiGLU)."""
    return _run(
        gate_up,
        weight,
        weight_scale,
        placed=placed,
        x_swiglu=True,
        trigger_early=trigger_early,
        cfg=cfg,
        use_pdl=use_pdl,
        ar=ar,
    )


try:
    from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cute_dsl_fp8_block_scale_linear import (
        cute_dsl_fp8_block_scale_linear as _baseline,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.layer_norm import (
        layer_norm as _layer_norm,
    )
    from tensorrt_llm.logger import logger
except ImportError:  # pragma: no cover - CPU pure-Python tests without full trtllm runtime
    Arch = None  # type: ignore[assignment]
    Cell = None  # type: ignore[assignment]
    OpWrapper = object  # type: ignore[assignment,misc]
    _baseline = None  # type: ignore[assignment]
    _layer_norm = None  # type: ignore[assignment]
    logger = None  # type: ignore[assignment]


class Ml4MxGemv(OpWrapper):
    """ML4 low-M FP8 GEMV (1x128 quant + tcgen05 cluster split-K + optional Paradigm-B uGPU)."""

    if Arch is not None:
        ARCHS = frozenset({Arch.SM_107})
        CELLS = tuple(
            Cell(why=f"ML4 SM107 {name} ({n}x{k}), M={m}", spec=dict(m=m, n=n, k=k))
            for name, (n, k) in ML4_PROJECTIONS.items()
            for m in (1, 4, 8)
            if serves(n, k, m)
        )

    def __call__(
        self,
        input: "torch.Tensor",
        weight: "torch.Tensor",
        weight_scale: "torch.Tensor",
        *,
        placed: Optional[PlacedGemvWeight] = None,
        trigger_early: bool = True,
        cfg: Optional[tuple] = None,
        use_pdl: Optional[bool] = None,
        ar: Optional[tuple] = None,
    ) -> Optional["torch.Tensor"]:
        out = _run(
            input,
            weight,
            weight_scale,
            placed=placed,
            trigger_early=trigger_early,
            cfg=cfg,
            use_pdl=use_pdl,
            ar=ar,
        )
        if out is not None:
            ek = (input.device, input.shape[0], weight.shape[0], input.shape[1], placed is not None)
            if ek not in _ENGAGED:
                _ENGAGED.add(ek)
                if logger is not None:
                    logger.info(
                        f"[ML4 sm_107] ml4_mx_gemv engaged: M={input.shape[0]}, "
                        f"N={weight.shape[0]}, K={input.shape[1]}, ugpu={placed is not None}"
                    )
        return out

    def reference(
        self,
        input: "torch.Tensor",
        weight: "torch.Tensor",
        weight_scale: "torch.Tensor",
        **kwargs,
    ) -> "torch.Tensor":
        assert _baseline is not None
        return _baseline.reference(input, weight, weight_scale)


class Ml4MxGemvLayerNorm(OpWrapper):
    """Fused ``LayerNorm + 1x128 FP8 quant + tcgen05 cluster GEMV (+ optional Paradigm-B uGPU)``
    for Mistral Large 4's 44-layer ``ix_q_input_norm + ix_wq_b`` (4096 x 6144)."""

    if Arch is not None:
        ARCHS = frozenset({Arch.SM_107})
        CELLS = tuple(
            Cell(
                why=f"ML4 SM107 fused ix_q_input_norm + ix_wq_b (4096x6144), M={m}",
                spec=dict(m=m, n=4096, k=6144),
            )
            for m in (1, 2, 4, 8)
        )

    def __call__(
        self,
        input: "torch.Tensor",
        norm_weight: "torch.Tensor",
        norm_bias: "torch.Tensor",
        eps: float,
        weight: "torch.Tensor",
        weight_scale: "torch.Tensor",
        *,
        placed: Optional[PlacedGemvWeight] = None,
        trigger_early: bool = True,
        cfg: Optional[tuple] = None,
        use_pdl: Optional[bool] = None,
    ) -> Optional["torch.Tensor"]:
        return _run(
            input,
            weight,
            weight_scale,
            placed=placed,
            ln_weight=norm_weight,
            ln_bias=norm_bias,
            ln_eps=eps,
            trigger_early=trigger_early,
            cfg=cfg,
            use_pdl=use_pdl,
        )

    def reference(
        self,
        input: "torch.Tensor",
        norm_weight: "torch.Tensor",
        norm_bias: "torch.Tensor",
        eps: float,
        weight: "torch.Tensor",
        weight_scale: "torch.Tensor",
        **kwargs,
    ) -> "torch.Tensor":
        assert _baseline is not None and _layer_norm is not None
        normed = _layer_norm.reference(input, norm_weight, norm_bias, eps)
        return _baseline.reference(normed, weight, weight_scale)


ml4_mx_gemv = Ml4MxGemv()
ml4_mx_gemv_layernorm = Ml4MxGemvLayerNorm()
