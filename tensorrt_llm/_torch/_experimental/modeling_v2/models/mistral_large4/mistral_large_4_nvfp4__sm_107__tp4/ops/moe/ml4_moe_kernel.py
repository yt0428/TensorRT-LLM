# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mistral Large 4 persistent CuTe DSL MoE engine for 1..8 decode tokens on SM107 (Rubin VR200).

Covers TP16 (``I=384``), TP8 (``I=768``), and TP4 (``I=1536``) with:
* ML4 BF16 router GEMV + softmax top-4 renormalized routing (exact match to ``renorm_moe_routing_op``),
* Standard SwiGLU (``silu(gate) * up``) for both routed NVFP4 experts and FP8 shared expert,
* Lossless ``FP8_PB_WO`` (128x128 power-of-2 block scales) -> 1x32 UE8M0 R128c4 verification and
  expansion (with explicit non-power-of-2 fallback),
* Paradigm-B single-kernel die-aware Rubin uGPU via :class:`LocalityMap` in both ``n_split`` and
  ``expert_half`` modes,
* Optional direct multicast all-reduce push (``ar_push_desc``, ``world_size`` in ``{4, 8, 16}``).
"""

from __future__ import annotations

import importlib.util
import os
import struct
import sys
from typing import Literal, Optional, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - CPU pure-Python helpers stay importable without torch
    torch = None  # type: ignore[assignment]

from ..gemm.locality import (
    LocalityMap,
    PlacedSplit,
    required_moe_ctas_per_domain,
    resolve_moe_locality_mode,
)

__all__ = [
    "AR_PUSH_LAMPORT_3SLOT",
    "AR_PUSH_M3_MC",
    "AR_PUSH_MCAST_HOP",
    "HIDDEN",
    "MIN_CTAS",
    "NUM_EXPERTS",
    "SHARED_CTAS",
    "SUPPORTED_INTER",
    "SUPPORTED_WORLD_SIZES",
    "TOP_K",
    "Ml4MoeEngine",
    "Ml4MoeLayer",
    "Ml4MoeState",
    "ar_push_desc",
    "ar_push_desc_from_lamport",
    "expand_fp8_128x128_scale_to_ue8m0_r128c4",
    "fine_ranges",
    "interleave_expert_slots_by_die",
    "is_exact_power_of_two_ue8m0",
    "is_supported",
    "make_phase_plan",
    "make_phase_plan_host",
    "ml4_moe_engine",
    "ml4_moe_reference",
    "ml4_moe_route_reference",
    "ml4_shared_expert_reference",
    "ml4_swiglu_quant_reference",
    "pretile_shared",
    "unswizzle_sf",
]

HIDDEN = 6144
NUM_EXPERTS = 128
TOP_K = 4
SHARED_CTAS = 24
MIN_CTAS = 64
SUPPORTED_INTER = (384, 768, 1536)
SUPPORTED_WORLD_SIZES = (4, 8, 16)
_S1 = 6
_LAG = 2

# All-reduce push descriptor wire modes (arp[6]).
AR_PUSH_M3_MC = 0  # 2-slot parity multicast exchange
AR_PUSH_LAMPORT_3SLOT = 1  # 3-slot LamportWorkspace (catalog/comm/_lamport.py)
AR_PUSH_MCAST_HOP = 2  # 3-slot McastGPUBuffer / glmMnnvlHop ([3][t][rank][H])

LocalityModeStr = Literal["none", "n_split", "expert_half"]
_LOCALITY_TO_INT: dict[str, int] = {"none": 0, "n_split": 1, "expert_half": 2}

# Per-token-count VR200 (SM107) tuning tables for I=768 (TP8) and I=384 (TP16).
_TP8_FLAT_MAX_TOKENS = 4
_TP8_FLAT_STACK = {"b2img": 0, "lg_repl": 16, "ar_pre": 0, "late_trigger": 0}
_TP8_HEAVY_FC2_ONLY_BY_M = (0, 0, 0, 1, 0, 0, 0, 1)
_TP8_STAGES_BY_M = (0, 0, 0, 0, 0, 0, 0, 5)
_TP8_B2_POLL_NS_BY_M = (0, 0, 0, 0, 0, 0, 0, 50)
_TP8_EARLY0_BY_M = (1, 1, 1, 1, 1, 1, 1, 0)

_TP16_B2IMG_BY_M = (0, 0, 0, 0, 0, 0, 1, 1)
_TP16_AR_PRE_BY_M = (1, 0, 1, 1, 1, 1, 1, 1)
_TP16_STAGES_BY_M = (0, 0, 0, 0, 0, 0, 0, 6)
_TP16_HB_REPL_BY_M = (32, 32, 8, 8, 1, 1, 1, 1)
_TP16_LG_REPL_MAX_TOKENS = 4

_DSL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "_ml4_moe_dsl.py"))
_MODULES: dict[tuple, object] = {}
_COMPILED: dict[tuple, object] = {}
_PRETILED: dict[tuple, tuple] = {}
_ENGAGED: set[tuple] = set()


# =============================================================================
# Pure-Python CPU-testable planning, slot-interleaving, and scale helpers.
# =============================================================================


def _f32_to_u32(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_exact_power_of_two_ue8m0(scales) -> bool:
    """Return True iff every element of ``scales`` is an exact positive normal power of two
    (IEEE-754 fp32 sign=0, mantissa=0, 0 < exponent < 255), so UE8M0 represents it losslessly."""
    if torch is not None and isinstance(scales, torch.Tensor):
        if scales.is_meta:
            return True
        if scales.dtype != torch.float32:
            scales = scales.float()
        bits = scales.contiguous().view(torch.int32)
        exp = (bits >> 23) & 0xFF
        exact = ((bits & 0x807FFFFF) == 0) & (exp > 0) & (exp < 255)
        return bool(exact.all().item())

    # Pure-Python fallback for nested lists / iterables of floats.
    def _walk(obj) -> bool:
        if isinstance(obj, (list, tuple)):
            return all(_walk(x) for x in obj)
        bits = _f32_to_u32(float(obj))
        exp = (bits >> 23) & 0xFF
        return ((bits & 0x807FFFFF) == 0) and (0 < exp < 255)

    return _walk(scales)


def expand_fp8_128x128_scale_to_ue8m0_r128c4(
    scale_f32,
    n: int,
    k: int,
    *,
    strict: bool = False,
):
    """Losslessly convert ML4 ``FP8_PB_WO`` 128x128 fp32 block scales ``[N/128, K/128]`` (or
    ``[N/128, 1, K/128, 1]``) into 128x4-swizzled (R128c4) ``1x32`` UE8M0 uint8 scales.

    Returns ``(r128c4_u8, True)`` when every scale is an exact positive normal power of two.
    If any scale is NOT a power of two:
    * raises ``ValueError`` when ``strict=True``;
    * returns ``(None, False)`` when ``strict=False`` so callers cleanly fall back to a
      non-UE8M0 path without silently quantizing scales.
    """
    if n % 128 != 0 or k % 128 != 0:
        raise ValueError(f"FP8_PB_WO expansion requires N, K divisible by 128, got ({n}, {k})")
    if torch is None or not isinstance(scale_f32, torch.Tensor):
        if not is_exact_power_of_two_ue8m0(scale_f32):
            if strict:
                raise ValueError(
                    "non-power-of-two FP8 block scale cannot be losslessly converted to UE8M0"
                )
            return None, False
        # Pure-Python flat byte list matching R128c4 layout for CPU tests without torch.
        n_blk, k_blk = n // 128, k // 128
        grid = scale_f32
        if len(grid) == n_blk and isinstance(grid[0], (list, tuple)) and len(grid[0]) == 1:
            grid = [[cell[0] for cell in row[0]] for row in grid]
        if len(grid) != n_blk or any(len(row) != k_blk for row in grid):
            raise ValueError(f"expected scale grid [{n_blk}, {k_blk}]")
        out = bytearray(n * (k // 32))
        for nb in range(n_blk):
            for kb in range(k_blk):
                exp = (_f32_to_u32(float(grid[nb][kb])) >> 23) & 0xFF
                atom_off = (nb * k_blk + kb) * 512
                out[atom_off : atom_off + 512] = bytes([exp]) * 512
        return bytes(out), True

    s = scale_f32
    if s.dim() == 4 and s.shape[1] == 1 and s.shape[3] == 1:
        s = s.reshape(s.shape[0], s.shape[2])
    if tuple(s.shape) != (n // 128, k // 128):
        raise ValueError(
            f"expected FP8 128x128 scale shape ({n // 128}, {k // 128}), got {tuple(s.shape)}"
        )
    if not is_exact_power_of_two_ue8m0(s):
        if strict:
            raise ValueError(
                "FP8 block scale contains non-power-of-two entries; cannot expand losslessly to UE8M0"
            )
        return None, False
    exps = ((s.contiguous().float().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    # Every 128x128 block maps to one contiguous 512-byte R128c4 atom whose 512 bytes are all equal
    # to that block's single UE8M0 exponent (128 rows x 4 K32 sub-blocks = 512 bytes).
    r128c4 = exps.unsqueeze(-1).expand(n // 128, k // 128, 512).contiguous().reshape(-1)
    return r128c4, True


def fine_ranges(weights: Sequence[float], total: int, *, stages: int = _S1) -> list[int]:
    """Contiguous FC1 item ranges over CTAs where every unit (``stages`` items) has at most one
    CTA boundary strictly inside it."""
    n = len(weights)
    items = [0] * n
    s = 0
    acc = 0.0
    for c in range(n):
        acc += weights[c]
        e = total if c == n - 1 else min(total, max(s, int(acc + 0.5)))
        if s % stages and e < total:
            e = max(e, min(total, (s // stages + 1) * stages))
        items[c] = e - s
        s = e
    if s != total:
        raise RuntimeError(f"fine_ranges sum mismatch: {s} != {total}")
    return items


def interleave_expert_slots_by_die(expert_ids: Sequence[int], half_e: int = 64) -> list[int]:
    """Deterministic warp-equivalent slot ordering for ``expert_half`` mode.

    Given distinct active expert ids, pairs up die-0 (``e < half_e``) and die-1 (``e >= half_e``)
    experts into 4-expert groups ``[d0, d0, d1, d1]`` so die-0 CTAs stream die-0 experts and
    die-1 CTAs stream die-1 experts; remaining tail experts follow (die-0 first, then die-1).
    """
    d0 = [e for e in expert_ids if e < half_e]
    d1 = [e for e in expert_ids if e >= half_e]
    k_pairs = min(len(d0) // 2, len(d1) // 2)
    out = [-1] * len(expert_ids)
    for r0, e in enumerate(d0):
        slot = (r0 // 2) * 4 + (r0 & 1) if r0 < 2 * k_pairs else 4 * k_pairs + (r0 - 2 * k_pairs)
        out[slot] = e
    for r1, e in enumerate(d1):
        slot = (
            (r1 // 2) * 4 + 2 + (r1 & 1)
            if r1 < 2 * k_pairs
            else 4 * k_pairs + (len(d0) - 2 * k_pairs) + (r1 - 2 * k_pairs)
        )
        out[slot] = e
    return out


def _plan_fc1_group(
    fc2: list[int],
    total: int,
    *,
    stages: int,
    item_bytes: int,
    fine: bool,
    no_fc1: list[bool],
) -> list[int]:
    num_ctas = len(fc2)
    items = [0] * num_ctas
    active_ctas = sum(1 for skip in no_fc1 if not skip)
    if total == 0:
        return items
    if total >= stages * active_ctas and active_ctas > 0:

        def counts_for(limit: int) -> list[int]:
            return [
                0 if no_fc1[c] else max(stages, (limit - f) // item_bytes)
                for c, f in enumerate(fc2)
            ]

        lo = 0
        hi = max(f for c, f in enumerate(fc2) if not no_fc1[c]) + total * item_bytes
        while lo < hi:
            mid = (lo + hi) // 2
            if sum(counts_for(mid)) >= total:
                hi = mid
            else:
                lo = mid + 1
        items = counts_for(lo)
        extra = sum(items) - total
        while extra > 0:
            c = min(
                (c for c in range(num_ctas) if items[c] > stages),
                key=lambda c: (-(fc2[c] + items[c] * item_bytes), c),
            )
            items[c] -= 1
            extra -= 1
    elif fine:
        lo_f, hi_f = 0.0, float(max(fc2) + total * item_bytes)
        for _ in range(100):
            mid = 0.5 * (lo_f + hi_f)
            if sum(max(0.0, (mid - f) / item_bytes) for f in fc2) >= total:
                hi_f = mid
            else:
                lo_f = mid
        wts = [max(0.0, (hi_f - f) / item_bytes) for f in fc2]
        sw = sum(wts)
        items = fine_ranges([w * total / sw for w in wts], total, stages=stages)
    else:
        chosen = sorted(sorted(range(num_ctas), key=lambda c: (fc2[c], c))[: total // stages])
        for c in chosen:
            items[c] = stages
    return items


def make_phase_plan_host(
    num_ctas: int,
    m_max: int,
    *,
    i_tp: int = 768,
    lag: int = _LAG,
    tail_w: float = 1.0,
    early0: bool = True,
    fine: bool = True,
    heavy_fc2_only: bool = False,
    die_aware: bool = False,
) -> tuple[list[list[list[int]]], int]:
    """Pure-Python host phase planner for ML4 MoE (both standard and Paradigm-B ``n_split``).

    Returns ``(plan[4*m_max+1][num_ctas][m_max+1], max_fc2_blocks_per_cta)``.
    When ``die_aware=True`` (``n_split`` mode), CTAs ``0..num_ctas//2-1`` (die 0) partition FC2
    blocks ``[0, 96)`` and local FC1 units ``[0, U1//2)`` independently of CTAs
    ``num_ctas//2..num_ctas-1`` (die 1), which partition FC2 blocks ``[96, 192)`` and local FC1
    units ``[0, U1//2)`` (mapped to global units ``[U1//2, U1)`` inside the kernel).
    """
    if i_tp not in SUPPORTED_INTER:
        raise ValueError(f"unsupported i_tp={i_tp}; expected one of {SUPPORTED_INTER}")
    gcap = TOP_K * m_max
    nb2 = HIDDEN // 32
    u1_full = 2 * i_tp // 32
    unit_bytes = 32 * (HIDDEN // 2 + HIDDEN // 16)
    row2_bytes = i_tp // 2 + i_tp // 16
    item_bytes = unit_bytes // _S1
    blk_bytes = 32 * row2_bytes
    plan = [[[0] * (m_max + 1) for _ in range(num_ctas)] for _ in range(gcap + 1)]
    big_w = 1 << 40

    if die_aware:
        if num_ctas % 2 != 0:
            raise ValueError(f"die_aware phase plan requires even num_ctas, got {num_ctas}")
        half_ctas = num_ctas // 2
        half_nb2 = nb2 // 2
        half_u1 = u1_full // 2
        nbs_half = [
            half_nb2 // half_ctas + (1 if c < half_nb2 % half_ctas else 0) for c in range(half_ctas)
        ]
        b0s_half = [sum(nbs_half[:c]) for c in range(half_ctas)]
        min_nb = min(nbs_half)
        no_fc1_half = [heavy_fc2_only and nbs_half[c] > min_nb for c in range(half_ctas)]
        for g in range(TOP_K, gcap + 1):
            ng = (g + 3) // 4
            for dom in range(2):
                blk_base = dom * half_nb2
                cta_base = dom * half_ctas
                for c in range(half_ctas):
                    plan[g][cta_base + c][m_max] = (blk_base + b0s_half[c]) | (nbs_half[c] << 16)
            for p in range(ng):
                n_p = min(4, g - 4 * p)
                total_half = n_p * half_u1 * _S1
                fc2_h = [0] * half_ctas
                for c in range(half_ctas):
                    if p >= lag:
                        fc2_h[c] += nbs_half[c] * min(4, g - 4 * (p - lag)) * blk_bytes
                    if p == ng - 1 and not (early0 and p == 0 and m_max >= 2):
                        for q in range(max(0, ng - lag), ng):
                            fc2_h[c] += int(tail_w * nbs_half[c] * min(4, g - 4 * q) * blk_bytes)
                    if no_fc1_half[c]:
                        fc2_h[c] += big_w
                items_h = _plan_fc1_group(
                    fc2_h,
                    total_half,
                    stages=_S1,
                    item_bytes=item_bytes,
                    fine=fine,
                    no_fc1=no_fc1_half,
                )
                i0 = 0
                for c in range(half_ctas):
                    val = (i0 | ((i0 + items_h[c]) << 16)) if items_h[c] else 0
                    plan[g][c][p] = val
                    plan[g][half_ctas + c][p] = val
                    i0 += items_h[c]
                if i0 != total_half:
                    raise RuntimeError(f"die_aware plan mismatch: {i0} != {total_half}")
        return plan, max(nbs_half)

    nbs = [nb2 // num_ctas + (1 if c < nb2 % num_ctas else 0) for c in range(num_ctas)]
    b0s = [sum(nbs[:c]) for c in range(num_ctas)]
    min_nb = min(nbs)
    no_fc1 = [heavy_fc2_only and nbs[c] > min_nb for c in range(num_ctas)]
    for g in range(TOP_K, gcap + 1):
        ng = (g + 3) // 4
        for c in range(num_ctas):
            plan[g][c][m_max] = b0s[c] | (nbs[c] << 16)
        for p in range(ng):
            n_p = min(4, g - 4 * p)
            total = n_p * u1_full * _S1
            fc2 = [0] * num_ctas
            for c in range(num_ctas):
                if p >= lag:
                    fc2[c] += nbs[c] * min(4, g - 4 * (p - lag)) * blk_bytes
                if p == ng - 1 and not (early0 and p == 0 and m_max >= 2):
                    for q in range(max(0, ng - lag), ng):
                        fc2[c] += int(tail_w * nbs[c] * min(4, g - 4 * q) * blk_bytes)
                if no_fc1[c]:
                    fc2[c] += big_w
            items = _plan_fc1_group(
                fc2, total, stages=_S1, item_bytes=item_bytes, fine=fine, no_fc1=no_fc1
            )
            i0 = 0
            for c in range(num_ctas):
                if items[c]:
                    plan[g][c][p] = i0 | ((i0 + items[c]) << 16)
                i0 += items[c]
            if i0 != total:
                raise RuntimeError(f"phase plan mismatch: {i0} != {total}")
    return plan, max(nbs)


def make_phase_plan(
    num_ctas: int,
    m_max: int,
    *,
    i_tp: int = 768,
    lag: int = _LAG,
    tail_w: float = 1.0,
    early0: bool = True,
    fine: bool = True,
    heavy_fc2_only: bool = False,
    die_aware: bool = False,
) -> tuple["torch.Tensor", int]:
    """Torch-tensor wrapper over :func:`make_phase_plan_host`."""
    if torch is None:
        raise RuntimeError("make_phase_plan requires PyTorch")
    plan_list, nb_cap = make_phase_plan_host(
        num_ctas,
        m_max,
        i_tp=i_tp,
        lag=lag,
        tail_w=tail_w,
        early0=early0,
        fine=fine,
        heavy_fc2_only=heavy_fc2_only,
        die_aware=die_aware,
    )
    return torch.tensor(plan_list, dtype=torch.int32), nb_cap


# =============================================================================
# All-reduce push descriptor helpers (world_size in {4, 8, 16}).
# =============================================================================


def ar_push_desc(
    mc_ptr: int,
    flags_or_state_ptr: int,
    rank: int,
    world: int,
    max_tokens: int,
    device,
    *,
    mode: int = AR_PUSH_M3_MC,
    bytes_per_buffer: int = 0,
) -> "torch.Tensor":
    """Build the ``int64 [8]`` ``arp`` descriptor enabling direct multicast AR push from
    :class:`Ml4MoeEngine`.

    * ``world`` must be in ``{4, 8, 16}`` and ``0 <= rank < world``.
    * ``mode == AR_PUSH_M3_MC`` (0): 2-slot parity multicast exchange.
    * ``mode == AR_PUSH_LAMPORT_3SLOT`` (1): 3-slot ``LamportWorkspace`` multicast buffer
      (``flags_or_state_ptr = workspace.state.data_ptr()``, 64-bit offsets).
    * ``mode == AR_PUSH_MCAST_HOP`` (2): 3-slot ``McastGPUBuffer`` / ``glmMnnvlHop``
      (``flags_or_state_ptr = buffer_flags.data_ptr()``, ``bytes_per_buffer > 0`` multiple of 16).
    """
    if world not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"ar_push_desc: world={world} not in {SUPPORTED_WORLD_SIZES}")
    if not (0 <= rank < world):
        raise ValueError(f"ar_push_desc: rank={rank} out of range [0, {world})")
    if mode not in (AR_PUSH_M3_MC, AR_PUSH_LAMPORT_3SLOT, AR_PUSH_MCAST_HOP):
        raise ValueError(f"ar_push_desc: unsupported mode={mode}")
    if mc_ptr == 0 or flags_or_state_ptr == 0:
        raise ValueError("ar_push_desc: multicast and flags/state pointers must be non-zero")
    if mode == AR_PUSH_MCAST_HOP and (bytes_per_buffer <= 0 or bytes_per_buffer % 16 != 0):
        raise ValueError(
            f"ar_push_desc(AR_PUSH_MCAST_HOP): bytes_per_buffer={bytes_per_buffer} must be positive multiple of 16"
        )
    return torch.tensor(
        [
            1,
            int(mc_ptr),
            int(flags_or_state_ptr),
            int(rank),
            int(world),
            int(max_tokens),
            int(mode),
            int(bytes_per_buffer),
        ],
        dtype=torch.int64,
        device=device,
    )


def ar_push_desc_from_lamport(workspace, max_tokens: int | None = None) -> Optional["torch.Tensor"]:
    """Build a ``mode=1`` push descriptor from ML4's ``LamportWorkspace``, or None if multicast is
    unavailable or ``world`` is unsupported."""
    mc = int(getattr(workspace, "multicast_base", 0) or 0)
    world = int(getattr(workspace, "world", 0))
    if mc == 0 or world not in SUPPORTED_WORLD_SIZES:
        return None
    cap = workspace.max_rows if max_tokens is None else max_tokens
    return ar_push_desc(
        mc,
        workspace.state.data_ptr(),
        workspace.rank,
        world,
        cap,
        workspace.state.device,
        mode=AR_PUSH_LAMPORT_3SLOT,
    )


# =============================================================================
# Shared expert pre-tiling and runtime support.
# =============================================================================


def is_supported(device=None) -> tuple[bool, str]:
    if torch is None or not torch.cuda.is_available():
        return False, "no CUDA"
    major, _ = torch.cuda.get_device_capability(device)
    if major != 10:
        return False, "requires SM100-family GPU (tcgen05 block-scaled MMA)"
    try:
        import cutlass  # noqa: F401
        from cutlass.experimental import primitives  # noqa: F401
    except ImportError:
        return False, "CuTe DSL (nvidia-cutlass-dsl) is not installed"
    return True, ""


def _view(t: "torch.Tensor", align: int):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align)


def _kernel_module(config: dict):
    key = tuple(sorted(config.items()))
    mod = _MODULES.get(key)
    if mod is None:
        name = f"{__name__}__dsl_" + "_".join(f"{k}{v}" for k, v in key)
        spec = importlib.util.spec_from_file_location(name, _DSL_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        mod.ML4_CONFIG = dict(config)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _MODULES[key] = mod
    return mod


def pretile_shared(
    gu8: "torch.Tensor",
    gusf8: "torch.Tensor",
    dn8: "torch.Tensor",
    dnsf8: "torch.Tensor",
    n_cta: int = SHARED_CTAS,
    *,
    shared_inter: int,
) -> "torch.Tensor":
    """Pre-tile ML4 shared-expert FP8 weights and R128c4 UE8M0 scales into the exact 16 KB + 512 B
    SMEM stage images streamed by the 24 shared CTAs. Works for ``shared_inter`` in ``{384, 768, 1536}``.
    """
    if shared_inter not in SUPPORTED_INTER:
        raise ValueError(f"pretile_shared: unsupported shared_inter={shared_inter}")
    dev = gu8.device
    h_, i_ = HIDDEN, shared_inter
    p = i_ // n_cta
    split = 128 // (2 * p)
    gt = h_ // 128 // split
    dkt = i_ // 128
    gu = gu8.view(torch.uint8).reshape(2 * i_, h_)
    dn = dn8.view(torch.uint8).reshape(h_, i_)
    gusf = gusf8.view(torch.uint8).reshape(-1)
    dnsf = dnsf8.view(torch.uint8).reshape(-1)
    gkb, dkb = h_ // 32, i_ // 32

    def sf_bytes(sf, n, b, kbn):
        return sf[
            ((n >> 7) * (kbn // 4) + (b >> 2)) * 512 + (n & 31) * 16 + ((n >> 5) & 3) * 4 + (b & 3)
        ]

    r = torch.arange(128, device=dev)
    q = r >> 5
    ri = r & 31
    octant = r // p
    chunk = torch.arange(8, device=dev)
    swz = chunk[None, :] ^ (r[:, None] & 7)
    out = torch.empty(n_cta, gt + 2 * dkt, 16384 + 512, dtype=torch.uint8, device=dev)
    for s in range(n_cta):
        nrow = (octant & 1) * i_ + p * s + r % p
        sl = octant >> 1
        for g in range(gt):
            kt = sl * gt + g
            tile = gu[nrow[:, None], kt[:, None] * 128 + torch.arange(128, device=dev)[None, :]]
            img = tile.view(128, 8, 16)[r[:, None], swz]
            out[s, g, :16384] = img.reshape(-1)
            kb = torch.arange(4, device=dev)
            codes = sf_bytes(gusf, nrow[:, None], kt[:, None] * 4 + kb[None, :], gkb)
            atom = torch.empty(512, dtype=torch.uint8, device=dev)
            atom[(ri[:, None] * 16 + q[:, None] * 4 + kb[None, :]).reshape(-1)] = codes.reshape(-1)
            out[s, g, 16384:] = atom
        for gd in range(2 * dkt):
            jt, kt = gd // dkt, gd % dkt
            row0 = 256 * s + 128 * jt
            tile = dn[row0 : row0 + 128, kt * 128 : (kt + 1) * 128]
            img = tile.reshape(128, 8, 16)[r[:, None], swz]
            out[s, gt + gd, :16384] = img.reshape(-1)
            atom_off = ((row0 >> 7) * (dkb // 4) + kt) * 512
            out[s, gt + gd, 16384:] = dnsf[atom_off : atom_off + 512]
    return out.view(-1)


def _cached_pretiled(
    gu8: "torch.Tensor",
    gusf8: "torch.Tensor",
    dn8: "torch.Tensor",
    dnsf8: "torch.Tensor",
    shared_inter: int,
) -> "torch.Tensor":
    key = (
        gu8.device,
        shared_inter,
        gu8.data_ptr(),
        gusf8.data_ptr(),
        dn8.data_ptr(),
        dnsf8.data_ptr(),
    )
    hit = _PRETILED.get(key)
    if hit is None:
        pt = pretile_shared(gu8, gusf8, dn8, dnsf8, shared_inter=shared_inter)
        hit = _PRETILED[key] = ((gu8, gusf8, dn8, dnsf8), pt)
    return hit[1]


# =============================================================================
# Ml4MoeLayer / Ml4MoeState / Ml4MoeEngine
# =============================================================================


class Ml4MoeLayer:
    """One ML4 MoE layer bound to an :class:`Ml4MoeState`.

    Reads ML4's in-place TRTLLM-Gen NVFP4 buffers and BF16 router weight without duplicating them
    (except when ``locality_mode='n_split'`` or ``'expert_half'`` places die-local VMM copies, in
    which case pre-placed :class:`PlacedSplit` objects may be supplied directly from ``weights.py``).
    """

    def __init__(
        self,
        state: "Ml4MoeState",
        fc1_w: "torch.Tensor",
        fc1_s: "torch.Tensor",
        fc2_w: "torch.Tensor",
        fc2_s: "torch.Tensor",
        out1_gate: "torch.Tensor",
        out1: "torch.Tensor",
        out2: "torch.Tensor",
        router_weight: "torch.Tensor",
        gscale: "torch.Tensor",
        shared_weights: Optional[
            tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]
        ] = None,
        *,
        placed_fc1_w: Optional[PlacedSplit] = None,
        placed_fc1_s: Optional[PlacedSplit] = None,
        placed_fc2_w: Optional[PlacedSplit] = None,
        placed_fc2_s: Optional[PlacedSplit] = None,
    ) -> None:
        inter = state.inter
        e = NUM_EXPERTS
        expected = {
            "fc1_w": (fc1_w, (e, 2 * inter, HIDDEN // 2)),
            "fc1_s": (fc1_s, (e, 2 * inter, HIDDEN // 16)),
            "fc2_w": (fc2_w, (e, HIDDEN, inter // 2)),
            "fc2_s": (fc2_s, (e, HIDDEN, inter // 16)),
        }
        for name, (t, shape) in expected.items():
            if t.element_size() != 1 or tuple(t.shape) != shape or not t.is_contiguous():
                raise ValueError(
                    f"ml4_moe: {name} is {t.dtype} {tuple(t.shape)}, expected contiguous 1-byte {shape}"
                )
        if (
            router_weight.dtype != torch.bfloat16
            or tuple(router_weight.shape) != (e, HIDDEN)
            or not router_weight.is_contiguous()
        ):
            raise ValueError(
                f"ml4_moe: router_weight must be contiguous bf16 [{e}, {HIDDEN}], "
                f"got {router_weight.dtype} {tuple(router_weight.shape)}"
            )
        if gscale.dtype != torch.float32 or gscale.numel() != 1:
            raise ValueError("ml4_moe: gscale must be fp32 [1]")

        self.state = state
        self.fc1_w = fc1_w.view(torch.uint8)
        self.fc1_s = fc1_s.view(torch.uint8)
        self.fc2_w = fc2_w.view(torch.uint8)
        self.fc2_s = fc2_s.view(torch.uint8)
        self.router_weight = router_weight
        self.gscale = gscale.contiguous()
        # Pack [out1_gate, out1, out2, 0] into fp32 [4, 128].
        zero = torch.zeros_like(out1_gate, dtype=torch.float32)
        self.escal = (
            torch.stack([out1_gate.float(), out1.float(), out2.float(), zero], dim=0)
            .reshape(4, e)
            .contiguous()
        )
        # Die-local views according to state.locality_mode.
        self._placed_refs: list[object] = []
        if state.locality_mode == "n_split":
            lm = state.locality_map
            assert lm is not None
            p1w = placed_fc1_w or lm.split_rows_padded(self.fc1_w, dim=1, mode="n_split")
            p1s = placed_fc1_s or lm.split_rows_padded(self.fc1_s, dim=1, mode="n_split")
            p2w = placed_fc2_w or lm.split_rows_padded(self.fc2_w, dim=1, mode="n_split")
            p2s = placed_fc2_s or lm.split_rows_padded(self.fc2_s, dim=1, mode="n_split")
            self._placed_refs.extend([p1w, p1s, p2w, p2s])
            self.a1_d0, self.a1_d1 = p1w.half0.view(torch.uint8), p1w.half1.view(torch.uint8)
            self.a2_d0, self.a2_d1 = p2w.half0.view(torch.uint8), p2w.half1.view(torch.uint8)
            self.s1w_d0 = p1s.half0.view(torch.uint8).reshape(-1).view(torch.int32)
            self.s1w_d1 = p1s.half1.view(torch.uint8).reshape(-1).view(torch.int32)
            self.s2w_d0 = p2s.half0.view(torch.uint8).reshape(-1).view(torch.int32)
            self.s2w_d1 = p2s.half1.view(torch.uint8).reshape(-1).view(torch.int32)
        elif state.locality_mode == "expert_half":
            lm = state.locality_map
            assert lm is not None
            p1w = placed_fc1_w or lm.place_expert_half(self.fc1_w, allow_padded=False)
            p1s = placed_fc1_s or lm.place_expert_half(self.fc1_s, allow_padded=False)
            p2w = placed_fc2_w or lm.place_expert_half(self.fc2_w, allow_padded=False)
            p2s = placed_fc2_s or lm.place_expert_half(self.fc2_s, allow_padded=True)
            assert (
                p1w.contiguous is not None
                and p1s.contiguous is not None
                and p2w.contiguous is not None
            )
            self._placed_refs.extend([p1w, p1s, p2w, p2s])
            self.a1_d0 = self.a1_d1 = p1w.contiguous.view(torch.uint8)
            self.a2_d0 = self.a2_d1 = p2w.contiguous.view(torch.uint8)
            self.s1w_d0 = self.s1w_d1 = (
                p1s.contiguous.view(torch.uint8).reshape(-1).view(torch.int32)
            )
            self.s2w_d0 = p2s.half0.view(torch.uint8).reshape(-1).view(torch.int32)
            self.s2w_d1 = p2s.half1.view(torch.uint8).reshape(-1).view(torch.int32)
        else:
            self.a1_d0 = self.a1_d1 = self.fc1_w
            self.a2_d0 = self.a2_d1 = self.fc2_w
            self.s1w_d0 = self.s1w_d1 = self.fc1_s.reshape(-1).view(torch.int32)
            self.s2w_d0 = self.s2w_d1 = self.fc2_s.reshape(-1).view(torch.int32)

        self.shared_mxfp8 = None
        self.sh_pretiled = state._dummy.view(torch.uint8)
        if state.shared:
            if shared_weights is None:
                raise ValueError("ml4_moe (shared=True): shared_weights are required")
            gu_w, gu_sf, dn_w, dn_sf = shared_weights
            sh_i = state.shared_inter
            if (
                gu_w.dtype != torch.float8_e4m3fn
                or tuple(gu_w.shape) != (2 * sh_i, HIDDEN)
                or not gu_w.is_contiguous()
            ):
                raise ValueError(
                    f"ml4_moe (shared): gate_up weight must be contiguous e4m3 ({2 * sh_i}, {HIDDEN})"
                )
            if (
                dn_w.dtype != torch.float8_e4m3fn
                or tuple(dn_w.shape) != (HIDDEN, sh_i)
                or not dn_w.is_contiguous()
            ):
                raise ValueError(
                    f"ml4_moe (shared): down weight must be contiguous e4m3 ({HIDDEN}, {sh_i})"
                )
            if gu_sf.numel() != 2 * sh_i * HIDDEN // 32 or dn_sf.numel() != HIDDEN * sh_i // 32:
                raise ValueError("ml4_moe (shared): UE8M0 scales must be R128c4 [N * K // 32]")
            self.shared_mxfp8 = (gu_w, gu_sf, dn_w, dn_sf)
            self.sh_pretiled = _cached_pretiled(gu_w, gu_sf, dn_w, dn_sf, sh_i)

    def __call__(
        self,
        hidden: "torch.Tensor",
        out: Optional["torch.Tensor"] = None,
        arp: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        return self.state.run(self, hidden, out=out, arp=arp)


class Ml4MoeState:
    """Shared workspace and compiled CuTe DSL kernel for ML4 MoE (1..8 decode tokens)."""

    def __init__(
        self,
        device,
        num_tokens: int,
        *,
        inter: int = 768,
        shared_inter: Optional[int] = None,
        num_ctas: Optional[int] = None,
        shared: bool = True,
        locality_mode: LocalityModeStr | Literal["auto"] = "none",
        early_ar_release: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("Ml4MoeState requires PyTorch")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Ml4MoeState allocates and compiles: build before CUDA-graph capture"
            )
        if inter not in SUPPORTED_INTER:
            raise ValueError(f"ml4_moe: inter={inter} not in {SUPPORTED_INTER}")
        if not (1 <= num_tokens <= 8):
            raise ValueError(f"ml4_moe: takes 1..8 tokens per call, got {num_tokens}")

        self.device = torch.device(device)
        ok, why = is_supported(self.device)
        if not ok:
            raise RuntimeError(f"ml4_moe: {why}")
        self.inter = inter
        self.shared_inter = inter if shared_inter is None else shared_inter
        self.shared = bool(shared)
        self.m = num_tokens

        sms_all = torch.cuda.get_device_properties(self.device).multi_processor_count
        max_routed = HIDDEN // 32  # 192
        nsh = SHARED_CTAS if self.shared else 0
        default_ctas = min(max_routed, sms_all - nsh)
        req_per_domain = required_moe_ctas_per_domain(
            sms_all,
            shared=self.shared,
            shared_ctas=SHARED_CTAS,
            max_routed=max_routed,
            num_ctas=num_ctas,
        )
        resolved_mode: LocalityModeStr
        resolved_mode, _ = resolve_moe_locality_mode(locality_mode, inter, NUM_EXPERTS)

        self.locality_map: Optional[LocalityMap] = None
        if resolved_mode != "none":
            try:
                lm = LocalityMap.get(self.device)
            except (ImportError, RuntimeError) as err:
                if logger is not None:
                    logger.warning(
                        f"[ML4 sm_107] ml4_moe_engine: LocalityMap unavailable on {self.device} "
                        f"({err}); falling back to 'none'"
                    )
                resolved_mode = "none"
            else:
                resolved_mode, fallback_reason = resolve_moe_locality_mode(
                    resolved_mode,
                    inter,
                    NUM_EXPERTS,
                    sms_per_domain=lm.sms_per_domain,
                    required_per_domain=req_per_domain,
                )
                if fallback_reason is not None:
                    if logger is not None:
                        logger.warning(
                            f"[ML4 sm_107] ml4_moe_engine on {self.device}: {fallback_reason}"
                        )
                elif resolved_mode != "none":
                    self.locality_map = lm

        resolved_ctas = default_ctas if num_ctas is None else num_ctas
        if resolved_mode != "none" and resolved_ctas % 2 != 0:
            resolved_ctas -= 1
        if not (MIN_CTAS <= resolved_ctas <= max_routed):
            raise ValueError(
                f"ml4_moe: num_ctas={resolved_ctas} must be in [{MIN_CTAS}, {max_routed}]"
            )
        if resolved_ctas + nsh > sms_all:
            raise ValueError(
                f"ml4_moe: {resolved_ctas} routed + {nsh} shared exceeds {sms_all} SMs"
            )
        self.num_ctas = resolved_ctas
        self.locality_mode: LocalityModeStr = resolved_mode
        grid_ctas = sms_all if resolved_mode != "none" else (self.num_ctas + nsh)

        if self.locality_map is not None:
            lmap, _ = self.locality_map.logical_map(
                per_domain=self.num_ctas // 2,
                extra=nsh,
                allow_idle=(self.num_ctas + nsh < sms_all),
            )
            self.lmap = lmap
        else:
            self.lmap = torch.zeros(1, dtype=torch.int32, device=self.device)

        heavy_fc2 = bool(_TP8_HEAVY_FC2_ONLY_BY_M[num_tokens - 1]) if inter == 768 else False
        early0 = (
            bool(_TP8_EARLY0_BY_M[num_tokens - 1]) if inter == 768 else True
        ) and resolved_mode != "expert_half"

        plan, nb_cap = make_phase_plan(
            self.num_ctas,
            num_tokens,
            i_tp=inter,
            lag=_LAG,
            tail_w=0.0 if num_tokens == 1 else 1.0,
            early0=early0,
            fine=True,
            heavy_fc2_only=heavy_fc2,
            die_aware=(resolved_mode == "n_split"),
        )

        self.config: dict[str, int] = {
            "m_max": num_tokens,
            "num_ctas": self.num_ctas,
            "nb2_cap": nb_cap,
            "fused_router": 1,
            "timing": 0,
            "i_tp": inter,
            "locality_mode": _LOCALITY_TO_INT[resolved_mode],
            "grid_ctas": grid_ctas,
            "early_ar": int(bool(early_ar_release)),
            "sh_pretile": 1,
            "shared": int(self.shared),
            "sh_i": self.shared_inter,
        }
        if inter == 768:
            if _TP8_STAGES_BY_M[num_tokens - 1] > 0:
                self.config["stages"] = _TP8_STAGES_BY_M[num_tokens - 1]
            if _TP8_B2_POLL_NS_BY_M[num_tokens - 1] > 0:
                self.config["b2_poll_ns"] = _TP8_B2_POLL_NS_BY_M[num_tokens - 1]
            if not early0:
                self.config["early0"] = 0
            if num_tokens <= _TP8_FLAT_MAX_TOKENS:
                self.config.update(_TP8_FLAT_STACK)
        elif inter == 384:
            self.config["b2img"] = _TP16_B2IMG_BY_M[num_tokens - 1]
            self.config["ar_pre"] = _TP16_AR_PRE_BY_M[num_tokens - 1]
            if _TP16_HB_REPL_BY_M[num_tokens - 1] > 1:
                self.config["hb_repl"] = _TP16_HB_REPL_BY_M[num_tokens - 1]
            if num_tokens <= _TP16_LG_REPL_MAX_TOKENS:
                self.config["lg_repl"] = 32
            if _TP16_STAGES_BY_M[num_tokens - 1] > 0:
                self.config["stages"] = _TP16_STAGES_BY_M[num_tokens - 1]
            if not early0:
                self.config["early0"] = 0
        else:
            if not early0:
                self.config["early0"] = 0

        self.mod = mod = _kernel_module(self.config)
        kw = dict(device=self.device)
        self.plan = plan.to(self.device)
        self.partials = torch.full(
            (mod.G_CAP * mod.U1 * num_tokens * 32,), -1, dtype=torch.int32, **kw
        )
        hb_words = (
            mod.HB_REPL * mod.HB_STRIDE if mod.HB_REPL > 1 else mod.G_CAP * num_tokens * mod.U1 * 4
        )
        self.hbuf32 = torch.zeros(hb_words, dtype=torch.int32, **kw)
        self.epochs = torch.zeros(self.num_ctas + mod.NSH, dtype=torch.int32, **kw)
        self.route_dbg = torch.zeros(num_tokens, 8, dtype=torch.int32, **kw)
        self.tdbg = torch.zeros(self.num_ctas + mod.NSH, 64, dtype=torch.int64, **kw)
        self.lgbuf = torch.zeros(mod.LG_REPL * mod.LG_STRIDE, dtype=torch.int64, **kw)
        self._dummy = torch.zeros(4, dtype=torch.int32, **kw)
        self._ar_off = torch.zeros(8, dtype=torch.int64, **kw)
        self.img32 = torch.zeros(max(1, mod.NG_CAP * mod.IMG_BYTES // 4), dtype=torch.int32, **kw)
        self.cnt = torch.zeros(mod.NG_CAP * 8 + 8, dtype=torch.int32, **kw)
        self.hx = torch.zeros(max(1, num_tokens * self.shared_inter), dtype=torch.int32, **kw)
        self.dnx = torch.zeros(max(1, num_tokens * HIDDEN), dtype=torch.int32, **kw)
        self._fn = None

    def layer(
        self,
        fc1_w: "torch.Tensor",
        fc1_s: "torch.Tensor",
        fc2_w: "torch.Tensor",
        fc2_s: "torch.Tensor",
        out1_gate: "torch.Tensor",
        out1: "torch.Tensor",
        out2: "torch.Tensor",
        router_weight: "torch.Tensor",
        gscale: "torch.Tensor",
        shared_weights=None,
        *,
        placed_fc1_w: Optional[PlacedSplit] = None,
        placed_fc1_s: Optional[PlacedSplit] = None,
        placed_fc2_w: Optional[PlacedSplit] = None,
        placed_fc2_s: Optional[PlacedSplit] = None,
    ) -> Ml4MoeLayer:
        return Ml4MoeLayer(
            self,
            fc1_w,
            fc1_s,
            fc2_w,
            fc2_s,
            out1_gate,
            out1,
            out2,
            router_weight,
            gscale,
            shared_weights,
            placed_fc1_w=placed_fc1_w,
            placed_fc1_s=placed_fc1_s,
            placed_fc2_w=placed_fc2_w,
            placed_fc2_s=placed_fc2_s,
        )

    def precompile(self, layer: Ml4MoeLayer) -> None:
        """Compile the CuTe DSL kernel ahead of CUDA-graph capture without launching it."""
        if self._fn is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ml4_moe compiles on its first call: warm up before graph capture")
        import cutlass.cute as cute

        dummy_hidden = torch.empty(self.m, HIDDEN, dtype=torch.bfloat16, device=self.device)
        dummy_out = torch.empty(self.m, HIDDEN, dtype=torch.bfloat16, device=self.device)
        args = self._build_args(layer, dummy_hidden, dummy_out, arp=None)
        self._ensure_compiled(args, cute)

    def _build_args(
        self,
        layer: Ml4MoeLayer,
        hidden: "torch.Tensor",
        out: "torch.Tensor",
        arp: Optional["torch.Tensor"],
    ) -> tuple:
        return (
            layer.a1_d0,
            layer.a1_d1,
            layer.a2_d0,
            layer.a2_d1,
            self._dummy.view(torch.float32),
            self._dummy,
            self._dummy,
            layer.escal,
            self.plan.view(-1),
            self.partials,
            self.hbuf32,
            self.epochs,
            out.view(-1),
            self.route_dbg.view(-1),
            self.tdbg.view(-1),
            hidden.view(-1).view(torch.int32),
            layer.router_weight.view(-1).view(torch.int32),
            layer.gscale.view(-1),
            self.lgbuf,
            layer.sh_pretiled,
            self.hx,
            self.dnx,
            self._ar_off if arp is None else arp,
            layer.s1w_d0,
            layer.s1w_d1,
            layer.s2w_d0,
            layer.s2w_d1,
            self.img32,
            self.cnt,
            self.lmap,
        )

    def _ensure_compiled(self, args: tuple, cute) -> None:
        aligns = (
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            4,
            16,
            16,
            16,
            16,
            16,
            4,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            4,
        )
        sig = [_view(t, a) for t, a in zip(args, aligns)]
        key = tuple(sorted(self.config.items()))
        fn = _COMPILED.get(key)
        if fn is None:
            fn = _COMPILED[key] = cute.compile(
                self.mod.ml4_moe,
                *sig,
                cute.runtime.make_fake_stream(),
                options="--enable-tvm-ffi",
            )
        self._fn = fn

    def run(
        self,
        layer: Ml4MoeLayer,
        hidden: "torch.Tensor",
        out: Optional["torch.Tensor"] = None,
        arp: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        m = self.m
        if (
            hidden is None
            or hidden.dtype != torch.bfloat16
            or tuple(hidden.shape) != (m, HIDDEN)
            or not hidden.is_contiguous()
            or hidden.data_ptr() % 16 != 0
        ):
            raise ValueError(
                f"ml4_moe: expects 16-byte-aligned contiguous bf16 hidden [{m}, {HIDDEN}]"
            )
        if out is None:
            out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=self.device)
        elif (
            out.dtype != torch.bfloat16
            or tuple(out.shape) != (m, HIDDEN)
            or not out.is_contiguous()
            or out.data_ptr() % 16 != 0
        ):
            raise ValueError(f"ml4_moe: out must be contiguous bf16 [{m}, {HIDDEN}]")
        if arp is not None and (
            arp.dtype != torch.int64 or arp.numel() != 8 or not arp.is_contiguous()
        ):
            raise ValueError("ml4_moe: arp must be contiguous int64 [8]")

        args = self._build_args(layer, hidden, out, arp)
        if self._fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "ml4_moe compiles on its first call: warm up before graph capture"
                )
            import cutlass.cute as cute

            self._ensure_compiled(args, cute)
        self._fn(*args, torch.cuda.current_stream(self.device).cuda_stream)
        return out


# =============================================================================
# Exact PyTorch reference implementations (CPU & GPU runnable).
# =============================================================================


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_RCP6 = 0.16666667163372039795


def unswizzle_sf(sf: "torch.Tensor", rows: int, blocks: int) -> "torch.Tensor":
    """Invert 128x4 scale swizzle into logical ``[rows, blocks]`` uint8."""
    flat = sf.reshape(-1).view(torch.uint8)
    r = torch.arange(rows, device=flat.device).view(-1, 1)
    b = torch.arange(blocks, device=flat.device).view(1, -1)
    padded_blocks = (blocks + 3) // 4 * 4
    idx = (
        (r // 128) * (128 * padded_blocks)
        + (b // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (b % 4)
    )
    return flat[idx]


def _e2m1_decode(packed: "torch.Tensor") -> "torch.Tensor":
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    lo = lut[(packed & 0xF).long()]
    hi = lut[(packed >> 4).long()]
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def _e2m1_round(x: "torch.Tensor") -> "torch.Tensor":
    a = x.abs()
    q = torch.full_like(a, 6.0)
    for hi, val, inc in (
        (5.0, 4.0, True),
        (3.5, 3.0, False),
        (2.5, 2.0, True),
        (1.75, 1.5, False),
        (1.25, 1.0, True),
        (0.75, 0.5, False),
        (0.25, 0.0, True),
    ):
        q = torch.where(a <= hi if inc else a < hi, torch.full_like(a, val), q)
    return torch.where(x < 0, -q, q)


def _e4m3_round(x: "torch.Tensor") -> "torch.Tensor":
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def _k128_quant_dequant(x: "torch.Tensor") -> "torch.Tensor":
    """Exact ML4 1x128 FP8 quantization round-trip (matches ``fp8_quantize_1x128_packed_ue8m0``)."""
    m, k = x.shape
    blocks = x.float().reshape(m, k // 128, 128)
    amax = blocks.abs().amax(dim=-1).clamp(min=1.0e-10)
    mant, exp = torch.frexp(amax.double() / 448.0)
    e = torch.where(mant == 0.5, exp - 1, exp).clamp(max=127)
    scale = torch.ldexp(torch.ones_like(e, dtype=torch.float64), e).float().unsqueeze(-1)
    return (_e4m3_round(blocks / scale) * scale).reshape(m, k)


def _r128c4_dequant(w: "torch.Tensor", sf: "torch.Tensor") -> "torch.Tensor":
    n, k = w.shape
    codes = unswizzle_sf(sf, n, k // 32).float()
    return w.view(torch.float8_e4m3fn).float() * torch.exp2(codes - 127.0).repeat_interleave(
        32, dim=1
    )


def ml4_moe_route_reference(
    hidden: "torch.Tensor", router_weight: "torch.Tensor"
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """ML4 softmax top-4 renormalized routing matching ``renorm_moe_routing_op``."""
    logits = (hidden.double() @ router_weight.double().t()).float()
    ids = torch.sort(logits, dim=-1, descending=True, stable=True).indices[:, :TOP_K]
    selected = logits.gather(1, ids).float()
    e = torch.exp(selected - selected.max(dim=-1, keepdim=True).values)
    weights = (e / e.sum(dim=-1, keepdim=True)).to(torch.bfloat16).float()
    return ids, weights


def ml4_swiglu_quant_reference(
    up_raw: "torch.Tensor",
    gate_raw: "torch.Tensor",
    out1_gate: "torch.Tensor",
    out1: "torch.Tensor",
) -> "torch.Tensor":
    """ML4 standard SwiGLU + NVFP4 requantization matching ``_swiglu_quant_ml4``."""
    g = gate_raw.float() * out1_gate.float()
    hc = (up_raw.float() * out1.float()) * (g * torch.sigmoid(g))
    blocks = hc.reshape(-1, 16)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    sff = _e4m3_round(amax * _RCP6)
    osc = torch.where(sff != 0, 1.0 / sff, torch.zeros_like(sff))
    return (_e2m1_round(blocks * osc) * sff).reshape(-1)


def ml4_shared_expert_reference(
    hidden: "torch.Tensor",
    shared_weights: tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"],
) -> "torch.Tensor":
    """ML4 shared expert reference: K128 FP8 gate_up -> standard SwiGLU -> K128 FP8 down."""
    gu_w, gu_sf, dn_w, dn_sf = shared_weights
    sh_i = dn_w.shape[1]
    gu = (
        (_k128_quant_dequant(hidden).double() @ _r128c4_dequant(gu_w, gu_sf).double().t())
        .float()
        .to(torch.bfloat16)
        .float()
    )
    gate, up = gu[:, :sh_i], gu[:, sh_i:]
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
    return (
        (_k128_quant_dequant(act).double() @ _r128c4_dequant(dn_w, dn_sf).double().t())
        .float()
        .to(torch.bfloat16)
        .float()
    )


def _block32_perm_ref(rows: int, device) -> "torch.Tensor":
    j = torch.arange(32)
    one = torch.empty(32, dtype=torch.long)
    one[(j % 4) * 8 + j // 4] = j
    blocks = torch.arange(rows // 32).repeat_interleave(32) * 32
    return (one.repeat(rows // 32) + blocks).to(device)


def _fc1_perm_ref(rows: int, device) -> "torch.Tensor":
    interleave = torch.empty(rows, dtype=torch.long)
    interleave[0::2] = torch.arange(rows // 2)
    interleave[1::2] = torch.arange(rows // 2, rows)
    return interleave.to(device)[_block32_perm_ref(rows, device)]


def _nvfp4_expert_ref(
    w: "torch.Tensor", sf: "torch.Tensor", perm_inv: "torch.Tensor"
) -> "torch.Tensor":
    rows, half = w.shape
    codes = unswizzle_sf(sf, rows, 2 * half // 16).view(torch.float8_e4m3fn).float()
    vals = _e2m1_decode(w) * codes.repeat_interleave(16, dim=1)
    return vals[perm_inv]


def _nvfp4_act_ref(x: "torch.Tensor", gscale: "torch.Tensor") -> "torch.Tensor":
    rows, k = x.shape
    sc = gscale.reshape(()).float()
    blocks = x.float().reshape(rows, k // 16, 16)
    vmax = blocks.abs().amax(dim=-1, keepdim=True)
    sff = _e4m3_round(sc * (vmax * _RCP6))
    osc = torch.where(vmax != 0, sc / sff, torch.zeros_like(vmax))
    return (_e2m1_round(blocks * osc) * sff).reshape(rows, k)


def ml4_moe_reference(
    hidden: "torch.Tensor", layer: Ml4MoeLayer, shared: Optional[bool] = None
) -> "torch.Tensor":
    """Plain-PyTorch reference matching :class:`Ml4MoeEngine` (BF16 ``[M, 6144]``)."""
    use_shared = layer.state.shared if shared is None else shared
    inter = layer.state.inter
    x = hidden.float()
    m = x.shape[0]
    ids, weights = ml4_moe_route_reference(x, layer.router_weight)
    xr = _nvfp4_act_ref(x, layer.gscale)
    inv31 = torch.argsort(_fc1_perm_ref(2 * inter, x.device))
    inv2 = torch.argsort(_block32_perm_ref(HIDDEN, x.device))
    y = torch.zeros(m, TOP_K, HIDDEN, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        w31 = _nvfp4_expert_ref(layer.fc1_w[e], layer.fc1_s[e], inv31)
        w2 = _nvfp4_expert_ref(layer.fc2_w[e], layer.fc2_s[e], inv2)
        esc = layer.escal[:, e]
        for t, k in (ids == e).nonzero().tolist():
            acc1 = (xr[t : t + 1].double() @ w31.double().t()).float()[0]
            h = ml4_swiglu_quant_reference(acc1[:inter], acc1[inter:], esc[0], esc[1])
            y[t, k] = (
                ((h.unsqueeze(0).double() @ w2.double().t()).float()[0] * esc[2])
                .to(torch.bfloat16)
                .float()
            )
    acc = torch.zeros(m, HIDDEN, dtype=torch.float64, device=x.device)
    for k in range(TOP_K):
        acc = (acc + weights[:, k : k + 1].double() * y[:, k].double()).float().double()
    routed = acc.float()
    if not use_shared:
        return routed.to(torch.bfloat16)
    sh = ml4_shared_expert_reference(x, layer.shared_mxfp8)
    return (sh + routed.to(torch.bfloat16).float()).to(torch.bfloat16)


try:
    from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
    from tensorrt_llm.logger import logger
except ImportError:  # pragma: no cover - standalone CPU unit-test fallback
    Arch = None  # type: ignore[assignment]
    Cell = None  # type: ignore[assignment]
    OpWrapper = object  # type: ignore[assignment,misc]
    logger = None  # type: ignore[assignment]


class Ml4MoeEngine(OpWrapper):
    """Mistral Large 4 fused persistent CuTe DSL MoE op wrapper (SM107 TP4/TP8/TP16)."""

    if Arch is not None:
        ARCHS = frozenset({Arch.SM_107})
        CELLS = tuple(
            Cell(
                why=f"ML4 SM107 {tag} (I={inter}), {m} tokens, locality={loc}",
                spec=dict(tokens=m, inter=inter, locality_mode=loc),
            )
            for tag, inter, locs in (
                ("TP8", 768, ("none", "n_split", "expert_half")),
                ("TP16", 384, ("none", "n_split", "expert_half")),
            )
            for m in (1, 2, 4, 8)
            for loc in locs
        )

    def __call__(
        self,
        hidden: "torch.Tensor",
        layer: Ml4MoeLayer,
        out: Optional["torch.Tensor"] = None,
        arp: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        state = layer.state
        key = (state.device, state.m, state.inter, state.locality_mode, state.shared)
        if key not in _ENGAGED:
            _ENGAGED.add(key)
            if logger is not None:
                logger.info(
                    f"[ML4 sm_107] ml4_moe_engine engaged: M={state.m}, I={state.inter}, "
                    f"locality={state.locality_mode}, shared={state.shared}, "
                    f"ctas={state.num_ctas}+{state.mod.NSH}, stages={state.mod.STAGES} on {state.device}"
                )
        return state.run(layer, hidden, out=out, arp=arp)

    def reference(
        self,
        hidden: "torch.Tensor",
        layer: Ml4MoeLayer,
        out: Optional["torch.Tensor"] = None,
        arp: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        return ml4_moe_reference(hidden, layer, shared=layer.state.shared)


ml4_moe_engine = Ml4MoeEngine()
