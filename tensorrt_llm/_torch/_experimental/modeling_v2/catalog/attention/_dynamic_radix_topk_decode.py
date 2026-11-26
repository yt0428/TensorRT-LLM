# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic length-proportional multi-CTA decode radix top-k kernel (SM107 / VR200) and CPU verifier.

This module implements the next-generation decode top-k architecture designed for SM107 DEP8
(VR200, 212 SMs, 1..32 requests x next_n=4 = 4..128 rows, contexts up to 1M):

1. **Zero GPC cluster fragmentation**: ``cluster=(1, 1, 1)`` — eliminates the 8-CTA-per-GPC
   co-scheduling penalty observed when ``ClusterRadixTopkDecodeKernel`` (C=8) runs beside
   side-stream query GEMMs.
2. **Length-proportional CTA allocation with zero-barrier short-row prologue**: Every short or
   padding dummy row (``end <= TOP_K``) is resolved independently in a short-row prologue assigned
   to the highest-indexed CTAs so CTA 0.. immediately start the longest straggler row. All
   ``total_ctas`` CTAs then partition the long rows (``end > TOP_K``) strictly proportional to
   ``end[r]`` (capped at ``max_ctas_per_row``), making kernel latency scale with total per-rank KV
   rather than ``max_ctx_len`` (directly attacking both the top-k SOL gap and dispatch wait).
3. **L2-resident sparse histogram reduction + self-cleaning persistent state + Pass-2 empty-slice
   skip**: Multi-CTA rows reduce only ``[global_lo, global_cut]`` into a small L2-resident
   zero-initialized state buffer (``MAX_ROWS * STATE_WORDS * 4 B ~= 3.16 MiB``) using
   ``red.relaxed.gpu`` + release/acquire arrival barriers. CTAs whose slice contributes zero
   elements at or above threshold skip Pass 2 entirely (bit-exact). The first ``c - 1`` CTAs exit
   immediately after signaling completion; the single last arriving CTA resolves any gather list,
   writes the ``-1`` tail, and zeroes only the sparse ``[rlo, rcut]`` bins and 17 control words so
   the state buffer returns to all-zeros for the next CUDA Graph replay.

The pure-Python planner (``plan_dynamic_cta_allocation``) and bit-exact multi-CTA emulator
(``emulate_dynamic_radix_topk_decode``) mirror ``DynamicRadixTopkDecodeKernel``'s exact integer
allocation and radix-select protocol so CPU tests verify every edge case without GPU dependencies.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Sequence

TOP_K = 2048
PARTITIONS = 10
LEVELS = ((20, 0xFFF, 4096), (10, 0x3FF, 1024), (0, 0x3FF, 1024))
GATHER_CAPACITY = 4096
PAD_KEY = 0xFF7FFFFF  # key(-FLT_MAX)
NEG_INF_KEY = 0xFF800000  # key(-inf); strictly larger keys are negative NaNs
MAX_ROWS = 128
MAX_REQUESTS = 64
DEFAULT_TOTAL_CTAS = 212  # 1 CTA/SM on 212-SM VR200 (guaranteed co-resident; 424 for 2/SM)
DEFAULT_MAX_CTAS_PER_ROW = 32
SLICE_ALIGN = 8  # 32-byte alignment for vectorized fp32 loads

# Persistent per-row L2 state layout (all zero at entry and restored to zero at exit).
_G_HIST0 = 0
_G_HIST1 = 4096
_G_HIST2 = 5120
_G_CTRL = 6144
_G_CTRL_WORDS = 32
STATE_WORDS_PER_ROW = _G_CTRL + _G_CTRL_WORDS  # 6176 int32 words = 24.7 KiB/row (~3.16 MiB @ 128)

# Control word offsets relative to _G_CTRL.
_C_LO0 = 0
_C_CUT0 = 1
_C_NEG = 2
_C_ARR_R0 = 3
_C_ARR_H0 = 4
_C_LO1 = 5
_C_CUT1 = 6
_C_ARR_R1 = 7
_C_ARR_H1 = 8
_C_LO2 = 9
_C_CUT2 = 10
_C_ARR_R2 = 11
_C_ARR_H2 = 12
_C_ABOVE = 13
_C_LIST = 14
_C_TIE = 15
_C_ARR_DONE = 16
_C_USED_WORDS = 17

# Gather-list layout inside aux_indices[row * PARTITIONS * TOP_K + ...].
_AUX_LKEY = 0
_AUX_LCOL = GATHER_CAPACITY


def float32_to_bits(x: float) -> int:
    """IEEE-754 binary32 bit pattern of ``x`` as an unsigned 32-bit int."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_to_float32(bits: int) -> float:
    """Convert unsigned 32-bit IEEE-754 pattern back to Python float (fp32 value)."""
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def ordered_key_from_bits(bits: int) -> int:
    """Order-preserving uint32 key matching ``_radix_topk_decode._key`` (ascending = better)."""
    bits &= 0xFFFFFFFF
    if (bits & 0x80000000) == 0:
        return (bits ^ 0xFFFFFFFF) & 0x7FFFFFFF
    return bits


def ordered_key_f32(x: float) -> int:
    """Order-preserving uint32 key of a float32 score."""
    return ordered_key_from_bits(float32_to_bits(x))


def _fp16_bin_from_bits(bits: int) -> int:
    """Exact fp16 first-pass bin matching ``_radix_topk_decode._fp16_bin``."""
    x = bits_to_float32(bits)
    if math.isnan(x):
        return 0  # hardware cvt.rn.f16.f32 turns any NaN (including -NaN) into +NaN -> top bin 0
    if math.isinf(x):
        hbits = 0xFC00 if x < 0 else 0x7C00
    else:
        try:
            hbits = struct.unpack("<H", struct.pack("<e", x))[0]
        except OverflowError:
            hbits = 0xFC00 if x < 0 else 0x7C00
    key16 = hbits if (hbits & 0x8000) else ((hbits ^ 0xFFFF) & 0x7FFF)
    return (key16 >> 5) & 0x7FF


def row_causal_ends(seq_lens: Sequence[int], next_n: int, width: int) -> list[int]:
    """Per-row causal window end ``clamp(seq_lens[b] - next_n + t + 1, 0, width)``."""
    ends: list[int] = []
    for length in seq_lens:
        for t in range(next_n):
            end = int(length) - next_n + t + 1
            if end < 0:
                end = 0
            elif end > width:
                end = width
            ends.append(end)
    return ends


def op_pad_count(end: int, partitions: int = PARTITIONS, top_k: int = TOP_K) -> int:
    """Virtual ``-FLT_MAX`` pads added by the native 10-partition split kernel."""
    block = end // partitions
    last = end - block * (partitions - 1)
    pad = (partitions - 1) * max(0, top_k - block) + max(0, top_k - last)
    return pad


@dataclass(frozen=True)
class DynamicTopkPlan:
    """Deterministic CTA assignment for one decode top-k call."""

    row_ends: tuple[int, ...]
    ctas_per_row: tuple[int, ...]
    cta_offsets: tuple[int, ...]
    long_order: tuple[int, ...]
    short_rows: tuple[int, ...]
    total_long_ctas_used: int
    total_ctas_used: int
    max_keys_per_cta: int
    mean_long_keys_per_cta: float

    def slice_bounds(self, row: int, local_rank: int) -> tuple[int, int]:
        """Contiguous 32B-aligned slice ``[lo, hi)`` owned by ``local_rank`` of ``row``."""
        end = self.row_ends[row]
        c = self.ctas_per_row[row]
        if c <= 1 or end <= 0:
            return (0, end) if local_rank == 0 else (end, end)
        span = (end + c - 1) // c
        span = ((span + SLICE_ALIGN - 1) // SLICE_ALIGN) * SLICE_ALIGN
        lo = min(local_rank * span, end)
        hi = min(lo + span, end)
        return lo, hi


def plan_dynamic_cta_allocation(
    seq_lens: Sequence[int],
    *,
    next_n: int = 4,
    width: int = 1048584,
    total_ctas: int = DEFAULT_TOTAL_CTAS,
    max_ctas_per_row: int = DEFAULT_MAX_CTAS_PER_ROW,
    top_k: int = TOP_K,
) -> DynamicTopkPlan:
    """Assign CTAs to decode rows strictly proportional to causal length ``end[r]``.

    Short/dummy rows (``end <= top_k``) run in the kernel's zero-barrier short-row prologue (1 CTA
    work item each, mapped from the highest CTA ids so CTA 0.. immediately start the longest row).
    All ``total_ctas`` CTAs then partition the long rows (``end > top_k``) longest-first using a
    2-pass exact integer proportional allocation (capped at ``max_ctas_per_row``) followed by a
    longest-first remainder sweep. This matches ``DynamicRadixTopkDecodeKernel._plan`` bit-for-bit.
    """
    if next_n not in (1, 4):
        raise ValueError(f"next_n must be 1 or 4; got {next_n}")
    num_rows = len(seq_lens) * next_n
    if num_rows > MAX_ROWS:
        raise ValueError(f"num_rows ({num_rows}) exceeds MAX_ROWS ({MAX_ROWS})")
    if total_ctas < num_rows:
        raise ValueError(f"total_ctas ({total_ctas}) must be >= total rows ({num_rows})")
    if max_ctas_per_row < 1:
        raise ValueError(f"max_ctas_per_row must be >= 1; got {max_ctas_per_row}")

    ends = row_causal_ends(seq_lens, next_n, width)
    short_rows = [r for r, e in enumerate(ends) if e <= top_k]
    long_rows = [r for r, e in enumerate(ends) if e > top_k]
    # Descending by causal end, ascending by original row index on ties.
    long_order = sorted(long_rows, key=lambda r: (-ends[r], r))

    ctas = [1] * num_rows
    n_long = len(long_order)
    long_ctas = [1] * n_long

    if n_long > 0:
        budget = total_ctas - n_long
        # Lower-bound proportional seed: since sum(max(1, ceil(e_i/T*))) <= n_long + S/T*,
        # we have 1/T* >= (total_ctas - n_long)/S, so floor(budget * e_i / S) <= c_i* for every
        # row. Seeding from this provable lower bound guarantees the subsequent water-fill reaches
        # the exact global integer minimax optimum (min max_i ceil(e_i / c_i)) in <= n_long steps.
        if budget > 0:
            budget0 = budget
            sum_long = sum(ends[r] for r in long_order)
            if sum_long > 0:
                for i, r in enumerate(long_order):
                    if budget <= 0:
                        break
                    seed = (budget0 * ends[r]) // sum_long
                    add = min(max_ctas_per_row - 1, max(0, seed - 1))
                    if add > 0:
                        if add > budget:
                            add = budget
                        long_ctas[i] += add
                        budget -= add

        # Exact integer water-fill for rounding leftovers: always give the next CTA to the
        # uncapped row with the largest current keys/CTA (end_i / c_i), tie-breaking by longer end
        # then earlier order index. Uses exact 64-bit cross-multiplication to match GPU bit-exact.
        while budget > 0:
            best_i = -1
            best_end = 0
            best_c = 1
            for i in range(n_long):
                ci = long_ctas[i]
                if ci >= max_ctas_per_row:
                    continue
                ei = ends[long_order[i]]
                if best_i < 0:
                    best_i = i
                    best_end = ei
                    best_c = ci
                else:
                    lhs = ei * best_c
                    rhs = best_end * ci
                    if lhs > rhs or (lhs == rhs and ei > best_end):
                        best_i = i
                        best_end = ei
                        best_c = ci
            if best_i < 0:
                break
            long_ctas[best_i] += 1
            budget -= 1

        for i, r in enumerate(long_order):
            ctas[r] = long_ctas[i]

    offsets = [-1] * num_rows
    acc = 0
    for i, r in enumerate(long_order):
        offsets[r] = acc
        acc += long_ctas[i]
    total_long_used = acc

    max_keys = 0
    long_loads: list[float] = []
    for r, e in enumerate(ends):
        if e > 0:
            per = (e + ctas[r] - 1) // ctas[r]
            if per > max_keys:
                max_keys = per
            if e > top_k:
                long_loads.append(e / ctas[r])
    mean_long = sum(long_loads) / len(long_loads) if long_loads else 0.0

    return DynamicTopkPlan(
        row_ends=tuple(ends),
        ctas_per_row=tuple(ctas),
        cta_offsets=tuple(offsets),
        long_order=tuple(long_order),
        short_rows=tuple(short_rows),
        total_long_ctas_used=total_long_used,
        total_ctas_used=max(total_long_used, len(short_rows)),
        max_keys_per_cta=max_keys,
        mean_long_keys_per_cta=mean_long,
    )


def _emulate_op_stage(
    row_bits: Sequence[int], cols: Sequence[int], pad: int, top_k: int = TOP_K
) -> tuple[list[int], int]:
    """Exact emulation of one native split/merge stage for the -NaN fallback path."""
    bins = [_fp16_bin_from_bits(row_bits[c]) for c in cols]
    pad_fp16_bin = 0x7E0  # fp16(-FLT_MAX) = -inf -> 0xFC00 -> bin 0x7E0
    counts = [0] * 2048
    for b in bins:
        counts[b] += 1
    counts[pad_fp16_bin] += pad
    acc = 0
    t0 = 0
    for b in range(2048):
        if acc + counts[b] >= top_k:
            t0 = b
            break
        acc += counts[b]
    before = acc
    count = counts[t0]
    if count > top_k:
        return _select_by_key(row_bits, cols, pad, top_k)
    above = [c for c, b in zip(cols, bins) if b < t0]
    inbin = [c for c, b in zip(cols, bins) if b == t0]
    inbin_pad = pad if pad_fp16_bin == t0 else 0
    pads_above = pad if pad_fp16_bin < t0 else 0
    picked, picked_pads = _select_by_key(row_bits, inbin, inbin_pad, top_k - before)
    return above + picked, pads_above + picked_pads


def _select_by_key(
    row_bits: Sequence[int], cols: Sequence[int], pad: int, need: int
) -> tuple[list[int], int]:
    """Select ``need`` best entries from ``cols`` + ``pad`` virtual pads (columns win ties)."""
    ranked = sorted((ordered_key_from_bits(row_bits[c]), 0, c) for c in cols)
    pad_entries = [(PAD_KEY, 1, -1)] * pad
    merged = sorted(ranked + pad_entries)[:need]
    selected = [c for _, is_pad, c in merged if not is_pad]
    return selected, need - len(selected)


@dataclass(frozen=True)
class EmulationStats:
    """Diagnostic counters from CPU emulation of ``DynamicRadixTopkDecodeKernel``."""

    total_long_ctas: int
    pass2_skipped_ctas: int
    narrowing_rounds_per_row: tuple[int, ...]
    fallback_rows: int


def emulate_dynamic_radix_topk_decode(
    logits_bits: Sequence[Sequence[int]],
    seq_lens: Sequence[int],
    *,
    next_n: int = 4,
    total_ctas: int = DEFAULT_TOTAL_CTAS,
    max_ctas_per_row: int = DEFAULT_MAX_CTAS_PER_ROW,
    top_k: int = TOP_K,
) -> tuple[list[list[int]], DynamicTopkPlan, EmulationStats]:
    """Bit-exact CPU emulator of ``DynamicRadixTopkDecodeKernel`` (including Pass-2 empty-slice skip).

    Accepts raw IEEE-754 uint32 bit patterns ``logits_bits[row][col]`` so special values
    (``-0.0``, ``-FLT_MAX``, ``+-inf``, ``+NaN``, ``-NaN``) are preserved without Python float
    canonicalization. Returns ``(output_indices, plan, stats)`` where every row holds valid
    selected columns first and ``-1`` after.
    """
    rows = len(logits_bits)
    width = len(logits_bits[0]) if rows else 0
    plan = plan_dynamic_cta_allocation(
        seq_lens,
        next_n=next_n,
        width=width,
        total_ctas=total_ctas,
        max_ctas_per_row=max_ctas_per_row,
        top_k=top_k,
    )
    assert len(plan.row_ends) == rows

    outputs: list[list[int]] = []
    total_long_ctas = 0
    pass2_skipped_ctas = 0
    rounds_per_row: list[int] = []
    fallback_rows = 0

    for row in range(rows):
        end = plan.row_ends[row]
        rb = logits_bits[row]
        if end <= top_k:
            has_negnan = any(ordered_key_from_bits(rb[c]) > NEG_INF_KEY for c in range(end))
            if not has_negnan:
                valid = [c for c in range(end) if ordered_key_from_bits(rb[c]) <= PAD_KEY]
                outputs.append(valid + [-1] * (top_k - len(valid)))
            else:
                fallback_rows += 1
                outputs.append(_emulate_negnan_row(rb, end, top_k))
            rounds_per_row.append(0)
            continue

        c = plan.ctas_per_row[row]
        total_long_ctas += c
        slices = [plan.slice_bounds(row, rk) for rk in range(c)]
        pad = op_pad_count(end, PARTITIONS, top_k)

        need = top_k
        pad_in = pad
        prefix = 0
        shift = 32
        a_total = 0
        my_above = [0] * c
        my_eq = [0] * c
        count_cols = 0
        mode = "narrow"
        rounds = 0

        for level, (sh, mask, nbins) in enumerate(LEVELS):
            if mode != "narrow":
                break
            rounds += 1
            local_hists = [[0] * nbins for _ in range(c)]
            neg_total = 0
            for rk, (lo, hi) in enumerate(slices):
                lh = local_hists[rk]
                for col in range(lo, hi):
                    key = ordered_key_from_bits(rb[col])
                    if level == 0:
                        lh[(key >> sh) & mask] += 1
                        if key > NEG_INF_KEY:
                            neg_total += 1
                    else:
                        up = LEVELS[level - 1][0]
                        if (key >> up) == (prefix >> up):
                            lh[(key >> sh) & mask] += 1
            if pad_in > 0:
                pad_digit = (PAD_KEY >> sh) & mask
                local_hists[0][pad_digit] += pad_in

            if neg_total > 0:
                mode = "fallback"
                break

            # Mirror zero-initialized atomic-max reduction of (nbins - 1 - lo/cut).
            max_inv_lo = 0
            max_inv_cut = 0
            for rk in range(c):
                lh = local_hists[rk]
                first = nbins
                acc = 0
                cut = nbins - 1
                found_cut = False
                for b in range(nbins):
                    v = lh[b]
                    if v and first == nbins:
                        first = b
                    acc += v
                    if not found_cut and acc >= need:
                        cut = b
                        found_cut = True
                if first < nbins:
                    inv_lo = (nbins - 1) - first
                    if inv_lo > max_inv_lo:
                        max_inv_lo = inv_lo
                inv_cut = (nbins - 1) - cut
                if inv_cut > max_inv_cut:
                    max_inv_cut = inv_cut
            rlo = (nbins - 1) - max_inv_lo
            rcut = (nbins - 1) - max_inv_cut

            ghist = [0] * nbins
            for b in range(rlo, rcut + 1):
                ghist[b] = sum(local_hists[rk][b] for rk in range(c))

            acc = 0
            t = rcut
            for b in range(rlo, rcut + 1):
                if acc + ghist[b] >= need:
                    t = b
                    break
                acc += ghist[b]
            before = acc
            count = ghist[t]

            pad_digit = (PAD_KEY >> sh) & mask
            pads_before = pad_in if pad_digit < t else 0
            new_pad_in = pad_in if pad_digit == t else 0
            for rk in range(c):
                below = sum(local_hists[rk][:t])
                eq = local_hists[rk][t]
                if rk == 0:
                    below -= pads_before
                    eq -= new_pad_in
                my_above[rk] += below
                my_eq[rk] = eq

            a_total += before - pads_before
            need -= before
            pad_in = new_pad_in
            count_cols = count - pad_in
            prefix |= t << sh
            shift = sh
            if level == 2:
                mode = "direct"
            elif count_cols <= GATHER_CAPACITY:
                mode = "gather"

        rounds_per_row.append(rounds)
        if mode == "fallback":
            fallback_rows += 1
            outputs.append(_emulate_negnan_row(rb, end, top_k))
            continue

        gather = mode == "gather"
        take = min(need, count_cols)
        out = [-1] * top_k
        base_a = 0
        base_e = 0
        gathered: list[tuple[int, int]] = []

        for rk, (lo, hi) in enumerate(slices):
            cta_base_a = base_a
            if my_above[rk] > 0:
                base_a += my_above[rk]
            cta_base_e = base_e
            if my_eq[rk] > 0:
                base_e += my_eq[rk]

            emit_eq = (my_eq[rk] > 0) and (gather or cta_base_e < take)
            if not ((my_above[rk] > 0) or emit_eq):
                pass2_skipped_ctas += 1
                continue

            pa = cta_base_a
            pe = cta_base_e
            for col in range(lo, hi):
                key = ordered_key_from_bits(rb[col])
                top = key >> shift
                if top < (prefix >> shift):
                    out[pa] = col
                    pa += 1
                elif top == (prefix >> shift):
                    if gather:
                        gathered.append((key, col))
                    elif pe < take:
                        out[a_total + pe] = col
                    pe += 1

        if gather:
            g_cols = [col for _, col in gathered]
            picked, _ = _select_by_key(rb, g_cols, pad_in, need)
            for idx, col in enumerate(picked):
                out[a_total + idx] = col

        outputs.append(out)

    stats = EmulationStats(
        total_long_ctas=total_long_ctas,
        pass2_skipped_ctas=pass2_skipped_ctas,
        narrowing_rounds_per_row=tuple(rounds_per_row),
        fallback_rows=fallback_rows,
    )
    return outputs, plan, stats


def _emulate_negnan_row(row_bits: Sequence[int], end: int, top_k: int) -> list[int]:
    """Exact 10-partition split + merge emulation for a row holding a negative NaN."""
    block = end // PARTITIONS
    pad = op_pad_count(end, PARTITIONS, top_k)
    candidates: list[int] = []
    for p in range(PARTITIONS):
        lo = p * block
        hi = end if p == PARTITIONS - 1 else (p + 1) * block
        cols = list(range(lo, hi))
        if hi - lo <= top_k:
            candidates.extend(cols)
        else:
            picked, _ = _emulate_op_stage(row_bits, cols, 0, top_k)
            candidates.extend(picked)
    winners, _ = _emulate_op_stage(row_bits, candidates, pad, top_k)
    return winners + [-1] * (top_k - len(winners))


# ---------------------------------------------------------------------------
# SM107 / VR200 CuTe DSL GPU Kernel Implementation
# ---------------------------------------------------------------------------

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils import SmemAllocator

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_self_sampling import (
        _ld_g_nc_v4_f32,
    )

    from ._cluster_radix_topk_decode import _passes, _score_floor
    from ._radix_topk_decode import (
        _N_OUT,
        _N_TIE,
        D1_BINS,
        LIST_CAPACITY,
        SplitRadixTopkDecodeKernel,
        _digit,
        _key,
    )

    _HAS_CUTLASS_DSL = True
except ImportError:
    _HAS_CUTLASS_DSL = False


if _HAS_CUTLASS_DSL:
    # Row resolution modes.
    _NARROW = 0
    _GATHER = 1
    _DIRECT = 2
    _FALLBACK = 3

    # Local scalar slots in SMEM.
    _S_OUT = _N_OUT
    _S_TIE = _N_TIE
    _S_ABOVE = 8
    _S_EQ = 9
    _S_EQN = 10
    _S_BASE_A = 11
    _S_BASE_E = 12
    _S_IS_LAST = 13
    _S_N_SHORT = 14
    _S_N_LONG = 15
    _S_USED_LONG = 16
    _S_MY_ROW = 17
    _S_MY_RANK = 18
    _S_MY_CTAS = 19
    _S_LO = 20
    _S_CUT = 21
    _SCALARS = 24

    @dsl_user_op
    def _red_add_relaxed_gpu(addr, val, *, loc=None, ip=None):
        """red.relaxed.gpu.global.add.s32 (no read-back)."""
        llvm.inline_asm(
            None,
            [addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "red.relaxed.gpu.global.add.s32 [$0], $1;",
            "l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _red_max_relaxed_gpu(addr, val, *, loc=None, ip=None):
        """red.relaxed.gpu.global.max.s32 (no read-back)."""
        llvm.inline_asm(
            None,
            [addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "red.relaxed.gpu.global.max.s32 [$0], $1;",
            "l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _atom_add_relaxed_gpu(addr, val, *, loc=None, ip=None):
        """atom.relaxed.gpu.global.add.s32 returning prior value."""
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
                "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;",
                "=r,l,r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _atom_add_release_gpu(addr, val, *, loc=None, ip=None):
        """atom.release.gpu.global.add.s32 returning prior value."""
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
                "atom.release.gpu.global.add.s32 $0, [$1], $2;",
                "=r,l,r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _ld_acquire_gpu(addr, *, loc=None, ip=None):
        """ld.acquire.gpu.global.b32."""
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [addr.ir_value(loc=loc, ip=ip)],
                "ld.acquire.gpu.global.b32 $0, [$1];",
                "=r,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _ld_relaxed_gpu(addr, *, loc=None, ip=None):
        """ld.relaxed.gpu.global.b32 (L2-coherent read)."""
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [addr.ir_value(loc=loc, ip=ip)],
                "ld.relaxed.gpu.global.b32 $0, [$1];",
                "=r,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _st_relaxed_gpu(addr, val, *, loc=None, ip=None):
        """st.relaxed.gpu.global.b32 (L2-coherent write)."""
        llvm.inline_asm(
            None,
            [addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "st.relaxed.gpu.global.b32 [$0], $1;",
            "l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _fence_acquire_gpu(*, loc=None, ip=None):
        """fence.acquire.gpu ensuring all prior released writes are visible."""
        llvm.inline_asm(
            None,
            [],
            "fence.acquire.gpu;",
            "",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    class DynamicRadixTopkDecodeKernel(SplitRadixTopkDecodeKernel):
        """Single-launch ``cluster=(1, 1, 1)`` dynamic length-proportional decode radix top-k.

        Launches ``total_ctas`` co-resident CTAs (``min_blocks`` CTAs/SM, default 1/SM = 212 CTAs
        on VR200 for unconditional deadlock freedom, or 2/SM = 424 CTAs when 2-CTA occupancy is
        enforced). Short/dummy rows execute in a zero-barrier prologue from the tail CTA ids, while
        long rows split all ``total_ctas`` proportionally to their causal window lengths up to
        ``max_ctas_per_row``.
        """

        def __init__(
            self,
            next_n: int,
            total_ctas: int = DEFAULT_TOTAL_CTAS,
            max_ctas_per_row: int = DEFAULT_MAX_CTAS_PER_ROW,
            threads: int = 512,
            unroll: int = 2,
            use_pdl: bool = True,
            pipe: bool = True,
            min_blocks: int = 1,
        ):
            super().__init__(next_n, PARTITIONS, use_pdl)
            assert threads % 32 == 0 and D1_BINS % threads == 0 and LEVELS[1][2] % threads == 0
            assert total_ctas >= MAX_ROWS and max_ctas_per_row >= 1
            self.total_ctas = total_ctas
            self.max_ctas_per_row = max_ctas_per_row
            self.T = threads
            self.W = threads // 32
            self.U = unroll
            self.pipe = pipe
            self.min_blocks = min_blocks

        # ---------------------------------------------------------- vectorized slice passes
        @cute.jit
        def _load(self, row_addr, lo, i0, n4, vals, tid):
            last = n4 - cutlass.Int32(1)
            for u in cutlass.range_constexpr(self.U):
                idx = i0 + cutlass.Int32(u * self.T) + tid
                if idx > last:
                    idx = last
                x = _ld_g_nc_v4_f32(row_addr + cutlass.Int64(lo + idx * cutlass.Int32(4)) * 4)
                for j in cutlass.range_constexpr(4):
                    vals[u * 4 + j] = x[j]

        @cute.jit
        def _count_batch(
            self, vals, i0, n4, lo, hi, s_hist, level: cutlass.Constexpr[int], prefix, neg, tid
        ):
            shift, mask, _ = LEVELS[level]
            for u in cutlass.range_constexpr(self.U):
                idx = i0 + cutlass.Int32(u * self.T) + tid
                for j in cutlass.range_constexpr(4):
                    c = lo + idx * cutlass.Int32(4) + cutlass.Int32(j)
                    if (idx < n4) & (c < hi):
                        key = _key(vals[u * 4 + j])
                        if cutlass.const_expr(level == 0):
                            self._count(s_hist, _digit(key, shift, mask), cutlass.Int32(1))
                            if key > cutlass.Uint32(NEG_INF_KEY):
                                neg = neg + cutlass.Int32(1)
                        else:
                            up = cutlass.const_expr(LEVELS[level - 1][0])
                            if (key >> up) == (prefix >> up):
                                self._count(s_hist, _digit(key, shift, mask), cutlass.Int32(1))
            return neg

        @cute.jit
        def _count_slice(
            self, row_addr, lo, hi, s_hist, level: cutlass.Constexpr[int], prefix, tid
        ):
            neg = cutlass.Int32(0)
            n4 = (hi - lo + cutlass.Int32(3)) // cutlass.Int32(4)
            step = cutlass.Int32(self.U * self.T)
            va = cute.make_rmem_tensor(cute.make_layout((self.U * 4,)), cutlass.Float32)
            vb = cute.make_rmem_tensor(cute.make_layout((self.U * 4,)), cutlass.Float32)
            i0 = cutlass.Int32(0)
            if cutlass.const_expr(self.pipe):
                if n4 > cutlass.Int32(0):
                    self._load(row_addr, lo, i0, n4, va, tid)
                while i0 < n4:
                    self._load(row_addr, lo, i0 + step, n4, vb, tid)
                    neg = self._count_batch(va, i0, n4, lo, hi, s_hist, level, prefix, neg, tid)
                    self._load(row_addr, lo, i0 + step + step, n4, va, tid)
                    neg = self._count_batch(
                        vb, i0 + step, n4, lo, hi, s_hist, level, prefix, neg, tid
                    )
                    i0 = i0 + step + step
            else:
                while i0 < n4:
                    self._load(row_addr, lo, i0, n4, va, tid)
                    neg = self._count_batch(va, i0, n4, lo, hi, s_hist, level, prefix, neg, tid)
                    i0 = i0 + step
            return neg

        @cute.jit
        def _claim_n(self, s_scal, slot: cutlass.Constexpr[int], n):
            lane = cute.arch.lane_idx()
            inclusive = self._warp_scan(n)
            total = cute.arch.shuffle_sync(inclusive, 31)
            base = cutlass.Int32(0)
            if (lane == cutlass.Int32(31)) & (total > cutlass.Int32(0)):
                base = cute.arch.atomic_add(
                    ptr=(s_scal.iterator + slot).llvm_ptr, val=total, sem="relaxed", scope="cta"
                )
            return cute.arch.shuffle_sync(base, 31) + inclusive - n

        @cute.jit
        def _emit_batch(
            self,
            vals,
            i0,
            n4,
            lo,
            hi,
            prefix,
            shift,
            floor,
            gather,
            take,
            a_total,
            dst,
            aux_indices,
            aux_base,
            s_scal,
            tid,
        ):
            zero = cutlass.Int32(0)
            n_above = zero
            n_eq = zero
            has = cutlass.Boolean(False)
            for v in cutlass.range_constexpr(self.U * 4):
                has = has | _passes(vals[v], floor)
            if has:
                for u in cutlass.range_constexpr(self.U):
                    idx = i0 + cutlass.Int32(u * self.T) + tid
                    for j in cutlass.range_constexpr(4):
                        c = lo + idx * cutlass.Int32(4) + cutlass.Int32(j)
                        if (idx < n4) & (c < hi):
                            top = _key(vals[u * 4 + j]) >> shift
                            if top < (prefix >> shift):
                                n_above = n_above + cutlass.Int32(1)
                            elif top == (prefix >> shift):
                                n_eq = n_eq + cutlass.Int32(1)
            if cute.arch.vote_ballot_sync((n_above + n_eq) > zero) != 0:
                pa = s_scal[_S_BASE_A] + self._claim_n(s_scal, _S_ABOVE, n_above)
                pe = s_scal[_S_BASE_E] + self._claim_n(s_scal, _S_EQ, n_eq)
                for u in cutlass.range_constexpr(self.U):
                    idx = i0 + cutlass.Int32(u * self.T) + tid
                    for j in cutlass.range_constexpr(4):
                        c = lo + idx * cutlass.Int32(4) + cutlass.Int32(j)
                        if (idx < n4) & (c < hi):
                            key = _key(vals[u * 4 + j])
                            top = key >> shift
                            if top < (prefix >> shift):
                                dst[pa] = c
                                pa = pa + cutlass.Int32(1)
                            elif top == (prefix >> shift):
                                if gather:
                                    k_i32 = cutlass.Int32(
                                        llvm.bitcast(cutlass.Int32.mlir_type, key.ir_value())
                                    )
                                    aux_indices[aux_base + cutlass.Int32(_AUX_LKEY) + pe] = k_i32
                                    aux_indices[aux_base + cutlass.Int32(_AUX_LCOL) + pe] = c
                                elif pe < take:
                                    dst[a_total + pe] = c
                                pe = pe + cutlass.Int32(1)

        @cute.jit
        def _emit_slice(
            self,
            row_addr,
            lo,
            hi,
            prefix,
            shift,
            gather,
            take,
            a_total,
            dst,
            aux_indices,
            aux_base,
            s_scal,
            tid,
        ):
            n4 = (hi - lo + cutlass.Int32(3)) // cutlass.Int32(4)
            step = cutlass.Int32(self.U * self.T)
            floor = _score_floor(
                (((prefix >> shift) + cutlass.Uint32(1)) << shift) - cutlass.Uint32(1)
            )
            va = cute.make_rmem_tensor(cute.make_layout((self.U * 4,)), cutlass.Float32)
            vb = cute.make_rmem_tensor(cute.make_layout((self.U * 4,)), cutlass.Float32)
            i0 = cutlass.Int32(0)
            if cutlass.const_expr(self.pipe):
                if n4 > cutlass.Int32(0):
                    self._load(row_addr, lo, i0, n4, va, tid)
                while i0 < n4:
                    self._load(row_addr, lo, i0 + step, n4, vb, tid)
                    self._emit_batch(
                        va, i0, n4, lo, hi, prefix, shift, floor, gather, take, a_total,
                        dst, aux_indices, aux_base, s_scal, tid,
                    )  # fmt: skip
                    self._load(row_addr, lo, i0 + step + step, n4, va, tid)
                    self._emit_batch(
                        vb, i0 + step, n4, lo, hi, prefix, shift, floor, gather, take, a_total,
                        dst, aux_indices, aux_base, s_scal, tid,
                    )  # fmt: skip
                    i0 = i0 + step + step
            else:
                while i0 < n4:
                    self._load(row_addr, lo, i0, n4, va, tid)
                    self._emit_batch(
                        va, i0, n4, lo, hi, prefix, shift, floor, gather, take, a_total,
                        dst, aux_indices, aux_base, s_scal, tid,
                    )  # fmt: skip
                    i0 = i0 + step

        # ---------------------------------------------------------- L2 sync & range helpers
        @cute.jit
        def _local_range(self, s_hist, nbins: cutlass.Constexpr[int], need, s_scal, s_warp, tid):
            """Compute this CTA's first nonzero bin (_S_LO) and crossing bin (_S_CUT) in SMEM."""
            per = cutlass.const_expr(nbins // self.T)
            lane = cute.arch.lane_idx()
            warp = tid // cutlass.Int32(32)
            base = tid * cutlass.Int32(per)
            if tid == cutlass.Int32(0):
                s_scal[_S_LO] = cutlass.Int32(nbins)
                s_scal[_S_CUT] = cutlass.Int32(nbins - 1)
            cute.arch.barrier()
            local = cutlass.Int32(0)
            first = cutlass.Int32(nbins)
            for j in cutlass.range_constexpr(per):
                v = s_hist[base + cutlass.Int32(j)]
                local = local + v
                if (v != cutlass.Int32(0)) & (first == cutlass.Int32(nbins)):
                    first = base + cutlass.Int32(j)
            if first < cutlass.Int32(nbins):
                cute.arch.atomic_min(
                    ptr=(s_scal.iterator + _S_LO).llvm_ptr, val=first, sem="relaxed", scope="cta"
                )
            inclusive = self._warp_scan(local)
            if lane == cutlass.Int32(31):
                s_warp[warp] = inclusive
            cute.arch.barrier()
            prefix, total = self._warp_offsets(s_warp, warp)
            prefix = prefix + inclusive - local
            for j in cutlass.range_constexpr(per):
                count = s_hist[base + cutlass.Int32(j)]
                if (prefix < need) & (prefix + count >= need):
                    s_scal[_S_CUT] = base + cutlass.Int32(j)
                prefix = prefix + count
            cute.arch.barrier()

        @cute.jit
        def _global_barrier(self, ctrl_addr, slot: cutlass.Constexpr[int], c, tid):
            """Release prior global writes of this CTA and wait until all `c` CTAs arrive."""
            cute.arch.barrier()
            if tid == cutlass.Int32(0):
                addr = ctrl_addr + cutlass.Int64(slot * 4)
                _atom_add_release_gpu(addr, cutlass.Int32(1))
                while _ld_acquire_gpu(addr) < c:
                    pass
            cute.arch.barrier()

        @cute.jit
        def _split_at(self, s_hist, nbins: cutlass.Constexpr[int], t, s_red, s_scal, tid):
            per = cutlass.const_expr(nbins // self.T)
            lane = cute.arch.lane_idx()
            warp = tid // cutlass.Int32(32)
            base = tid * cutlass.Int32(per)
            below = cutlass.Int32(0)
            for j in cutlass.range_constexpr(per):
                b = base + cutlass.Int32(j)
                v = s_hist[b]
                if b < t:
                    below = below + v
                if b == t:
                    s_scal[_S_EQN] = v
            below = cute.arch.warp_reduction(below, lambda x, y: x + y)
            if lane == cutlass.Int32(0):
                s_red[warp] = below
            cute.arch.barrier()
            total = cutlass.Int32(0)
            if lane < cutlass.Int32(self.W):
                total = s_red[lane]
            return cute.arch.warp_reduction(total, lambda x, y: x + y)

        @cute.jit
        def _short_row(
            self, logits, seq_lens, aux_indices, aux_logits, out, row, end, dst, s_hist, s_out,
            s_lkey, s_lcol, s_warp, s_scal, tid,
        ):  # fmt: skip
            lane = cute.arch.lane_idx()
            warp = tid // cutlass.Int32(32)
            score = logits[row, None]
            per = cutlass.const_expr(TOP_K // self.T)
            c0 = tid * cutlass.Int32(per)
            if tid == cutlass.Int32(0):
                s_scal[0] = cutlass.Int32(0)
            cute.arch.barrier()
            keep = cute.make_rmem_tensor(cute.make_layout((per,)), cutlass.Int32)
            neg = cutlass.Int32(0)
            local = cutlass.Int32(0)
            for e in cutlass.range_constexpr(per):
                keep[e] = cutlass.Int32(0)
                if c0 + cutlass.Int32(e) < end:
                    key = _key(score[c0 + cutlass.Int32(e)])
                    if key <= cutlass.Uint32(PAD_KEY):
                        keep[e] = cutlass.Int32(1)
                    if key > cutlass.Uint32(NEG_INF_KEY):
                        neg = neg + cutlass.Int32(1)
                local = local + keep[e]
            if neg > cutlass.Int32(0):
                self._count(s_scal, cutlass.Int32(0), neg)
            inclusive = self._warp_scan(local)
            if lane == cutlass.Int32(31):
                s_warp[warp] = inclusive
            cute.arch.barrier()
            if s_scal[0] == cutlass.Int32(0):
                pos, total = self._warp_offsets(s_warp, warp)
                pos = pos + inclusive - local
                for e in cutlass.range_constexpr(per):
                    if keep[e] == cutlass.Int32(1):
                        dst[pos] = c0 + cutlass.Int32(e)
                    pos = pos + keep[e]
                k = total + tid
                while k < cutlass.Int32(TOP_K):
                    dst[k] = cutlass.Int32(-1)
                    k = k + cutlass.Int32(self.T)
            else:
                cute.arch.barrier()
                for part in cutlass.range_constexpr(PARTITIONS):
                    self._partition(
                        logits, seq_lens, aux_indices, aux_logits, row, cutlass.Int32(part),
                        s_hist, s_out, s_lkey, s_lcol, s_warp, s_scal, tid,
                    )  # fmt: skip
                    cute.arch.barrier()
                self._merge(
                    aux_indices, aux_logits, out, row, s_hist, s_out, s_lkey, s_lcol, s_warp,
                    s_scal, tid,
                )  # fmt: skip

        @cute.jit
        def _resolve_list(
            self, n_list, need, pad_in, prefix, shift, a_total, dst, s_hist, s_lkey, s_lcol,
            s_warp, s_thr, s_scal, tid,
        ):  # fmt: skip
            lane = cute.arch.lane_idx()
            warp = tid // cutlass.Int32(32)
            pad_key = cutlass.Uint32(PAD_KEY)
            if tid == cutlass.Int32(0):
                s_scal[_S_OUT] = cutlass.Int32(0)
                s_scal[_S_TIE] = cutlass.Int32(0)
            cute.arch.barrier()
            for level in cutlass.range_constexpr(1, 3):
                sh, mask, nbins = LEVELS[level]
                up = cutlass.const_expr(LEVELS[level - 1][0])
                if shift > cutlass.Int32(sh):
                    self._clear(s_hist, nbins, tid)
                    j = tid
                    while j < n_list:
                        key = s_lkey[j]
                        if (key >> up) == (prefix >> up):
                            self._count(s_hist, _digit(key, sh, mask), cutlass.Int32(1))
                        j = j + cutlass.Int32(self.T)
                    if (tid == cutlass.Int32(0)) & (pad_in > cutlass.Int32(0)):
                        self._count(s_hist, _digit(pad_key, sh, mask), pad_in)
                    cute.arch.barrier()
                    t, before, count = self._threshold(s_hist, s_warp, s_thr, nbins, need, tid)
                    need = need - before
                    prefix = prefix | (cutlass.Uint32(t) << sh)
                    if _digit(pad_key, sh, mask) != t:
                        pad_in = cutlass.Int32(0)
                    if cutlass.const_expr(level == 2):
                        take = need
                        if count - pad_in < take:
                            take = count - pad_in
                        lbase = warp * cutlass.Int32(32)
                        while lbase < n_list:
                            j = lbase + lane
                            live = j < n_list
                            key = prefix
                            col = cutlass.Int32(-1)
                            if live:
                                key = s_lkey[j]
                                col = s_lcol[j]
                            above = live & (key < prefix)
                            share = live & (key == prefix)
                            pos = self._claim(s_scal, _S_OUT, above)
                            if above:
                                dst[a_total + pos] = col
                            slot = self._claim(s_scal, _S_TIE, share)
                            win = share & (slot < take)
                            pos = self._claim(s_scal, _S_OUT, win)
                            if win:
                                dst[a_total + pos] = col
                            lbase = lbase + cutlass.Int32(self.T)
                        cute.arch.barrier()
            return s_scal[_S_OUT]

        # ---------------------------------------------------------- in-kernel CTA planner
        @cute.jit
        def _plan(
            self, logits, seq_lens, rows, grid_ctas, bid, s_ends, s_short, s_order, s_ctas,
            s_scal, tid,
        ):  # fmt: skip
            """Compute deterministic CTA plan in SMEM identical to ``plan_dynamic_cta_allocation``."""
            zero = cutlass.Int32(0)
            one = cutlass.Int32(1)
            top_k = cutlass.Int32(TOP_K)
            max_c = cutlass.Int32(self.max_ctas_per_row)

            if tid < rows:
                s_ends[tid] = self._row_end(logits, seq_lens, tid)
            cute.arch.barrier()

            # Partition into short_rows and long_rows (ranked descending by end, ascending by id).
            if tid == zero:
                n_short = zero
                n_long = zero
                r = zero
                while r < rows:
                    if s_ends[r] <= top_k:
                        s_short[n_short] = r
                        n_short = n_short + one
                    else:
                        n_long = n_long + one
                    r = r + one
                s_scal[_S_N_SHORT] = n_short
                s_scal[_S_N_LONG] = n_long
            cute.arch.barrier()

            n_long = s_scal[_S_N_LONG]
            if tid < rows:
                mine = s_ends[tid]
                if mine > top_k:
                    place = zero
                    b = zero
                    while b < rows:
                        other = s_ends[b]
                        if (other > top_k) & ((other > mine) | ((other == mine) & (b < tid))):
                            place = place + one
                        b = b + one
                    s_order[place] = tid
                    s_ctas[place] = one
            cute.arch.barrier()

            if tid == zero:
                s_scal[_S_MY_ROW] = cutlass.Int32(-1)
                s_scal[_S_MY_RANK] = zero
                s_scal[_S_MY_CTAS] = zero
                s_scal[_S_USED_LONG] = zero
                if n_long > zero:
                    budget = grid_ctas - n_long
                    # Provable lower-bound proportional seed: c_i = max(1, floor(budget0 * e_i / S))
                    # never exceeds the exact integer minimax optimum c_i*.
                    if budget > zero:
                        budget0 = cutlass.Int64(budget)
                        sum_long = cutlass.Int64(0)
                        i = zero
                        while i < n_long:
                            sum_long = sum_long + cutlass.Int64(s_ends[s_order[i]])
                            i = i + one
                        if sum_long > cutlass.Int64(0):
                            i = zero
                            while (i < n_long) & (budget > zero):
                                seed = cutlass.Int32(
                                    (budget0 * cutlass.Int64(s_ends[s_order[i]])) // sum_long
                                )
                                add = seed - one
                                if add > zero:
                                    room = max_c - one
                                    if add > room:
                                        add = room
                                    if add > budget:
                                        add = budget
                                    s_ctas[i] = one + add
                                    budget = budget - add
                                i = i + one
                    # Exact 64-bit integer water-fill for rounding leftovers: always give the next
                    # CTA to the uncapped row with the largest current keys/CTA (end_i / c_i).
                    done_wf = zero
                    while (budget > zero) & (done_wf == zero):
                        best_i = cutlass.Int32(-1)
                        best_end = cutlass.Int64(0)
                        best_c = cutlass.Int64(1)
                        i = zero
                        while i < n_long:
                            ci = s_ctas[i]
                            if ci < max_c:
                                ei = cutlass.Int64(s_ends[s_order[i]])
                                if best_i < zero:
                                    best_i = i
                                    best_end = ei
                                    best_c = cutlass.Int64(ci)
                                else:
                                    lhs = ei * best_c
                                    rhs = best_end * cutlass.Int64(ci)
                                    if (lhs > rhs) | ((lhs == rhs) & (ei > best_end)):
                                        best_i = i
                                        best_end = ei
                                        best_c = cutlass.Int64(ci)
                            i = i + one
                        if best_i < zero:
                            done_wf = one
                        else:
                            s_ctas[best_i] = s_ctas[best_i] + one
                            budget = budget - one
                    # Resolve this CTA's assigned (row, local_rank, ctas).
                    acc = zero
                    i = zero
                    while i < n_long:
                        ci = s_ctas[i]
                        if (bid >= acc) & (bid < acc + ci):
                            s_scal[_S_MY_ROW] = s_order[i]
                            s_scal[_S_MY_RANK] = bid - acc
                            s_scal[_S_MY_CTAS] = ci
                        acc = acc + ci
                        i = i + one
                    s_scal[_S_USED_LONG] = acc
            cute.arch.barrier()

        # ---------------------------------------------------------- main kernel
        @cute.kernel
        def dynamic_kernel(
            self,
            logits: cute.Tensor,
            seq_lens: cute.Tensor,
            aux_indices: cute.Tensor,
            aux_logits: cute.Tensor,
            state: cute.Tensor,
            out: cute.Tensor,
        ):
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            bid, _, _ = cute.arch.block_idx()
            grid_ctas, _, _ = cute.arch.grid_dim()
            tid, _, _ = cute.arch.thread_idx()
            rows = cute.size(logits, mode=[0])
            zero = cutlass.Int32(0)
            one = cutlass.Int32(1)
            top_k = cutlass.Int32(TOP_K)

            smem = SmemAllocator()
            s_hist = smem.allocate_tensor(cutlass.Int32, cute.make_layout((D1_BINS,)), 128)
            s_ghist = smem.allocate_tensor(cutlass.Int32, cute.make_layout((D1_BINS,)), 128)
            s_out = cute.make_tensor(s_ghist.iterator, cute.make_layout((TOP_K,)))
            s_lkey = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((LIST_CAPACITY,)), 128)
            s_lcol = smem.allocate_tensor(cutlass.Int32, cute.make_layout((LIST_CAPACITY,)), 128)
            s_warp = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.W,)), 128)
            s_red = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.W,)), 128)
            s_thr = smem.allocate_tensor(cutlass.Int32, cute.make_layout((8,)), 16)
            s_scal = smem.allocate_tensor(cutlass.Int32, cute.make_layout((_SCALARS,)), 16)
            s_fb = smem.allocate_tensor(cutlass.Int32, cute.make_layout((8,)), 16)
            s_ends = smem.allocate_tensor(cutlass.Int32, cute.make_layout((MAX_ROWS,)), 16)
            s_short = smem.allocate_tensor(cutlass.Int32, cute.make_layout((MAX_ROWS,)), 16)
            s_order = smem.allocate_tensor(cutlass.Int32, cute.make_layout((MAX_ROWS,)), 16)
            s_ctas = smem.allocate_tensor(cutlass.Int32, cute.make_layout((MAX_ROWS,)), 16)

            self._plan(
                logits, seq_lens, rows, grid_ctas, bid, s_ends, s_short, s_order, s_ctas,
                s_scal, tid,
            )  # fmt: skip

            # Zero-barrier short-row prologue mapped from the tail CTA ids so CTA 0.. immediately
            # start the longest straggler row.
            n_short = s_scal[_S_N_SHORT]
            rev_bid = grid_ctas - one - bid
            k_short = rev_bid
            while k_short < n_short:
                srow = s_short[k_short]
                self._short_row(
                    logits, seq_lens, aux_indices, aux_logits, out, srow, s_ends[srow],
                    out[srow, None], s_hist, s_out, s_lkey, s_lcol, s_warp, s_fb, tid,
                )  # fmt: skip
                cute.arch.barrier()
                k_short = k_short + grid_ctas

            row = s_scal[_S_MY_ROW]
            if row >= zero:
                rank = s_scal[_S_MY_RANK]
                c = s_scal[_S_MY_CTAS]
                end = s_ends[row]
                span = (end + c - one) // c
                span = (
                    (span + cutlass.Int32(SLICE_ALIGN - 1)) // cutlass.Int32(SLICE_ALIGN)
                ) * cutlass.Int32(SLICE_ALIGN)
                lo = rank * span
                if lo > end:
                    lo = end
                hi = lo + span
                if hi > end:
                    hi = end

                block = end // cutlass.Int32(PARTITIONS)
                last = end - block * cutlass.Int32(PARTITIONS - 1)
                pad = zero
                if block < top_k:
                    pad = cutlass.Int32(PARTITIONS - 1) * (top_k - block)
                if last < top_k:
                    pad = pad + top_k - last

                row_addr = logits[row, None].iterator.toint()
                dst = out[row, None]
                aux_base = row * cutlass.Int32(PARTITIONS * TOP_K)
                row_state = state[row, None]
                state_addr = row_state.iterator.toint()
                ctrl_addr = state_addr + cutlass.Int64(_G_CTRL * 4)

                if tid < cutlass.Int32(13):
                    s_scal[tid] = zero
                cute.arch.barrier()

                mode = cutlass.Int32(_NARROW)
                need = top_k
                pad_in = pad
                prefix = cutlass.Uint32(0)
                shift = cutlass.Int32(32)
                a_total = zero
                my_above = zero
                my_eq = zero
                count_cols = zero

                # Track exact sparse [rlo, rcut] per level for minimal cleanup by last CTA.
                u0_used = zero
                u0_lo = zero
                u0_cut = zero
                u1_used = zero
                u1_lo = zero
                u1_cut = zero
                u2_used = zero
                u2_lo = zero
                u2_cut = zero

                g_hist_bases = (_G_HIST0, _G_HIST1, _G_HIST2)
                c_lo_slots = (_C_LO0, _C_LO1, _C_LO2)
                c_cut_slots = (_C_CUT0, _C_CUT1, _C_CUT2)
                c_arr_r_slots = (_C_ARR_R0, _C_ARR_R1, _C_ARR_R2)
                c_arr_h_slots = (_C_ARR_H0, _C_ARR_H1, _C_ARR_H2)

                for level in cutlass.range_constexpr(3):
                    sh, mask, nbins = LEVELS[level]
                    gh_base = g_hist_bases[level]
                    lo_slot = c_lo_slots[level]
                    cut_slot = c_cut_slots[level]
                    arr_r = c_arr_r_slots[level]
                    arr_h = c_arr_h_slots[level]
                    pad_digit = cutlass.const_expr((PAD_KEY >> sh) & mask)

                    if mode == cutlass.Int32(_NARROW):
                        self._clear(s_hist, nbins, tid)
                        neg = self._count_slice(row_addr, lo, hi, s_hist, level, prefix, tid)
                        if cutlass.const_expr(level == 0):
                            if neg > zero:
                                _red_add_relaxed_gpu(ctrl_addr + cutlass.Int64(_C_NEG * 4), neg)
                        if (rank == zero) & (tid == zero) & (pad_in > zero):
                            self._count(s_hist, cutlass.Int32(pad_digit), pad_in)
                        cute.arch.barrier()

                        self._local_range(s_hist, nbins, need, s_scal, s_warp, tid)
                        if tid == zero:
                            my_lo = s_scal[_S_LO]
                            my_cut = s_scal[_S_CUT]
                            if my_lo < cutlass.Int32(nbins):
                                _red_max_relaxed_gpu(
                                    ctrl_addr + cutlass.Int64(lo_slot * 4),
                                    cutlass.Int32(nbins - 1) - my_lo,
                                )
                            inv_cut = cutlass.Int32(nbins - 1) - my_cut
                            if inv_cut > zero:
                                _red_max_relaxed_gpu(
                                    ctrl_addr + cutlass.Int64(cut_slot * 4), inv_cut
                                )
                        self._global_barrier(ctrl_addr, arr_r, c, tid)

                        if cutlass.const_expr(level == 0):
                            rneg = _ld_relaxed_gpu(ctrl_addr + cutlass.Int64(_C_NEG * 4))
                            if rneg > zero:
                                mode = cutlass.Int32(_FALLBACK)
                        if mode == cutlass.Int32(_NARROW):
                            rlo = cutlass.Int32(nbins - 1) - _ld_relaxed_gpu(
                                ctrl_addr + cutlass.Int64(lo_slot * 4)
                            )
                            rcut = cutlass.Int32(nbins - 1) - _ld_relaxed_gpu(
                                ctrl_addr + cutlass.Int64(cut_slot * 4)
                            )
                            if cutlass.const_expr(level == 0):
                                u0_used = one
                                u0_lo = rlo
                                u0_cut = rcut
                            elif cutlass.const_expr(level == 1):
                                u1_used = one
                                u1_lo = rlo
                                u1_cut = rcut
                            else:
                                u2_used = one
                                u2_lo = rlo
                                u2_cut = rcut

                            # Sparse reduction over [rlo, rcut] only into L2 global histogram.
                            b = rlo + tid
                            while b <= rcut:
                                v = s_hist[b]
                                if v != zero:
                                    _red_add_relaxed_gpu(
                                        state_addr
                                        + cutlass.Int64(cutlass.Int32(gh_base) + b)
                                        * cutlass.Int64(4),
                                        v,
                                    )
                                b = b + cutlass.Int32(self.T)
                            self._global_barrier(ctrl_addr, arr_h, c, tid)

                            # Copy [rlo, rcut] into SMEM s_ghist and zero outside.
                            per = cutlass.const_expr(nbins // self.T)
                            base = tid * cutlass.Int32(per)
                            for j in cutlass.range_constexpr(per):
                                idx_b = base + cutlass.Int32(j)
                                if (idx_b < rlo) | (idx_b > rcut):
                                    s_ghist[idx_b] = zero
                                else:
                                    s_ghist[idx_b] = _ld_relaxed_gpu(
                                        state_addr
                                        + cutlass.Int64(cutlass.Int32(gh_base) + idx_b)
                                        * cutlass.Int64(4)
                                    )
                            cute.arch.barrier()

                            t, before, count = self._threshold(
                                s_ghist, s_warp, s_thr, nbins, need, tid
                            )
                            below = self._split_at(s_hist, nbins, t, s_red, s_scal, tid)
                            eq = s_scal[_S_EQN]
                            pads_before = zero
                            if cutlass.Int32(pad_digit) < t:
                                pads_before = pad_in
                            if cutlass.Int32(pad_digit) != t:
                                pad_in = zero
                            if rank == zero:
                                below = below - pads_before
                                eq = eq - pad_in
                            a_total = a_total + before - pads_before
                            need = need - before
                            my_above = my_above + below
                            my_eq = eq
                            count_cols = count - pad_in
                            prefix = prefix | (cutlass.Uint32(t) << sh)
                            shift = cutlass.Int32(sh)
                            if cutlass.const_expr(level == 2):
                                mode = cutlass.Int32(_DIRECT)
                            else:
                                if count_cols <= cutlass.Int32(GATHER_CAPACITY):
                                    mode = cutlass.Int32(_GATHER)

                if mode == cutlass.Int32(_FALLBACK):
                    part = rank
                    while part < cutlass.Int32(PARTITIONS):
                        self._partition(
                            logits, seq_lens, aux_indices, aux_logits, row, part,
                            s_hist, s_out, s_lkey, s_lcol, s_warp, s_fb, tid,
                        )  # fmt: skip
                        cute.arch.barrier()
                        part = part + c
                else:
                    gather = mode == cutlass.Int32(_GATHER)
                    if tid == zero:
                        if my_above > zero:
                            s_scal[_S_BASE_A] = _atom_add_relaxed_gpu(
                                ctrl_addr + cutlass.Int64(_C_ABOVE * 4), my_above
                            )
                        if my_eq > zero:
                            cslot = cutlass.Int32(_C_TIE)
                            if gather:
                                cslot = cutlass.Int32(_C_LIST)
                            s_scal[_S_BASE_E] = _atom_add_relaxed_gpu(
                                ctrl_addr + cutlass.Int64(cslot) * cutlass.Int64(4), my_eq
                            )
                    cute.arch.barrier()
                    take = need
                    if count_cols < take:
                        take = count_cols
                    emit_eq = (my_eq > zero) & (gather | (s_scal[_S_BASE_E] < take))
                    if (my_above > zero) | emit_eq:
                        self._emit_slice(
                            row_addr, lo, hi, prefix, shift, gather, take, a_total, dst,
                            aux_indices, aux_base, s_scal, tid,
                        )  # fmt: skip

                # Final row arrival: first c - 1 CTAs exit immediately; last CTA finalizes & cleans.
                cute.arch.barrier()
                if tid == zero:
                    old_done = _atom_add_release_gpu(
                        ctrl_addr + cutlass.Int64(_C_ARR_DONE * 4), one
                    )
                    is_last = zero
                    if old_done == c - one:
                        _fence_acquire_gpu()
                        is_last = one
                    s_scal[_S_IS_LAST] = is_last
                cute.arch.barrier()

                if s_scal[_S_IS_LAST] == one:
                    if mode == cutlass.Int32(_FALLBACK):
                        self._merge(
                            aux_indices, aux_logits, out, row, s_hist, s_out, s_lkey, s_lcol,
                            s_warp, s_fb, tid,
                        )  # fmt: skip
                    else:
                        gather = mode == cutlass.Int32(_GATHER)
                        take = need
                        if count_cols < take:
                            take = count_cols
                        written = take
                        if gather:
                            n_list = _ld_relaxed_gpu(ctrl_addr + cutlass.Int64(_C_LIST * 4))
                            aux_addr = aux_indices.iterator.toint()
                            j = tid
                            while j < n_list:
                                k_i32 = _ld_relaxed_gpu(
                                    aux_addr
                                    + cutlass.Int64(aux_base + cutlass.Int32(_AUX_LKEY) + j)
                                    * cutlass.Int64(4)
                                )
                                s_lkey[j] = cutlass.Uint32(
                                    llvm.bitcast(cutlass.Uint32.mlir_type, k_i32.ir_value())
                                )
                                s_lcol[j] = _ld_relaxed_gpu(
                                    aux_addr
                                    + cutlass.Int64(aux_base + cutlass.Int32(_AUX_LCOL) + j)
                                    * cutlass.Int64(4)
                                )
                                j = j + cutlass.Int32(self.T)
                            cute.arch.barrier()
                            written = self._resolve_list(
                                n_list, need, pad_in, prefix, shift, a_total, dst, s_hist,
                                s_lkey, s_lcol, s_warp, s_thr, s_scal, tid,
                            )  # fmt: skip
                        k = a_total + written + tid
                        while k < top_k:
                            dst[k] = cutlass.Int32(-1)
                            k = k + cutlass.Int32(self.T)

                    # Self-clean sparse histogram ranges and control words back to all-zeros.
                    if u0_used == one:
                        b = u0_lo + tid
                        while b <= u0_cut:
                            _st_relaxed_gpu(
                                state_addr
                                + cutlass.Int64(cutlass.Int32(_G_HIST0) + b) * cutlass.Int64(4),
                                zero,
                            )
                            b = b + cutlass.Int32(self.T)
                    if u1_used == one:
                        b = u1_lo + tid
                        while b <= u1_cut:
                            _st_relaxed_gpu(
                                state_addr
                                + cutlass.Int64(cutlass.Int32(_G_HIST1) + b) * cutlass.Int64(4),
                                zero,
                            )
                            b = b + cutlass.Int32(self.T)
                    if u2_used == one:
                        b = u2_lo + tid
                        while b <= u2_cut:
                            _st_relaxed_gpu(
                                state_addr
                                + cutlass.Int64(cutlass.Int32(_G_HIST2) + b) * cutlass.Int64(4),
                                zero,
                            )
                            b = b + cutlass.Int32(self.T)
                    if tid < cutlass.Int32(_C_USED_WORDS):
                        _st_relaxed_gpu(ctrl_addr + cutlass.Int64(tid) * cutlass.Int64(4), zero)

        @cute.jit
        def __call__(
            self,
            logits: cute.Tensor,
            seq_lens: cute.Tensor,
            aux_indices: cute.Tensor,
            aux_logits: cute.Tensor,
            state: cute.Tensor,
            out: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.dynamic_kernel(logits, seq_lens, aux_indices, aux_logits, state, out).launch(
                grid=(self.total_ctas, 1, 1),
                block=(self.T, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
                min_blocks_per_mp=self.min_blocks,
                use_pdl=self.use_pdl,
            )
