# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic length-proportional multi-CTA decode radix top-k prototype and CPU verifier.

This module implements the next-generation decode top-k architecture designed for SM107 DEP8
(VR200, 212 SMs, 1..32 requests x next_n=4 = 4..128 rows, contexts up to 1M):

1. **Zero GPC cluster fragmentation**: ``cluster=(1, 1, 1)`` — eliminates the 8-CTA-per-GPC
   co-scheduling penalty observed when ``ClusterRadixTopkDecodeKernel`` (C=8) runs beside
   side-stream query GEMMs (which shrank standalone savings by 30-45% in DEP4/DEP8 nsys).
2. **Length-proportional CTA allocation**: Every short/dummy row (``end <= TOP_K``) gets at most
   1 CTA (or is coalesced with other short rows into 1 CTA), while long rows receive CTAs
   strictly proportional to ``end[r]`` (up to ``max_ctas_per_row``, e.g. 24-32 CTAs on a 720k
   straggler row). Every long CTA scans roughly the same number of keys, making kernel latency
   scale with total per-rank KV rather than ``max_ctx_len`` (directly attacking both the 2.32 ms
   top-k SOL gap and the 0.91 ms dispatch-wait straggler with R^2=0.60).
3. **L2-resident sparse histogram reduction + Pass-2 empty-slice skip**: Multi-CTA rows reduce
   only the ``[global_lo, global_cut]`` digit range into L2-resident scratch (reusing the
   already-allocated ``aux_indices`` buffer: 128 x 10 x 2048 x 4 B = 10 MiB, zero extra memory)
   via ``red.relaxed.gpu`` + ``ld.acquire.gpu``. CTAs whose slice contributes zero elements at or
   above threshold skip Pass 2 entirely (bit-exact).
4. **256-bit vector streaming** (or TMA 1D bulk ring buffer in GPU follow-up): doubles in-flight
   L2->SM bytes per thread relative to 128-bit ``v4`` loads.

The pure-Python planner (``plan_dynamic_cta_allocation``) and bit-exact multi-CTA emulator
(``emulate_dynamic_radix_topk_decode``) run on CPU with zero GPU/CuTe dependencies so every
edge case (dummy padding, ties, ``-FLT_MAX`` pads, ``+-inf``, ``+NaN``, ``-NaN`` fallback,
empty-slice Pass-2 skipping) is verified deterministically before VR200 GPU compilation.
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
DEFAULT_TOTAL_CTAS = 424  # 2 CTAs/SM on 212-SM VR200
DEFAULT_MAX_CTAS_PER_ROW = 32
SLICE_ALIGN = 8  # 32-byte alignment for 256-bit fp32 vector loads


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

    Short/dummy rows (``end <= top_k``) receive 1 CTA (or 0 if ``end == 0`` and handled in a
    short-row sweep; here every row receives at least 1 CTA so every row owns a well-defined
    rank-0 CTA that writes its ``-1`` tail without inter-row coupling).
    Long rows (``end > top_k``) share the remaining budget ``P_rem`` in proportion to ``end[r]``,
    clamped to ``[1, max_ctas_per_row]``, then water-filled longest-first for any rounding
    remainder so the critical longest row always gets maximum bandwidth.

    This exact O(R) integer algorithm maps directly to one 32-thread warp in SMEM (< 0.3 us).
    """
    if next_n not in (1, 4):
        raise ValueError(f"next_n must be 1 or 4; got {next_n}")
    if total_ctas < len(seq_lens) * next_n:
        raise ValueError(
            f"total_ctas ({total_ctas}) must be >= total rows ({len(seq_lens) * next_n})"
        )
    if max_ctas_per_row < 1:
        raise ValueError(f"max_ctas_per_row must be >= 1; got {max_ctas_per_row}")

    ends = row_causal_ends(seq_lens, next_n, width)
    num_rows = len(ends)
    ctas = [1] * num_rows
    long_rows = [r for r, e in enumerate(ends) if e > top_k]

    if long_rows:
        # Budget left after reserving 1 CTA for every row (short and long alike).
        extra_budget = total_ctas - num_rows
        sum_long = sum(ends[r] for r in long_rows)
        # First pass: floor proportional share of extra CTAs, capped at max_ctas_per_row - 1.
        for r in long_rows:
            desired_total = max(1, (total_ctas - (num_rows - len(long_rows))) * ends[r] // sum_long)
            add = min(max_ctas_per_row - 1, max(0, desired_total - 1))
            ctas[r] += add
            extra_budget -= add

        # Water-fill any remaining extra CTAs to the row with the largest current keys/CTA
        # (deterministic tie-break: longer end, then lower row index).
        while extra_budget > 0:
            best_r = -1
            best_load = -1.0
            for r in long_rows:
                if ctas[r] >= max_ctas_per_row:
                    continue
                load = ends[r] / ctas[r]
                if load > best_load or (
                    load == best_load and (best_r < 0 or ends[r] > ends[best_r])
                ):
                    best_load = load
                    best_r = r
            if best_r < 0:
                break
            ctas[best_r] += 1
            extra_budget -= 1

    offsets: list[int] = []
    acc = 0
    for c in ctas:
        offsets.append(acc)
        acc += c

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
        total_ctas_used=acc,
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

            # Mirror sparse [global_lo, global_cut] L2 histogram reduction.
            lo_bins = []
            cut_bins = []
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
                lo_bins.append(first)
                cut_bins.append(cut)
            rlo = min(lo_bins)
            rcut = min(cut_bins)

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
            # Rank 0 resolves remaining digits over gathered list.
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
# GPU CuTe DSL Kernel Architecture Specification (SM107 / VR200)
# ---------------------------------------------------------------------------
# Scratch layout inside ``aux_indices`` (int32[rows * 10 * 2048], >= 10 MiB at 128 rows):
#   Per row ``r`` (row stride = 10 * 2048 = 20480 int32 words = 80 KiB, fits in L2!):
#     [0 .. 4095]       : L2 global histogram buffer 0 (level 0 / 2)
#     [4096 .. 5119]    : L2 global histogram buffer 1 (level 1)
#     [5120 .. 5135]    : per-row control words (lo_bin, cut_bin, neg_count, arrival_l0..l2,
#                         c_above, c_list, c_tie, arrival_emit)
#     [5136 .. 9231]    : gather list keys   (4096 uint32)
#     [9232 .. 13327]   : gather list cols   (4096 int32)
#   Total per row = 13328 int32 words < 20480 words already allocated per row!
#   Reset discipline: last CTA arriving at final row barrier clears the 16 control words and
#   zeroes only ``[rlo, rcut]`` of the used histogram buffers (zero extra launch, zero full clear).
#
# Launch geometry on VR200 (212 SMs):
#   grid=(total_ctas, 1, 1)  # 424 CTAs (2/SM, 512 threads/CTA) or 212 CTAs (1/SM, 1024 threads/CTA)
#   cluster=(1, 1, 1)        # NO GPC cluster restriction -> zero contention with side-stream GEMMs
#   Memory pass: 256-bit vectorized loads (VEC=8 fp32) + 2-stage register software pipeline
#                (or TMA 1D bulk 4-stage SMEM ring buffer for 64 KiB in-flight per CTA).
_AUX_ROW_WORDS = PARTITIONS * TOP_K  # 20480 int32 words per row
_AUX_HIST0_OFF = 0
_AUX_HIST1_OFF = 4096
_AUX_CTRL_OFF = 5120
_AUX_GATHER_KEY_OFF = 5136
_AUX_GATHER_COL_OFF = 5136 + GATHER_CAPACITY
_AUX_REQUIRED_WORDS_PER_ROW = _AUX_GATHER_COL_OFF + GATHER_CAPACITY
assert _AUX_REQUIRED_WORDS_PER_ROW <= _AUX_ROW_WORDS
