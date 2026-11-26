# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Triton kernels of the attention-DP expert path's one-shot collectives.

`dispatch_kernel` is `lamport_moe_dispatch`'s gather of every rank's routed-expert inputs, and
`combine_kernel` is `lamport_moe_combine`'s reduce-scatter of the expert partials with the
add of the shared expert (and, on request, of the residual and the next layer's RMSNorm).
Both move rows through a `MoeCommWorkspace` (`_moe_workspace.py`): `[3, world, region]` 16-bit
words of symmetric memory per rank, region `p` of a slot holding what rank `p` sent this rank.

Lamport readiness, in band: every word of a slot holds bf16 -0.0 (0x8000) until a peer's row
lands on it, a sender never stores that pattern (it stores +0.0 in its place), and a receiver
polls each row until none of its words is the sentinel, reads it, and stores the sentinel back.
A 16-bit store is never torn, so a word that is not the sentinel holds the sender's value.

Slots rotate on the device: every launch reads the current slot from `state[0]` and its last
program to finish advances it (`state[1]` counts the arrivals), so a captured graph replays
onto whichever slot is current. A peer writes slot `s` again only three collectives later,
after it has received this rank's rows of the two in between, which this rank sends only after
its earlier launch -- the one that read slot `s` and stored the sentinel back -- completed.

Each kernel runs in two phases per program: every send first, then every receive. With no more
programs than SMs every program is resident, so a program polling for a peer's row never holds
back a program of its own rank that still has to send.
"""

from __future__ import annotations

import triton
import triton.language as tl

#: bf16 -0.0 as int16: a word whose row has not landed.
SENTINEL = tl.constexpr(-32768)
#: Slots the device counter rotates over.
SLOTS = tl.constexpr(3)
#: Maximum number of peer ranks polled concurrently per wave in `combine_kernel`.
#: Four 16-bit vectors of length `BLOCK=8192` across 8 warps keep register usage below
#: spill thresholds while overlapping up to 4 NVLink peer loads per burst (1 wave for
#: DEP4, 2 waves for DEP8).
COMBINE_POLL_WAVE_SIZE = 4


def resolve_combine_wave_size(world: int) -> int:
    """Return the parallel peer-poll wave size for ``combine_kernel`` at ``world`` ranks."""
    if world < 1 or world > 8:
        raise ValueError(f"combine_kernel supports world in 1..8, got {world}")
    return min(world, COMBINE_POLL_WAVE_SIZE)


@triton.jit
def _count(q, c0, c1, c2, c3, c4, c5, c6, c7):
    """Rank `q`'s row count."""
    n = tl.where(q == 0, c0, 0) + tl.where(q == 1, c1, 0) + tl.where(q == 2, c2, 0)
    n += tl.where(q == 3, c3, 0) + tl.where(q == 4, c4, 0) + tl.where(q == 5, c5, 0)
    return n + tl.where(q == 6, c6, 0) + tl.where(q == 7, c7, 0)


@triton.jit
def _offset(q, c0, c1, c2, c3, c4, c5, c6, c7):
    """Rank `q`'s first gathered row: the counts of the ranks before it."""
    n = tl.where(q > 0, c0, 0) + tl.where(q > 1, c1, 0) + tl.where(q > 2, c2, 0)
    n += tl.where(q > 3, c3, 0) + tl.where(q > 4, c4, 0) + tl.where(q > 5, c5, 0)
    return n + tl.where(q > 6, c6, 0)


@triton.jit
def _widest(c0, c1, c2, c3, c4, c5, c6, c7):
    """The largest rank's row count."""
    return tl.maximum(
        tl.maximum(tl.maximum(c0, c1), tl.maximum(c2, c3)),
        tl.maximum(tl.maximum(c4, c5), tl.maximum(c6, c7)),
    )


@triton.jit
def _advance(state, programs):
    """The last program to arrive moves the workspace to its next slot. Every program read the
    slot before it arrived, so none reads the advanced one."""
    arrived = tl.atomic_add(state + 1, 1, sem="acq_rel", scope="gpu")
    if arrived == programs - 1:
        slot = tl.load(state, volatile=True)
        tl.store(state + 1, 0)
        tl.store(state, (slot + 1) % SLOTS)


@triton.jit
def _peer(peers, q):
    """Rank `q`'s buffer, mapped into this rank. Symmetric buffers are page-aligned; saying so
    lets the stores through it vectorize to 16 B."""
    return tl.multiple_of(tl.load(peers + q), 16).to(tl.pointer_type(tl.int16))


@triton.jit
def _or_combine(a, b):
    """Bitwise-OR reduction combinator for single-barrier multi-peer sentinel checks."""
    return a | b


@triton.jit
def _poll(src, mask):
    """`src`'s words once none of them is the sentinel."""
    words = tl.load(src, mask=mask, other=0, volatile=True)
    while tl.max(tl.where(words == SENTINEL, 1, 0), axis=0) > 0:
        words = tl.load(src, mask=mask, other=0, volatile=True)
    return words


@triton.jit
def _poll_wave4(
    buf,
    slot_base,
    region,
    row_off,
    live,
    rank,
    own_v,
    acc,
    P0: tl.constexpr,
    WORLD: tl.constexpr,
):
    """Poll up to four consecutive ranks ``[P0, P0 + 4)`` in parallel and accumulate into
    ``acc`` in strict ascending rank order.

    Issues straight-line vectorized volatile loads for every active remote peer in the wave
    so NVLink read latencies overlap, packs all four peers' sentinel status bits into a single
    CTA-wide bitwise-OR reduction (one ``bar.sync`` per wave instead of one per peer), reloads
    only peers whose bit remains set, and folds ``own_v`` at ``p == rank`` for bit-identical
    FP32 rank-order summation. Sentinel clearing is deferred until after the output stores.
    """
    HAS0: tl.constexpr = P0 < WORLD
    HAS1: tl.constexpr = (P0 + 1) < WORLD
    HAS2: tl.constexpr = (P0 + 2) < WORLD
    HAS3: tl.constexpr = (P0 + 3) < WORLD

    rem0 = (P0 != rank) if HAS0 else False
    rem1 = ((P0 + 1) != rank) if HAS1 else False
    rem2 = ((P0 + 2) != rank) if HAS2 else False
    rem3 = ((P0 + 3) != rank) if HAS3 else False

    src0 = buf + slot_base + P0 * region + row_off
    src1 = buf + slot_base + (P0 + 1) * region + row_off
    src2 = buf + slot_base + (P0 + 2) * region + row_off
    src3 = buf + slot_base + (P0 + 3) * region + row_off

    # Straight-line parallel burst: issue all active remote peer loads back-to-back.
    if HAS0:
        w0 = tl.load(src0, mask=live & rem0, other=0, volatile=True)
    if HAS1:
        w1 = tl.load(src1, mask=live & rem1, other=0, volatile=True)
    if HAS2:
        w2 = tl.load(src2, mask=live & rem2, other=0, volatile=True)
    if HAS3:
        w3 = tl.load(src3, mask=live & rem3, other=0, volatile=True)

    # Single CTA-wide bitwise-OR reduction checks all peers in this wave with one barrier.
    bits = tl.zeros(live.shape, dtype=tl.int32)
    if HAS0:
        bits = bits | tl.where((w0 == SENTINEL) & rem0, 1, 0)
    if HAS1:
        bits = bits | tl.where((w1 == SENTINEL) & rem1, 2, 0)
    if HAS2:
        bits = bits | tl.where((w2 == SENTINEL) & rem2, 4, 0)
    if HAS3:
        bits = bits | tl.where((w3 == SENTINEL) & rem3, 8, 0)
    pending = tl.reduce(bits, axis=0, combine_fn=_or_combine)

    while pending != 0:
        next_bits = tl.zeros(live.shape, dtype=tl.int32)
        if HAS0:
            if (pending & 1) != 0:
                w0 = tl.load(src0, mask=live, other=0, volatile=True)
                next_bits = next_bits | tl.where(w0 == SENTINEL, 1, 0)
        if HAS1:
            if (pending & 2) != 0:
                w1 = tl.load(src1, mask=live, other=0, volatile=True)
                next_bits = next_bits | tl.where(w1 == SENTINEL, 2, 0)
        if HAS2:
            if (pending & 4) != 0:
                w2 = tl.load(src2, mask=live, other=0, volatile=True)
                next_bits = next_bits | tl.where(w2 == SENTINEL, 4, 0)
        if HAS3:
            if (pending & 8) != 0:
                w3 = tl.load(src3, mask=live, other=0, volatile=True)
                next_bits = next_bits | tl.where(w3 == SENTINEL, 8, 0)
        pending = tl.reduce(next_bits, axis=0, combine_fn=_or_combine)

    if HAS0:
        acc += tl.where(rem0, w0.to(tl.bfloat16, bitcast=True).to(tl.float32), own_v.to(tl.float32))
    if HAS1:
        acc += tl.where(rem1, w1.to(tl.bfloat16, bitcast=True).to(tl.float32), own_v.to(tl.float32))
    if HAS2:
        acc += tl.where(rem2, w2.to(tl.bfloat16, bitcast=True).to(tl.float32), own_v.to(tl.float32))
    if HAS3:
        acc += tl.where(rem3, w3.to(tl.bfloat16, bitcast=True).to(tl.float32), own_v.to(tl.float32))
    return acc


@triton.jit
def _dispatch_row(codes, scales, ids, weights, j, lanes, CODES, SCALES, IDS, WEIGHTS):
    """Row `j` of the four inputs as one row of words: codes, scales, ids, weights."""
    j = tl.cast(j, tl.int64)
    s0 = CODES
    s1 = s0 + SCALES
    s2 = s1 + IDS
    s3 = s2 + WEIGHTS
    row = tl.load(codes + j * CODES + lanes, mask=lanes < s0, other=0)
    row += tl.load(scales + j * SCALES + (lanes - s0), mask=(lanes >= s0) & (lanes < s1), other=0)
    row += tl.load(ids + j * IDS + (lanes - s1), mask=(lanes >= s1) & (lanes < s2), other=0)
    row += tl.load(weights + j * WEIGHTS + (lanes - s2), mask=(lanes >= s2) & (lanes < s3), other=0)
    return row


@triton.jit
def _dispatch_store(codes, scales, ids, weights, r, row, lanes, CODES, SCALES, IDS, WEIGHTS):
    """One row of words into gathered row `r` of the four outputs."""
    r = tl.cast(r, tl.int64)
    s0 = CODES
    s1 = s0 + SCALES
    s2 = s1 + IDS
    s3 = s2 + WEIGHTS
    tl.store(codes + r * CODES + lanes, row, mask=lanes < s0)
    tl.store(scales + r * SCALES + (lanes - s0), row, mask=(lanes >= s0) & (lanes < s1))
    tl.store(ids + r * IDS + (lanes - s1), row, mask=(lanes >= s1) & (lanes < s2))
    tl.store(weights + r * WEIGHTS + (lanes - s2), row, mask=(lanes >= s2) & (lanes < s3))


@triton.jit(do_not_specialize=["rank", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"])
def dispatch_kernel(
    codes,
    scales,
    ids,
    weights,
    out_codes,
    out_scales,
    out_ids,
    out_weights,
    buf,
    peers,
    state,
    rank,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    region,
    CODES: tl.constexpr,
    SCALES: tl.constexpr,
    IDS: tl.constexpr,
    WEIGHTS: tl.constexpr,
    ROW: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Every rank's rows of the four inputs, gathered in rank order.

    All operands are 16-bit words: `codes [n, CODES]`, `scales [n, SCALES]`, `ids [n, IDS]` and
    `weights [n, WEIGHTS]` this rank's `n = c<rank>` rows, the four outputs the same widths over
    `sum(c)` rows. A row travels as `ROW` words (the four widths, zero-padded to 16 B) at
    stride `ROW` in the slot region. The sentinel pattern is stored as +0 in every gathered
    row, this rank's own included, so every rank's outputs hold the same bits.
    """
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    slot = tl.load(state, volatile=True)
    region = region.to(tl.int64)
    slot_base = slot.to(tl.int64) * WORLD * region
    lanes = tl.arange(0, BLOCK)
    in_row = lanes < ROW
    n_own = _count(rank, c0, c1, c2, c3, c4, c5, c6, c7)
    own = _offset(rank, c0, c1, c2, c3, c4, c5, c6, c7)
    widest = _widest(c0, c1, c2, c3, c4, c5, c6, c7)
    # Send: this rank's row j to rank q's slot region `rank`, or to its own outputs.
    # Rotate peer order by `rank` so all ranks do not contend on peer 0 first.
    for i in range(pid, n_own * WORLD, programs):
        j = i // WORLD
        q = (i % WORLD + rank) % WORLD
        row = _dispatch_row(codes, scales, ids, weights, j, lanes, CODES, SCALES, IDS, WEIGHTS)
        row = tl.where(row == SENTINEL, 0, row).to(tl.int16)
        if q == rank:
            _dispatch_store(
                out_codes, out_scales, out_ids, out_weights, own + j, row, lanes,
                CODES, SCALES, IDS, WEIGHTS,
            )  # fmt: skip
        else:
            dst = _peer(peers, q) + slot_base + rank * region + j * ROW
            tl.store(dst + lanes, row, mask=in_row)
    # Receive: rank q's row j out of this rank's slot region q, clearing sentinels with
    # non-polluting streaming stores after the output writes.
    sentinel = tl.full([BLOCK], SENTINEL, tl.int16)
    for i in range(pid, WORLD * widest, programs):
        q = (i // widest + rank) % WORLD
        j = i % widest
        if (q != rank) & (j < _count(q, c0, c1, c2, c3, c4, c5, c6, c7)):
            src = buf + slot_base + q * region + j * ROW + lanes
            row = _poll(src, in_row)
            _dispatch_store(
                out_codes, out_scales, out_ids, out_weights,
                _offset(q, c0, c1, c2, c3, c4, c5, c6, c7) + j, row, lanes,
                CODES, SCALES, IDS, WEIGHTS,
            )  # fmt: skip
            tl.store(src, sentinel, mask=in_row, cache_modifier=".cs")
    _advance(state, programs)


@triton.jit(do_not_specialize=["rank", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "eps"])
def combine_kernel(
    partial,
    shared,
    residual,
    residual_out,
    weight,
    out,
    buf,
    peers,
    state,
    rank,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    region,
    eps,
    H: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
    NORM: tl.constexpr,
):
    """This rank's rows of the sum of every rank's `partial`, plus `shared`.

    `partial` is `[sum(c), H]` bf16, every rank's rows in rank order; this rank's are rows
    `[offset, offset + c<rank>)` of the result. The sum is fp32 in rank order and rounds once to
    bf16 (`routed`), and `mlp = bf16(routed + shared)`. Without `NORM`, `out = mlp`. With it,
    `residual_out = bf16(residual + mlp)` (it may be `residual` itself) and
    `out = bf16(residual_out * rsqrt(mean(residual_out^2) + eps) * weight)`, fp32 throughout.
    A sender stores +0.0 for -0.0, which changes no sum but an all -0.0 one.
    """
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    slot = tl.load(state, volatile=True)
    region = region.to(tl.int64)
    slot_base = slot.to(tl.int64) * WORLD * region
    cols = tl.arange(0, BLOCK)
    live = cols < H
    widest = _widest(c0, c1, c2, c3, c4, c5, c6, c7)
    # Send: rank q's rows of this partial to its slot region `rank` (rotated by `rank` to
    # spread NVLink write traffic across peers).
    for i in range(pid, WORLD * widest, programs):
        q = (i % WORLD + rank) % WORLD
        j = i // WORLD
        if (q != rank) & (j < _count(q, c0, c1, c2, c3, c4, c5, c6, c7)):
            r = tl.cast(_offset(q, c0, c1, c2, c3, c4, c5, c6, c7) + j, tl.int64)
            words = tl.load(partial + r * H + cols, mask=live, other=0.0).to(tl.int16, bitcast=True)
            words = tl.where(words == SENTINEL, 0, words).to(tl.int16)
            dst = _peer(peers, q) + slot_base + rank * region + j * H
            tl.store(dst + cols, words, mask=live)
    # Receive and reduce: preload this rank's own partial, poll peers in parallel 4-peer waves
    # while accumulating in strict rank order, write outputs, then clear sentinels off the
    # critical path with non-polluting streaming stores before advancing the slot counter.
    n_own = _count(rank, c0, c1, c2, c3, c4, c5, c6, c7)
    own = _offset(rank, c0, c1, c2, c3, c4, c5, c6, c7)
    sentinel = tl.full([BLOCK], SENTINEL, tl.int16)
    for j in range(pid, n_own, programs):
        row = tl.cast(j, tl.int64) * H + cols
        own_v = tl.load(partial + tl.cast(own + j, tl.int64) * H + cols, mask=live, other=0.0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for wave_base in tl.static_range(0, WORLD, 4):
            acc = _poll_wave4(buf, slot_base, region, row, live, rank, own_v, acc, wave_base, WORLD)
        s = tl.load(shared + row, mask=live, other=0.0).to(tl.float32)
        mlp = (acc.to(tl.bfloat16).to(tl.float32) + s).to(tl.bfloat16)
        if NORM:
            r = tl.load(residual + row, mask=live, other=0.0).to(tl.float32)
            h = (r + mlp.to(tl.float32)).to(tl.bfloat16)
            tl.store(residual_out + row, h, mask=live)
            hf = h.to(tl.float32)
            inv = tl.math.rsqrt(tl.sum(hf * hf, axis=0) / H + eps)
            w = tl.load(weight + cols, mask=live, other=0.0).to(tl.float32)
            tl.store(out + row, (hf * inv * w).to(tl.bfloat16), mask=live)
        else:
            tl.store(out + row, mlp, mask=live)
        for p in tl.static_range(WORLD):
            if p != rank:
                tl.store(
                    buf + slot_base + p * region + row,
                    sentinel,
                    mask=live,
                    cache_modifier=".cs",
                )
    _advance(state, programs)
