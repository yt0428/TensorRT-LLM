# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Triton kernels of the attention-DP expert path's one-shot collectives.

`dispatch_kernel` is `lamport_moe_dispatch`'s gather of every rank's routed-expert inputs, and
`combine_kernel` is `lamport_moe_combine`'s reduce-scatter of the expert partials with the
add of the shared expert (and, on request, of the residual and the next layer's RMSNorm).
Both move rows through a `MoeCommWorkspace` (`_lamport.py`): `[3, world, region]` 16-bit
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
def _poll(src, mask):
    """`src`'s words once none of them is the sentinel."""
    words = tl.load(src, mask=mask, other=0, volatile=True)
    while tl.max(tl.where(words == SENTINEL, 1, 0), axis=0) > 0:
        words = tl.load(src, mask=mask, other=0, volatile=True)
    return words


@triton.jit
def _dispatch_row(codes, scales, ids, weights, j, lanes, CODES, SCALES, IDS, WEIGHTS):
    """Row `j` of the four inputs as one row of words: codes, scales, ids, weights."""
    j = j.to(tl.int64)
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
    r = r.to(tl.int64)
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
    for i in range(pid, n_own * WORLD, programs):
        j = i // WORLD
        q = i % WORLD
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
    # Receive: rank q's row j out of this rank's slot region q.
    for i in range(pid, WORLD * widest, programs):
        q = i // widest
        j = i % widest
        if (q != rank) & (j < _count(q, c0, c1, c2, c3, c4, c5, c6, c7)):
            src = buf + slot_base + q * region + j * ROW + lanes
            row = _poll(src, in_row)
            _dispatch_store(
                out_codes, out_scales, out_ids, out_weights,
                _offset(q, c0, c1, c2, c3, c4, c5, c6, c7) + j, row, lanes,
                CODES, SCALES, IDS, WEIGHTS,
            )  # fmt: skip
            tl.store(src, tl.full([BLOCK], SENTINEL, tl.int16), mask=in_row)
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
    # Send: rank q's rows of this partial to its slot region `rank`.
    for i in range(pid, WORLD * widest, programs):
        q = i % WORLD
        j = i // WORLD
        if (q != rank) & (j < _count(q, c0, c1, c2, c3, c4, c5, c6, c7)):
            r = (_offset(q, c0, c1, c2, c3, c4, c5, c6, c7) + j).to(tl.int64)
            words = tl.load(partial + r * H + cols, mask=live, other=0.0).to(tl.int16, bitcast=True)
            words = tl.where(words == SENTINEL, 0, words).to(tl.int16)
            dst = _peer(peers, q) + slot_base + rank * region + j * H
            tl.store(dst + cols, words, mask=live)
    # Receive and reduce: this rank's row j from every rank, in rank order.
    n_own = _count(rank, c0, c1, c2, c3, c4, c5, c6, c7)
    own = _offset(rank, c0, c1, c2, c3, c4, c5, c6, c7)
    for j in range(pid, n_own, programs):
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for p in tl.static_range(WORLD):
            if p == rank:
                v = tl.load(partial + (own + j).to(tl.int64) * H + cols, mask=live, other=0.0)
                v = v.to(tl.float32)
            else:
                src = buf + slot_base + p * region + j * H + cols
                words = _poll(src, live)
                tl.store(src, tl.full([BLOCK], SENTINEL, tl.int16), mask=live)
                v = words.to(tl.bfloat16, bitcast=True).to(tl.float32)
            acc += v
        row = j.to(tl.int64) * H + cols
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
    _advance(state, programs)
