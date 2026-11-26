# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CuTe DSL kernel behind `cluster_radix_topk_decode`: one thread-block cluster per decode row.

It selects what `SplitRadixTopkDecodeKernel`'s partition and merge launches select (see
`_radix_topk_decode`): the `top_k` best of the row's columns and of the pads the op's ten blocks
add (`pad` entries at `key(-FLT_MAX)`, ranked after the columns of that key), ties at the k-th key
taken in no particular order. The selected columns come first and every selected pad as a -1 after
them.

The row's `C` CTAs each own a contiguous slice of its causal window, read in 16-byte vectors. An
exact radix select over the full 32-bit key runs digit by digit (bits 31..20, 19..10, 9..0):

1. Each CTA counts its slice's digit (rank 0 adds the row's pads) into a shared-memory histogram
   and publishes the first nonzero bin and the bin where its own cumulative count reaches the
   number still needed. The row's threshold bin lies between the smallest of the first bins and
   the smallest of the crossing bins, so after a cluster barrier each CTA sums only that range of
   the C histograms over distributed shared memory and finds the threshold bin, the count above
   it and the count in it, identically in every CTA.
2. While the threshold bin holds more columns than rank 0's gather list, the next digit is counted
   over the slice among the keys sharing the prefix, into the other histogram buffer.
3. One pass over the slice writes each column above the prefix to the output, at a base each CTA
   claims with one atomic on rank 0's counter; a thread whose scores all fall below the prefix's
   floor (`_score_floor`) skips its batch, and a warp of such threads skips the claims. A column
   sharing the prefix goes to rank 0's gather list, which rank 0 resolves over the remaining
   digits after a cluster barrier; with all 32 bits resolved, each CTA instead claims its share of
   the tie group from a counter on rank 0. Rank 0 then writes the -1s.

A row of at most `top_k` columns takes every column whose key is at most key(-FLT_MAX), in column
order, in rank 0 alone; the other CTAs of its cluster exit at once. The clusters take the rows
longest request first.

A row holding a negative NaN (counted in the first pass, which the op's fp16 first pass ranks
differently from its key) runs the partition and merge launches' code instead: the CTAs take the
ten partitions round-robin into the aux buffers, and rank 0 merges them after a cluster barrier
(a short row: rank 0 alone, all ten).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_self_sampling import (
    _ld_g_nc_v4_f32,
)

from ._radix_topk_decode import (
    _N_OUT,
    _N_TIE,
    D1_BINS,
    LIST_CAPACITY,
    NEG_INF_KEY,
    PAD_KEY,
    TOP_K,
    SplitRadixTopkDecodeKernel,
    _digit,
    _key,
)

#: The op's blocks per row on the long-row split this kernel replaces.
PARTITIONS = 10
#: The radix digits, most significant first: (shift, mask, bins).
LEVELS = ((20, 0xFFF, 4096), (10, 0x3FF, 1024), (0, 0x3FF, 1024))
#: Columns rank 0 gathers for the remaining digits once the threshold bin holds at most this many.
GATHER_CAPACITY = LIST_CAPACITY
#: The most requests a call holds (its rows, at most 64, over next_n).
MAX_REQUESTS = 64
#: Words per CTA of the diagnostic `trace` build: %globaltimer_lo at its start, after its first
#: pass, after the first cluster barrier, after the first digit's threshold, before and after its
#: output pass, after the last cluster barrier and at its end, written to `aux_indices` past the
#: rows' partition slots.
TRACE_WORDS = 8

# How the row is being resolved.
_NARROW = 0  # counting the next digit over the slice
_GATHER = 1  # the threshold bin's columns go to rank 0's list
_DIRECT = 2  # the threshold key is known: ties are claimed from rank 0's counter
_FALLBACK = 3  # a negative NaN: the partition and merge code

# A CTA's published words per level (two sets, by level parity), read by its peers.
_I_LO = 0  # first nonzero bin
_I_CUT = 1  # bin where the CTA's cumulative count reaches the count still needed
_I_NEG = 2  # negative NaNs in the slice (first level)
_INFO_WORDS = 4

# Rank 0's counters, claimed by every CTA of the cluster.
_C_ABOVE = 0  # output positions of the columns above the prefix
_C_LIST = 1  # gather list slots
_C_TIE = 2  # tie group slots
_COUNTERS = 4

# Local scalars.
_S_OUT = _N_OUT  # rank 0: list winners written (the base class's slot)
_S_TIE = _N_TIE  # rank 0: list tie slots claimed (the base class's slot)
_S_ABOVE = 8  # local claims of the CTA's above columns
_S_EQ = 9  # local claims of the CTA's prefix-sharing columns
_S_EQN = 10  # the CTA's count in the threshold bin
_S_BASE_A = 11  # the CTA's first output position
_S_BASE_E = 12  # the CTA's first gather or tie slot
_SCALARS = 16
assert len({_S_ABOVE, _S_EQ, _S_EQN, _S_BASE_A, _S_BASE_E, _S_OUT, _S_TIE}) == 7


@dsl_user_op
def _mapa(saddr, rank, *, loc=None, ip=None):
    """The shared::cluster address of this CTA's shared address `saddr` in CTA `rank`."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [saddr.ir_value(loc=loc, ip=ip), rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_cluster(caddr, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [caddr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.b32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_cluster(caddr, value, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [caddr.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_add_cluster(caddr, value, *, loc=None, ip=None):
    """atom.shared::cluster.add: the counter's value before adding `value`."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [caddr.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
            "atom.shared::cluster.add.u32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _globaltimer_lo(*, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %globaltimer_lo;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _score_floor(key_max):
    """A floor of the scores whose keys are at most `key_max`: every such score but a NaN compares
    >= it. Past key(-inf) (negative NaNs' keys) it is -inf; below key(+inf) a NaN, which only the
    positive NaNs, the sole scores with such keys, pass (by `_passes`' NaN test)."""
    bits = key_max
    if (key_max & cutlass.Uint32(0x80000000)) == cutlass.Uint32(0):
        bits = (key_max ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
    if key_max > cutlass.Uint32(NEG_INF_KEY):
        bits = cutlass.Uint32(NEG_INF_KEY)
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, bits.ir_value()))


@cute.jit
def _passes(x, floor):
    """Whether a score may hold a key at most the one `floor` came from: not below it, or a NaN."""
    return (x >= floor) | (x != x)


def _saddr(tensor, index):
    """The shared address of `tensor[index]` (a 1-D shared-memory tensor)."""
    return (tensor.iterator + index).toint()


class ClusterRadixTopkDecodeKernel(SplitRadixTopkDecodeKernel):
    """`cluster` CTAs of `threads` threads per row, `unroll` 16-byte vectors per thread in each
    batch of a pass over the slice, the next batch's loads issued before a batch is processed
    (`pipe`), at least `min_blocks` CTAs per SM when nonzero; `next_n` is a compile-time
    constant."""

    def __init__(
        self,
        next_n: int,
        cluster: int,
        threads: int,
        unroll: int,
        use_pdl: bool,
        pipe: bool = True,
        min_blocks: int = 0,
        trace: bool = False,
    ):
        super().__init__(next_n, PARTITIONS, use_pdl)
        assert threads % 32 == 0 and D1_BINS % threads == 0 and LEVELS[1][2] % threads == 0
        self.C = cluster
        self.T = threads
        self.W = threads // 32
        self.U = unroll
        self.pipe = pipe
        self.min_blocks = min_blocks
        self.trace = trace

    # ------------------------------------------------------------------ passes over the slice
    @cute.jit
    def _load(self, row_addr, lo, i0, n4, vals, tid):
        """The thread's `self.U` vectors of the batch at vector `i0` into `vals`; a vector past
        the slice reads its last one again."""
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
        """Digit `level` of the batch's keys that share `prefix` above it into `s_hist`; returns
        `neg` plus the batch's negative NaNs (first level only)."""
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
    def _count_slice(self, row_addr, lo, hi, s_hist, level: cutlass.Constexpr[int], prefix, tid):
        """Digit `level` of the slice's keys that share `prefix` above it into `s_hist`. Returns
        the thread's count of negative NaNs (first level only). With `pipe`, each batch's loads
        are in flight while the previous batch is counted."""
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
                neg = self._count_batch(
                    va,
                    i0,
                    n4,
                    lo,
                    hi,
                    s_hist,
                    level,
                    prefix,
                    neg,
                    tid,
                )
                self._load(row_addr, lo, i0 + step + step, n4, va, tid)
                neg = self._count_batch(
                    vb,
                    i0 + step,
                    n4,
                    lo,
                    hi,
                    s_hist,
                    level,
                    prefix,
                    neg,
                    tid,
                )
                i0 = i0 + step + step
        else:
            while i0 < n4:
                self._load(row_addr, lo, i0, n4, va, tid)
                neg = self._count_batch(
                    va,
                    i0,
                    n4,
                    lo,
                    hi,
                    s_hist,
                    level,
                    prefix,
                    neg,
                    tid,
                )
                i0 = i0 + step
        return neg

    @cute.jit
    def _claim_n(self, s_scal, slot: cutlass.Constexpr[int], n):
        """The first of `n` consecutive positions of the counter `s_scal[slot]` per thread, one
        atomic per warp."""
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
        self, vals, i0, n4, lo, hi, prefix, shift, floor, gather, take, a_total, dst, s_lkey,
        s_lcol, s_scal, tid,
    ):  # fmt: skip
        """`_emit_slice` over one batch: the thread's counts, one claim per warp and class, then
        its writes."""
        zero = cutlass.Int32(0)
        n_above = zero
        n_eq = zero
        # Almost every batch of a long slice holds nothing to write: a thread whose scores all
        # fall below `floor` skips them, and a warp of such threads skips the claims.
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
                                _st_cluster(_mapa(_saddr(s_lkey, pe), zero), key)
                                _st_cluster(_mapa(_saddr(s_lcol, pe), zero), c)
                            elif pe < take:
                                dst[a_total + pe] = c
                            pe = pe + cutlass.Int32(1)

    @cute.jit
    def _emit_slice(
        self, row_addr, lo, hi, prefix, shift, gather, take, a_total, dst, s_lkey, s_lcol,
        s_scal, tid,
    ):  # fmt: skip
        """Each column of the slice above the prefix to `dst` from the CTA's base; each one
        sharing it to rank 0's list (`gather`) or, among the first `take` of the tie group, to
        `dst` after the `a_total` columns above."""
        n4 = (hi - lo + cutlass.Int32(3)) // cutlass.Int32(4)
        step = cutlass.Int32(self.U * self.T)
        # Every column at or above the prefix (above it or sharing it) scores at least this.
        floor = _score_floor((((prefix >> shift) + cutlass.Uint32(1)) << shift) - cutlass.Uint32(1))
        va = cute.make_rmem_tensor(cute.make_layout((self.U * 4,)), cutlass.Float32)
        vb = cute.make_rmem_tensor(cute.make_layout((self.U * 4,)), cutlass.Float32)
        i0 = cutlass.Int32(0)
        if cutlass.const_expr(self.pipe):
            if n4 > cutlass.Int32(0):
                self._load(row_addr, lo, i0, n4, va, tid)
            while i0 < n4:
                self._load(row_addr, lo, i0 + step, n4, vb, tid)
                self._emit_batch(
                    va, i0, n4, lo, hi, prefix, shift, floor, gather, take, a_total, dst,
                    s_lkey, s_lcol, s_scal, tid,
                )  # fmt: skip
                self._load(row_addr, lo, i0 + step + step, n4, va, tid)
                self._emit_batch(
                    vb, i0 + step, n4, lo, hi, prefix, shift, floor, gather, take, a_total, dst,
                    s_lkey, s_lcol, s_scal, tid,
                )  # fmt: skip
                i0 = i0 + step + step
        else:
            while i0 < n4:
                self._load(row_addr, lo, i0, n4, va, tid)
                self._emit_batch(
                    va, i0, n4, lo, hi, prefix, shift, floor, gather, take, a_total, dst,
                    s_lkey, s_lcol, s_scal, tid,
                )  # fmt: skip
                i0 = i0 + step

    # ------------------------------------------------------------------ one digit
    @cute.jit
    def _publish(self, s_hist, nbins: cutlass.Constexpr[int], need, s_info, s_warp, tid):
        """The CTA's first nonzero bin and the bin where its cumulative count reaches `need` (the
        last bin when it never does) into `s_info`, whose first-bin word holds `nbins`."""
        per = cutlass.const_expr(nbins // self.T)
        lane = cute.arch.lane_idx()
        warp = tid // cutlass.Int32(32)
        base = tid * cutlass.Int32(per)
        local = cutlass.Int32(0)
        first = cutlass.Int32(nbins)
        for j in cutlass.range_constexpr(per):
            v = s_hist[base + cutlass.Int32(j)]
            local = local + v
            if (v != cutlass.Int32(0)) & (first == cutlass.Int32(nbins)):
                first = base + cutlass.Int32(j)
        if first < cutlass.Int32(nbins):
            cute.arch.atomic_min(
                ptr=(s_info.iterator + _I_LO).llvm_ptr, val=first, sem="relaxed", scope="cta"
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
                s_info[_I_CUT] = base + cutlass.Int32(j)
            prefix = prefix + count
        if (tid == cutlass.Int32(0)) & (total < need):
            s_info[_I_CUT] = cutlass.Int32(nbins - 1)

    @cute.jit
    def _peer_range(self, s_info, s_peer, tid):
        """(first bin, crossing bin, negative NaNs) over the cluster: the smallest first bin and
        crossing bin, and the summed count. Every CTA reads the same words."""
        if tid < cutlass.Int32(self.C):
            for i in cutlass.range_constexpr(3):
                s_peer[tid * cutlass.Int32(3) + cutlass.Int32(i)] = _ld_cluster(
                    _mapa(_saddr(s_info, cutlass.Int32(i)), tid)
                )
        cute.arch.barrier()
        lo = s_peer[_I_LO]
        cut = s_peer[_I_CUT]
        neg = s_peer[_I_NEG]
        for r in cutlass.range_constexpr(1, self.C):
            if s_peer[r * 3 + _I_LO] < lo:
                lo = s_peer[r * 3 + _I_LO]
            if s_peer[r * 3 + _I_CUT] < cut:
                cut = s_peer[r * 3 + _I_CUT]
            neg = neg + s_peer[r * 3 + _I_NEG]
        return lo, cut, neg

    @cute.jit
    def _sum_range(self, s_hist, s_ghist, nbins: cutlass.Constexpr[int], lo, cut, tid):
        """The cluster's histogram over bins [lo, cut] into `s_ghist`, zero elsewhere. The bins of
        the range go to the threads in turn (each reads the C peers' counts of its bins); the
        rest are zeroed by the thread that owns them in the base class's `_threshold`. Ends with
        a barrier."""
        per = cutlass.const_expr(nbins // self.T)
        base = tid * cutlass.Int32(per)
        for j in cutlass.range_constexpr(per):
            b = base + cutlass.Int32(j)
            if (b < lo) | (b > cut):
                s_ghist[b] = cutlass.Int32(0)
        b = lo + tid
        while b <= cut:
            addr = _saddr(s_hist, b)
            g = cutlass.Int32(0)
            for r in cutlass.range_constexpr(self.C):
                g = g + _ld_cluster(_mapa(addr, cutlass.Int32(r)))
            s_ghist[b] = g
            b = b + cutlass.Int32(self.T)
        cute.arch.barrier()

    @cute.jit
    def _short_row(
        self, logits, seq_lens, aux_indices, aux_logits, out, row, end, dst, s_hist, s_out,
        s_lkey, s_lcol, s_warp, s_scal, tid,
    ):  # fmt: skip
        """Rank 0 alone, a row of at most top_k columns: every column whose key is at most
        key(-FLT_MAX), in column order, then -1; a row holding a negative NaN runs the ten
        partitions and the merge here."""
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
                aux_indices, aux_logits, out, row, s_hist, s_out, s_lkey, s_lcol, s_warp, s_scal,
                tid,
            )  # fmt: skip

    @cute.jit
    def _split_at(self, s_hist, nbins: cutlass.Constexpr[int], t, s_red, s_scal, tid):
        """The CTA's count below bin `t`; its count in it goes to `s_scal[_S_EQN]`."""
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

    # ------------------------------------------------------------------ rank 0's list
    @cute.jit
    def _resolve_list(
        self, n_list, need, pad_in, prefix, shift, a_total, dst, s_hist, s_lkey, s_lcol, s_warp,
        s_thr, s_scal, tid,
    ):  # fmt: skip
        """Rank 0: the digits below `shift` over the gathered list and the `pad_in` pads that
        share the prefix; the winners to `dst` after the `a_total` columns above. Returns how many
        columns it wrote."""
        lane = cute.arch.lane_idx()
        warp = tid // cutlass.Int32(32)
        pad_key = cutlass.Uint32(PAD_KEY)
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
                    # Above the threshold key: selected; equal to it: `take` of them.
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

    @cute.jit
    def _stamp(self, aux_indices, trace_base, bid, k: cutlass.Constexpr[int], tid):
        if cutlass.const_expr(self.trace):
            if tid == cutlass.Int32(0):
                aux_indices[trace_base + bid * cutlass.Int32(TRACE_WORDS) + cutlass.Int32(k)] = (
                    _globaltimer_lo()
                )

    # ------------------------------------------------------------------ kernel
    @cute.kernel
    def cluster_kernel(
        self,
        logits: cute.Tensor,
        seq_lens: cute.Tensor,
        aux_indices: cute.Tensor,
        aux_logits: cute.Tensor,
        out: cute.Tensor,
    ):
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        bid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        rank = bid % cutlass.Int32(self.C)
        top_k = cutlass.Int32(TOP_K)
        zero = cutlass.Int32(0)
        trace_base = cute.size(logits, mode=[0]) * cutlass.Int32(PARTITIONS * TOP_K)
        self._stamp(aux_indices, trace_base, bid, 0, tid)

        smem = SmemAllocator()
        # Histograms by level parity (read by the peers; the second level's has 1024 bins), and
        # the cluster's sum, whose space the negative-NaN path takes for its selection.
        s_hist0 = smem.allocate_tensor(cutlass.Int32, cute.make_layout((D1_BINS,)), 128)
        s_hist1 = smem.allocate_tensor(cutlass.Int32, cute.make_layout((LEVELS[1][2],)), 128)
        s_hists = (s_hist0, s_hist1)
        s_ghist = smem.allocate_tensor(cutlass.Int32, cute.make_layout((D1_BINS,)), 128)
        s_out = cute.make_tensor(s_ghist.iterator, cute.make_layout((TOP_K,)))
        s_lkey = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((LIST_CAPACITY,)), 128)
        s_lcol = smem.allocate_tensor(cutlass.Int32, cute.make_layout((LIST_CAPACITY,)), 128)
        s_warp = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.W,)), 128)
        s_red = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.W,)), 128)
        s_thr = smem.allocate_tensor(cutlass.Int32, cute.make_layout((8,)), 16)
        s_scal = smem.allocate_tensor(cutlass.Int32, cute.make_layout((_SCALARS,)), 16)
        s_fb = smem.allocate_tensor(cutlass.Int32, cute.make_layout((8,)), 16)
        s_info0 = smem.allocate_tensor(cutlass.Int32, cute.make_layout((_INFO_WORDS,)), 16)
        s_info1 = smem.allocate_tensor(cutlass.Int32, cute.make_layout((_INFO_WORDS,)), 16)
        s_infos = (s_info0, s_info1)
        s_peer = smem.allocate_tensor(cutlass.Int32, cute.make_layout((3 * self.C,)), 16)
        s_cnt = smem.allocate_tensor(cutlass.Int32, cute.make_layout((_COUNTERS,)), 16)
        s_order = smem.allocate_tensor(cutlass.Int32, cute.make_layout((MAX_REQUESTS,)), 16)

        # The clusters take the requests longest first (ties by index): not every cluster of a
        # call fits on the GPU at once, and those it starts late then hold the short rows.
        requests = cute.size(seq_lens, mode=[0])
        if tid < requests:
            mine = seq_lens[tid]
            place = zero
            b = zero
            while b < requests:
                other = seq_lens[b]
                if (other > mine) | ((other == mine) & (b < tid)):
                    place = place + cutlass.Int32(1)
                b = b + cutlass.Int32(1)
            s_order[place] = tid
        cute.arch.barrier()
        slot = bid // cutlass.Int32(self.C)
        row = s_order[slot // cutlass.Int32(self.next_n)] * cutlass.Int32(self.next_n)
        row = row + slot % cutlass.Int32(self.next_n)

        # The causal end, its slice, and the op's pads (its blocks of at most top_k columns
        # each pad to top_k).
        end = self._row_end(logits, seq_lens, row)
        span = (end + cutlass.Int32(self.C - 1)) // cutlass.Int32(self.C)
        span = (span + cutlass.Int32(7)) // cutlass.Int32(8) * cutlass.Int32(8)
        lo = rank * span
        if lo > end:
            lo = end
        hi = lo + span
        if hi > end:
            hi = end
        block = end // cutlass.Int32(PARTITIONS)
        last = end - block * cutlass.Int32(PARTITIONS - 1)
        pad = cutlass.Int32(0)
        if block < top_k:
            pad = cutlass.Int32(PARTITIONS - 1) * (top_k - block)
        if last < top_k:
            pad = pad + top_k - last
        row_addr = logits[row, None].iterator.toint()
        dst = out[row, None]

        if end <= top_k:
            # A short row (a padding dummy among them): rank 0 alone, the others exit.
            if rank == zero:
                self._short_row(
                    logits, seq_lens, aux_indices, aux_logits, out, row, end, dst, s_hist0, s_out,
                    s_lkey, s_lcol, s_warp, s_fb, tid,
                )  # fmt: skip
        else:
            i = tid
            while i < cutlass.Int32(D1_BINS):
                s_hist0[i] = zero
                if i < cutlass.Int32(LEVELS[1][2]):
                    s_hist1[i] = zero
                i = i + cutlass.Int32(self.T)
            if tid < cutlass.Int32(_SCALARS):
                s_scal[tid] = zero
            if tid < cutlass.Int32(_INFO_WORDS):
                s_info0[tid] = zero
                s_info1[tid] = zero
            if tid < cutlass.Int32(_COUNTERS):
                s_cnt[tid] = zero
            cute.arch.barrier()
            # A level's first-bin word starts at its bin count (the third level's is reset below);
            # the barrier after each level's pass orders this before its atomics.
            if tid == zero:
                s_info0[_I_LO] = cutlass.Int32(LEVELS[0][2])
                s_info1[_I_LO] = cutlass.Int32(LEVELS[1][2])

            mode = cutlass.Int32(_NARROW)
            need = top_k
            pad_in = pad
            prefix = cutlass.Uint32(0)
            shift = cutlass.Int32(32)
            a_total = zero  # columns above the prefix, over the row
            my_above = zero  # of them in this slice
            my_eq = zero  # this slice's columns sharing the prefix
            count_cols = zero  # the row's columns sharing the prefix
            for level in cutlass.range_constexpr(3):
                sh, mask, nbins = LEVELS[level]
                s_hist = s_hists[level % 2]
                s_info = s_infos[level % 2]
                pad_digit = cutlass.const_expr((PAD_KEY >> sh) & mask)
                if mode == cutlass.Int32(_NARROW):
                    if cutlass.const_expr(level == 2):
                        # The first level's buffers: every peer read them before the second
                        # level's cluster barrier.
                        if tid == zero:
                            s_info[_I_LO] = cutlass.Int32(nbins)
                        self._clear(s_hist, nbins, tid)
                    neg = self._count_slice(row_addr, lo, hi, s_hist, level, prefix, tid)
                    if neg > zero:
                        self._count(s_info, cutlass.Int32(_I_NEG), neg)
                    if (rank == zero) & (tid == zero) & (pad_in > zero):
                        self._count(s_hist, cutlass.Int32(pad_digit), pad_in)
                    cute.arch.barrier()
                    if cutlass.const_expr(level == 0):
                        self._stamp(aux_indices, trace_base, bid, 1, tid)
                    self._publish(s_hist, nbins, need, s_info, s_warp, tid)
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                    if cutlass.const_expr(level == 0):
                        self._stamp(aux_indices, trace_base, bid, 2, tid)
                    rlo, rcut, rneg = self._peer_range(s_info, s_peer, tid)
                    if rneg > zero:
                        mode = cutlass.Int32(_FALLBACK)
                    else:
                        self._sum_range(s_hist, s_ghist, nbins, rlo, rcut, tid)
                        t, before, count = self._threshold(s_ghist, s_warp, s_thr, nbins, need, tid)
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
                        if cutlass.const_expr(level == 0):
                            self._stamp(aux_indices, trace_base, bid, 3, tid)

            if mode == cutlass.Int32(_FALLBACK):
                part = rank
                while part < cutlass.Int32(PARTITIONS):
                    self._partition(
                        logits, seq_lens, aux_indices, aux_logits, row, part,
                        s_hists[0], s_out, s_lkey, s_lcol, s_warp, s_fb, tid,
                    )  # fmt: skip
                    cute.arch.barrier()
                    part = part + cutlass.Int32(self.C)
                cute.arch.fence_acq_rel_gpu()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                if rank == zero:
                    self._merge(
                        aux_indices, aux_logits, out, row, s_hists[0], s_out, s_lkey, s_lcol, s_warp,
                        s_fb, tid,
                    )  # fmt: skip
            else:
                gather = mode == cutlass.Int32(_GATHER)
                if tid == zero:
                    if my_above > zero:
                        s_scal[_S_BASE_A] = _atom_add_cluster(
                            _mapa(_saddr(s_cnt, _C_ABOVE), zero), my_above
                        )
                    if my_eq > zero:
                        slot = cutlass.Int32(_C_TIE)
                        if gather:
                            slot = cutlass.Int32(_C_LIST)
                        s_scal[_S_BASE_E] = _atom_add_cluster(
                            _mapa(_saddr(s_cnt, slot), zero), my_eq
                        )
                cute.arch.barrier()
                take = need
                if count_cols < take:
                    take = count_cols
                self._stamp(aux_indices, trace_base, bid, 4, tid)
                emit_eq = (my_eq > zero) & (gather | (s_scal[_S_BASE_E] < take))
                if (my_above > zero) | emit_eq:
                    self._emit_slice(
                        row_addr, lo, hi, prefix, shift, gather, take, a_total, dst, s_lkey, s_lcol,
                        s_scal, tid,
                    )  # fmt: skip
                self._stamp(aux_indices, trace_base, bid, 5, tid)
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                self._stamp(aux_indices, trace_base, bid, 6, tid)
                if rank == zero:
                    written = take
                    if gather:
                        written = self._resolve_list(
                            s_cnt[_C_LIST], need, pad_in, prefix, shift, a_total, dst, s_hists[0],
                            s_lkey, s_lcol, s_warp, s_thr, s_scal, tid,
                        )  # fmt: skip
                    k = a_total + written + tid
                    while k < top_k:
                        dst[k] = cutlass.Int32(-1)
                        k = k + cutlass.Int32(self.T)
                    if cutlass.const_expr(self.trace):
                        cute.arch.barrier()
        self._stamp(aux_indices, trace_base, bid, 7, tid)

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        seq_lens: cute.Tensor,
        aux_indices: cute.Tensor,
        aux_logits: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        rows = cute.size(logits, mode=[0])
        self.cluster_kernel(logits, seq_lens, aux_indices, aux_logits, out).launch(
            grid=(rows * self.C, 1, 1),
            block=(self.T, 1, 1),
            cluster=(self.C, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.min_blocks,
            use_pdl=self.use_pdl,
        )
