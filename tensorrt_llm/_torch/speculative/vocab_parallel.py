# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Token selection over a vocab-sharded LM head without gathering the logits.

A column-parallel LM head leaves every TP rank with a contiguous vocab shard of
the logits. Picking one token per row does not need the gathered
``[rows, vocab]`` logits: each rank reduces its shard to one
``(value, global index)`` pair per row, the pairs are exchanged, and the winner
is taken in ``torch.argmax`` order (``NaN`` beats every number, equal maxima go
to the lowest global index). The selected ids are therefore bit-identical to
``torch.argmax`` over the gathered logits.

Optional exact tanh soft-capping (``cap * tanh(x / cap)``) is fused directly
into the local shard reduction so soft-capped models (e.g. Mistral Large 4 with
``cap=30.0``) remain bit-identical to gathered soft-capped argmax even at
saturation (``|x| >= 270.327``) and preserve exact sampling distributions under
temperature-only Gumbel-max sampling.

Consumers:

* :meth:`VocabParallelHead.argmax` -- greedy target tokens.
* :meth:`VocabParallelHead.sample` -- temperature-only sampling through the
  Gumbel-max trick: ``argmax(logits / T + G)`` with standard Gumbel noise ``G``
  is a draw from ``softmax(logits / T)``, exact up to the 23-bit uniform
  behind ``G``. The noise of every vocab entry comes from a Philox stream keyed
  by the row's ``(seed, offset)``, so all ranks agree on the winner.
* :meth:`VocabParallelHead.markov_tokens` -- the DSpark vanilla-Markov greedy
  block chain on shard logits.
* :meth:`VocabParallelHead.rnn_markov_tokens` -- the DSpark recurrent (RNN)
  Markov greedy block chain on shard logits with vocab-sharded
  ``draft_markov_output`` (``[V / tp, rank]``).

The pairs travel through one all-reduce (sum) of a zero-padded float32 buffer in
which each rank writes only its own columns; ``x + 0`` is exact and indices
below ``2**24`` are exact in float32, so the sum acts as an all-gather. Launch
shapes depend only on ``(rows, shard width)`` and nothing synchronizes with the
host, so every path is CUDA-graph capturable.
"""

from __future__ import annotations

import os
from typing import Callable, NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..distributed import AllReduce, AllReduceStrategy, allgather
from ..modules.embedding import LMHead
from ..modules.linear import TensorParallelMode
from ..pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

_INT32_MAX = tl.constexpr(2**31 - 1)
# Index published by a split without valid columns: exact in float32 and above
# every supported vocab id, so it never wins a tie.
_EMPTY_INDEX = tl.constexpr(2**24)
_MAX_VOCAB = 2**24
_LOCAL_BLOCK_V = 2048
_REDUCE_BLOCK_ROWS = 32
# Programs the local reductions aim for (rows x column splits): rows are few in
# decode, so each row is split over several programs.
_TARGET_PROGRAMS = 256
# Batch-dependent layout arguments stay unspecialized so varying batch sizes
# never recompile Triton variants mid-serving.
_PAIR_LAYOUT_ARGS = ["stride_out", "slot0", "zero_lo", "zero_hi"]


def _gathers_even_vocab_shards(lm_head: LMHead) -> bool:
    mapping = lm_head.mapping
    return (
        mapping.tp_size > 1
        and not mapping.enable_attention_dp
        and lm_head.tp_mode == TensorParallelMode.COLUMN
        and lm_head.gather_output
        and lm_head.padding_size == 0
        and lm_head.gather_output_sizes is None
        and lm_head.num_embeddings < _MAX_VOCAB
    )


def is_vocab_sharded_lm_head(lm_head: torch.nn.Module) -> bool:
    """Whether ``lm_head`` is an evenly vocab-sharded gathered TP ``LMHead``."""
    return type(lm_head) is LMHead and _gathers_even_vocab_shards(lm_head)


def _softcap_params(softcap: Optional[float]) -> tuple[bool, float, float]:
    """Return ``(has_softcap, inv_cap, cap)`` matching ``torch.tanh(x / cap) * cap``."""
    if softcap is None or softcap == 0.0:
        return False, 1.0, 1.0
    cap_f = float(softcap)
    inv_cap = float(np.float32(1.0) / np.float32(cap_f))
    return True, inv_cap, cap_f


def _use_libdevice_softcap() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") != "1"


@triton.jit
def _mul_rn_f32(a, b):
    """fp32 product rounded to nearest with subnormals kept, matching torch CUDA kernels."""
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _tanh_softcap_f32(x, inv_cap, cap, USE_LIBDEVICE: tl.constexpr):
    """Bit-identical to ``torch.tanh(x / cap) * cap`` on fp32 CUDA tensors.

    Triton links libdevice ``tanhf`` with flush-to-zero, so subnormal quotients
    (where ``tanhf`` is the identity) bypass ``libdevice.tanh``. Under
    ``TRITON_INTERPRET=1`` (CPU-only tests), falls back to an exact exp form
    because the Triton interpreter does not execute PTX inline asm or libdevice.
    """
    if USE_LIBDEVICE:
        y = _mul_rn_f32(x, inv_cap)
        t = tl.where(tl.abs(y) < 1.1754943508222875e-38, y, libdevice.tanh(y))
        return _mul_rn_f32(t, cap)
    y = x * inv_cap
    # Stable tanh(y) = sign(y) * (1 - exp(-2|y|)) / (1 + exp(-2|y|)) without overflow.
    ay = tl.abs(y)
    e = tl.exp(-2.0 * tl.minimum(ay, 20.0))
    t_abs = tl.where(ay >= 10.0, 1.0, (1.0 - e) / (1.0 + e))
    t = tl.where(y < 0.0, -t_abs, t_abs)
    # Preserve NaN: NaN comparisons are false, so NaN must propagate directly.
    t = tl.where(y != y, y, t)
    return t * cap


@triton.jit
def argmax_combine(val_a, idx_a, val_b, idx_b):
    """``tl.reduce`` combine in ``torch.argmax`` order: NaN beats any number,
    equal values go to the lower index."""
    nan_a = val_a != val_a
    nan_b = val_b != val_b
    take_a = tl.where(
        nan_a,
        (~nan_b) | (idx_a < idx_b),
        (~nan_b) & ((val_a > val_b) | ((val_a == val_b) & (idx_a < idx_b))),
    )
    return tl.where(take_a, val_a, val_b), tl.where(take_a, idx_a, idx_b)


@triton.jit
def _gumbel(r):
    # Standard Gumbel sample -log(-log(u)) from 32 random bits. The top 23 bits
    # map to the open interval (0, 1): (k + 0.5) / 2^23 is exact in float32, so
    # u never rounds to 0 or 1, and the sample lies in about [-2.8, 16.6].
    u = ((r >> 9).to(tl.float32) + 0.5) * (1.0 / 8388608.0)
    return -tl.log(-tl.log(u))


@triton.jit
def _global_index(best_idx, col_base):
    return tl.where(best_idx == _INT32_MAX, _EMPTY_INDEX, col_base + best_idx)


@triton.jit
def _store_pair(
    out_ptr,
    row,
    stride_out,
    slot,
    best_val,
    gidx,
    zero_lo,
    zero_hi,
    fill,
    ZERO_WIDTH: tl.constexpr,
):
    # Publishes (value, global index) at columns [2 * slot, 2 * slot + 1]. With
    # ZERO_WIDTH > 0 the program flagged `fill` also zeroes the row outside
    # [zero_lo, zero_hi), i.e. the other ranks' columns.
    base = out_ptr + row.to(tl.int64) * stride_out
    tl.store(base + 2 * slot, best_val)
    tl.store(base + 2 * slot + 1, gidx.to(tl.float32))
    if ZERO_WIDTH > 0:
        if fill:
            offs = tl.arange(0, ZERO_WIDTH)
            tl.store(
                base + offs,
                tl.zeros([ZERO_WIDTH], tl.float32),
                mask=(offs < zero_lo) | (offs >= zero_hi),
            )


@triton.jit(do_not_specialize=_PAIR_LAYOUT_ARGS)
def _local_argmax_kernel(
    x_ptr,
    stride_x,
    n_cols,
    col_base,
    chunk,
    inv_cap,
    cap,
    out_ptr,
    stride_out,
    slot0,
    zero_lo,
    zero_hi,
    HAS_SOFTCAP: tl.constexpr,
    USE_LIBDEVICE: tl.constexpr,
    ZERO_WIDTH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Program (row, split) reduces columns [split * chunk, (split + 1) * chunk).
    row = tl.program_id(0)
    split = tl.program_id(1)
    lo = split * chunk
    hi = tl.minimum(lo + chunk, n_cols)
    offs = tl.arange(0, BLOCK_V)
    x_row = x_ptr + row.to(tl.int64) * stride_x
    best_val = tl.full([BLOCK_V], float("-inf"), tl.float32)
    best_idx = tl.full([BLOCK_V], _INT32_MAX, tl.int32)
    for start in range(lo, hi, BLOCK_V):
        cols = start + offs
        mask = cols < hi
        vals = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_SOFTCAP:
            vals = tl.where(
                mask, _tanh_softcap_f32(vals, inv_cap, cap, USE_LIBDEVICE), float("-inf")
            )
        idx = tl.where(mask, cols, _INT32_MAX)
        best_val, best_idx = argmax_combine(best_val, best_idx, vals, idx)
    val, idx = tl.reduce((best_val, best_idx), 0, argmax_combine)
    _store_pair(
        out_ptr,
        row,
        stride_out,
        slot0 + split,
        val,
        _global_index(idx, col_base),
        zero_lo,
        zero_hi,
        split == 0,
        ZERO_WIDTH,
    )


@triton.jit(do_not_specialize=_PAIR_LAYOUT_ARGS)
def _local_sample_kernel(
    x_ptr,
    stride_x,
    n_cols,
    col_base,
    chunk,
    temp_ptr,
    seed_ptr,
    offset_ptr,
    greedy_threshold,
    inv_cap,
    cap,
    out_ptr,
    stride_out,
    slot0,
    zero_lo,
    zero_hi,
    HAS_SOFTCAP: tl.constexpr,
    USE_LIBDEVICE: tl.constexpr,
    ZERO_WIDTH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Gumbel-max over one split of this rank's shard. The noise of global vocab
    # index v is output (v % 4) of Philox(seed, counter=(v // 4, offset lo,
    # offset hi, 0)): the same on every rank, distinct per (seed, offset).
    # col_base and chunk are multiples of 4, so a 4-column group never
    # straddles a split or a shard.
    GROUPS: tl.constexpr = BLOCK_V // 4
    row = tl.program_id(0)
    split = tl.program_id(1)
    lo = split * chunk
    hi = tl.minimum(lo + chunk, n_cols)
    offs_g = tl.arange(0, GROUPS)
    offs_l = tl.arange(0, 4)
    x_row = x_ptr + row.to(tl.int64) * stride_x
    temperature = tl.load(temp_ptr + row).to(tl.float32)
    is_greedy = temperature <= greedy_threshold
    inv_t = 1.0 / tl.where(is_greedy, 1.0, temperature)
    seed = tl.load(seed_ptr + row).to(tl.int64)
    offset = tl.load(offset_ptr + row).to(tl.int64)
    c1 = tl.full([GROUPS], 0, tl.uint32) + (offset & 0xFFFFFFFF).to(tl.uint32)
    c2 = tl.full([GROUPS], 0, tl.uint32) + ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c3 = tl.full([GROUPS], 0, tl.uint32)
    best_val = tl.full([GROUPS, 4], float("-inf"), tl.float32)
    best_idx = tl.full([GROUPS, 4], _INT32_MAX, tl.int32)
    for start in range(lo, hi, BLOCK_V):
        cols = start + offs_g[:, None] * 4 + offs_l[None, :]
        mask = cols < hi
        vals = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_SOFTCAP:
            vals = tl.where(
                mask, _tanh_softcap_f32(vals, inv_cap, cap, USE_LIBDEVICE), float("-inf")
            )
        if not is_greedy:
            counter = ((col_base + start) // 4 + offs_g).to(tl.uint32)
            r0, r1, r2, r3 = tl.philox(seed, counter, c1, c2, c3)
            # [GROUPS, 2, 2] -> [GROUPS, 4] in lane order r0, r1, r2, r3.
            noise = tl.join(
                tl.join(_gumbel(r0), _gumbel(r2)), tl.join(_gumbel(r1), _gumbel(r3))
            ).reshape([GROUPS, 4])
            vals = tl.where(mask, vals * inv_t + noise, float("-inf"))
        idx = tl.where(mask, cols, _INT32_MAX)
        best_val, best_idx = argmax_combine(best_val, best_idx, vals, idx)
    best_val, best_idx = tl.reduce((best_val, best_idx), 1, argmax_combine)
    val, idx = tl.reduce((best_val, best_idx), 0, argmax_combine)
    _store_pair(
        out_ptr,
        row,
        stride_out,
        slot0 + split,
        val,
        _global_index(idx, col_base),
        zero_lo,
        zero_hi,
        split == 0,
        ZERO_WIDTH,
    )


@triton.jit(do_not_specialize=["stride_row", "n_rows", "n_slots"])
def _reduce_pairs_kernel(
    pairs_ptr,
    stride_row,
    n_rows,
    n_slots,
    out_ptr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    slots = tl.arange(0, BLOCK_SLOTS)
    mask = (rows[:, None] < n_rows) & (slots[None, :] < n_slots)
    ptrs = pairs_ptr + rows[:, None].to(tl.int64) * stride_row + slots[None, :] * 2
    vals = tl.load(ptrs, mask=mask, other=float("-inf"))
    gidx = tl.load(ptrs + 1, mask=mask, other=0.0).to(tl.int32)
    gidx = tl.where(mask, gidx, _INT32_MAX)
    _, best = tl.reduce((vals, gidx), 1, argmax_combine)
    tl.store(out_ptr + rows, best.to(out_ptr.dtype.element_ty), mask=rows < n_rows)


@triton.jit(do_not_specialize=["stride_row", "n_slots"])
def _markov_reduce_embed_kernel(
    pairs_ptr,
    stride_row,
    n_slots,
    anchor_ptr,
    w1_ptr,
    emb_ptr,
    tokens_ptr,
    stride_tokens,
    vocab,
    step,
    RANK: tl.constexpr,
    FROM_ANCHOR: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Resolves prev_step for one request -- the anchor at step 0, else the
    # global argmax of the previous step, which is also draft token step - 1 --
    # and gathers its markov_w1 row; an out-of-vocab prev gives a zero row.
    req = tl.program_id(0)
    if FROM_ANCHOR:
        prev = tl.load(anchor_ptr + req).to(tl.int64)
    else:
        slots = tl.arange(0, BLOCK_SLOTS)
        smask = slots < n_slots
        ptrs = pairs_ptr + req.to(tl.int64) * stride_row + slots * 2
        vals = tl.load(ptrs, mask=smask, other=float("-inf"))
        gidx = tl.load(ptrs + 1, mask=smask, other=0.0).to(tl.int32)
        gidx = tl.where(smask, gidx, _INT32_MAX)
        _, best = tl.reduce((vals, gidx), 0, argmax_combine)
        prev = best.to(tl.int64)
        tl.store(tokens_ptr + req.to(tl.int64) * stride_tokens + step - 1, prev.to(tl.int32))
    valid = (prev >= 0) & (prev < vocab)
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < RANK
    w1_row = tl.load(
        w1_ptr + tl.where(valid, prev, 0) * RANK + offs_r, mask=mask_r & valid, other=0.0
    )
    tl.store(emb_ptr + req * RANK + offs_r, w1_row, mask=mask_r)


@triton.jit(do_not_specialize=["stride_pairs", "n_slots", "hid_s_row"])
def _rnn_markov_reduce_input_kernel(
    pairs_ptr,
    stride_pairs,
    n_slots,
    anchor_ptr,
    emb_ptr,
    vocab,
    state_ptr,
    hid_ptr,
    hid_s_row,
    out_ptr,
    tokens_ptr,
    stride_tokens,
    step,
    RANK: tl.constexpr,
    HIDDEN: tl.constexpr,
    HID_BLOCK: tl.constexpr,
    FROM_ANCHOR: tl.constexpr,
    HAS_STATE: tl.constexpr,
    BUILD_INPUT: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Fused pair reduction + RNN Markov joint input ``[state | emb(prev) | hidden]``.

    At ``step == 0`` (``FROM_ANCHOR=True``), loads ``prev`` from ``anchor_ptr``.
    At ``step > 0`` (``FROM_ANCHOR=False``), reduces ``pairs`` to ``prev`` in
    ``torch.argmax`` order and stores ``tokens[req, step - 1] = prev``.
    When ``BUILD_INPUT=True``, also emits ``[state | emb(prev) | hidden]``.
    """
    req = tl.program_id(0).to(tl.int64)
    if FROM_ANCHOR:
        prev = tl.load(anchor_ptr + req).to(tl.int64)
    else:
        slots = tl.arange(0, BLOCK_SLOTS)
        smask = slots < n_slots
        ptrs = pairs_ptr + req * stride_pairs + slots * 2
        vals = tl.load(ptrs, mask=smask, other=float("-inf"))
        gidx = tl.load(ptrs + 1, mask=smask, other=0.0).to(tl.int32)
        gidx = tl.where(smask, gidx, _INT32_MAX)
        _, best = tl.reduce((vals, gidx), 0, argmax_combine)
        prev = best.to(tl.int64)
        tl.store(tokens_ptr + req * stride_tokens + step - 1, prev.to(tl.int32))
    if BUILD_INPUT:
        offs_r = tl.arange(0, BLOCK_R)
        mask_r = offs_r < RANK
        valid = (prev >= 0) & (prev < vocab)
        idx = tl.where(valid, prev, 0)
        emb = tl.load(emb_ptr + idx * RANK + offs_r, mask=mask_r, other=0.0)
        emb = tl.where(valid, emb, tl.zeros_like(emb))
        out = out_ptr + req * (2 * RANK + HIDDEN)
        if HAS_STATE:
            state = tl.load(state_ptr + req * RANK + offs_r, mask=mask_r, other=0.0)
        else:
            state = tl.zeros_like(emb)
        tl.store(out + offs_r, state, mask=mask_r)
        tl.store(out + RANK + offs_r, emb, mask=mask_r)
        for c in tl.static_range(0, HIDDEN, HID_BLOCK):
            hd = c + tl.arange(0, HID_BLOCK)
            hmask = hd < HIDDEN
            tl.store(
                out + 2 * RANK + hd,
                tl.load(hid_ptr + req * hid_s_row + hd, mask=hmask, other=0.0),
                mask=hmask,
            )


@triton.jit(do_not_specialize=_PAIR_LAYOUT_ARGS)
def _markov_add_local_argmax_kernel(
    base_ptr,
    stride_base,
    n_cols,
    col_base,
    chunk,
    bias_ptr,
    stride_bias,
    biased_ptr,
    stride_biased,
    inv_cap,
    cap,
    out_ptr,
    stride_out,
    slot0,
    zero_lo,
    zero_hi,
    HAS_SOFTCAP: tl.constexpr,
    USE_LIBDEVICE: tl.constexpr,
    STORE_BIASED: tl.constexpr,
    ZERO_WIDTH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # base (optionally soft-capped) + bias, both in float32 like the gathered
    # reference, optionally stored, then the argmax of one split.
    req = tl.program_id(0)
    split = tl.program_id(1)
    lo = split * chunk
    hi = tl.minimum(lo + chunk, n_cols)
    offs = tl.arange(0, BLOCK_V)
    base_row = base_ptr + req.to(tl.int64) * stride_base
    bias_row = bias_ptr + req.to(tl.int64) * stride_bias
    biased_row = biased_ptr + req.to(tl.int64) * stride_biased
    best_val = tl.full([BLOCK_V], float("-inf"), tl.float32)
    best_idx = tl.full([BLOCK_V], _INT32_MAX, tl.int32)
    for start in range(lo, hi, BLOCK_V):
        cols = start + offs
        mask = cols < hi
        base = tl.load(base_row + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_SOFTCAP:
            base = _tanh_softcap_f32(base, inv_cap, cap, USE_LIBDEVICE)
        bias = tl.load(bias_row + cols, mask=mask, other=0.0).to(tl.float32)
        summed = base + bias
        if STORE_BIASED:
            tl.store(biased_row + cols, summed, mask=mask)
        vals = tl.where(mask, summed, float("-inf"))
        idx = tl.where(mask, cols, _INT32_MAX)
        best_val, best_idx = argmax_combine(best_val, best_idx, vals, idx)
    val, idx = tl.reduce((best_val, best_idx), 0, argmax_combine)
    _store_pair(
        out_ptr,
        req,
        stride_out,
        slot0 + split,
        val,
        _global_index(idx, col_base),
        zero_lo,
        zero_hi,
        split == 0,
        ZERO_WIDTH,
    )


class _PairLayout(NamedTuple):
    """Where the local kernels of one exchange publish their pairs."""

    buffer: torch.Tensor  # [rows, width] float32
    slot0: int  # first pair slot owned by this rank
    zero_lo: int  # columns [zero_lo, zero_hi) are this rank's; the rest is zeroed
    zero_hi: int
    zero_width: int  # 0 when there is nothing to zero (a single shard)
    splits: int  # column splits per rank, i.e. pair slots per rank
    chunk: int  # columns per split


class _PairExchange:
    """Lays out and exchanges ``(value, global index)`` pairs across a TP group.

    Each rank owns ``splits`` consecutive pair slots of a zero-padded
    ``[rows, width]`` float32 buffer and one all-reduce sums the buffers. A
    single shard (``all_reduce`` None) publishes its pairs and exchanges nothing.
    """

    def __init__(self, all_reduce: Optional[AllReduce], tp_size: int, tp_rank: int):
        self.all_reduce = all_reduce
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    def layout(self, rows: int, n_cols: int, device: torch.device) -> _PairLayout:
        max_splits = triton.cdiv(n_cols, _LOCAL_BLOCK_V)
        splits = max(1, min(max_splits, triton.cdiv(_TARGET_PROGRAMS, max(rows, 1))))
        chunk = triton.cdiv(triton.cdiv(n_cols, splits), _LOCAL_BLOCK_V) * _LOCAL_BLOCK_V
        splits = max(1, triton.cdiv(n_cols, chunk))
        if self.tp_size == 1:
            buf = torch.empty((rows, 2 * splits), dtype=torch.float32, device=device)
            return _PairLayout(buf, 0, 0, 0, 0, splits, chunk)
        width = triton.next_power_of_2(2 * self.tp_size * splits)
        buf = torch.empty((rows, width), dtype=torch.float32, device=device)
        lo = 2 * self.tp_rank * splits
        return _PairLayout(buf, self.tp_rank * splits, lo, lo + 2 * splits, width, splits, chunk)

    def exchange(self, layout: _PairLayout) -> tuple[torch.Tensor, int]:
        """Every rank's pairs, ``[rows, width]``, and the number of pair slots."""
        if self.all_reduce is None:
            return layout.buffer, layout.splits
        return self.all_reduce(layout.buffer), self.tp_size * layout.splits


def _select_tokens(
    kernel,
    local: torch.Tensor,
    extra_args: tuple,
    col_base: int,
    exchange: _PairExchange,
    out_dtype: torch.dtype,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """Run a local selection ``kernel`` over ``[rows, n_cols]`` (a shard of the)
    logits, exchange the pairs and reduce them to one global id per row."""
    assert local.dim() == 2 and local.stride(-1) == 1
    rows, n_cols = local.shape
    has_softcap, inv_cap, cap = _softcap_params(softcap)
    layout = exchange.layout(rows, n_cols, local.device)
    if rows > 0:
        kernel[(rows, layout.splits)](
            local,
            local.stride(0),
            n_cols,
            col_base,
            layout.chunk,
            *extra_args,
            inv_cap,
            cap,
            layout.buffer,
            layout.buffer.stride(0),
            layout.slot0,
            layout.zero_lo,
            layout.zero_hi,
            HAS_SOFTCAP=has_softcap,
            USE_LIBDEVICE=_use_libdevice_softcap(),
            ZERO_WIDTH=layout.zero_width,
            BLOCK_V=_LOCAL_BLOCK_V,
            num_warps=8,
            enable_fp_fusion=not has_softcap,
        )
    pairs, n_slots = exchange.exchange(layout)
    out = torch.empty((rows,), dtype=out_dtype, device=local.device)
    if rows > 0:
        _reduce_pairs_kernel[(triton.cdiv(rows, _REDUCE_BLOCK_ROWS),)](
            pairs,
            pairs.stride(0),
            rows,
            n_slots,
            out,
            BLOCK_ROWS=_REDUCE_BLOCK_ROWS,
            BLOCK_SLOTS=triton.next_power_of_2(n_slots),
            num_warps=4,
        )
    return out


def _markov_chain_tokens(
    block_logits: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2_rows: torch.Tensor,
    col_base: int,
    exchange: _PairExchange,
    biased_out: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """Greedy vanilla-Markov block chain over (a shard of) the block logits."""
    num_reqs, block_size, n_cols = block_logits.shape
    device = block_logits.device
    tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
    if num_reqs == 0 or block_size == 0:
        return tokens
    assert markov_w1.is_contiguous() and block_logits.stride(-1) == 1
    vocab, rank = markov_w1.shape
    has_softcap, inv_cap, cap = _softcap_params(softcap)
    emb = torch.empty((num_reqs, rank), dtype=markov_w1.dtype, device=device)
    layout = exchange.layout(num_reqs, n_cols, device)
    pairs, n_slots = first_prev_tokens, 1
    for step in range(block_size + 1):
        _markov_reduce_embed_kernel[(num_reqs,)](
            pairs,
            pairs.stride(0) if step > 0 else 0,
            n_slots,
            first_prev_tokens,
            markov_w1,
            emb,
            tokens,
            tokens.stride(0),
            vocab,
            step,
            RANK=rank,
            FROM_ANCHOR=step == 0,
            BLOCK_SLOTS=triton.next_power_of_2(n_slots),
            BLOCK_R=triton.next_power_of_2(rank),
            num_warps=4,
        )
        if step == block_size:
            break
        bias = F.linear(emb, markov_w2_rows)
        step_logits = block_logits[:, step]
        step_biased = step_logits if biased_out is None else biased_out[:, step]
        _markov_add_local_argmax_kernel[(num_reqs, layout.splits)](
            step_logits,
            step_logits.stride(0),
            n_cols,
            col_base,
            layout.chunk,
            bias,
            bias.stride(0),
            step_biased,
            step_biased.stride(0),
            inv_cap,
            cap,
            layout.buffer,
            layout.buffer.stride(0),
            layout.slot0,
            layout.zero_lo,
            layout.zero_hi,
            HAS_SOFTCAP=has_softcap,
            USE_LIBDEVICE=_use_libdevice_softcap(),
            STORE_BIASED=biased_out is not None,
            ZERO_WIDTH=layout.zero_width,
            BLOCK_V=_LOCAL_BLOCK_V,
            num_warps=8,
            enable_fp_fusion=not has_softcap,
        )
        pairs, n_slots = exchange.exchange(layout)
    return tokens


def _default_rnn_markov_gate(
    joint: torch.Tensor, state: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    gate, candidate, emitted = joint.chunk(3, dim=-1)
    if state is None:
        state = torch.zeros_like(gate)
    gate = torch.sigmoid(gate)
    return gate * state + (1 - gate) * torch.tanh(candidate), torch.tanh(emitted)


def _default_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    return F.linear(x, weight, bias)


def rnn_markov_chain_tokens(
    block_logits: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    hidden_states: torch.Tensor,
    markov_embedding: torch.Tensor,
    markov_joint: torch.Tensor,
    markov_joint_bias: torch.Tensor,
    markov_output_rows: torch.Tensor,
    col_base: int,
    exchange: _PairExchange,
    *,
    softcap: Optional[float] = None,
    biased_out: Optional[torch.Tensor] = None,
    need_final_tokens: bool = True,
    linear_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    ] = None,
    gate_fn: Optional[
        Callable[[torch.Tensor, Optional[torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]
    ] = None,
) -> torch.Tensor:
    """Vocab-parallel greedy RNN-Markov block chain over shard block logits.

    Keeps the tiny rank-dimensional recurrence (``embedding`` lookup,
    ``joint_proj`` and gated state update) replicated across TP ranks while
    sharding the large ``markov_output`` projection (``[V / tp, rank]``) and
    fusing optional exact tanh soft-capping + bias add + local argmax on each
    rank's vocab shard. Each step exchanges only ``[B, 2 * splits * tp]``
    float32 pairs.

    Args:
        block_logits: ``[B, K, n_cols]`` block logits (unit vocab stride), raw
            bf16 with ``softcap`` or pre-capped fp32 with ``softcap=None``.
        first_prev_tokens: ``[B]`` anchor token ids.
        hidden_states: ``[B, K, hidden]`` draft backbone hidden states.
        markov_embedding: ``[vocab, rank]`` contiguous token embedding table.
        markov_joint: ``[3 * rank, hidden + 2 * rank]`` recurrent joint weight.
        markov_joint_bias: ``[3 * rank]`` recurrent joint bias.
        markov_output_rows: ``[n_cols, rank]`` this rank's vocab rows of
            ``draft_markov_output``.
        col_base: global vocab id of column 0.
        exchange: the TP pair exchange.
        softcap: optional tanh soft-cap applied to ``block_logits`` before
            adding the Markov bias.
        biased_out: optional ``[B, K, n_cols]`` float32 destination for the
            biased shard logits (unit vocab stride).
        need_final_tokens: if False (caller only consumes ``biased_out``),
            skips the final pair exchange and reduction after step ``K - 1``.
        linear_fn: optional ``(x, weight, bias) -> y`` linear callable.
        gate_fn: optional ``(joint, state) -> (new_state, emitted)`` callable.

    Returns:
        ``[B, K]`` int32 draft tokens (column ``K - 1`` is undefined when
        ``need_final_tokens=False``).
    """
    num_reqs, block_size, n_cols = block_logits.shape
    device = block_logits.device
    tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
    if num_reqs == 0 or block_size == 0:
        return tokens
    assert markov_embedding.is_contiguous() and block_logits.stride(-1) == 1
    vocab, rank = markov_embedding.shape
    hidden_dim = hidden_states.shape[-1]
    has_softcap, inv_cap, cap = _softcap_params(softcap)
    lin = _default_linear if linear_fn is None else linear_fn
    gate = _default_rnn_markov_gate if gate_fn is None else gate_fn

    joint_in = hidden_states.new_empty((num_reqs, 2 * rank + hidden_dim))
    layout = exchange.layout(num_reqs, n_cols, device)
    hid_block = min(2048, triton.next_power_of_2(hidden_dim))
    block_r = triton.next_power_of_2(rank)
    state: Optional[torch.Tensor] = None
    pairs: torch.Tensor = first_prev_tokens
    n_slots = 1
    total_steps = block_size + (1 if need_final_tokens else 0)
    for step in range(total_steps):
        build_input = step < block_size
        step_hidden = hidden_states[:, step] if build_input else hidden_states[:, 0]
        _rnn_markov_reduce_input_kernel[(num_reqs,)](
            pairs,
            pairs.stride(0) if step > 0 else 0,
            n_slots,
            first_prev_tokens,
            markov_embedding,
            vocab,
            state if state is not None else joint_in,
            step_hidden,
            step_hidden.stride(0),
            joint_in,
            tokens,
            tokens.stride(0),
            step,
            RANK=rank,
            HIDDEN=hidden_dim,
            HID_BLOCK=hid_block,
            FROM_ANCHOR=step == 0,
            HAS_STATE=state is not None,
            BUILD_INPUT=build_input,
            BLOCK_SLOTS=triton.next_power_of_2(n_slots),
            BLOCK_R=block_r,
            num_warps=4,
        )
        if not build_input:
            break
        joint = lin(joint_in, markov_joint, markov_joint_bias)
        state, emitted = gate(joint, state)
        local_bias = lin(emitted, markov_output_rows, None)
        step_logits = block_logits[:, step]
        step_biased = step_logits if biased_out is None else biased_out[:, step]
        _markov_add_local_argmax_kernel[(num_reqs, layout.splits)](
            step_logits,
            step_logits.stride(0),
            n_cols,
            col_base,
            layout.chunk,
            local_bias,
            local_bias.stride(0),
            step_biased,
            step_biased.stride(0),
            inv_cap,
            cap,
            layout.buffer,
            layout.buffer.stride(0),
            layout.slot0,
            layout.zero_lo,
            layout.zero_hi,
            HAS_SOFTCAP=has_softcap,
            USE_LIBDEVICE=_use_libdevice_softcap(),
            STORE_BIASED=biased_out is not None,
            ZERO_WIDTH=layout.zero_width,
            BLOCK_V=_LOCAL_BLOCK_V,
            num_warps=8,
            enable_fp_fusion=not has_softcap,
        )
        if step + 1 < total_steps:
            pairs, n_slots = exchange.exchange(layout)
    return tokens


def tp_pair_exchange(mapping: Mapping, strategy: AllReduceStrategy) -> _PairExchange:
    """The pair exchange over ``mapping``'s TP group, one float32 all-reduce
    with ``strategy``."""
    return _PairExchange(
        AllReduce(mapping, strategy=strategy, dtype=torch.float32),
        mapping.tp_size,
        mapping.tp_rank,
    )


def shard_argmax(
    local: torch.Tensor,
    col_base: int,
    exchange: _PairExchange,
    out_dtype: torch.dtype = torch.int64,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """Global argmax over the vocab of shard logits ``local`` ``[rows, shard]``
    (unit vocab stride) whose column 0 is global vocab id ``col_base``: the
    ids ``torch.argmax`` gives on the gathered (optionally soft-capped) logits."""
    return _select_tokens(
        _local_argmax_kernel, local, (), col_base, exchange, out_dtype, softcap=softcap
    )


def markov_chain_logits(
    base_logits: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    tp_exchange: Optional[_PairExchange] = None,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """The DSpark vanilla-Markov greedy block chain, returning the biased logits."""
    assert base_logits.dtype == torch.float32 or softcap is not None
    n_cols = base_logits.shape[-1]
    if tp_exchange is None:
        col_base, exchange = 0, _PairExchange(None, 1, 0)
    else:
        assert n_cols * tp_exchange.tp_size == markov_w2.shape[0]
        col_base, exchange = tp_exchange.tp_rank * n_cols, tp_exchange
    biased = torch.empty(base_logits.shape, dtype=torch.float32, device=base_logits.device)
    _markov_chain_tokens(
        base_logits,
        first_prev_tokens,
        markov_w1,
        markov_w2[col_base : col_base + n_cols],
        col_base,
        exchange,
        biased_out=biased,
        softcap=softcap,
    )
    return biased


class VocabParallelHead:
    """Token selection over a column-parallel LM head from this rank's vocab shard.

    Args:
        lm_head: an ``LMHead`` for which :meth:`supports` holds.
        softcap: optional final-logit tanh soft-cap (``cap * tanh(x / cap)``)
            applied to logits before selection or gathering.
        gather_fn: optional ``(local: Tensor) -> Tensor`` override to materialize
            full-vocab float32 (already soft-capped) logits on fallback paths.
    """

    def __init__(
        self,
        lm_head: LMHead,
        *,
        softcap: Optional[float] = None,
        gather_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.lm_head = lm_head
        mapping = lm_head.mapping
        self.vocab_size = lm_head.num_embeddings
        self.shard_width = lm_head.out_features
        self.col_base = mapping.tp_rank * self.shard_width
        self.softcap = float(softcap) if softcap else None
        self._gather_fn = gather_fn
        # The pairs carry token indices as float32 and must sum exactly, so the
        # exchange picks its own exact strategy rather than the model's
        # allreduce_strategy, which may be lossy (LOWPRECISION) or need user
        # buffers (UB).
        self._exchange = _PairExchange(
            AllReduce(mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.float32),
            mapping.tp_size,
            mapping.tp_rank,
        )
        logger.info(
            f"Vocab-parallel LM-head token selection: tp={mapping.tp_size}, "
            f"shard={self.shard_width}, vocab={self.vocab_size}, softcap={self.softcap}"
        )

    @property
    def exchange(self) -> _PairExchange:
        return self._exchange

    @staticmethod
    def supports(lm_head: torch.nn.Module) -> bool:
        """Whether ``lm_head`` is a gathered, evenly vocab-sharded TP ``LMHead``.

        Each shard must be a multiple of 4 columns (the sampler draws Philox
        noise in 4-column groups of the global vocab) and the vocab below
        ``2**24`` (indices travel as float32).
        """
        return (
            isinstance(lm_head, LMHead)
            and _gathers_even_vocab_shards(lm_head)
            and lm_head.out_features % 4 == 0
        )

    def local_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's ``[rows, shard]`` logits in the head's output dtype."""
        return self.lm_head.apply_linear(hidden_states, self.lm_head.bias)

    def shard_logits(self, hidden_states: torch.Tensor) -> VocabShardedLogits:
        """This rank's logits of ``hidden_states``, wrapped for a spec worker."""
        return VocabShardedLogits(self, self.local_logits(hidden_states))

    def gather_logits(self, local: torch.Tensor) -> torch.Tensor:
        """The gathered ``[rows, vocab]`` float32 (optionally soft-capped) logits."""
        if self._gather_fn is not None:
            return self._gather_fn(local)
        logits = allgather(local, self.lm_head.mapping, dim=-1).float()
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def argmax(self, local: torch.Tensor, out_dtype: torch.dtype = torch.int64) -> torch.Tensor:
        """Global argmax over the vocab of shard logits ``[rows, shard]``."""
        return shard_argmax(local, self.col_base, self._exchange, out_dtype, softcap=self.softcap)

    def sample(
        self,
        local: torch.Tensor,
        temperatures: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
        greedy_threshold: float,
    ) -> torch.Tensor:
        """One temperature-only draw per row of shard logits ``[rows, shard]``.

        Rows whose temperature is at or below ``greedy_threshold`` take the
        argmax. ``temperatures`` / ``seeds`` / ``offsets`` hold one entry per
        row (float32 / int64 / int64).
        """
        return _select_tokens(
            _local_sample_kernel,
            local,
            (temperatures, seeds, offsets, float(greedy_threshold)),
            self.col_base,
            self._exchange,
            torch.int64,
            softcap=self.softcap,
        )

    def markov_tokens(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy DSpark vanilla-Markov block draft tokens from shard block logits."""
        w2_rows = (
            markov_w2
            if markov_w2.shape[0] == self.shard_width
            else markov_w2[self.col_base : self.col_base + self.shard_width]
        )
        return _markov_chain_tokens(
            local_block,
            first_prev_tokens,
            markov_w1,
            w2_rows,
            self.col_base,
            self._exchange,
            softcap=self.softcap,
        )

    def rnn_markov_tokens(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        markov_embedding: torch.Tensor,
        markov_joint: torch.Tensor,
        markov_joint_bias: torch.Tensor,
        markov_output: torch.Tensor,
        *,
        linear_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
        ] = None,
        gate_fn: Optional[
            Callable[[torch.Tensor, Optional[torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> torch.Tensor:
        """Greedy DSpark RNN-Markov block draft tokens from shard block logits."""
        w_out_rows = (
            markov_output
            if markov_output.shape[0] == self.shard_width
            else markov_output[self.col_base : self.col_base + self.shard_width]
        )
        return rnn_markov_chain_tokens(
            local_block,
            first_prev_tokens,
            hidden_states,
            markov_embedding,
            markov_joint,
            markov_joint_bias,
            w_out_rows,
            self.col_base,
            self._exchange,
            softcap=self.softcap,
            linear_fn=linear_fn,
            gate_fn=gate_fn,
        )


class VocabShardedLogits:
    """This rank's vocab shard of the target logits, handed to a spec worker in
    place of the gathered float32 logits.

    The worker selects tokens from it (:meth:`select_tokens`) when the batch's
    sampling allows, and otherwise materializes the gathered logits
    (:meth:`gather`).
    """

    def __init__(self, head: VocabParallelHead, local: torch.Tensor):
        self.head = head
        self.local = local

    @property
    def device(self) -> torch.device:
        return self.local.device

    @property
    def num_rows(self) -> int:
        return self.local.shape[0]

    def gather(self) -> torch.Tensor:
        """The gathered ``[rows, vocab]`` float32 logits."""
        return self.head.gather_logits(self.local)

    def select_tokens(self, spec_metadata) -> torch.Tensor:
        """One target token per row: the argmax for an all-greedy batch, else a
        temperature-only draw (the caller guarantees the sampling mode)."""
        if spec_metadata.is_all_greedy_sample:
            return self.head.argmax(self.local)
        rows = self.num_rows
        return self.head.sample(
            self.local,
            spec_metadata.temperatures[:rows],
            spec_metadata.seeds[:rows],
            spec_metadata.offsets[:rows],
            GREEDY_TEMPERATURE_THRESHOLD,
        )
