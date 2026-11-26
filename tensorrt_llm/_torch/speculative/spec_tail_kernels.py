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
"""Triton kernels for the bookkeeping of a one-model speculative-decoding step.

Each kernel replaces a chain of small eager PyTorch ops with one launch and
produces the same values:

* :func:`strict_accept` -- strict (token-equality) acceptance of the previous
  draft against the sampled target tokens.
* :func:`decode_draft_block_table` -- the DFlash draft-KV block-table decode
  (extent per request, clamped and divided block offsets).
* :func:`dflash_paged_latent_prepare` -- the per-request bookkeeping of a
  DFlash generation step whose drafter context is a paged single-latent cache.
* :func:`store_context_latents` -- the masked or unmasked store of the context
  latents of every drafter layer into its page pool.
* :func:`compute_gen_ctx_positions` -- pre-step context RoPE positions for the
  overlapped context-projection stream.
* :func:`prepare_next_new_tokens` -- fused ``[last_accepted | next_draft]``
  assembly for the overlap scheduler.

Launch shapes depend only on host batch sizes, so all of them are CUDA-graph
capturable.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

_BLOCK_TABLE_BLOCK = 2048
# Rows per program of _strict_accept_kernel. Fixed, like the arguments in its
# do_not_specialize list, so one compiled variant per draft length serves every
# batch without mid-serving recompiles.
_STRICT_ACCEPT_BLOCK_B = 128


@triton.jit(
    do_not_specialize=["draft_ptr", "stride_draft_row", "stride_draft_col", "num_ctx", "num_gen"]
)
def _strict_accept_kernel(
    target_ptr,
    draft_ptr,
    stride_draft_row,
    stride_draft_col,
    accepted_ptr,
    num_accepted_ptr,
    num_ctx,
    num_gen,
    DRAFT_LEN: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    width = DRAFT_LEN + 1
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)

    # Context requests: accept the sampled token only.
    ctx_mask = rows < num_ctx
    t_ctx = tl.load(target_ptr + rows, mask=ctx_mask, other=0)
    tl.store(accepted_ptr + rows * width, t_ctx.to(tl.int32), mask=ctx_mask)
    tl.store(num_accepted_ptr + rows, tl.full([BLOCK_B], 1, tl.int32), mask=ctx_mask)

    # Generation requests: the accepted row is the K+1 target tokens, and the
    # count is 1 + the number of leading draft tokens equal to them.
    gen_mask = rows < num_gen
    tile_mask = gen_mask[:, None] & (cols[None, :] < width)
    t_gen = tl.load(
        target_ptr + num_ctx + rows[:, None] * width + cols[None, :], mask=tile_mask, other=0
    ).to(tl.int32)
    tl.store(
        accepted_ptr + (num_ctx + rows[:, None]) * width + cols[None, :], t_gen, mask=tile_mask
    )
    draft_mask = gen_mask[:, None] & (cols[None, :] < DRAFT_LEN)
    draft = tl.load(
        draft_ptr + rows[:, None] * stride_draft_row + cols[None, :] * stride_draft_col,
        mask=draft_mask,
        other=0,
    ).to(tl.int32)
    mismatch = draft_mask & (draft != t_gen)
    first_mismatch = tl.min(tl.where(mismatch, cols[None, :], DRAFT_LEN), axis=1)
    tl.store(num_accepted_ptr + num_ctx + rows, (1 + first_mismatch).to(tl.int32), mask=gen_mask)


def strict_accept(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict acceptance of a uniform-width draft in one launch.

    Args:
        target_tokens: ``[num_contexts + num_gens * (K + 1)]`` sampled target
            tokens (any integer dtype).
        draft_tokens: ``[num_gens, K]`` previous draft tokens.
        num_contexts, batch_size: request counts.
    Returns:
        ``accepted_tokens [batch_size, K + 1]`` and ``num_accepted_tokens
        [batch_size]``, both int32. Context rows define only column 0.
    """
    num_gens = batch_size - num_contexts
    draft_len = draft_tokens.shape[-1]
    device = target_tokens.device
    accepted = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
    num_accepted = torch.empty((batch_size,), dtype=torch.int32, device=device)
    if batch_size == 0:
        return accepted, num_accepted
    block_b = _STRICT_ACCEPT_BLOCK_B
    grid = (triton.cdiv(max(num_contexts, num_gens, 1), block_b),)
    _strict_accept_kernel[grid](
        target_tokens,
        draft_tokens,
        draft_tokens.stride(0),
        draft_tokens.stride(1),
        accepted,
        num_accepted,
        num_contexts,
        num_gens,
        DRAFT_LEN=draft_len,
        BLOCK_B=block_b,
        BLOCK_K=triton.next_power_of_2(draft_len + 1),
        num_warps=4,
    )
    return accepted, num_accepted


@triton.jit
def _decode_block_table_kernel(
    src_ptr,
    stride_src_row,
    stride_src_col,
    n_blocks,
    divisor,
    block_offset,
    table_ptr,
    stride_table,
    counts_ptr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    src_row = src_ptr + row.to(tl.int64) * stride_src_row
    table_row = table_ptr + row.to(tl.int64) * stride_table
    extent = tl.zeros([BLOCK], dtype=tl.int64)
    for start in range(0, n_blocks, BLOCK):
        cols = start + offs
        mask = cols < n_blocks
        encoded = tl.load(src_row + cols * stride_src_col, mask=mask, other=-1).to(tl.int64)
        encoded -= block_offset
        extent = tl.maximum(extent, tl.where(encoded >= 0, cols.to(tl.int64) + 1, 0))
        tl.store(table_row + cols, (tl.maximum(encoded, 0) // divisor).to(tl.int32), mask=mask)
    tl.store(counts_ptr + row, tl.max(extent, axis=0))


def decode_draft_block_table(
    src: torch.Tensor,
    divisor: int,
    table: torch.Tensor,
    counts: torch.Tensor,
    *,
    block_offset: int = 0,
) -> None:
    """Decode layer-relative block offsets without modifying ``src``.

    For ``encoded = src - block_offset``, computes
    ``table[i] = encoded[i].clamp(min=0) // divisor`` and ``counts[i]`` = one
    past the last ``j`` with ``encoded[i, j] >= 0`` (0 when none).
    """
    rows, n_blocks = src.shape
    assert table.shape[-1] == n_blocks and table.stride(-1) == 1
    if rows == 0:
        return
    _decode_block_table_kernel[(rows,)](
        src,
        src.stride(0),
        src.stride(1),
        n_blocks,
        int(divisor),
        int(block_offset),
        table,
        table.stride(0),
        counts,
        BLOCK=min(_BLOCK_TABLE_BLOCK, triton.next_power_of_2(max(n_blocks, 1))),
        num_warps=4,
    )


class DraftBlockPlacement(NamedTuple):
    """Where a paged drafter context places each request's own block."""

    pages: torch.Tensor  # [G, block] int64
    offsets: torch.Tensor  # [G, block] int64
    seq_lens: torch.Tensor  # [G] int32
    kv_bounds: torch.Tensor  # [G * block] int32


class PagedLatentPrepare(NamedTuple):
    """Outputs of :func:`dflash_paged_latent_prepare` for ``G`` generation requests."""

    bonus: torch.Tensor  # [G] int64, the last accepted token
    query_positions: torch.Tensor  # [G, block] int64
    ctx_positions: torch.Tensor  # [G * width] int64, RoPE positions of the target latents
    store_pages: torch.Tensor  # [G * width] int64, destination page of each latent
    store_offsets: torch.Tensor  # [G * width] int64, offset inside that page
    store_mask: torch.Tensor  # [G * width] int8, 1 for accepted tokens
    num_ctx: torch.Tensor  # [G] int64, context length the block attends
    blk_pages: Optional[torch.Tensor]  # [G, block] int64 or None
    blk_offsets: Optional[torch.Tensor]  # [G, block] int64 or None
    blk_seq_lens: Optional[torch.Tensor]  # [G] int32 or None
    blk_kv_bounds: Optional[torch.Tensor]  # [G * block] int32 or None


@triton.jit(
    do_not_specialize=["num_gens", "first_row"],
    do_not_specialize_on_alignment=["slots_ptr"],
)
def _paged_latent_prepare_kernel(
    num_accepted_ptr,
    accepted_ptr,
    stride_accepted,
    slots_ptr,
    ctx_len_ptr,
    block_counts_ptr,
    table_ptr,
    stride_table,
    max_blocks,
    seq_lens_ptr,
    bonus_ptr,
    query_pos_ptr,
    ctx_pos_ptr,
    store_pages_ptr,
    store_offsets_ptr,
    store_mask_ptr,
    num_ctx_ptr,
    blk_pages_ptr,
    blk_offsets_ptr,
    blk_seq_lens_ptr,
    blk_kv_bounds_ptr,
    num_gens,
    first_row,
    dummy_slot,
    max_ctx,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_W: tl.constexpr,
    USE_COUNTS: tl.constexpr,
    COMPUTE_BLK: tl.constexpr,
):
    g = tl.arange(0, BLOCK_G)
    j = tl.arange(0, BLOCK_W)
    gm = g < num_gens
    row = first_row + g
    nacc = tl.load(num_accepted_ptr + row, mask=gm, other=1).to(tl.int64)
    slot = tl.load(slots_ptr + g, mask=gm, other=0).to(tl.int64)
    old = tl.load(ctx_len_ptr + slot, mask=gm, other=0).to(tl.int64)
    if USE_COUNTS:
        capacity = tl.load(block_counts_ptr + row, mask=gm, other=0).to(tl.int64) * PAGE_SIZE
        trow_idx = row
    else:
        capacity = tl.full([BLOCK_G], max_ctx, tl.int64)
        trow_idx = slot
    table_row = table_ptr + trow_idx.to(tl.int64) * stride_table

    bonus = tl.load(
        accepted_ptr + row.to(tl.int64) * stride_accepted + tl.maximum(nacc - 1, 0),
        mask=gm,
        other=0,
    )
    tl.store(bonus_ptr + g, bonus.to(tl.int64), mask=gm)

    # The block continues from the context length advanced by this step.
    now = tl.minimum(old + nacc, max_ctx)
    qmask = gm[:, None] & (j[None, :] < BLOCK_SIZE)
    blk = g[:, None] * BLOCK_SIZE + j[None, :]
    tl.store(query_pos_ptr + blk, now[:, None] + j[None, :], mask=qmask)

    # Context latents of this step's target tokens: RoPE position, destination
    # (column clamped into the request's allocation) and write mask.
    wmask = gm[:, None] & (j[None, :] < WIDTH)
    flat = g[:, None] * WIDTH + j[None, :]
    pos = old[:, None] + j[None, :]
    tl.store(ctx_pos_ptr + flat, pos, mask=wmask)
    col = tl.maximum(tl.minimum(pos, capacity[:, None] - 1), 0)
    page = tl.load(table_row[:, None] + tl.minimum(col // PAGE_SIZE, max_blocks - 1), mask=wmask)
    tl.store(store_pages_ptr + flat, page.to(tl.int64), mask=wmask)
    tl.store(store_offsets_ptr + flat, col % PAGE_SIZE, mask=wmask)
    tl.store(store_mask_ptr + flat, (j[None, :] < nacc[:, None]).to(tl.int8), mask=wmask)

    # Advance the context length; the shared dummy slot stays empty.
    new_len = tl.where(slot == dummy_slot, 0, now)
    tl.debug_barrier()
    tl.store(ctx_len_ptr + slot, new_len, mask=gm)
    tl.store(ctx_len_ptr + dummy_slot, tl.zeros([], dtype=tl.int64))

    if USE_COUNTS:
        num_ctx = tl.minimum(new_len, tl.maximum(capacity - BLOCK_SIZE, 0))
    else:
        num_ctx = new_len
    tl.store(num_ctx_ptr + g, num_ctx, mask=gm)
    tl.store(
        seq_lens_ptr + row,
        tl.full([BLOCK_G], WIDTH, tl.int32).to(seq_lens_ptr.dtype.element_ty),
        mask=gm,
    )

    if COMPUTE_BLK:
        bpos = num_ctx[:, None] + j[None, :]
        bentry = tl.minimum(bpos // PAGE_SIZE, max_blocks - 1)
        bpage = tl.load(table_row[:, None] + bentry, mask=qmask)
        tl.store(blk_pages_ptr + blk, bpage.to(tl.int64), mask=qmask)
        tl.store(blk_offsets_ptr + blk, bpos % PAGE_SIZE, mask=qmask)
        seq_len = (num_ctx + BLOCK_SIZE).to(tl.int32)
        tl.store(blk_seq_lens_ptr + g, seq_len, mask=gm)
        tl.store(
            blk_kv_bounds_ptr + blk,
            tl.zeros([BLOCK_G, BLOCK_W], tl.int32) + seq_len[:, None],
            mask=qmask,
        )


def dflash_paged_latent_prepare(
    num_accepted_tokens: torch.Tensor,
    accepted_tokens: torch.Tensor,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    block_counts: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    seq_lens_cuda: torch.Tensor,
    first_row: int,
    dummy_slot: int,
    max_ctx: int,
    page_size: int,
    block_size: int,
    width: int,
    *,
    compute_block_placement: bool = False,
) -> PagedLatentPrepare:
    """Fused DFlash generation-step bookkeeping for a paged single-latent context.

    Replaces the eager bonus-gather, position construction, page-table lookup,
    write-mask computation, ``ctx_len`` advance (with dummy-slot zeroing),
    advertised context clamping, and ``seq_lens_cuda`` update with one launch.
    """
    num_gens = slots.shape[0]
    device = ctx_len.device
    i64 = dict(dtype=torch.int64, device=device)
    bonus = torch.empty((num_gens,), **i64)
    query_positions = torch.empty((num_gens, block_size), **i64)
    ctx_positions = torch.empty((num_gens * width,), **i64)
    store_pages = torch.empty((num_gens * width,), **i64)
    store_offsets = torch.empty((num_gens * width,), **i64)
    store_mask = torch.empty((num_gens * width,), dtype=torch.int8, device=device)
    num_ctx = torch.empty((num_gens,), **i64)
    if compute_block_placement:
        blk_pages = torch.empty((num_gens, block_size), **i64)
        blk_offsets = torch.empty((num_gens, block_size), **i64)
        blk_seq_lens = torch.empty((num_gens,), dtype=torch.int32, device=device)
        blk_kv_bounds = torch.empty((num_gens * block_size,), dtype=torch.int32, device=device)
    else:
        blk_pages = blk_offsets = blk_seq_lens = blk_kv_bounds = None

    out = PagedLatentPrepare(
        bonus=bonus,
        query_positions=query_positions,
        ctx_positions=ctx_positions,
        store_pages=store_pages,
        store_offsets=store_offsets,
        store_mask=store_mask,
        num_ctx=num_ctx,
        blk_pages=blk_pages,
        blk_offsets=blk_offsets,
        blk_seq_lens=blk_seq_lens,
        blk_kv_bounds=blk_kv_bounds,
    )
    if num_gens == 0:
        return out
    assert accepted_tokens.stride(-1) == 1 and block_tables.stride(-1) == 1
    use_counts = block_counts is not None
    _paged_latent_prepare_kernel[(1,)](
        num_accepted_tokens,
        accepted_tokens,
        accepted_tokens.stride(0),
        slots,
        ctx_len,
        block_counts if use_counts else ctx_len,
        block_tables,
        block_tables.stride(0),
        block_tables.shape[1],
        seq_lens_cuda,
        bonus,
        query_positions,
        ctx_positions,
        store_pages,
        store_offsets,
        store_mask,
        num_ctx,
        blk_pages if compute_block_placement else bonus,
        blk_offsets if compute_block_placement else bonus,
        blk_seq_lens if compute_block_placement else seq_lens_cuda,
        blk_kv_bounds if compute_block_placement else seq_lens_cuda,
        num_gens,
        first_row,
        dummy_slot,
        max_ctx,
        PAGE_SIZE=page_size,
        BLOCK_SIZE=block_size,
        WIDTH=width,
        BLOCK_G=triton.next_power_of_2(num_gens),
        BLOCK_W=triton.next_power_of_2(max(width, block_size)),
        USE_COUNTS=use_counts,
        COMPUTE_BLK=compute_block_placement,
        num_warps=4,
    )
    return out


@triton.jit
def _store_context_latents_kernel(
    latent_ptr,
    stride_latent_tok,
    stride_latent_layer,
    pool_ptrs_ptr,
    stride_pool_page,
    stride_pool_off,
    pages_ptr,
    offsets_ptr,
    mask_ptr,
    head_dim,
    HAS_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tok = tl.program_id(0)
    layer = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    dmask = offs < head_dim
    page = tl.load(pages_ptr + tok).to(tl.int64)
    off = tl.load(offsets_ptr + tok).to(tl.int64)
    src = (
        latent_ptr + tok.to(tl.int64) * stride_latent_tok + layer.to(tl.int64) * stride_latent_layer
    )
    vals = tl.load(src + offs, mask=dmask, other=0.0)
    if HAS_MASK:
        keep = tl.load(mask_ptr + tok).to(tl.float32)
        vals = (vals.to(tl.float32) * keep).to(vals.dtype)
    pool = tl.load(pool_ptrs_ptr + layer).to(tl.pointer_type(vals.dtype))
    dst_off = page * stride_pool_page + off * stride_pool_off
    tl.store(pool + dst_off + offs, vals, mask=dmask)


def store_context_latents(
    latent: torch.Tensor,
    pools: list[torch.Tensor],
    pool_ptrs: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """``pools[l][pages[t], 0, 0, offsets[t]] = latent[t, l] (* mask[t])`` for
    every token ``t`` and layer ``l`` in one launch."""
    num_tokens, num_layers, head_dim = latent.shape
    if num_tokens == 0:
        return
    pool0 = pools[0]
    assert latent.stride(-1) == 1 and pool0.stride(-1) == 1
    assert all(p.stride() == pool0.stride() and p.dtype == latent.dtype for p in pools)
    has_mask = mask is not None
    _store_context_latents_kernel[(num_tokens, num_layers)](
        latent,
        latent.stride(0),
        latent.stride(1),
        pool_ptrs,
        pool0.stride(0),
        pool0.stride(3),
        pages,
        offsets,
        mask if has_mask else pages,
        head_dim,
        HAS_MASK=has_mask,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


@triton.jit
def _ctx_scatter_table_kernel(
    latent_ptr,
    stride_latent_tok,
    stride_latent_layer,
    pool_ptrs_ptr,
    stride_pool_page,
    stride_pool_off,
    rows_ptr,
    pos_ptr,
    table_ptr,
    stride_table,
    page_size,
    head_dim,
    BLOCK_D: tl.constexpr,
):
    """Unmasked single-latent context scatter resolving pages via ``table``."""
    tok = tl.program_id(0)
    layer = tl.program_id(1)
    row = tl.load(rows_ptr + tok).to(tl.int64)
    pos = tl.load(pos_ptr + tok).to(tl.int64)
    page = tl.load(table_ptr + row * stride_table + pos // page_size).to(tl.int64)
    off = pos % page_size
    offs = tl.arange(0, BLOCK_D)
    dmask = offs < head_dim
    src = (
        latent_ptr + tok.to(tl.int64) * stride_latent_tok + layer.to(tl.int64) * stride_latent_layer
    )
    vals = tl.load(src + offs, mask=dmask, other=0.0)
    pool = tl.load(pool_ptrs_ptr + layer).to(tl.pointer_type(vals.dtype))
    dst_off = page * stride_pool_page + off * stride_pool_off
    tl.store(pool + dst_off + offs, vals, mask=dmask)


def scatter_context_latents(
    latent: torch.Tensor,
    pools: list[torch.Tensor],
    pool_ptrs: torch.Tensor,
    rows: torch.Tensor,
    positions: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> None:
    """Prefill single-latent scatter into ``pools`` through ``table`` in one launch."""
    num_tokens, num_layers, head_dim = latent.shape
    if num_tokens == 0:
        return
    pool0 = pools[0]
    assert latent.stride(-1) == 1 and pool0.stride(-1) == 1 and table.stride(-1) == 1
    assert all(p.stride() == pool0.stride() and p.dtype == latent.dtype for p in pools)
    _ctx_scatter_table_kernel[(num_tokens, num_layers)](
        latent,
        latent.stride(0),
        latent.stride(1),
        pool_ptrs,
        pool0.stride(0),
        pool0.stride(3),
        rows,
        positions,
        table,
        table.stride(0),
        page_size,
        head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


@triton.jit(do_not_specialize=["num_gens"])
def _gen_ctx_positions_kernel(
    slots_ptr,
    ctx_len_ptr,
    out_ptr,
    num_gens,
    WIDTH: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    g = tl.arange(0, BLOCK_G)
    j = tl.arange(0, BLOCK_W)
    gm = g < num_gens
    slot = tl.load(slots_ptr + g, mask=gm, other=0).to(tl.int64)
    old = tl.load(ctx_len_ptr + slot, mask=gm, other=0).to(tl.int64)
    wmask = gm[:, None] & (j[None, :] < WIDTH)
    tl.store(out_ptr + g[:, None] * WIDTH + j[None, :], old[:, None] + j[None, :], mask=wmask)


def compute_gen_ctx_positions(
    slots: torch.Tensor, ctx_len: torch.Tensor, width: int
) -> torch.Tensor:
    """``(ctx_len[slots, None] + arange(width)).reshape(-1)`` in one launch."""
    num_gens = slots.shape[0]
    out = torch.empty((num_gens * width,), dtype=torch.int64, device=ctx_len.device)
    if num_gens == 0:
        return out
    _gen_ctx_positions_kernel[(1,)](
        slots,
        ctx_len,
        out,
        num_gens,
        WIDTH=width,
        BLOCK_G=triton.next_power_of_2(num_gens),
        BLOCK_W=triton.next_power_of_2(max(width, 1)),
        num_warps=4,
    )
    return out


@triton.jit(do_not_specialize=["batch_size", "stride_acc", "stride_draft"])
def _prepare_next_new_tokens_kernel(
    accepted_ptr,
    stride_acc,
    draft_ptr,
    stride_draft,
    num_accepted_ptr,
    out_ptr,
    batch_size,
    DRAFT_LEN: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    rmask = rows < batch_size
    nacc = tl.load(num_accepted_ptr + rows, mask=rmask, other=1).to(tl.int64)
    last_idx = tl.maximum(nacc - 1, 0)
    last_tok = tl.load(
        accepted_ptr + rows.to(tl.int64) * stride_acc + last_idx, mask=rmask, other=0
    ).to(tl.int32)
    width = DRAFT_LEN + 1
    tl.store(out_ptr + rows.to(tl.int64) * width, last_tok, mask=rmask)
    cols = tl.arange(0, BLOCK_K)
    cmask = rmask[:, None] & (cols[None, :] < DRAFT_LEN)
    draft = tl.load(
        draft_ptr + rows[:, None].to(tl.int64) * stride_draft + cols[None, :],
        mask=cmask,
        other=0,
    ).to(tl.int32)
    tl.store(out_ptr + rows[:, None].to(tl.int64) * width + 1 + cols[None, :], draft, mask=cmask)


def prepare_next_new_tokens(
    accepted_tokens: torch.Tensor,
    next_draft_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """``concat([accepted[i, num_accepted[i] - 1], next_draft[i]], dim=1)`` in one launch."""
    draft_len = next_draft_tokens.shape[1]
    out = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=accepted_tokens.device)
    if batch_size == 0:
        return out
    assert accepted_tokens.stride(-1) == 1 and next_draft_tokens.stride(-1) == 1
    block_b = _STRICT_ACCEPT_BLOCK_B
    grid = (triton.cdiv(batch_size, block_b),)
    _prepare_next_new_tokens_kernel[grid](
        accepted_tokens,
        accepted_tokens.stride(0),
        next_draft_tokens,
        next_draft_tokens.stride(0),
        num_accepted_tokens,
        out,
        batch_size,
        DRAFT_LEN=draft_len,
        BLOCK_B=block_b,
        BLOCK_K=triton.next_power_of_2(max(draft_len, 1)),
        num_warps=4,
    )
    return out
