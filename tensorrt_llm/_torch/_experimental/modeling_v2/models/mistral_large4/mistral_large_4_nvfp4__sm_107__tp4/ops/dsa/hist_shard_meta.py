# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This rank's key shard of the generation request: its context length and block table."""

from __future__ import annotations

from typing import Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .. import _cpp_fakes  # noqa: F401 — registers torch.library fake ops
from ._hist_topk import shard_capacity, shard_size

#: ML4 TP16 c1: 16 ranks, 32-token pages, 1M + DSpark headroom capacity.
_ML4_TP16_P32 = dict(group_size=16, tokens_per_block=32, capacity=shard_capacity(1_048_584, 16))
#: ML4 TP16 c1 with 64-token pages.
_ML4_TP16_P64 = dict(group_size=16, tokens_per_block=64, capacity=shard_capacity(1_048_584, 16))
#: ML4 TP8 c1: 8 ranks, 32-token pages, 780K AgentX capacity (<= 8 * 98304 = 786432).
_ML4_TP8_P32 = dict(group_size=8, tokens_per_block=32, capacity=shard_capacity(786_432, 8))


def _spec(base: dict, kv_len: int, rank: int, **overrides) -> dict:
    spec = dict(base, kv_len=kv_len, rank=rank)
    spec.update(overrides)
    return spec


class GlmHistShardMeta(OpWrapper):
    """``shard_ctx[0]`` and ``shard_block_table[0, :]`` of rank ``rank`` for one request.

    With ``S = shard_size(kv_lens[0], group_size)`` (ceil(kv_len / group_size) rounded up to 128),
    the rank holds the keys [rank * S, rank * S + n), ``n = clamp(kv_len - rank * S, 0, S)``:

    - ``shard_ctx[0] = max(n, 1)`` (an empty shard scores one key no row reaches);
    - ``shard_block_table[0, j] = block_table[rank * S / tokens_per_block + j]`` for the shard's
      ``ceil(n / tokens_per_block)`` pages, 0 for the rest of the row.

    ``block_table`` is the request's INDEX_KEY block-table row (one-dimensional). One CTA. Returns
    ``(shard_ctx, shard_block_table)``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tp16 p32: a 530,000-key request on rank 0", spec=_spec(_ML4_TP16_P32, 530_000, 0)
        ),
        Cell(
            why="tp16 p32: the same request's last rank, a partial shard",
            spec=_spec(_ML4_TP16_P32, 530_000, 15),
        ),
        Cell(
            why="tp16 p32: full 1,048,584-key ML4 DSpark capacity on rank 15",
            spec=_spec(_ML4_TP16_P32, 1_048_584, 15),
        ),
        Cell(
            why="tp16 p32: short request where rank 13 holds no key",
            spec=_spec(_ML4_TP16_P32, 3_000, 13),
        ),
        Cell(
            why="tp16 p32: shard ending inside a 32-token page",
            spec=_spec(_ML4_TP16_P32, 1_000_003, 15),
        ),
        Cell(
            why="tp16 p64: full 1,048,584-key ML4 DSpark capacity on rank 15",
            spec=_spec(_ML4_TP16_P64, 1_048_584, 15),
        ),
        Cell(
            why="tp8 p32: 780,000-key AgentX request on rank 7",
            spec=_spec(_ML4_TP8_P32, 780_000, 7),
        ),
        Cell(
            why="tp8 p32: max ISS-8 786,432-key capacity on rank 7",
            spec=_spec(_ML4_TP8_P32, 786_432, 7),
        ),
    )

    note = """
    Metadata consumed: none. The target passes the generation request's indexer KV length (int32,
    on the device), its row of the INDEX_KEY block offsets, persistent ``shard_ctx`` [1] and
    ``shard_block_table`` [1, ceil(capacity / tokens_per_block)] int32 buffers (captured CUDA
    graphs read them), the group size (8 or 16), this rank's TP rank and the page size (32 or 64).

    The kernel refuses a group above 16 ranks and a page size that does not divide 128.
    """

    def __call__(
        self,
        kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        shard_ctx: torch.Tensor,
        shard_block_table: torch.Tensor,
        group_size: int,
        rank: int,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.ops.trtllm.glm_dsa_hist_shard_meta(
            kv_lens, block_table, shard_ctx, shard_block_table, group_size, rank, tokens_per_block
        )
        return shard_ctx, shard_block_table

    def reference(
        self,
        kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        shard_ctx: torch.Tensor,
        shard_block_table: torch.Tensor,
        group_size: int,
        rank: int,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fresh ``(shard_ctx, shard_block_table)`` computed on the host."""
        kv_len = int(kv_lens[0])
        size = shard_size(kv_len, group_size)
        base = rank * size
        n = min(max(kv_len - base, 0), size)
        first_page = base // tokens_per_block
        pages = -(-n // tokens_per_block)
        row = block_table.tolist()
        width = shard_block_table.numel()
        table = [
            row[first_page + j] if j < pages and first_page + j < len(row) else 0
            for j in range(width)
        ]
        ctx = torch.tensor([max(n, 1)], dtype=torch.int32, device=shard_ctx.device)
        out = torch.tensor(table, dtype=torch.int32, device=shard_block_table.device)
        return ctx, out.view(shard_block_table.shape)

    def is_valid(
        self,
        kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        shard_ctx: torch.Tensor,
        shard_block_table: torch.Tensor,
        group_size: int,
        rank: int,
        tokens_per_block: int,
    ) -> None:
        assert block_table.dim() == 1, "block_table must be the request's one-dimensional row"
        kv_len = int(kv_lens[0])
        size = shard_size(kv_len, group_size)
        n = min(max(kv_len - rank * size, 0), size)
        assert shard_block_table.numel() * tokens_per_block >= n, (
            f"shard_block_table holds {shard_block_table.numel()} pages, the shard {n} keys"
        )
        if n > 0:
            first_page = rank * size // tokens_per_block
            assert first_page + -(-n // tokens_per_block) <= block_table.numel(), (
                "the shard's pages run past the request's block-table row"
            )


hist_shard_meta = GlmHistShardMeta()
