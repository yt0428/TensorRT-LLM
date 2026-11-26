# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ML4 SM107 hist_shard_meta entry (32/64-token pages, ISS-8 and ISS-16)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.dsa._hist_topk import (  # noqa: E501
    shard_size,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.dsa.hist_shard_meta import (  # noqa: E501
    hist_shard_meta as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="requires VR200/SM107",
)


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    tpb = spec["tokens_per_block"]
    pages = -(-spec["kv_len"] // tpb) + 7
    row = torch.randperm(4 * pages, generator=gen)[:pages].to(torch.int32)
    width = -(-spec["capacity"] // tpb)
    return dict(
        kv_lens=torch.tensor([spec["kv_len"]], dtype=torch.int32, device="cuda"),
        block_table=row.cuda(),
        shard_ctx=torch.full((1,), -7, dtype=torch.int32, device="cuda"),
        shard_block_table=torch.full((1, width), -7, dtype=torch.int32, device="cuda"),
        group_size=spec["group_size"],
        rank=spec["rank"],
        tokens_per_block=tpb,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref_ctx, ref_table = op.reference(**args)
    ctx, table = op(**args)
    torch.cuda.synchronize()
    assert ctx is args["shard_ctx"] and table is args["shard_block_table"]
    assert torch.equal(ctx, ref_ctx), (ctx.tolist(), ref_ctx.tolist())
    assert torch.equal(table, ref_table)


@pytest.mark.parametrize("cell_idx", (0, 6), ids=("tp16_p32", "tp8_p32"))
def test_shards_of_all_ranks_tile_the_request(cell_idx: int):
    spec = op.CELLS[cell_idx].spec
    args = _build(spec, seed=61 + cell_idx)
    pages, keys = [], 0
    size = shard_size(spec["kv_len"], spec["group_size"])
    for rank in range(spec["group_size"]):
        ctx, table = op(**dict(args, rank=rank))
        n = int(ctx[0])
        held = min(max(spec["kv_len"] - rank * size, 0), size)
        assert n == max(held, 1)
        keys += held
        pages += table[0, : -(-held // spec["tokens_per_block"])].tolist()
    assert keys == spec["kv_len"]
    total_pages = -(-spec["kv_len"] // spec["tokens_per_block"])
    assert pages == args["block_table"][:total_pages].tolist()


def test_guard_refuses_a_two_dimensional_table():
    args = _build(op.CELLS[0].spec, seed=62)
    with pytest.raises(AssertionError, match="one-dimensional"):
        with validating(op):
            op(**dict(args, block_table=args["block_table"][None, :]))


def test_guard_refuses_a_shard_table_narrower_than_the_shard():
    args = _build(op.CELLS[0].spec, seed=63)
    with pytest.raises(AssertionError, match="shard_block_table holds"):
        with validating(op):
            op(**dict(args, shard_block_table=args["shard_block_table"][:, :100].contiguous()))
