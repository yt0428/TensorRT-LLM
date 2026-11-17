# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The scatter of `symm_logits_gather_softcap`: each rank's lm_head shard pushed into every
rank's symmetric slot. The slot's widened, soft-capped read is the catalog's
`_widen_softcap_kernel.softcap_gather_kernel`."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def scatter_shard_kernel(
    src_ptr, src_s_row, peers_ptr, rank, rows_cap, shard, WORLD: tl.constexpr, BLOCK_V: tl.constexpr
):
    """Write this rank's bf16 vocabulary shard of row `r` into block `rank` of every rank's
    `[world, rows_cap, shard]` symmetric slot, its own included: NVLink stores, no reads."""
    r = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    v = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = v < shard
    x = tl.load(src_ptr + r * src_s_row + v, mask=mask)
    offset = (rank * rows_cap + r) * shard + v
    for peer in tl.static_range(WORLD):
        dst = tl.load(peers_ptr + peer).to(tl.pointer_type(tl.bfloat16))
        tl.store(dst + offset, x, mask=mask)
