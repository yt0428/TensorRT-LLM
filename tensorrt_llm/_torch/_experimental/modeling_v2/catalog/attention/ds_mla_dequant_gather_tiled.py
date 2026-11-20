# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather context requests' fp8_ds_mla latent rows out of the paged pool, dequantized to bf16,
a tile of rows per program."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._op import Arch, Cell
from .ds_mla_dequant_gather import DsMlaDequantGather

#: Workspace rows per program and its warps: 8 rows x 656 bytes in and 8 x 1152 bytes out over
#: 64 threads, every access a 16-byte vector. Of 8 / 16 / 32 / 64 rows at 2 / 4 / 8 warps, the
#: fastest on sm_107 at each AgentX step size the cells name (`note`).
ROWS = 8
WARPS = 2


@triton.jit
def _gather_tiled_kernel(
    pool_ptr,  # uint8 [blocks, owners, tpb, 656]
    owners,
    row,
    kv_req_ptr,  # int32 [N]
    kv_pos_ptr,  # int32 [N]
    block_table_ptr,
    bt_stride,
    out_ptr,  # bf16 [N, 576] (row stride out_stride)
    out_stride,
    n,
    TPB: tl.constexpr,
    ROWS: tl.constexpr,
):
    w = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = w < n
    req = tl.load(kv_req_ptr + w, mask=live, other=0).to(tl.int64)
    pos = tl.load(kv_pos_ptr + w, mask=live, other=0).to(tl.int64)
    block = tl.load(block_table_ptr + req * bt_stride + pos // TPB, mask=live, other=0)
    # Every row starts on a 16-byte boundary (656 = 41 x 16), so each access below is a
    # 16-byte vector.
    src = tl.multiple_of(((block.to(tl.int64) * owners + row) * TPB + pos % TPB) * 656, 16)
    dst = w * out_stride
    tile = tl.arange(0, 4)
    col = tl.arange(0, 128)
    cols = tile[None, :, None] * 128 + col[None, None, :]
    live3 = live[:, None, None]
    codes = tl.load(pool_ptr + src[:, None, None] + cols, mask=live3, other=0)
    codes = codes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scale_ptr = (pool_ptr + src[:, None] + 512 + tile[None, :] * 4).to(tl.pointer_type(tl.float32))
    scale = tl.load(scale_ptr, mask=live[:, None], other=0.0)
    tl.store(
        out_ptr + dst[:, None, None] + cols,
        (codes * scale[:, :, None]).to(tl.bfloat16),
        mask=live3,
    )
    r = tl.arange(0, 64)
    rope_ptr = (pool_ptr + src[:, None] + 528).to(tl.pointer_type(tl.bfloat16)) + r[None, :]
    rope = tl.load(rope_ptr, mask=live[:, None], other=0.0)
    tl.store(out_ptr + dst[:, None] + 512 + r[None, :], rope, mask=live[:, None])


class DsMlaDequantGatherTiled(DsMlaDequantGather):
    """`ds_mla_dequant_gather`'s contract, `reference` and guard, with a kernel that moves a
    tile of `ROWS` whole workspace rows per program instead of a fifth of one row::

        out[w, :512]   = e4m3(code) * tile_scale    (exact in bf16: 3 mantissa bits times 2^k)
        out[w, 512:576] = the stored bf16 rope

    `pool`: uint8 `[blocks, owners, tokens_per_block, 656]`; `kv_req` / `kv_pos`: int32
    `[N]`; `out`: bf16 `[N, 576]` (row stride free). Returns `out`.
    """

    ARCHS = frozenset({Arch.SM_107})

    # spec: as `ds_mla_dequant_gather`'s. The sm_107 dep4, tep4 and tp4 targets' cells, plus
    # the dep4 context steps of an AgentX prefill chunk: 16384 new rows behind a cached prefix
    # of 0, 100000 and 300000 tokens (the workspace path through N = 316384, the shared
    # source kept beside the step's rows through N = 2**18), and the dep4 workspace of an
    # engine at max_seq_len 1048576, 2**20 rows: one request at max_seq_len fills it.
    CELLS: tuple[Cell, ...] = (
        *(
            c
            for c in DsMlaDequantGather.CELLS
            if not c.why.startswith("1M tep8") and not c.spec.get("large")
        ),
        *(
            Cell(
                why=f"dep4 AgentX chunk: 16384 new rows behind a {cached}-token prefix "
                f"(N = {cached + 16384}) into the 2**19-row buffer, owner row 22, its pages "
                "anywhere in a 16500-block pool",
                spec=dict(
                    cached=[cached],
                    new=[16384],
                    blocks=16500,
                    owners=63,
                    tokens_per_block=32,
                    row=22,
                    out_rows=524288,
                    table_rows=16,
                    table_width=32772,
                    prefixes=cached == 300000,
                ),
            )
            for cached in (0, 100000, 300000)
        ),
        Cell(
            why="dep4 AgentX packed chunk: three requests cached 150000/120000/0 + new "
            "8192/6144/2048 (N = 286384), owner row 35, zero rope",
            spec=dict(
                cached=[150000, 120000, 0],
                new=[8192, 6144, 2048],
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=35,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
                nope=True,
            ),
        ),
        Cell(
            why="1M dep4 boot-sized pool, large: 74183 blocks (91.4 GiB), the 491520 + 32768 "
            "request's 16384 pages its top ones (byte offsets to 9.81e10), N = B, row 62",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=74183,
                owners=63,
                tokens_per_block=32,
                row=62,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
                far=True,
                large=True,
            ),
        ),
        Cell(
            why="1M dep4 boot-sized pool, large: one request at max_seq_len, 1032192 cached + "
            "16384 new (N = 2**20, the 2**20-row buffer whole), its 32768 pages the pool's top "
            "ones, row 62",
            spec=dict(
                cached=[1032192],
                new=[16384],
                blocks=74183,
                owners=63,
                tokens_per_block=32,
                row=62,
                out_rows=1048576,
                table_rows=16,
                table_width=32772,
                far=True,
                large=True,
            ),
        ),
    )

    note = """
    Call sites: the Mistral Large 4 sm_107 dep4, tep4 and tp4 targets' context workspace
    path (`PrefillTarget._attn_ctx`), once per owner layer and once per step for the NoPE
    source (its sharers reuse that workspace). Bitwise equal to `reference` (the cells) and
    so to `ds_mla_dequant_gather`, whose cells this entry drives too (bar the tep8 ones, a
    target sm_107 has none of). Launch: Triton, grid (ceil(N / 8),), 2 warps; not captured
    (context steps).

    Each program reads its 8 rows' kv_req / kv_pos and block-table entries, then moves the
    rows whole: [8, 4, 128] e4m3 codes and [8, 4] fp32 scales in, [8, 512] bf16 out, and
    [8, 64] bf16 rope copied, each access a 16-byte vector off the row's 16-byte aligned
    start; consecutive workspace rows of one request are consecutive slots of a page.
    `ds_mla_dequant_gather` runs five programs per row, each one 128-byte or 512-byte access.

    Measured on sm_107 (one request of 16384 new rows behind its cached prefix, a 63-owner
    pool, 1808 bytes moved per row; L2 flushed): N = 16384 / 116384 / 316384 takes 48.8 /
    362 / 1182 us with `ds_mla_dequant_gather` (0.5 - 0.6 TB/s) and 8.1 / 24.2 / 54.4 us
    here (3.6 / 8.7 / 10.5 TB/s; a plain device copy reaches 13.6 TB/s on the part).

    Hard limits (sm_107): as `ds_mla_dequant_gather`'s (int64 pool and out offsets, int32
    kv_req / kv_pos / block table widened in the kernel), with `w` int64 from the program id.
    The 16-byte vectors need the pool's base and out's base and row stride 16-byte aligned,
    which Triton's argument specialization checks per launch: an unaligned one compiles a
    variant with narrower accesses, slower and bitwise the same. The guard is the parent's.
    """

    def __call__(
        self,
        pool: torch.Tensor,
        row: int,
        kv_req: torch.Tensor,
        kv_pos: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        n = kv_req.shape[0]
        if n:
            _gather_tiled_kernel[(triton.cdiv(n, ROWS),)](
                pool, pool.shape[1], row, kv_req, kv_pos, block_table, block_table.stride(0),
                out, out.stride(0), n, TPB=pool.shape[2], ROWS=ROWS, num_warps=WARPS,
            )  # fmt: skip
        return out


ds_mla_dequant_gather_tiled = DsMlaDequantGatherTiled()
