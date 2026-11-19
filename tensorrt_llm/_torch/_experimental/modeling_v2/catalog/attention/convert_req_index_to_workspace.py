# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map a DSA request-local KV position to its row in a context workspace that holds several
requests' KV runs back to back."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._op import Arch, Cell, OpWrapper

#: Selections per program: half a row of 2048, 8 int32 a thread over 4 warps.
_BLOCK = 1024


@triton.jit
def _offset_kernel(
    req_ptr,  # int32 [rows]
    sel_ptr,  # int32 [rows, topk] (row stride sel_stride)
    sel_stride,
    off_ptr,  # int32 [requests]
    out_ptr,  # int32 [rows, topk] (row stride out_stride)
    out_stride,
    topk,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = c < topk
    off = tl.load(off_ptr + tl.load(req_ptr + r))
    sel = tl.load(sel_ptr + r * sel_stride + c, mask=live, other=-1)
    tl.store(out_ptr + r * out_stride + c, tl.where(sel >= 0, sel + off, sel), mask=live)


class ConvertReqIndexToWorkspace(OpWrapper):
    """`out[r, j] = sel[r, j] + ws_off[req_id[r]]` where `sel[r, j] >= 0`, else `sel[r, j]`::

        request-local KV position (or -1)  ->  its row in the workspace (or -1)

    The rows a context workspace hands FlashMLA's sparse prefill when it holds several
    requests' KV, each request's run starting at `ws_off[req]`; -1 (an empty slot) stays -1.
    `req_id`: int32 `[rows]`; `sel`: int32 `[rows, topk]`, row stride free; `ws_off`: int32
    `[requests]`; `out`: int32 `[rows, topk]`, row stride free. Writes `out` and returns it.
    """

    ARCHS = frozenset({Arch.SM_107})

    # spec: context requests of `new` query rows each, `kv` KV rows each laid back to back in
    # the workspace from `base` (the group's first KV row is the step's `base`-th), `topk`
    # selections a row shaped like the indexer's (every earlier position then -1 while the
    # row's position is below topk, else topk positions in [0, p]); `lead` rows of the
    # selection buffer before the call's rows (a group behind others in the step), `requests`
    # the step's request count (`ws_off`'s length), `first` the group's first request.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep4 AgentX packed chunk: three requests of 100000 / 60000 / 0 cached + "
            "8192 / 4096 / 4096 new, 16384 rows of the step's first group",
            spec=dict(kv=[108192, 64096, 4096], new=[8192, 4096, 4096], topk=2048),
        ),
        Cell(
            why="dep4 second group of a step past the held rows: requests 3..5 of six, "
            "150000 + 2048 KV rows each, behind 6144 rows of the selection buffer",
            spec=dict(kv=[152048] * 3, new=[2048] * 3, topk=2048, lead=6144, requests=6, first=3),
        ),
        Cell(
            why="dep4 sixteen requests filling the 2**19-row workspace: 30720 + 2048 KV rows "
            "each, 32768 rows (the chunked-prefill step)",
            spec=dict(kv=[32768] * 16, new=[2048] * 16, topk=2048),
        ),
        Cell(
            why="short requests: 1, 2, 7 and 2047-row fresh prompts, every row within topk "
            "(dense, -1 padded)",
            spec=dict(kv=[1, 2, 7, 2047], new=[1, 2, 7, 2047], topk=2048),
        ),
        Cell(
            why="the largest offsets: two requests of 262144 KV rows, the second's run at "
            "row 262144 of the 2**19-row workspace, selections to its last row",
            spec=dict(kv=[262144, 262144], new=[64, 64], topk=2048),
        ),
        Cell(
            why="topk past one program's block and not a multiple of it: 3000 selections",
            spec=dict(kv=[5000, 4000], new=[33, 17], topk=3000),
        ),
    )

    note = """
    Call site: the Mistral Large 4 sm_107 dep4 target's context workspace path
    (`PrefillTarget._attn_ctx`), once per selection (a layer that selects, and a step's first
    layer) for a workspace group of more than one request, into a buffer the group holds
    for the step. Bitwise equal to `reference` (integer arithmetic). Launch: Triton, grid
    (rows, ceil(topk / 1024)), 4 warps; not captured (context steps).

    Each program loads its row's request id and offset, then 1024 selections as 16-byte
    vectors, and stores them offset where they are >= 0: one read and one write of the
    selections. The torch expression it replaces (`torch.where(sel >= 0, sel +
    ws_off[req_id][:, None], sel)`) runs a compare, a gather, an add and a select, each a
    pass over the [rows, topk] selections. Measured on sm_107 at 16384 x 2048 over three
    requests (L2 flushed): 32.4 us here, 162.6 us for that expression in int32 and 390.9 us
    with int64 offsets and a cast back.

    Hard limits (sm_107)
      sel, out    r * row stride + c, int64 in the kernel     not reached (32768 x 2048)
      values      int32: a workspace row is below 2**19, so sel + off < 2**20 in every
                  cell; no overflow check (an offset past 2**31 - topk would wrap)
      req_id      read as given: a row whose request id is past ws_off's length reads
                  past ws_off (not checked: `is_valid` does no device reads)
    """

    def __call__(
        self,
        req_id: torch.Tensor,
        sel: torch.Tensor,
        ws_off: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        rows, topk = sel.shape
        if rows:
            _offset_kernel[(rows, triton.cdiv(topk, _BLOCK))](
                req_id, sel, sel.stride(0), ws_off, out, out.stride(0), topk,
                BLOCK=_BLOCK, num_warps=4,
            )  # fmt: skip
        return out

    def reference(
        self,
        req_id: torch.Tensor,
        sel: torch.Tensor,
        ws_off: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """A new int32 tensor shaped like `out`, offset in int64 so it cannot wrap."""
        off = ws_off.long()[req_id.long()][:, None]
        return torch.where(sel >= 0, sel.long() + off, sel.long()).to(torch.int32)

    def is_valid(
        self,
        req_id: torch.Tensor,
        sel: torch.Tensor,
        ws_off: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if any(t.dtype != torch.int32 for t in (req_id, sel, ws_off, out)):
            raise ValueError("req_id, sel, ws_off and out must be int32")
        if out.shape != sel.shape or req_id.shape != sel.shape[:1]:
            raise ValueError(
                f"out {tuple(out.shape)} and req_id {tuple(req_id.shape)} must match sel "
                f"{tuple(sel.shape)}: the kernel writes a row of out per row of sel"
            )
        if sel.stride(1) != 1 or out.stride(1) != 1:
            raise ValueError("sel and out must have contiguous rows")


convert_req_index_to_workspace = ConvertReqIndexToWorkspace()
