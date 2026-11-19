# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the convert_req_index_to_workspace catalog entry: integer arithmetic, so every
output is gated bitwise against the entry's reference, the rows of `out`'s buffer past the call
are held to a sentinel, and every valid selection must land inside its own request's run."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.convert_req_index_to_workspace import (  # noqa: E501
    convert_req_index_to_workspace as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "convert_req_index_to_workspace requires a CUDA device"

#: Fills `out`'s buffer rows the call must leave alone.
_UNWRITTEN = -7
_PAD = 8
#: The KV rows of a step's requests outside the cell's group (their offsets only move ws_off).
_OTHER_KV = 100000


def _build(spec, seed):
    """The call's arguments and `out`'s buffer: the group's requests' rows in order, each row's
    selection request-local and shaped like the indexer's, `lead` rows of the selection buffer
    before them, ws_off over the step's `requests` (the group's KV runs back to back from 0,
    the others' offsets relative to the group's first row), `out` the first rows of a
    sentinel-filled buffer."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    kv, new, topk = spec["kv"], spec["new"], spec["topk"]
    first = spec.get("first", 0)
    requests = spec.get("requests", len(kv))
    lens = [_OTHER_KV] * requests
    lens[first : first + len(kv)] = kv
    starts = torch.tensor([0, *lens[:-1]], dtype=torch.int64).cumsum(0)
    ws_off = (starts - starts[first]).to(torch.int32).to(dev)
    req_id = torch.cat(
        [torch.full((n,), first + r, dtype=torch.int32) for r, n in enumerate(new)]
    ).to(dev)
    pos = torch.cat([torch.arange(k - n, k) for k, n in zip(kv, new)]).to(dev)
    col = torch.arange(topk, device=dev)[None]
    draw = (torch.rand(pos.numel(), topk, device=dev, generator=g) * (pos[:, None] + 1)).long()
    draw[:, -1] = pos
    sel = torch.where(pos[:, None] < topk, torch.where(col <= pos[:, None], col, -1), draw)
    lead = spec.get("lead", 0)
    sel_buf = torch.randint(-1, 1000, (lead + pos.numel(), topk), device=dev, generator=g)
    sel_buf[lead:] = sel
    sel_buf = sel_buf.to(torch.int32)
    out_buf = torch.full((pos.numel() + _PAD, topk), _UNWRITTEN, dtype=torch.int32, device=dev)
    return (req_id, sel_buf[lead:], ws_off, out_buf[: pos.numel()]), out_buf, kv


def _check(args, out, out_buf, kv) -> None:
    req_id, sel, ws_off, _ = args
    ref = op.reference(*args)
    assert torch.equal(out, ref), f"{int((out != ref).sum())} of {out.numel()} differ"
    assert bool((out_buf[out.shape[0] :] == _UNWRITTEN).all()), "wrote past the call's rows"
    assert torch.equal(out < 0, sel < 0), "an empty slot moved, or a selection became one"
    lo = ws_off[req_id.long()].long()[:, None]
    hi = lo + torch.tensor(kv, device=out.device)[req_id.long() - int(req_id.min())][:, None]
    valid = out >= 0
    assert bool(((out >= lo) & (out < hi))[valid].all()), "a selection left its request's run"


@pytest.mark.parametrize("cell", [pytest.param(c, id=c.why[:60]) for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args, out_buf, kv = _build(cell.spec, zlib.crc32(cell.why.encode()))
    out = op(*args)
    assert out.data_ptr() == args[3].data_ptr()
    _check(args, out, out_buf, kv)


def test_strided_rows() -> None:
    """`sel` and `out` rows spaced past topk (views of wider buffers): the same answer."""
    spec = dict(kv=[9000, 5000], new=[300, 200], topk=2048)
    (req_id, sel, ws_off, out), out_buf, kv = _build(spec, 3)
    wide_sel = torch.full((sel.shape[0], 2048 + 96), 5, dtype=torch.int32, device="cuda")
    wide_sel[:, :2048] = sel
    wide_out = torch.full((out.shape[0], 2048 + 32), _UNWRITTEN, dtype=torch.int32, device="cuda")
    with validating(op):
        got = op(req_id, wide_sel[:, :2048], ws_off, wide_out[:, :2048])
    assert torch.equal(got, op.reference(req_id, sel, ws_off, out))
    assert bool((wide_out[:, 2048:] == _UNWRITTEN).all()), "wrote past a row's topk columns"


def test_is_valid_refuses_a_short_out_and_strided_columns() -> None:
    """Unguarded, an `out` of fewer rows than `sel` takes the extra rows past its end, and a
    column-strided `sel` is read as if contiguous; guarded, both raise before the kernel."""
    (req_id, sel, ws_off, out), out_buf, kv = _build(dict(kv=[4000], new=[64], topk=2048), 5)
    with validating(op), pytest.raises(ValueError, match="must match sel"):
        op(req_id, sel, ws_off, out[:40])
    with validating(op), pytest.raises(ValueError, match="contiguous rows"):
        op(req_id, sel.t().contiguous().t(), ws_off, out)
    with validating(op), pytest.raises(ValueError, match="int32"):
        op(req_id, sel.long(), ws_off, out)
    assert bool((out_buf == _UNWRITTEN).all()), "refused, yet the kernel wrote"
