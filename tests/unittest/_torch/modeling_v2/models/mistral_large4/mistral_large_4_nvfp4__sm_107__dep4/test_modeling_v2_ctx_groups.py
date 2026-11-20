# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral-Large-4-1T-A52B NVFP4 sm_107 dep4 target's context rows, request group by group.

`PrefillTarget._ctx_workspace` groups the step's context requests, in order, on this rank's
host ints: a request whose own KV (cached and new rows) fits the held workspace joins a
workspace group, which takes the requests after it while their KV together still fits; a run of
requests each past the held rows is a decode group. `PrefillTarget._attn_ctx` runs each group's
rows on its path: the workspace group's KV gathered, dequantized, into the held workspace's
first rows (`ds_mla_dequant_gather_tiled`) for FlashMLA's bf16 sparse prefill; the decode
group's rows as one-row sequences of FlashMLA's sparse fp8 decode over the pool, in sub-batches
of at most `_CTX_DECODE_ROWS`. Layer 35's rows are kept beside the step's only when the step is
one workspace group whose KV fits twice; otherwise every sharer gathers them again. A step of
context rows alone that is one workspace group returns the prefill kernel's own output, which
the V up-projection reads in place of the latent scratch (`own_out`).

Host only, with stand-ins for the kernels: the held workspace's rows for an engine's
max_num_tokens, max_num_requests, max_seq_len and runtime features (room for a request at
max_seq_len, within 2**19..2**20 rows), the groups, their rows, request and position per
workspace row, offsets, kept rows and sub-batches over a 40-row held workspace, and which
kernel each group's rows reach, with which slices, and which steps return the kernel's own
output.

On one sm_107 GPU, over a 63-owner fp8_ds_mla pool (every byte 0xFF, a NaN in every field, but
the owner rows of layers 1, 35 and 36), queries at all 64 heads and selections shaped like the
indexer's, steps over an 8192-row held workspace: one group fitting twice and once, three
requests in two workspace groups, five alternating workspace and decode groups, and a decode
group of two sub-batches behind a workspace group. For layers 1, 35, 36 and 39 (35's sharer), in
that order on one step's groups:

* every workspace group's request rows are bitwise the sparse prefill over that request's own KV
  read straight from the pool, as on a step of that request alone, and every decode group's
  rows bitwise the sparse decode over the group's own rows in its sub-batches, as on a step of
  those requests alone;
* the step agrees within relative L2 1e-3 with the sparse decode over all of its context rows,
  the target's computation for a step past the held rows before it grouped requests;
* sampled rows are within relative L2 1e-2 of an fp32 softmax over the selected keys read
  straight from the pool;

and on a step that is one workspace group, the kernel's own output and the copy of it the step
makes with a generation row behind it are bitwise the same, as are their V up-projections.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.convert_req_index_to_global import (  # noqa: E501
    convert_req_index_to_global,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flash_mla_sparse_decode import (  # noqa: E501
    flash_mla_sparse_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flash_mla_sparse_prefill import (  # noqa: E501
    flash_mla_sparse_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__dep4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm.flash_mla import get_mla_metadata

_LAYERS = 72
_TPB = 32
_TOPK = 2048
#: Attention DP: a rank runs all 64 heads, FlashMLA's own head count.
_HEADS = 64
_LAT_DIM, _NOPE, _TILE, _ROW_BYTES = 576, 512, 128, 656
#: Every fourth layer from 39 on attends over layer 35's latent; the pool holds the 63 owners.
_SHARED = 35
_SOURCES = [_SHARED if i >= 39 and i % 4 == 3 else i for i in range(_LAYERS)]
_OWNERS = [i for i in range(_LAYERS) if _SOURCES[i] == i]
#: The owner rows the pool holds, and the layers each step runs, in order: a RoPE owner, the
#: NoPE source, the owner after it (which rewrites the step's workspace rows), and the
#: source's first sharer.
_WRITTEN = (1, 35, 36)
_RUN = (1, 35, 36, 39)
_SUB = M._CTX_DECODE_ROWS
#: The GPU steps' held workspace, small enough that a few thousand rows cross it.
_HELD = 8192
_BLOCKS = 1200
_TABLE = (16, 1024)
_SCALE = 128**-0.5
_SEED = 1077
#: The sm_103 dep4 ctx-bound test's band between the decode path and the workspace path over
#: one step, and its fp32 band.
_AGREE = 1e-3
_FP32_BAND = 1e-2
_SAMPLED = 128


def _cumsum(lens, dev=None) -> torch.Tensor:
    return torch.tensor([0, *lens], dtype=torch.int64, device=dev).cumsum(0)


def _host_step(contexts) -> dict:
    """What `_ctx_workspace` and `_attn_ctx` read of a step of context requests of (cached, new)
    tokens each, as `step_args` carries them: the KV prefix sum on the device, the new and the
    KV prefix sums on the host."""
    new = [n for _, n in contexts]
    kv = [c + n for c, n in contexts]
    return dict(
        num_contexts=len(contexts),
        num_ctx_tokens=sum(new),
        num_tokens=sum(new),
        ctx_cached_tokens=sum(c for c, _ in contexts),
        cu_ctx_kv=_cumsum(kv),
        host_cu_ctx_new=_cumsum(new).tolist(),
        host_cu_ctx_kv=_cumsum(kv).tolist(),
        skip_ctx=False,
        latent_pool=None,
        latent_kv=None,
        block_table=None,
        tokens_per_block=_TPB,
    )


def _target(held: int) -> M.PrefillTarget:
    """A prefill target over what its context workspace and attention read off the core: a held
    workspace of `held` rows, layer 35 the one shared source, and the 63-owner pool's
    addressing."""
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = SimpleNamespace(
        lat_dim=_LAT_DIM,
        ctx_workspace_rows=held,
        shared_source=[i == _SHARED for i in range(_LAYERS)],
        softmax_scale=_SCALE,
        idx_topk=_TOPK,
        _stride_factor=len(_OWNERS) * _TPB,
        _pool_row=[_OWNERS.index(_SOURCES[i]) for i in range(_LAYERS)],
    )
    return target


def _rows_of(view: torch.Tensor, base: torch.Tensor) -> tuple[int, int]:
    """The rows `[lo, hi)` of `base` (contiguous rows) a view of it covers."""
    assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    row = base[0].numel() * base.element_size()
    lo = (view.data_ptr() - base.data_ptr()) // row
    return lo, lo + view.shape[0]


# ---- host only: the groups and the kernels each group's rows reach ----


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "contexts, expected",
    [
        ([(7, 3), (0, 6)], [((0, 9), "kept", 16)]),
        ([(20, 5), (10, 5)], [((0, 10), "workspace", 40)]),
        ([(31, 10)], [((0, 10), "decode", [(0, 10)])]),
        ([(30, 10), (0, 1)], [((0, 10), "workspace", 40), ((10, 11), "workspace", 1)]),
        (
            [(10, 5), (10, 5), (10, 5)],
            [((0, 10), "workspace", 30), ((10, 15), "workspace", 15)],
        ),
        (
            [(5, 5), (31, 10), (35, 6), (5, 5)],
            [
                ((0, 5), "workspace", 10),
                ((5, 21), "decode", [(0, 16)]),
                ((21, 26), "workspace", 10),
            ],
        ),
    ],
    ids=["fits_twice", "fits_once", "one_past", "exact_then_one", "three_in_two", "alternating"],
)
def test_groups_by_request_kv(contexts, expected):
    """Over a held workspace of 40 rows: the requests, in order, join a workspace group while
    their KV together fits (40 rows exactly included), a request past 40 rows alone, or with the
    ones past it after it, is a decode group, in sub-batches from its first row. Every workspace
    group starts at the held buffer's first row and addresses its own requests (the step's
    request ids) at their own positions; one over several requests holds each request's offset
    in the group. Only a step that is one group fitting twice keeps rows after its own."""
    target = _target(40)
    step = _host_step(contexts)
    tc = step["num_ctx_tokens"]
    ctx = target._ctx_workspace(step, tc, torch.device("cpu"))
    buf = target._ws_buf
    assert buf.shape == (40, _LAT_DIM) and buf.dtype == torch.bfloat16
    kv = [c + n for c, n in contexts]
    starts = _cumsum(kv).tolist()
    new_starts = _cumsum([n for _, n in contexts]).tolist()
    assert [g.rows for g in ctx.groups] == [rows for rows, _, _ in expected]
    assert ctx.own_out is (len(expected) == 1 and expected[0][1] != "decode")
    sizes = set()
    for g, (rows, kind, shape) in zip(ctx.groups, expected):
        if kind == "decode":
            assert g.ws_buf is None and not hasattr(g, "kv_req")
            assert g.sub_batches == shape
            sizes.update(hi - lo for lo, hi in shape)
            continue
        reqs = [r for r in range(len(contexts)) if rows[0] <= new_starts[r] < rows[1]]
        assert _rows_of(g.ws_buf, buf) == (0, shape) and g.ws == {} and g.idx is None
        assert g.kv_req.tolist() == [r for r in reqs for _ in range(kv[r])]
        assert g.kv_pos.tolist() == [p for r in reqs for p in range(kv[r])]
        if len(reqs) == 1:
            assert g.ws_off is None
        else:
            assert g.ws_off.dtype == torch.int32
            assert g.ws_off[reqs].tolist() == [starts[r] - starts[reqs[0]] for r in reqs]
        if kind == "kept":
            assert _rows_of(g.kept, buf) == (shape, 2 * shape)
        else:
            assert g.kept is None
    assert sorted(ctx.sched) == sorted(sizes)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "features, tokens, requests, max_seq_len, rows",
    [
        ((False, False), 32768, 16, 32768, 65536),
        ((True, True), 32768, 16, 32768, 1 << 19),
        ((False, True), 16384, 64, 32768, 1 << 19),
        ((True, True), 16384, 32, 600000, 600000),
        ((True, True), 16384, 32, 1048576, 1 << 20),
        ((True, False), 16384, 32, 2097152, 1 << 20),
        ((True, True), 16, 1, 300000, 1 << 19),
    ],
    ids=["32k_new_only", "32k_chunked", "32k_reuse", "600k", "1m", "2m_capped", "short_step"],
)
def test_workspace_rows_fit_a_request_at_max_seq_len(features, tokens, requests, max_seq_len, rows):
    """The held workspace has room for the most context KV one step attends, twice over, up to
    the engine's max_seq_len within 2**19..2**20 rows: an engine of max_seq_len 32768 keeps the
    2**19 rows sixteen of its requests fill, one of max_seq_len 1048576 holds 2**20 so that a
    request at max_seq_len fits whole, and one past 2**20 stops there."""
    md = SimpleNamespace(
        runtime_features=SimpleNamespace(chunked_prefill=features[0], cache_reuse=features[1]),
        max_num_tokens=tokens,
        max_num_requests=requests,
        max_seq_len=max_seq_len,
    )
    assert M._ctx_workspace_rows(md) == rows


def _recorded(monkeypatch) -> list:
    """Stand-ins for the context attention's kernels, recording each call's pool row, rows and
    selections: the gather writes nothing, the slots are the selections, nothing is attended."""
    calls = []

    def gather(pool, row, kv_req, kv_pos, block_table, out):
        calls.append(("gather", row, kv_req.tolist(), out))
        return out

    def prefill(q, ws, idx, scale, out):
        calls.append(("prefill", q, ws, idx, out))

    def slots(req, table, sel, tpb, topk, stride, row):
        calls.append(("slots", row, req.tolist()))
        return sel

    def decode(q, kv, slot, sched, scale, out):
        calls.append(("decode", q, sched, out))

    def offset(req, sel, ws_off, out):
        calls.append(("offset", req.tolist(), out))
        return out.copy_(torch.where(sel >= 0, sel + ws_off[req.long()][:, None], sel))

    monkeypatch.setattr(M, "ds_mla_dequant_gather_tiled", gather)
    monkeypatch.setattr(M, "convert_req_index_to_workspace", offset)
    monkeypatch.setattr(M, "flash_mla_sparse_prefill", prefill)
    monkeypatch.setattr(M, "convert_req_index_to_global", slots)
    monkeypatch.setattr(M, "flash_mla_sparse_decode", decode)
    return calls


@pytest.mark.cpu_only
def test_attn_ctx_runs_each_group_on_its_rows(monkeypatch):
    """A step of requests of 15 + 5, 41 and 10 KV rows over 40 held rows: a workspace group over
    the first two (rows 0..7), a decode group (7..17) and a workspace group (17..22). Layer 35,
    then 39 (its sharer, selecting nothing): each workspace group gathers its own requests into
    the held rows and attends its rows over them, the two-request group's selections offset by
    10 for its second request once, into a buffer 39 reads again; the decode group's rows
    become slots and attend in one sub-batch; and 39 gathers layer 35 again, the step keeping
    no rows."""
    calls = _recorded(monkeypatch)
    pool_row = _OWNERS.index(_SHARED)
    target = _target(40)
    contexts = [(5, 5), (3, 2), (31, 10), (5, 5)]
    step = _host_step(contexts)
    ctx = target._ctx_workspace(step, 22, torch.device("cpu"))
    buf = target._ws_buf
    assert [g.rows for g in ctx.groups] == [(0, 7), (7, 17), (17, 22)]
    req_idx = torch.tensor([0] * 5 + [1] * 2 + [2] * 10 + [3] * 5, dtype=torch.int32)
    sel = torch.tensor([[0, 1, -1, 2]] * 22, dtype=torch.int32)
    q, out = torch.empty(22, 1), torch.empty(22, 1)
    for layer, selects in ((_SHARED, True), (39, False)):
        L = SimpleNamespace(kv_source=_SOURCES[layer], indexer=object() if selects else None)
        target._attn_ctx(step, L, q, out, req_idx, sel, ctx)
    offsets = [c for c in calls if c[0] == "offset"]
    assert len(offsets) == 1 and offsets[0][1] == [0] * 5 + [1] * 2
    calls = [c for c in calls if c[0] != "offset"]
    assert len(calls) == 12
    first = None
    for i, layer in enumerate((_SHARED, 39)):
        gather0, prefill0, slots, decode, gather2, prefill2 = calls[6 * i : 6 * i + 6]
        assert gather0[:3] == ("gather", pool_row, [0] * 10 + [1] * 5)
        assert _rows_of(gather0[3], buf) == (0, 15)
        assert prefill0[0] == "prefill" and prefill0[2] is gather0[3]
        assert _rows_of(prefill0[1], q) == (0, 7) and _rows_of(prefill0[4], out) == (0, 7)
        assert prefill0[3].tolist() == [[0, 1, -1, 2]] * 5 + [[10, 11, -1, 12]] * 2
        if first is None:
            first = prefill0[3]
            assert first is offsets[0][2]
        else:
            assert prefill0[3] is first, f"layer {layer} read another buffer"
        assert slots == ("slots", pool_row, [2] * 10)
        assert decode[0] == "decode" and decode[2] is ctx.sched[10]
        assert _rows_of(decode[1], q) == (7, 17) and _rows_of(decode[3], out) == (7, 17)
        assert gather2[:3] == ("gather", pool_row, [3] * 10)
        assert _rows_of(gather2[3], buf) == (0, 10)
        assert _rows_of(prefill2[1], q) == (17, 22) and prefill2[3].tolist() == sel[17:].tolist()
    assert all(g.ws == {} for g in ctx.groups if g.ws_buf is not None)


@pytest.mark.cpu_only
def test_decode_group_sub_batches_from_its_first_row(monkeypatch):
    """A 20000-row request past the held rows behind a 5-row one that fits: the decode group's
    rows 5..20005 attend in sub-batches of 8192, 8192 and 3616 from its first row, on those
    sizes' schedulers, and nothing of its KV is gathered."""
    calls = _recorded(monkeypatch)
    target = _target(40)
    step = _host_step([(5, 5), (100, 20000)])
    ctx = target._ctx_workspace(step, 20005, torch.device("cpu"))
    assert [g.rows for g in ctx.groups] == [(0, 5), (5, 20005)]
    assert ctx.groups[1].sub_batches == [(0, 8192), (8192, 16384), (16384, 20000)]
    assert sorted(ctx.sched) == [3616, 8192]
    req_idx = torch.tensor([0] * 5 + [1] * 20000, dtype=torch.int32)
    sel = torch.zeros(20005, 4, dtype=torch.int32)
    q, out = torch.empty(20005, 1), torch.empty(20005, 1)
    L = SimpleNamespace(kv_source=_SOURCES[1], indexer=object())
    target._attn_ctx(step, L, q, out, req_idx, sel, ctx)
    gathers = [c for c in calls if c[0] == "gather"]
    assert [kv_req for _, _, kv_req, _ in gathers] == [[0] * 10]
    decodes = [c for c in calls if c[0] == "decode"]
    assert [_rows_of(c[1], q) for c in decodes] == [(5, 8197), (8197, 16389), (16389, 20005)]
    assert [c[2] for c in decodes] == [ctx.sched[8192], ctx.sched[8192], ctx.sched[3616]]


@pytest.mark.cpu_only
@pytest.mark.parametrize("gen_rows, own", [(0, True), (2, False)], ids=["ctx_alone", "gen_behind"])
def test_one_group_alone_returns_the_kernel_output(monkeypatch, gen_rows, own):
    """A step that is one workspace group (16 KV rows over 40): with its context rows alone the
    prefill writes no latent scratch and its own output is returned for the V up-projection;
    with generation rows behind them the prefill writes the scratch's context rows, which are
    returned."""
    prefills = []
    kernel_out = torch.empty(9, 1)

    def prefill(q, ws, idx, scale, out):
        prefills.append(out)
        return kernel_out if out is None else out

    monkeypatch.setattr(M, "ds_mla_dequant_gather_tiled", lambda *args: args[-1])
    monkeypatch.setattr(M, "convert_req_index_to_workspace", lambda req, sel, off, out: out)
    monkeypatch.setattr(M, "flash_mla_sparse_prefill", prefill)
    target = _target(40)
    step = _host_step([(7, 3), (0, 6)])
    step["num_tokens"] += gen_rows
    ctx = target._ctx_workspace(step, 9, torch.device("cpu"))
    assert ctx.own_out is own
    lat_out = torch.empty(9, 1)
    L = SimpleNamespace(kv_source=_SOURCES[1], indexer=object())
    req_idx = torch.tensor([0] * 3 + [1] * 6, dtype=torch.int32)
    sel = torch.zeros(9, 4, dtype=torch.int32)
    got = target._attn_ctx(step, L, torch.empty(9, 1), lat_out, req_idx, sel, ctx)
    if own:
        assert prefills == [None] and got is kernel_out
    else:
        assert _rows_of(prefills[0], lat_out) == (0, 9) and got is lat_out


# ---- one GPU: the groups' numerics ----


def _dev_rows(n: int, g: torch.Generator, nope: bool) -> torch.Tensor:
    """`n` uint8 656-byte rows as the fp8_ds_mla writer stores them: e4m3 codes, power-of-two
    fp32 tile scales 2^-10..2^-4, bf16 rope (zero on a NoPE owner)."""
    dev = g.device
    codes = ((torch.rand(n, _NOPE, generator=g, device=dev) * 2 - 1) * 448).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-10, -3, (n, _NOPE // _TILE), generator=g, device=dev).float())
    if nope:
        rope = torch.zeros(n, _LAT_DIM - _NOPE, device=dev)
    else:
        rope = torch.randn(n, _LAT_DIM - _NOPE, generator=g, device=dev) * 0.5
    return torch.cat(
        [
            codes.view(torch.uint8),
            scale.view(torch.uint8),
            rope.to(torch.bfloat16).view(torch.uint8),
        ],
        dim=1,
    )


def _dequant(rows: torch.Tensor) -> torch.Tensor:
    """fp8_ds_mla rows `[n, 656]` to bf16 `[n, 576]`: e4m3 times its tile's scale (exact in
    bf16 for power-of-two scales), then the rope."""
    codes = rows[:, :_NOPE].contiguous().view(torch.float8_e4m3fn).float()
    scale = rows[:, _NOPE : _NOPE + 16].contiguous().view(torch.float32)
    nope = (codes.view(-1, _NOPE // _TILE, _TILE) * scale[..., None]).view(-1, _NOPE)
    rope = rows[:, _NOPE + 16 :].contiguous().view(torch.bfloat16)
    return torch.cat([nope.to(torch.bfloat16), rope], dim=1)


def _selections(pos: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """int32 `[rows, 2048]` request-local selections for rows at positions `pos`: below
    index_topk every earlier position in order then -1, else one draw from each of 2048 equal
    strata of [0, p] with p itself in the last, so distinct."""
    p = pos.long()[:, None]
    col = torch.arange(_TOPK, device=pos.device)[None]
    lo = col * (p + 1) // _TOPK
    hi = (col + 1) * (p + 1) // _TOPK
    draw = lo + (torch.rand(pos.numel(), _TOPK, generator=g, device=pos.device) * (hi - lo)).long()
    draw = torch.minimum(draw, hi - 1)
    draw[:, -1] = pos
    return torch.where(p < _TOPK, torch.where(col <= p, col, -1), draw).to(torch.int32)


@pytest.fixture(scope="module")
def env():
    """The pool and the target over an 8192-row held workspace."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() != 107:
        pytest.skip("the sm_107 dep4 target's context paths are certified on sm_107 only")
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(_SEED)
    pool = torch.full(
        (_BLOCKS, len(_OWNERS), _TPB, _ROW_BYTES), 0xFF, dtype=torch.uint8, device=dev
    )
    rows = {layer: _OWNERS.index(layer) for layer in _WRITTEN}
    for layer, row in rows.items():
        pool[:, row] = _dev_rows(_BLOCKS * _TPB, g, nope=layer == _SHARED).view(
            _BLOCKS, _TPB, _ROW_BYTES
        )
    return SimpleNamespace(
        dev=dev, pool=pool, kv=pool.view(-1, _TPB, 1, _ROW_BYTES), rows=rows, target=_target(_HELD)
    )


def _step(env, contexts, seed: int) -> SimpleNamespace:
    """A step of context requests of (cached, new) tokens: each request's pages distinct blocks
    of the pool, a query per row at all 64 heads, and its selection."""
    dev = env.dev
    g = torch.Generator(device=dev).manual_seed(_SEED + seed)
    kv_lens = [c + n for c, n in contexts]
    pages = [-(-k // _TPB) for k in kv_lens]
    assert sum(pages) <= _BLOCKS and max(pages) <= _TABLE[1] and len(kv_lens) <= _TABLE[0]
    order = torch.randperm(_BLOCKS, generator=g, device=dev).to(torch.int32)
    table = torch.zeros(_TABLE, dtype=torch.int32, device=dev)
    used = 0
    for r, n in enumerate(pages):
        table[r, :n] = order[used : used + n]
        used += n
    req_idx = torch.cat(
        [torch.full((n,), r, dtype=torch.int32, device=dev) for r, (_, n) in enumerate(contexts)]
    )
    pos = torch.cat([torch.arange(c, c + n, device=dev) for c, n in contexts])
    q = torch.randn(pos.numel(), _HEADS, _LAT_DIM, generator=g, device=dev).mul_(0.1)
    step = _host_step(contexts)
    step.update(
        cu_ctx_kv=step["cu_ctx_kv"].to(dev),
        block_table=table,
        latent_pool=env.pool,
        latent_kv=env.kv,
    )
    return SimpleNamespace(
        step=step, contexts=list(contexts), kv_lens=kv_lens, tc=pos.numel(), req_idx=req_idx,
        pos=pos, topk=_selections(pos, g), q=q.to(torch.bfloat16), table=table,
    )  # fmt: skip


def _request_kv(env, s, layer: int, r: int) -> torch.Tensor:
    """Request r's KV rows of `layer`, bf16, read through its block table."""
    p = torch.arange(s.kv_lens[r], device=env.dev)
    return _dequant(env.pool[s.table[r, p // _TPB].long(), env.rows[layer], p % _TPB])


def _decode(env, s, layer: int, lo: int, hi: int, bounds) -> torch.Tensor:
    """The sparse decode over context rows lo..hi alone, in sub-batches `bounds` from row lo."""
    slots = convert_req_index_to_global(
        s.req_idx[lo:hi], s.table, s.topk[lo:hi], _TPB, _TOPK, len(_OWNERS) * _TPB,
        _OWNERS.index(layer),
    )  # fmt: skip
    out = torch.empty(hi - lo, _HEADS, _NOPE, dtype=torch.bfloat16, device=env.dev)
    for a, b in bounds:
        flash_mla_sparse_decode(
            s.q[lo + a : lo + b], env.kv, slots[a:b], get_mla_metadata()[0], _SCALE, out[a:b]
        )
    return out


def _fp32_error(env, s, layer: int, out: torch.Tensor, rows: torch.Tensor) -> float:
    """`out`'s rows `rows` against an fp32 softmax over each row's selected keys read straight
    from the pool through its request's table row: the relative L2 error."""
    sel = s.topk[rows].long()
    req = s.req_idx[rows].long()
    valid = sel >= 0
    key = sel.clamp_min(0)
    blocks = s.table[req[:, None], key // _TPB].long()
    k = _dequant(env.pool[blocks, env.rows[layer], key % _TPB].view(-1, _ROW_BYTES)).float()
    k = k.view(rows.numel(), _TOPK, _LAT_DIM)
    with true_fp32_matmul():
        logits = torch.matmul(s.q[rows].float(), k.transpose(1, 2)) * _SCALE
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        exact = torch.matmul(torch.softmax(logits, -1), k[..., :_NOPE])
    return float((out[rows].float() - exact).norm() / exact.norm())


def _check_step(env, contexts, expected_groups, seed: int) -> tuple[SimpleNamespace, list]:
    """Run `_RUN` on one step's groups and check each layer's output (the module docstring's
    three checks); return the step's groups and which workspace groups kept rows."""
    s = _step(env, contexts, seed)
    target = env.target
    ctx = target._ctx_workspace(s.step, s.tc, env.dev)
    kinds = ["decode" if g.ws_buf is None else "workspace" for g in ctx.groups]
    assert list(zip([g.rows for g in ctx.groups], kinds)) == expected_groups
    new_starts = _cumsum([n for _, n in contexts]).tolist()
    edges = sorted({r for g in ctx.groups for r in (g.rows[0], g.rows[1] - 1)})
    g_cpu = torch.Generator().manual_seed(_SEED + seed)
    sampled = torch.randperm(s.tc, generator=g_cpu)[:_SAMPLED].tolist()
    rows = torch.tensor(sorted(set(sampled) | set(edges)), device=env.dev)
    for layer in _RUN:
        src = _SOURCES[layer]
        scratch = torch.full(
            (s.tc, _HEADS, _NOPE), float("nan"), dtype=torch.bfloat16, device=env.dev
        )
        L = SimpleNamespace(kv_source=src, indexer=None if layer == 39 else object())
        out = target._attn_ctx(s.step, L, s.q, scratch, s.req_idx, s.topk, ctx)
        assert out is scratch or ctx.own_out
        assert bool(torch.isfinite(out).all()), f"layer {layer}: a row read an unwritten byte"
        for g in ctx.groups:
            q0, q1 = g.rows
            if g.ws_buf is None:
                ref = _decode(env, s, src, q0, q1, g.sub_batches)
                assert torch.equal(out[q0:q1].view(torch.int16), ref.view(torch.int16)), (
                    layer,
                    g.rows,
                )
                continue
            for r in range(len(contexts)):
                a, b = new_starts[r], new_starts[r + 1]
                if not q0 <= a < q1:
                    continue
                own = torch.empty(b - a, _HEADS, _NOPE, dtype=torch.bfloat16, device=env.dev)
                flash_mla_sparse_prefill(
                    s.q[a:b], _request_kv(env, s, src, r), s.topk[a:b], _SCALE, own
                )
                assert torch.equal(out[a:b].view(torch.int16), own.view(torch.int16)), (layer, r)
        bounds = [(lo, min(lo + _SUB, s.tc)) for lo in range(0, s.tc, _SUB)]
        whole = _decode(env, s, src, 0, s.tc, bounds)
        rel = float((out.float() - whole.float()).norm() / whole.float().norm())
        err = _fp32_error(env, s, src, out, rows)
        print(
            f"CTXGROUPS {contexts} layer {layer}: groups {expected_groups}; against the decode "
            f"path over the step rel L2 {rel:.3e}; fp32 rel L2 {err:.3e}"
        )
        assert rel <= _AGREE, (layer, rel)
        assert err < _FP32_BAND, (layer, err)
    kept = [g.kept is not None for g in ctx.groups if g.ws_buf is not None]
    return ctx, kept


def test_one_group_fitting_twice_keeps_layer_35(env):
    """1000 + 500 and 500 + 500 KV rows (2500, twice within 8192): one workspace group, layer
    35's rows kept beside the step's and read there by 39 after 36 rewrote the step's rows."""
    ctx, kept = _check_step(env, [(1000, 500), (500, 500)], [((0, 1000), "workspace")], seed=1)
    assert kept == [True] and _SHARED in ctx.groups[0].ws


def test_one_group_fitting_once(env):
    """3000 + 1000 and 1000 + 500 KV rows (5500, once within 8192): one workspace group, layer
    35 gathered again for 39."""
    ctx, kept = _check_step(env, [(3000, 1000), (1000, 500)], [((0, 1500), "workspace")], seed=2)
    assert kept == [False] and ctx.groups[0].ws == {}


def test_three_requests_in_two_workspace_groups(env):
    """KV rows 4000, 2500 and 4800 (11300, past 8192): the first two in one workspace group,
    the third in another."""
    _check_step(
        env,
        [(3000, 1000), (2000, 500), (4000, 800)],
        [((0, 1500), "workspace"), ((1500, 2300), "workspace")],
        seed=3,
    )


def test_alternating_workspace_and_decode_groups(env):
    """KV rows 2000, 9700, 2300, 12600 and 900: the second and fourth requests are each past
    the held rows, decode groups between three workspace groups."""
    _check_step(
        env,
        [(1500, 500), (9000, 700), (2000, 300), (12000, 600), (500, 400)],
        [
            ((0, 500), "workspace"),
            ((500, 1200), "decode"),
            ((1200, 1500), "workspace"),
            ((1500, 2100), "decode"),
            ((2100, 2500), "workspace"),
        ],
        seed=4,
    )


def test_decode_group_of_two_sub_batches(env):
    """A 9000-row request over 18000 KV rows behind a 1000-row one: its decode group's rows
    1000..10000 in sub-batches of 8192 and 808 from its first row."""
    ctx, _ = _check_step(
        env,
        [(2000, 1000), (9000, 9000)],
        [((0, 1000), "workspace"), ((1000, 10000), "decode")],
        seed=5,
    )
    assert ctx.groups[1].sub_batches == [(0, _SUB), (_SUB, 9000)]


def test_kernel_output_feeds_the_v_projection_bitwise(env):
    """One workspace group (1000 + 500 and 500 + 500 KV rows), with its context rows alone and
    with a generation row behind them: the prefill kernel's own output the first returns and
    the latent scratch the second writes are bitwise the same on every layer run, and so are
    the V up-projections `bmm_out` makes of them with the target's transposed weight view."""
    s = _step(env, [(1000, 500), (500, 500)], seed=6)
    g = torch.Generator(device=env.dev).manual_seed(_SEED + 6)
    v_b_t = torch.transpose(
        (torch.randn(_HEADS, 128, _NOPE, generator=g, device=env.dev) * 0.05).to(torch.bfloat16),
        1,
        2,
    )
    runs = []
    for gen_rows in (0, 1):
        step = dict(s.step, num_tokens=s.tc + gen_rows)
        target = _target(_HELD)
        ctx = target._ctx_workspace(step, s.tc, env.dev)
        assert ctx.own_out is (gen_rows == 0)
        scratch = torch.full(
            (s.tc, _HEADS, _NOPE), float("nan"), dtype=torch.bfloat16, device=env.dev
        )
        per_layer = []
        for layer in _RUN:
            L = SimpleNamespace(
                kv_source=_SOURCES[layer], indexer=None if layer == 39 else object()
            )
            lat = target._attn_ctx(step, L, s.q, scratch, s.req_idx, s.topk, ctx)
            assert (lat is scratch) is (gen_rows == 1)
            attn = torch.empty(s.tc, _HEADS * 128, dtype=torch.bfloat16, device=env.dev)
            bmm_out(
                torch.transpose(lat, 0, 1),
                v_b_t,
                torch.transpose(attn.view(s.tc, _HEADS, 128), 0, 1),
            )
            per_layer.append((lat.clone(), attn))
        runs.append(per_layer)
    for layer, (own, copied) in zip(_RUN, zip(*runs)):
        assert torch.equal(own[0].view(torch.int16), copied[0].view(torch.int16)), layer
        assert torch.equal(own[1].view(torch.int16), copied[1].view(torch.int16)), layer
