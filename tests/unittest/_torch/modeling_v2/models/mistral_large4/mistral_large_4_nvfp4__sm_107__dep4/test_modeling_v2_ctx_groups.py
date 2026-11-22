# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral-Large-4-1T-A52B NVFP4 sm_107 dep4 target's context rows, request group by group.

`PrefillTarget._ctx_workspace` groups the step's context requests, in order, on this rank's
host ints: a request whose own KV (cached and new rows) fits the context workspace joins a
workspace group, which takes the requests after it while their KV together still fits; a run of
requests each past the workspace's rows is a decode group. `PrefillTarget._attn_ctx` runs each
group's rows on its path: the workspace group's KV gathered, dequantized, into the first rows
of the layer's workspace (`ds_mla_dequant_gather_tiled`) for FlashMLA's bf16 sparse prefill,
layer 35's again for each sharer; the decode group's rows as one-row sequences of FlashMLA's
sparse fp8 decode over the pool, in sub-batches of at most `_CTX_DECODE_ROWS`, into a latent
output of the group's own; and each group's latent output (the prefill kernel's own output, not
a copy) projected into its rows of the attention output (`_v_up`) as soon as it is written.
`PrefillTarget._attention` takes the layer's absorbed query and whole workspace as one
allocation, the attention output over the query's first bytes, and projects the generation
rows' latent output the same way.

Host only, with stand-ins for the kernels: the workspace's rows for an engine's
max_num_tokens, max_num_requests, max_seq_len and runtime features (room for a request at
max_seq_len, within 2**19..2**20 rows), the groups, their rows, request and position per
workspace row, offsets and sub-batches over a 40-row workspace, which kernel each group's rows
reach, with which slices, which latent output each group's projection reads and which rows it
writes, and the attention block's buffers.

On one sm_107 GPU, over a 63-owner fp8_ds_mla pool (every byte 0xFF, a NaN in every field, but
the owner rows of layers 1, 35 and 36), queries at all 64 heads and selections shaped like the
indexer's, steps over an 8192-row workspace refilled with NaN before every layer: one small
group and one near the workspace's rows, three requests in two workspace groups, five
alternating workspace and decode groups, and a decode group of two sub-batches behind a
workspace group. For layers 1, 35, 36 and 39 (35's sharer), in that order on one step's groups:

* every workspace group's request rows are bitwise the sparse prefill over that request's own KV
  read straight from the pool, as on a step of that request alone, and every decode group's
  rows bitwise the sparse decode over the group's own rows in its sub-batches, as on a step of
  those requests alone;
* the step agrees within relative L2 1e-3 with the sparse decode over all of its context rows,
  the target's computation for a step past its workspace rows before it grouped requests;
* sampled rows are within relative L2 1e-2 of an fp32 softmax over the selected keys read
  straight from the pool;

and the V up-projection group by group is bitwise the one over the step's whole latent output
(five groups), as `bmm_out` over any split of 20480 rows is.
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
#: The GPU steps' workspace rows, small enough that a few thousand rows cross them.
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


def _target(held: int, heads: int = _HEADS, kv_lora: int = _NOPE, v_dim: int = 128):
    """A prefill target over what its context workspace and attention read off the core: a
    workspace of `held` rows, the 63-owner pool's addressing, and the latent and attention
    outputs' widths (the host tests' stand-ins take 1 each)."""
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = SimpleNamespace(
        heads=heads,
        kv_lora=kv_lora,
        v_dim=v_dim,
        lat_dim=_LAT_DIM,
        ctx_workspace_rows=held,
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
        ([(7, 3), (0, 6)], [((0, 9), "workspace", 16)]),
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
    ids=["small", "fits", "one_past", "exact_then_one", "three_in_two", "alternating"],
)
def test_groups_by_request_kv(contexts, expected):
    """Over a workspace of 40 rows: the requests, in order, join a workspace group while their KV
    together fits (40 rows exactly included), a request past 40 rows alone, or with the ones past
    it after it, is a decode group, in sub-batches from its first row. Every workspace group
    holds its KV's row count, the workspace's first rows, and addresses its own requests (the
    step's request ids) at their own positions; one over several requests holds each request's
    offset in the group."""
    target = _target(40)
    step = _host_step(contexts)
    tc = step["num_ctx_tokens"]
    ctx = target._ctx_workspace(step, tc, torch.device("cpu"))
    kv = [c + n for c, n in contexts]
    starts = _cumsum(kv).tolist()
    new_starts = _cumsum([n for _, n in contexts]).tolist()
    assert [g.rows for g in ctx.groups] == [rows for rows, _, _ in expected]
    sizes = set()
    for g, (rows, kind, shape) in zip(ctx.groups, expected):
        if kind == "decode":
            assert g.n_kv is None and not hasattr(g, "kv_req")
            assert g.sub_batches == shape
            sizes.update(hi - lo for lo, hi in shape)
            continue
        reqs = [r for r in range(len(contexts)) if rows[0] <= new_starts[r] < rows[1]]
        assert g.n_kv == shape and g.idx is None
        assert g.kv_req.tolist() == [r for r in reqs for _ in range(kv[r])]
        assert g.kv_pos.tolist() == [p for r in reqs for p in range(kv[r])]
        if len(reqs) == 1:
            assert g.ws_off is None
        else:
            assert g.ws_off.dtype == torch.int32
            assert g.ws_off[reqs].tolist() == [starts[r] - starts[reqs[0]] for r in reqs]
    assert sorted(ctx.sched) == sorted(sizes)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "features, tokens, requests, max_seq_len, rows",
    [
        ((False, False), 32768, 16, 32768, 32768),
        ((True, True), 32768, 16, 32768, 1 << 19),
        ((False, True), 16384, 64, 32768, 1 << 19),
        ((True, True), 16384, 32, 600000, 600000),
        ((True, True), 16384, 32, 1048576, 1 << 20),
        ((True, False), 16384, 32, 2097152, 1 << 20),
        ((True, True), 16, 1, 300000, 300000),
    ],
    ids=["32k_new_only", "32k_chunked", "32k_reuse", "600k", "1m", "2m_capped", "short_step"],
)
def test_workspace_rows_fit_a_request_at_max_seq_len(features, tokens, requests, max_seq_len, rows):
    """The workspace has room for the most context KV one step attends, up to the engine's
    max_seq_len within 2**19..2**20 rows: an engine of max_seq_len 32768 keeps the 2**19 rows
    sixteen of its requests fill, one of max_seq_len 1048576 takes 2**20 so that a request at
    max_seq_len fits whole, and one past 2**20 stops there; a step that attends less (new rows
    only, or one request) takes that much."""
    md = SimpleNamespace(
        runtime_features=SimpleNamespace(chunked_prefill=features[0], cache_reuse=features[1]),
        max_num_tokens=tokens,
        max_num_requests=requests,
        max_seq_len=max_seq_len,
    )
    assert M._ctx_workspace_rows(md) == rows


def _recorded(monkeypatch) -> list:
    """Stand-ins for the context attention's kernels, recording each call's pool row, rows and
    selections: the gather writes nothing, the slots are the selections, the prefill returns an
    output of its own, nothing is attended, and each projection records what it reads and
    writes."""
    calls = []

    def gather(pool, row, kv_req, kv_pos, block_table, out):
        calls.append(("gather", row, kv_req.tolist(), out))
        return out

    def prefill(q, ws, idx, scale, out):
        own = torch.empty(q.shape[0], 1, 1)
        calls.append(("prefill", q, ws, idx, out, own))
        return own

    def slots(req, table, sel, tpb, topk, stride, row):
        calls.append(("slots", row, req.tolist()))
        return sel

    def decode(q, kv, slot, sched, scale, out):
        calls.append(("decode", q, sched, out))

    def offset(req, sel, ws_off, out):
        calls.append(("offset", req.tolist(), out))
        return out.copy_(torch.where(sel >= 0, sel + ws_off[req.long()][:, None], sel))

    def v_up(self, L, lat, out):
        calls.append(("v_up", lat, out))

    monkeypatch.setattr(M, "ds_mla_dequant_gather_tiled", gather)
    monkeypatch.setattr(M, "convert_req_index_to_workspace", offset)
    monkeypatch.setattr(M, "flash_mla_sparse_prefill", prefill)
    monkeypatch.setattr(M, "convert_req_index_to_global", slots)
    monkeypatch.setattr(M, "flash_mla_sparse_decode", decode)
    monkeypatch.setattr(M.PrefillTarget, "_v_up", v_up)
    return calls


@pytest.mark.cpu_only
def test_attn_ctx_runs_each_group_on_its_rows(monkeypatch):
    """A step of requests of 15 + 5, 41 and 10 KV rows over a 40-row workspace: a workspace group
    over the first two (rows 0..7), a decode group (7..17) and a workspace group (17..22). Layer
    35, then 39 (its sharer, selecting nothing): each workspace group gathers its own requests
    into the workspace's first rows, attends its rows over them, the two-request group's
    selections offset by 10 for its second request once, into a buffer 39 reads again, and
    projects the prefill's own output into its rows; the decode group's rows become slots,
    attend in one sub-batch into a latent output of the group's own, which is projected into
    its rows; and 39 gathers layer 35 again."""
    calls = _recorded(monkeypatch)
    pool_row = _OWNERS.index(_SHARED)
    target = _target(40, 1, 1, 1)
    contexts = [(5, 5), (3, 2), (31, 10), (5, 5)]
    step = _host_step(contexts)
    ctx = target._ctx_workspace(step, 22, torch.device("cpu"))
    buf = torch.empty(40, _LAT_DIM, dtype=torch.bfloat16)
    assert [g.rows for g in ctx.groups] == [(0, 7), (7, 17), (17, 22)]
    req_idx = torch.tensor([0] * 5 + [1] * 2 + [2] * 10 + [3] * 5, dtype=torch.int32)
    sel = torch.tensor([[0, 1, -1, 2]] * 22, dtype=torch.int32)
    q, out = torch.empty(22, 1), torch.empty(22, 1)
    for layer, selects in ((_SHARED, True), (39, False)):
        L = SimpleNamespace(kv_source=_SOURCES[layer], indexer=object() if selects else None)
        target._attn_ctx(step, L, q, out, req_idx, sel, ctx, buf)
    offsets = [c for c in calls if c[0] == "offset"]
    assert len(offsets) == 1 and offsets[0][1] == [0] * 5 + [1] * 2
    calls = [c for c in calls if c[0] != "offset"]
    assert len(calls) == 18
    for i, layer in enumerate((_SHARED, 39)):
        gather0, prefill0, v_up0, slots, decode, v_up1, gather2, prefill2, v_up2 = calls[
            9 * i : 9 * i + 9
        ]
        assert gather0[:3] == ("gather", pool_row, [0] * 10 + [1] * 5)
        assert _rows_of(gather0[3], buf) == (0, 15)
        assert prefill0[0] == "prefill" and prefill0[2] is gather0[3] and prefill0[4] is None
        assert _rows_of(prefill0[1], q) == (0, 7)
        assert prefill0[3].tolist() == [[0, 1, -1, 2]] * 5 + [[10, 11, -1, 12]] * 2
        # The offset's buffer, the one 35 wrote, is what each layer reads.
        assert prefill0[3] is offsets[0][2], f"layer {layer} read another buffer"
        assert v_up0[1] is prefill0[5] and _rows_of(v_up0[2], out) == (0, 7)
        assert slots == ("slots", pool_row, [2] * 10)
        assert decode[0] == "decode" and decode[2] is ctx.sched[10]
        assert _rows_of(decode[1], q) == (7, 17) and _rows_of(decode[3], v_up1[1]) == (0, 10)
        assert v_up1[1].shape == (10, 1, 1) and _rows_of(v_up1[2], out) == (7, 17)
        assert gather2[:3] == ("gather", pool_row, [3] * 10)
        assert _rows_of(gather2[3], buf) == (0, 10)
        assert _rows_of(prefill2[1], q) == (17, 22) and prefill2[3].tolist() == sel[17:].tolist()
        assert v_up2[1] is prefill2[5] and _rows_of(v_up2[2], out) == (17, 22)


@pytest.mark.cpu_only
def test_decode_group_sub_batches_from_its_first_row(monkeypatch):
    """A 20000-row request past the workspace's rows behind a 5-row one that fits: the decode
    group's rows 5..20005 attend in sub-batches of 8192, 8192 and 3616 from its first row, on
    those sizes' schedulers, into the group's own latent output, projected into rows 5..20005;
    nothing of its KV is gathered."""
    calls = _recorded(monkeypatch)
    target = _target(40, 1, 1, 1)
    step = _host_step([(5, 5), (100, 20000)])
    ctx = target._ctx_workspace(step, 20005, torch.device("cpu"))
    assert [g.rows for g in ctx.groups] == [(0, 5), (5, 20005)]
    assert ctx.groups[1].sub_batches == [(0, 8192), (8192, 16384), (16384, 20000)]
    assert sorted(ctx.sched) == [3616, 8192]
    req_idx = torch.tensor([0] * 5 + [1] * 20000, dtype=torch.int32)
    sel = torch.zeros(20005, 4, dtype=torch.int32)
    q, out = torch.empty(20005, 1), torch.empty(20005, 1)
    L = SimpleNamespace(kv_source=_SOURCES[1], indexer=object())
    target._attn_ctx(step, L, q, out, req_idx, sel, ctx, torch.empty(40, _LAT_DIM))
    gathers = [c for c in calls if c[0] == "gather"]
    assert [kv_req for _, _, kv_req, _ in gathers] == [[0] * 10]
    decodes = [c for c in calls if c[0] == "decode"]
    lat = [c for c in calls if c[0] == "v_up"][1][1]
    assert [_rows_of(c[1], q) for c in decodes] == [(5, 8197), (8197, 16389), (16389, 20005)]
    assert [_rows_of(c[3], lat) for c in decodes] == [(0, 8192), (8192, 16384), (16384, 20000)]
    assert [c[2] for c in decodes] == [ctx.sched[8192], ctx.sched[8192], ctx.sched[3616]]
    assert [_rows_of(c[2], out) for c in calls if c[0] == "v_up"] == [(0, 5), (5, 20005)]


@pytest.mark.cpu_only
@pytest.mark.parametrize("gen_rows", [0, 2], ids=["ctx_alone", "gen_behind"])
def test_attention_buffers_are_one_allocation(monkeypatch, gen_rows):
    """`_attention` on a step of one workspace group (9 context rows, 16 KV rows over a 40-row
    workspace): the absorbed query at all 64 heads and the whole workspace behind it are one
    allocation, and the attention output the context and generation rows are projected into
    is its first bytes. Generation rows behind them attend into a latent output of their own,
    outside that allocation, projected into their rows of the attention output."""
    target = _target(40)
    target.core.qk_dim, target.core.unfused_glue = 128, False
    tc = 9
    rows = tc + gen_rows
    step = _host_step([(7, 3), (0, 6)])
    step["num_tokens"] += gen_rows
    ctx = target._ctx_workspace(step, tc, torch.device("cpu"))
    seen = {"bmm": [], "v_up": []}

    def attn_ctx(step_, L_, fused_q, attn_out, req_idx, topk_buf, ctx_, ws_rows):
        seen.update(ctx_q=fused_q, ctx_attn=attn_out, ws=ws_rows)

    def attn_gen(step_, L_, fused_q, lat, slots, sched):
        seen.update(gen_q=fused_q, gen_lat=lat)

    monkeypatch.setattr(
        target, "_linear_fp8", lambda x, w, w_s: torch.zeros(x.shape[0], _HEADS * 128)
    )
    monkeypatch.setattr(target, "_attn_ctx", attn_ctx)
    monkeypatch.setattr(target, "_attn_gen", attn_gen)
    monkeypatch.setattr(target, "_v_up", lambda L, lat, out: seen["v_up"].append((lat, out)))
    monkeypatch.setattr(M, "ssmax_scaling_fused_q", lambda *args: None)
    monkeypatch.setattr(M, "convert_req_index_to_global", lambda *args: None)
    monkeypatch.setattr(M, "bmm_out", lambda a, b, out: seen["bmm"].append((a, out)))
    L = SimpleNamespace(
        wq=None, wq_s=None, ssmax_s=None, ssmax_b=None, rope=True, indexer=None, nope=64,
        k_b=None, owner=False, kv_source=_SOURCES[1], v_b_t=None,
    )  # fmt: skip
    s = SimpleNamespace(nope_lat=None, sched=None)
    attn = target._attention(
        step, L, torch.zeros(rows, 8), torch.zeros(rows), torch.zeros(rows, dtype=torch.int32),
        torch.zeros(rows, 4, dtype=torch.int32), s, ctx, tc, False, False,
    )  # fmt: skip
    q, ws = seen["ctx_q"], seen["ws"]
    assert q.shape == (tc, _HEADS, _LAT_DIM) and ws.shape == (40, _LAT_DIM)
    assert q.untyped_storage().data_ptr() == ws.untyped_storage().data_ptr()
    assert ws.data_ptr() - q.data_ptr() == rows * _HEADS * _LAT_DIM * 2
    assert attn.shape == (rows, _HEADS * 128) and attn.data_ptr() == q.data_ptr()
    assert seen["ctx_attn"].data_ptr() == attn.data_ptr() and seen["ctx_attn"].shape[0] == tc
    ((absorb_a, absorb_out),) = seen["bmm"]
    assert absorb_out.data_ptr() == q.data_ptr()
    if gen_rows == 0:
        assert "gen_q" not in seen and seen["v_up"] == []
        return
    lat = seen["gen_lat"]
    assert lat.shape == (gen_rows, _HEADS, _NOPE)
    assert lat.untyped_storage().data_ptr() != q.untyped_storage().data_ptr()
    assert seen["gen_q"].data_ptr() == q.data_ptr() + tc * _HEADS * _LAT_DIM * 2
    ((v_lat, v_out),) = seen["v_up"]
    assert v_lat is lat and _rows_of(v_out, attn) == (tc, rows)


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
    """The 63-owner pool, its owner rows of layers 1, 35 and 36 written."""
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
    return SimpleNamespace(dev=dev, pool=pool, kv=pool.view(-1, _TPB, 1, _ROW_BYTES), rows=rows)


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


def _check_step(env, contexts, expected_groups, seed: int) -> SimpleNamespace:
    """Run `_RUN` on one step's groups, the workspace refilled with NaN before every layer, and
    check each layer's output (the module docstring's three checks); return the step's
    groups."""
    s = _step(env, contexts, seed)
    target = _target(_HELD)
    ctx = target._ctx_workspace(s.step, s.tc, env.dev)
    ws = torch.empty(_HELD, _LAT_DIM, dtype=torch.bfloat16, device=env.dev)
    # The projection's stand-in copies each group's latent output into its rows of `out`.
    rows_of_attn = torch.empty(s.tc, 1, device=env.dev)
    out = torch.empty(s.tc, _HEADS, _NOPE, dtype=torch.bfloat16, device=env.dev)

    def capture(L, lat, attn_rows):
        lo, hi = _rows_of(attn_rows, rows_of_attn)
        out[lo:hi].copy_(lat)

    target._v_up = capture
    kinds = ["decode" if g.n_kv is None else "workspace" for g in ctx.groups]
    assert list(zip([g.rows for g in ctx.groups], kinds)) == expected_groups
    new_starts = _cumsum([n for _, n in contexts]).tolist()
    edges = sorted({r for g in ctx.groups for r in (g.rows[0], g.rows[1] - 1)})
    g_cpu = torch.Generator().manual_seed(_SEED + seed)
    sampled = torch.randperm(s.tc, generator=g_cpu)[:_SAMPLED].tolist()
    rows = torch.tensor(sorted(set(sampled) | set(edges)), device=env.dev)
    for layer in _RUN:
        src = _SOURCES[layer]
        out.fill_(float("nan"))
        ws.fill_(float("nan"))
        L = SimpleNamespace(kv_source=src, indexer=None if layer == 39 else object())
        target._attn_ctx(s.step, L, s.q, rows_of_attn, s.req_idx, s.topk, ctx, ws)
        assert bool(torch.isfinite(out).all()), f"layer {layer}: a row read an unwritten byte"
        for g in ctx.groups:
            q0, q1 = g.rows
            if g.n_kv is None:
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
    return ctx


def test_one_small_group(env):
    """1000 + 500 and 500 + 500 KV rows (2500 within 8192): one workspace group, layer 35
    gathered again for 39 after 36 rewrote the workspace's rows."""
    _check_step(env, [(1000, 500), (500, 500)], [((0, 1000), "workspace")], seed=1)


def test_one_group_near_the_workspace_rows(env):
    """3000 + 1000 and 1000 + 500 KV rows (5500 within 8192): one workspace group, layer 35
    gathered again for 39."""
    _check_step(env, [(3000, 1000), (1000, 500)], [((0, 1500), "workspace")], seed=2)


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
    the workspace's rows, decode groups between three workspace groups."""
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
    ctx = _check_step(
        env,
        [(2000, 1000), (9000, 9000)],
        [((0, 1000), "workspace"), ((1000, 10000), "decode")],
        seed=5,
    )
    assert ctx.groups[1].sub_batches == [(0, _SUB), (_SUB, 9000)]


def _v_b_t(dev, seed: int) -> torch.Tensor:
    """A V up-projection weight in the target's transposed view (`L.v_b_t`)."""
    g = torch.Generator(device=dev).manual_seed(_SEED + seed)
    w = (torch.randn(_HEADS, 128, _NOPE, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    return torch.transpose(w, 1, 2)


def test_v_projection_by_group_is_the_whole_steps_bitwise(env):
    """The alternating step's five groups (three workspace, two decode): every layer's attention
    output, projected group by group by `_attn_ctx`, is bitwise `bmm_out` over the step's whole
    latent output in one call, the projection before the step was split."""
    s = _step(env, [(1500, 500), (9000, 700), (2000, 300), (12000, 600), (500, 400)], seed=6)
    target = _target(_HELD)
    ctx = target._ctx_workspace(s.step, s.tc, env.dev)
    assert len(ctx.groups) == 5
    ws = torch.empty(_HELD, _LAT_DIM, dtype=torch.bfloat16, device=env.dev)
    attn = torch.empty(s.tc, _HEADS * 128, dtype=torch.bfloat16, device=env.dev)
    lat = torch.empty(s.tc, _HEADS, _NOPE, dtype=torch.bfloat16, device=env.dev)
    project = target._v_up

    def capture(L, group_lat, rows):
        lo, hi = _rows_of(rows, attn)
        lat[lo:hi].copy_(group_lat)
        project(L, group_lat, rows)

    target._v_up = capture
    whole = torch.empty_like(attn)
    for layer in _RUN:
        L = SimpleNamespace(
            kv_source=_SOURCES[layer], indexer=None if layer == 39 else object(),
            v_b_t=_v_b_t(env.dev, layer),
        )  # fmt: skip
        attn.fill_(float("nan"))
        target._attn_ctx(s.step, L, s.q, attn, s.req_idx, s.topk, ctx, ws)
        bmm_out(
            torch.transpose(lat, 0, 1), L.v_b_t,
            torch.transpose(whole.view(s.tc, _HEADS, 128), 0, 1),
        )  # fmt: skip
        assert torch.equal(attn.view(torch.int16), whole.view(torch.int16)), layer


def test_v_projection_splits_bitwise(env):
    """`bmm_out` with the target's transposed V weight over 20480 latent rows in one call, and
    over the same rows split at 1, 15, 16, 17, 255, 1000, 2051, 8192, 16387 and 20479: every
    piece's rows, at their offsets in one attention output, are bitwise the whole call's."""
    g = torch.Generator(device=env.dev).manual_seed(_SEED + 7)
    n = 20480
    lat = (torch.randn(n, _HEADS, _NOPE, generator=g, device=env.dev) * 0.5).to(torch.bfloat16)
    v_b_t = _v_b_t(env.dev, 7)
    whole = torch.empty(n, _HEADS * 128, dtype=torch.bfloat16, device=env.dev)
    bmm_out(torch.transpose(lat, 0, 1), v_b_t, torch.transpose(whole.view(n, _HEADS, 128), 0, 1))
    cuts = [0, 1, 15, 16, 17, 255, 1000, 2051, 8192, 16387, 20479, n]
    pieces = torch.full_like(whole, float("nan"))
    for lo, hi in zip(cuts, cuts[1:]):
        bmm_out(
            torch.transpose(lat[lo:hi], 0, 1), v_b_t,
            torch.transpose(pieces[lo:hi].view(hi - lo, _HEADS, 128), 0, 1),
        )  # fmt: skip
    assert torch.equal(pieces.view(torch.int16), whole.view(torch.int16))
