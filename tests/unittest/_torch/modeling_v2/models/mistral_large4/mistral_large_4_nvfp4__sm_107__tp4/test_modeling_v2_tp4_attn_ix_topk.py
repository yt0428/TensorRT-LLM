# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for sm_107__tp4 Attention/Indexer dual-stream scheduling, strided
indexer key updates, NoPE/Layer-35 KV sharing, and 1-launch Cluster Radix Top-K dispatch."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_radix_topk_decode import (
    CompiledClusterRadixTopkDecode,
    select_cluster_size,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4 import (
    modeling as tp4_modeling,
)

pytestmark = pytest.mark.cpu_only


class _FakeStream:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeEvent:
    def __init__(self, name: str, trace: list[tuple[str, str, str]], stream_ref: list[str]) -> None:
        self.name = name
        self._trace = trace
        self._stream_ref = stream_ref
        self.recorded = False

    def record(self) -> None:
        self.recorded = True
        self._trace.append(("event.record", self._stream_ref[0], self.name))

    def wait(self) -> None:
        assert self.recorded, f"{self.name}.wait() called before {self.name}.record() (deadlock)"
        self._trace.append(("event.wait", self._stream_ref[0], self.name))


class _RecordingTensor:
    """Thin wrapper around a CPU tensor that records `record_stream` calls."""

    def __init__(self, tensor: torch.Tensor, tag: str, trace: list[tuple[str, str, str]]) -> None:
        self.tensor = tensor
        self.tag = tag
        self._trace = trace

    def record_stream(self, stream) -> None:
        self._trace.append(("record_stream", self.tag, getattr(stream, "name", str(stream))))

    def __getitem__(self, item):
        return self.tensor[item]

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def device(self):
        return self.tensor.device


def _make_fake_layer(
    index: int,
    *,
    has_indexer: bool,
    owner: bool,
    rope: bool,
    kv_source: int,
) -> SimpleNamespace:
    ix = (
        SimpleNamespace(
            rope=rope,
            wk_wp="wk_wp",
            k_norm_w="k_norm_w",
            k_norm_b="k_norm_b",
            q_norm_w="q_norm_w",
            q_norm_b="q_norm_b",
            wq_b="wq_b",
            wq_b_s="wq_b_s",
            ssmax_s="ssmax_s",
            ssmax_b="ssmax_b",
        )
        if has_indexer
        else None
    )
    return SimpleNamespace(
        index=index,
        indexer=ix,
        owner=owner,
        rope=rope,
        kv_source=kv_source,
        nope=64 if rope else 128,
        wq="wq",
        wq_s="wq_s",
        ssmax_s="l_ssmax_s",
        ssmax_b="l_ssmax_b",
        k_b="k_b",
        v_b_t="v_b_t",
        wkv_a="wkv_a",
        kv_norm="kv_norm",
    )


def _setup_attention_harness(
    monkeypatch,
    *,
    multi_stream: bool,
    capturing: bool,
    rows: int = 4,
    max_ctx_len: int = 1048584,
):
    trace: list[tuple[str, str, str]] = []
    cur_stream = ["main"]
    main_stream = _FakeStream("main")
    side_stream = _FakeStream("side")

    @contextmanager
    def _fake_stream_ctx(s):
        prev = cur_stream[0]
        cur_stream[0] = s.name
        try:
            yield
        finally:
            cur_stream[0] = prev

    def _fake_maybe_parallel(fn0, fn1, ev0, ev1, aux_stream=None, disable_on_compile=False):
        if multi_stream and aux_stream is not None:
            ev0.record()
            r0 = fn0()
            with _fake_stream_ctx(aux_stream):
                ev0.wait()
                r1 = fn1()
                ev1.record()
            ev1.wait()
            return (r0, r1)
        return (fn0(), fn1())

    monkeypatch.setattr(tp4_modeling, "do_multi_stream", lambda: multi_stream)
    monkeypatch.setattr(tp4_modeling, "maybe_execute_in_parallel", _fake_maybe_parallel)
    monkeypatch.setattr(torch.cuda, "stream", _fake_stream_ctx)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **kw: main_stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    heads, qk_dim, kv_lora, rope_dim, v_dim = 8, 128, 512, 64, 128
    idx_heads, idx_dim, idx_topk = 32, 128, 2048
    rope_idx = torch.zeros(16, rope_dim)
    rope_id = torch.ones(16, rope_dim)

    core = SimpleNamespace(
        heads=heads,
        qk_dim=qk_dim,
        kv_lora=kv_lora,
        rope_dim=rope_dim,
        lat_dim=kv_lora + rope_dim,
        v_dim=v_dim,
        idx_heads=idx_heads,
        idx_dim=idx_dim,
        idx_topk=idx_topk,
        idx_eps=1e-6,
        idx_weight_scale=1.0,
        eps=1e-6,
        softmax_scale=1.0,
        unfused_glue=False,
        gen_attn_flash_mla=False,
        _rope=SimpleNamespace(indexer=rope_idx, identity=rope_id),
        _ix_k={True: "ix_k_contig", False: "ix_k_strided"},
        _ix_q=None,
        _ix_topk=None,
        _ix_topk_verify=None,
        _mla_decode="mla_decode",
        _max_ctx_len=max_ctx_len,
        _stride_factor=63 * 32,
        _pool_row=list(range(72)),
    )

    k_update_calls: list[dict] = []
    convert_calls: list[dict] = []
    rope_append_calls: list[dict] = []
    ssmax_fq_calls: list[dict] = []
    mla_calls: list[dict] = []

    def _cublas_mm(x, weight):
        if weight == "wk_wp":
            trace.append(("op", cur_stream[0], "wk_wp"))
            raw = torch.arange(rows * (idx_dim + idx_heads), dtype=torch.bfloat16).reshape(
                rows, idx_dim + idx_heads
            )
            return _RecordingTensor(raw, "kw", trace)
        if weight == "wkv_a":
            trace.append(("op", cur_stream[0], "wkv_a"))
            raw = torch.zeros(rows, kv_lora + rope_dim, dtype=torch.bfloat16)
            return _RecordingTensor(raw, "latent", trace)
        raise AssertionError(f"unexpected cublas_mm weight {weight}")

    def _k_cache_update(k, *args, **kwargs):
        trace.append(("op", cur_stream[0], "k_cache_update"))
        k_update_calls.append(
            dict(
                stream=cur_stream[0],
                shape=tuple(k.shape),
                stride=tuple(k.stride()),
                is_contiguous=k.is_contiguous(),
                table=args[3],
                k_pool=args[4],
                num_tokens=args[7],
            )
        )
        return args[4]

    def _layer_norm(x, *args):
        trace.append(("op", cur_stream[0], "q_input_norm"))
        return x

    def _linear_fp8(x, w, w_s):
        trace.append(("op", cur_stream[0], f"linear_fp8:{w}"))
        if w == "wq_b":
            return torch.zeros(rows, idx_heads * idx_dim, dtype=torch.bfloat16)
        if w == "wq":
            return torch.zeros(rows, heads * qk_dim, dtype=torch.bfloat16)
        raise AssertionError(f"unexpected linear_fp8 weight {w}")

    def _ssmax_ix_q(*args):
        trace.append(("op", cur_stream[0], "ssmax_ix_q"))
        return args[7], args[8]

    def _mqa_logits(*args):
        trace.append(("op", cur_stream[0], "mqa_logits"))
        return torch.zeros(rows, max_ctx_len, dtype=torch.float32)

    def _select_decode(logits, kv_lens, topk_buf, next_n, aux_i, aux_l):
        trace.append(("op", cur_stream[0], "select_indexer_decode"))
        return topk_buf

    def _convert(req_idx, block_table, topk_buf, tpb, topk, stride_factor, pool_row):
        trace.append(("op", cur_stream[0], "convert_req_index_to_global"))
        convert_calls.append(dict(stream=cur_stream[0], pool_row=pool_row))
        return torch.zeros(rows, topk, dtype=torch.int32)

    def _ssmax_fq(q3, pos, s, b, out_q, out_tail, rope):
        trace.append(("op", cur_stream[0], "ssmax_scaling_fused_q"))
        ssmax_fq_calls.append(dict(stream=cur_stream[0], rope=rope))

    def _bmm_out(a, b, out):
        trace.append(("op", cur_stream[0], f"bmm_out:{b}"))

    def _rmsnorm(x, w, eps, out=None):
        trace.append(("op", cur_stream[0], "kv_a_norm"))
        return out if out is not None else x

    def _rope_append(lat, fq, pos, rope, req_idx, block_table, pool, pool_row, is_rope):
        trace.append(("op", cur_stream[0], "ds_mla_rope_append"))
        rope_append_calls.append(
            dict(stream=cur_stream[0], pool_row=pool_row, is_rope=is_rope, lat=lat)
        )

    def _sparse_decode(*args, **kwargs):
        trace.append(("op", cur_stream[0], "sparse_mla_decode"))
        mla_calls.append(dict(args=args, kwargs=kwargs))

    monkeypatch.setattr(tp4_modeling, "cublas_mm", _cublas_mm)
    monkeypatch.setattr(tp4_modeling, "cute_dsl_indexer_k_cache_update", _k_cache_update)
    monkeypatch.setattr(tp4_modeling, "layer_norm", _layer_norm)
    monkeypatch.setattr(tp4_modeling, "ssmax_indexer_q_side_fused", _ssmax_ix_q)
    monkeypatch.setattr(tp4_modeling, "cute_dsl_fp8_paged_mqa_logits", _mqa_logits)
    monkeypatch.setattr(tp4_modeling, "convert_req_index_to_global", _convert)
    monkeypatch.setattr(tp4_modeling, "ssmax_scaling_fused_q", _ssmax_fq)
    monkeypatch.setattr(tp4_modeling, "bmm_out", _bmm_out)
    monkeypatch.setattr(tp4_modeling, "flashinfer_rmsnorm", _rmsnorm)
    monkeypatch.setattr(tp4_modeling, "ds_mla_rope_append", _rope_append)
    monkeypatch.setattr(tp4_modeling, "cute_dsl_ds_mla_sparse_decode", _sparse_decode)

    core._select_indexer_decode = _select_decode

    target = SimpleNamespace(
        core=core,
        _latent_stream=side_stream,
        _latent_fork=_FakeEvent("latent_fork", trace, cur_stream),
        _latent_join=_FakeEvent("latent_join", trace, cur_stream),
        _keys_fork=_FakeEvent("keys_fork", trace, cur_stream),
        _keys_ready=_FakeEvent("keys_ready", trace, cur_stream),
        _latent_ready=_FakeEvent("latent_ready", trace, cur_stream),
        _linear_fp8=_linear_fp8,
    )
    target._latent = MethodType(tp4_modeling.DecodeTarget._latent, target)
    target._indexer_keys = MethodType(tp4_modeling.DecodeTarget._indexer_keys, target)
    target._indexer = MethodType(tp4_modeling.DecodeTarget._indexer, target)
    target._attention = MethodType(tp4_modeling.DecodeTarget._attention, target)

    step = dict(
        num_seqs=rows // 4,
        ix_pools={i: torch.empty(1) for i in range(72)},
        slot_mapping_fp8="slot_fp8",
        slot_mapping_scale="slot_scale",
        kv_lens_cuda=torch.full((rows // 4,), 780000, dtype=torch.int32),
        indexer_k_cache_block_offsets=torch.zeros(rows // 4, 1, dtype=torch.int32),
        scheduler_metadata_buffer="sched",
        radix_aux_indices="aux_i",
        radix_aux_logits="aux_l",
        block_table="block_table",
        tokens_per_block=32,
        latent_pool="latent_pool",
        latent_kv="latent_kv",
    )
    scratch = SimpleNamespace(
        fused_q=torch.zeros(rows, heads, kv_lora + rope_dim, dtype=torch.bfloat16),
        nope_lat=torch.zeros(rows, kv_lora + rope_dim, dtype=torch.bfloat16),
        lat_out=torch.zeros(rows, heads, kv_lora, dtype=torch.bfloat16),
        attn_out=torch.zeros(rows, heads * v_dim, dtype=torch.bfloat16),
        sched=None,
    )
    x = torch.zeros(rows, 6144, dtype=torch.bfloat16)
    pos = torch.arange(rows, dtype=torch.int32)
    req_idx = torch.zeros(rows, dtype=torch.int32)
    topk_buf = torch.zeros(rows, idx_topk, dtype=torch.int32)

    return SimpleNamespace(
        target=target,
        core=core,
        step=step,
        scratch=scratch,
        x=x,
        pos=pos,
        req_idx=req_idx,
        topk_buf=topk_buf,
        trace=trace,
        k_update_calls=k_update_calls,
        convert_calls=convert_calls,
        rope_append_calls=rope_append_calls,
        ssmax_fq_calls=ssmax_fq_calls,
        mla_calls=mla_calls,
        rope_idx=rope_idx,
        rope_id=rope_id,
    )


@pytest.mark.parametrize("capturing", [True, False])
@pytest.mark.parametrize("rope", [True, False])
def test_indexer_layer_strided_keys_and_side_stream_overlap(monkeypatch, capturing, rope):
    """Strided `kw[:, :128]` is read in place without `.contiguous()`, and the key half leads on
    the side stream to overlap with `q_input_norm + wq_b` on the main stream."""
    h = _setup_attention_harness(monkeypatch, multi_stream=True, capturing=capturing)
    layer = _make_fake_layer(
        3 if not rope else 0,
        has_indexer=True,
        owner=True,
        rope=rope,
        kv_source=3 if not rope else 0,
    )
    h.target._attention(h.step, layer, h.x, h.pos, h.req_idx, h.topk_buf, h.scratch, skip_gen=False)

    # Strided slice [4, 128] with row stride 160 is passed directly without a contiguous copy.
    assert len(h.k_update_calls) == 1
    k_call = h.k_update_calls[0]
    assert k_call["stream"] == "side"
    assert k_call["shape"] == (4, 128)
    assert k_call["stride"] == (160, 1)
    assert not k_call["is_contiguous"]
    expected_table = (h.rope_idx if rope else h.rope_id).view(-1, 2, 32)
    assert torch.equal(k_call["table"], expected_table)

    # Key half runs on side stream; q_input_norm + wq_b run on main BEFORE keys_ready.wait().
    t = h.trace
    idx_kf_rec = t.index(("event.record", "main", "keys_fork"))
    idx_kf_wait = t.index(("event.wait", "side", "keys_fork"))
    idx_wk = t.index(("op", "side", "wk_wp"))
    idx_ku = t.index(("op", "side", "k_cache_update"))
    idx_kr_rec = t.index(("event.record", "side", "keys_ready"))
    idx_qn = t.index(("op", "main", "q_input_norm"))
    idx_wqb = t.index(("op", "main", "linear_fp8:wq_b"))
    idx_kr_wait = t.index(("event.wait", "main", "keys_ready"))
    idx_qs = t.index(("op", "main", "ssmax_ix_q"))
    idx_mqa = t.index(("op", "main", "mqa_logits"))
    idx_lj_wait = t.index(("event.wait", "main", "latent_join"))
    idx_mla = t.index(("op", "main", "sparse_mla_decode"))

    assert idx_kf_rec < idx_kf_wait < idx_wk < idx_ku < idx_kr_rec
    assert idx_qn < idx_wqb < idx_kr_wait < idx_qs < idx_mqa < idx_lj_wait < idx_mla
    # Query half + latent still run on side stream on indexer layers.
    assert ("op", "side", "linear_fp8:wq") in t
    assert ("op", "side", "wkv_a") in t
    assert ("op", "side", "ds_mla_rope_append") in t
    # record_stream only outside capture.
    assert (("record_stream", "kw", "main") in t) == (not capturing)


def test_indexer_layer_skip_gen_still_appends_keys_without_orphan_event(monkeypatch):
    """When skip_gen=True, _indexer_keys still updates the key pool on side stream, scoring is
    skipped, no orphan keys_ready event is recorded, and side joins cleanly via latent_join."""
    h = _setup_attention_harness(monkeypatch, multi_stream=True, capturing=True)
    layer = _make_fake_layer(0, has_indexer=True, owner=True, rope=True, kv_source=0)
    h.target._attention(h.step, layer, h.x, h.pos, h.req_idx, h.topk_buf, h.scratch, skip_gen=True)

    assert len(h.k_update_calls) == 1
    assert h.k_update_calls[0]["stream"] == "side"
    assert ("event.record", "side", "keys_ready") not in h.trace
    assert ("op", "main", "q_input_norm") not in h.trace
    assert ("op", "main", "mqa_logits") not in h.trace
    assert ("event.wait", "main", "latent_join") in h.trace


@pytest.mark.parametrize("capturing", [True, False])
def test_non_indexer_layer_latent_runs_on_main_stream(monkeypatch, capturing):
    """On non-indexer owner layers, wkv_a + kv_a_norm run on main stream and signal
    `_latent_ready`; side stream runs wq + ssmax + k_b before waiting on `_latent_ready` at
    ds_mla_rope_append, and SparseMLA resolves req indices inline without convert_req_index_to_global."""
    h = _setup_attention_harness(monkeypatch, multi_stream=True, capturing=capturing)
    # Layer 17 is a non-indexer RoPE owner (17 % 4 == 1).
    layer = _make_fake_layer(17, has_indexer=False, owner=True, rope=True, kv_source=17)
    h.target._attention(h.step, layer, h.x, h.pos, h.req_idx, h.topk_buf, h.scratch, skip_gen=False)

    t = h.trace
    assert len(h.k_update_calls) == 0
    assert len(h.convert_calls) == 0
    idx_wkva = t.index(("op", "main", "wkv_a"))
    idx_kvnorm = t.index(("op", "main", "kv_a_norm"))
    idx_lr_rec = t.index(("event.record", "main", "latent_ready"))
    idx_wq = t.index(("op", "side", "linear_fp8:wq"))
    idx_kb = t.index(("op", "side", "bmm_out:k_b"))
    idx_lr_wait = t.index(("event.wait", "side", "latent_ready"))
    idx_app = t.index(("op", "side", "ds_mla_rope_append"))
    idx_lj_wait = t.index(("event.wait", "main", "latent_join"))
    idx_mla = t.index(("op", "main", "sparse_mla_decode"))

    assert idx_wkva < idx_kvnorm < idx_lr_rec < idx_mla
    assert idx_wq < idx_kb < idx_lr_wait < idx_app < idx_lj_wait < idx_mla
    assert h.mla_calls[-1]["kwargs"]["layer_id"] == 17
    assert h.mla_calls[-1]["kwargs"]["req_id"] is h.req_idx
    assert h.mla_calls[-1]["kwargs"]["block_table"] == "block_table"
    assert (("record_stream", "latent", "side") in t) == (not capturing)


def test_layer35_nope_owner_and_layer39_kv_sharer_contract(monkeypatch):
    """Verify Layer 35 (NoPE owner) appends to pool row 35 and Layer 39 (NoPE KV sharer) skips
    latent/append while routing inline req-index lookup to pool row 35."""
    h = _setup_attention_harness(monkeypatch, multi_stream=True, capturing=True)
    layer35 = _make_fake_layer(35, has_indexer=True, owner=True, rope=False, kv_source=35)
    h.target._attention(
        h.step, layer35, h.x, h.pos, h.req_idx, h.topk_buf, h.scratch, skip_gen=False
    )
    assert len(h.rope_append_calls) == 1
    assert h.rope_append_calls[0]["pool_row"] == 35
    assert h.rope_append_calls[0]["is_rope"] is False
    assert len(h.convert_calls) == 0
    assert h.mla_calls[-1]["kwargs"]["layer_id"] == 35
    assert h.ssmax_fq_calls[-1]["rope"] is False

    h.trace.clear()
    h.rope_append_calls.clear()
    layer39 = _make_fake_layer(39, has_indexer=True, owner=False, rope=False, kv_source=35)
    h.target._attention(
        h.step, layer39, h.x, h.pos, h.req_idx, h.topk_buf, h.scratch, skip_gen=False
    )
    assert len(h.rope_append_calls) == 0
    assert ("op", "side", "wkv_a") not in h.trace
    assert ("op", "main", "wkv_a") not in h.trace
    assert len(h.convert_calls) == 0
    assert h.mla_calls[-1]["kwargs"]["layer_id"] == 35
    assert h.ssmax_fq_calls[-1]["rope"] is False


def test_fused_combine_vup_skips_convert_and_bmm_vup(monkeypatch):
    """When compiled supports fused combine+V-up for low rows, DecodeTarget._attention skips both
    convert_req_index_to_global and bmm_out(v_b_t) and writes directly into s.attn_out."""
    h = _setup_attention_harness(monkeypatch, multi_stream=True, capturing=True, rows=4)
    h.core._mla_decode = SimpleNamespace(combine_vup_fns={16: object(), 32: object()}, num_sms=212)
    layer = _make_fake_layer(0, has_indexer=True, owner=True, rope=True, kv_source=0)
    out = h.target._attention(
        h.step, layer, h.x, h.pos, h.req_idx, h.topk_buf, h.scratch, skip_gen=False
    )
    assert out is h.scratch.attn_out
    assert len(h.convert_calls) == 0
    assert ("op", "main", "bmm_out:v_b_t") not in h.trace
    assert len(h.mla_calls) == 1
    assert h.mla_calls[0]["kwargs"]["v_b_t"] == "v_b_t"
    assert h.mla_calls[0]["kwargs"]["req_id"] is h.req_idx
    assert tuple(h.mla_calls[0]["args"][4].shape) == (4, 8, 128)

    # Past GLOBAL_MAX_ROWS, combine+V-up falls back to bmm_out(v_b_t) while keeping inline req_idx.
    h16 = _setup_attention_harness(monkeypatch, multi_stream=True, capturing=True, rows=16)
    h16.core._mla_decode = SimpleNamespace(
        combine_vup_fns={16: object(), 32: object()}, num_sms=212
    )
    h16.target._attention(
        h16.step, layer, h16.x, h16.pos, h16.req_idx, h16.topk_buf, h16.scratch, skip_gen=False
    )
    assert len(h16.convert_calls) == 0
    assert ("op", "main", "bmm_out:v_b_t") in h16.trace
    assert "v_b_t" not in h16.mla_calls[0]["kwargs"]
    assert h16.mla_calls[0]["kwargs"]["req_id"] is h16.req_idx


def test_single_stream_mode_runs_all_ops_inline_on_main(monkeypatch):
    """Outside multi-stream (eager), neither indexer layers nor non-indexer layers switch streams."""
    for has_indexer, idx in ((True, 0), (False, 17)):
        h = _setup_attention_harness(monkeypatch, multi_stream=False, capturing=False)
        layer = _make_fake_layer(idx, has_indexer=has_indexer, owner=True, rope=True, kv_source=idx)
        h.target._attention(
            h.step, layer, h.x, h.pos, h.req_idx, h.topk_buf, h.scratch, skip_gen=False
        )
        assert all(entry[1] == "main" for entry in h.trace if entry[0] == "op")
        assert all(entry[0] != "event.record" for entry in h.trace)
        assert all(entry[0] != "event.wait" for entry in h.trace)


def test_select_indexer_topk_routing_and_override_hook(monkeypatch):
    """Verify `_select_indexer_topk` routes next_n=4 rows in {4,8,16,32,64} at 20480..1048584 to
    `cluster_radix_topk_decode` with 8/16/32 cluster resolution, falls back cleanly on out-of-range
    shapes, and exposes `_select_indexer_topk` as a single override point."""

    def _handle(c: int) -> CompiledClusterRadixTopkDecode:
        return CompiledClusterRadixTopkDecode(
            launch=object(),
            device_id=0,
            next_n=4,
            cluster=c,
            threads=512,
            unroll=2,
            pipe=True,
            min_blocks=0,
            use_pdl=True,
            sm_count=212,
        )

    variants = {8: _handle(8), 16: _handle(16), 32: _handle(32)}
    core = SimpleNamespace(idx_topk=2048, _ix_topk="radix_1cta", _ix_topk_verify=variants)
    core._select_indexer_topk = MethodType(
        tp4_modeling.MistralLarge4Tp4ModelingV2._select_indexer_topk, core
    )
    core._select_indexer_decode = MethodType(
        tp4_modeling.MistralLarge4Tp4ModelingV2._select_indexer_decode, core
    )

    cluster_mock = Mock(side_effect=lambda *a, **kw: a[2])
    radix_mock = Mock(side_effect=lambda *a, **kw: a[2])
    native_mock = Mock(side_effect=lambda *a, **kw: a[2])
    monkeypatch.setattr(tp4_modeling, "cluster_radix_topk_decode", cluster_mock)
    monkeypatch.setattr(tp4_modeling, "cute_dsl_radix_topk_decode", radix_mock)
    monkeypatch.setattr(tp4_modeling, "indexer_topk_decode", native_mock)

    for rows, width, exp_cluster in (
        (4, 780000, 32),
        (4, 262144, 32),
        (4, 131072, 16),
        (4, 65536, 8),
        (4, 32768, 8),
        (8, 780000, 16),
        (16, 780000, 8),
        (32, 1048584, 8),
        (64, 1048584, 8),
    ):
        cluster_mock.reset_mock()
        logits = torch.empty(rows, width, dtype=torch.float32)
        kv_lens = torch.full((rows // 4,), width, dtype=torch.int32)
        indices = torch.empty(rows, 2048, dtype=torch.int32)
        out = core._select_indexer_decode(logits, kv_lens, indices, 4, "aux_i", "aux_l")
        assert out is indices
        cluster_mock.assert_called_once()
        assert cluster_mock.call_args.kwargs["compiled"] is variants
        assert select_cluster_size(rows, width, 212, tuple(variants.keys())) == exp_cluster

    # Out-of-range rows (96) or width (16384) fall back to native op.
    for rows, width in ((96, 1048584), (4, 16384)):
        native_mock.reset_mock()
        logits = torch.empty(rows, width, dtype=torch.float32)
        kv_lens = torch.full((rows // 4,), width, dtype=torch.int32)
        indices = torch.empty(rows, 2048, dtype=torch.int32)
        core._select_indexer_decode(logits, kv_lens, indices, 4, "aux_i", "aux_l")
        native_mock.assert_called_once()

    # next_n=1 hits 1-CTA radix selector.
    radix_mock.reset_mock()
    core._select_indexer_decode(
        torch.empty(4, 32768, dtype=torch.float32),
        torch.full((4,), 32768, dtype=torch.int32),
        torch.empty(4, 2048, dtype=torch.int32),
        1,
        "aux_i",
        "aux_l",
    )
    radix_mock.assert_called_once()

    # Overriding `_select_indexer_topk` intercepts `_select_indexer_decode` at one hook point.
    custom_hook = Mock(return_value="custom_out")
    core._select_indexer_topk = custom_hook
    assert (
        core._select_indexer_decode(
            torch.empty(4, 780000, dtype=torch.float32),
            torch.tensor([780000], dtype=torch.int32),
            torch.empty(4, 2048, dtype=torch.int32),
            4,
            "aux_i",
            "aux_l",
        )
        == "custom_out"
    )
    custom_hook.assert_called_once()


def test_decode_indexer_positional_compat(monkeypatch):
    """Verify backward-compatible positional signature `DecodeTarget._indexer(target, step, x,
    pos, L, output, False)` (as called by `test_native_dspark.py`)."""
    h = _setup_attention_harness(monkeypatch, multi_stream=False, capturing=False, rows=4)
    # Strip `_indexer_keys` and `_select_indexer_topk` to mirror test_native_dspark's SimpleNamespace.
    bare_core = SimpleNamespace(**vars(h.core))
    bare_core._ix_topk_verify = None
    bare_core._select_indexer_decode = MethodType(
        tp4_modeling.MistralLarge4Tp4ModelingV2._select_indexer_decode, bare_core
    )
    native_mock = Mock(side_effect=lambda *a, **kw: a[2])
    monkeypatch.setattr(tp4_modeling, "indexer_topk_decode", native_mock)

    bare_target = SimpleNamespace(core=bare_core, _linear_fp8=h.target._linear_fp8)
    layer = _make_fake_layer(0, has_indexer=True, owner=True, rope=True, kv_source=0)
    tp4_modeling.DecodeTarget._indexer(bare_target, h.step, h.x, h.pos, layer, h.topk_buf, False)
    assert len(h.k_update_calls) == 1
    assert h.k_update_calls[0]["stride"] == (160, 1)
    native_mock.assert_called_once()
