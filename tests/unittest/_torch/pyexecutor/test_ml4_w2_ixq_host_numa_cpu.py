# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for fused Indexer Q contracts, NUMA-local pinned memory,
incremental block-offset updates, and steady DSA/DSpark prepare fast paths."""

from __future__ import annotations

import importlib
import math
from types import MethodType, SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4 import (
    modeling as tp4_modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4 import (
    runtime_hooks as ml4_hooks,
)
from tensorrt_llm._torch.pyexecutor import h2d_cache, numa_pinned

_TARGET_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models."
    "mistral_large4.mistral_large_4_nvfp4__sm_107__tp4"
)
ml4_ix_q = importlib.import_module(f"{_TARGET_PKG}.ops.gemm.ml4_ix_q_fused")

pytestmark = pytest.mark.cpu_only


# =============================================================================
# 1. ml4_ix_q_fused host configuration, sizing, and exact log tests
# =============================================================================


def test_ix_q_candidate_configs_fit_smem_and_tmem():
    for cfg in ml4_ix_q.CANDIDATE_CONFIGS:
        resolved = ml4_ix_q.get_config(cfg)
        assert resolved is not None, f"config {cfg} should resolve"
        assert ml4_ix_q.smem_bytes(resolved) <= 220 * 1024
        my_tiles = ml4_ix_q.K_TILES // resolved.split
        tcols = ml4_ix_q.tmem_cols(resolved.ring, my_tiles, resolved.a_tmem)
        assert tcols <= 512
        # split=8 configs target 2 CTAs/SM: verify SMEM <= 112 KB and TMEM <= 256 cols.
        if resolved.split == 8:
            assert ml4_ix_q.smem_bytes(resolved) <= 112 * 1024
            assert tcols <= 256
    assert ml4_ix_q.get_config(None) == ml4_ix_q.DEFAULT_CONFIG
    assert ml4_ix_q.get_config(ml4_ix_q.IxQFusedConfig("cluster", 6, 8)) is None
    assert ml4_ix_q.get_config(ml4_ix_q.IxQFusedConfig("l2", 5, 4)) is None
    assert ml4_ix_q.serves(1)
    assert ml4_ix_q.serves(4)
    assert ml4_ix_q.serves(8)
    assert not ml4_ix_q.serves(0)
    assert not ml4_ix_q.serves(9)


def test_exact_log_bf16_host_matches_math_log_across_1m_range():
    positions = list(range(0, 2048)) + [
        4095,
        4096,
        8191,
        16384,
        32767,
        65535,
        131072,
        262144,
        384000,
        524288,
        785008,
        1048574,
        1048575,
    ]
    for p in positions:
        got = torch.tensor(ml4_ix_q.exact_log_bf16_host(p), dtype=torch.float32).to(torch.bfloat16)
        ref = torch.tensor(math.log(float(p) + 1.0), dtype=torch.float32).to(torch.bfloat16)
        assert torch.equal(got, ref), f"mismatch at pos={p}: got={got}, ref={ref}"


# =============================================================================
# 2. DecodeTarget._indexer fused-ix-q wiring & stream-ordering contract tests
# =============================================================================


class _FakeEvent:
    def __init__(self, name: str, trace: list[str]) -> None:
        self.name = name
        self._trace = trace
        self.wait_count = 0

    def record(self) -> None:
        self._trace.append(f"record:{self.name}")

    def wait(self) -> None:
        self.wait_count += 1
        self._trace.append(f"wait:{self.name}")


@pytest.mark.parametrize(
    ("rows", "fused_returns", "unfused_glue", "expect_fused"),
    [
        (4, True, False, True),
        (4, False, False, False),
        (16, True, False, False),
        (4, True, True, False),
    ],
)
def test_decode_target_indexer_fused_ix_q_ordering_and_fallback(
    monkeypatch, rows: int, fused_returns: bool, unfused_glue: bool, expect_fused: bool
):
    trace: list[str] = []
    idx_heads, idx_dim, idx_topk = 32, 128, 2048

    def _fake_fused(*args, wp=None, **kwargs):
        assert wp is not None
        trace.append("fused_ix_q:inline_wp")
        return fused_returns

    def _fake_ln(x, *a):
        trace.append("layer_norm")
        return x

    def _fake_linear_fp8(self_unused, x, w, w_s):
        trace.append("linear_fp8")
        return torch.zeros(rows, idx_heads * idx_dim, dtype=torch.bfloat16)

    def _fake_ssmax(*args):
        trace.append("ssmax_ix_q")
        return args[7], args[8]

    def _fake_ssmax_scaling(q, *a):
        trace.append("ssmax_scaling")
        return q

    def _fake_q_side(*args, **kwargs):
        trace.append("q_side_fused")
        return args[5], args[6]

    def _fake_mqa(*args):
        trace.append("mqa_logits")
        return torch.zeros(rows, 2048, dtype=torch.float32)

    def _fake_select(*args):
        trace.append("select_decode")

    monkeypatch.setattr(tp4_modeling, "ml4_ix_q_fused", _fake_fused)
    monkeypatch.setattr(tp4_modeling, "ml4_ix_q_fused_serves", lambda r: r <= 8)
    monkeypatch.setattr(tp4_modeling, "layer_norm", _fake_ln)
    monkeypatch.setattr(tp4_modeling, "ssmax_indexer_q_side_fused", _fake_ssmax)
    monkeypatch.setattr(tp4_modeling, "ssmax_scaling", _fake_ssmax_scaling)
    monkeypatch.setattr(tp4_modeling, "cute_dsl_indexer_q_side_fused", _fake_q_side)
    monkeypatch.setattr(tp4_modeling, "cute_dsl_fp8_paged_mqa_logits", _fake_mqa)

    core = SimpleNamespace(
        idx_heads=idx_heads,
        idx_dim=idx_dim,
        idx_topk=idx_topk,
        idx_eps=1e-6,
        idx_weight_scale=1.0,
        unfused_glue=unfused_glue,
        _ix_q="compiled_ix_q",
        _rope=SimpleNamespace(indexer=torch.zeros(16, 64), identity=torch.ones(16, 64)),
        _max_ctx_len=2048,
        _select_indexer_decode=_fake_select,
    )
    target = SimpleNamespace(core=core)
    target._linear_fp8 = MethodType(_fake_linear_fp8, target)
    ix = SimpleNamespace(
        q_norm_w=torch.ones(6144, dtype=torch.bfloat16),
        q_norm_b=torch.zeros(6144, dtype=torch.bfloat16),
        wq_b="wq_b",
        wq_b_s="wq_b_s",
        wp=torch.zeros(idx_heads, 6144, dtype=torch.bfloat16),
        ssmax_s=torch.zeros(idx_heads, dtype=torch.bfloat16),
        ssmax_b=torch.ones(idx_heads, dtype=torch.bfloat16),
        rope=True,
    )
    layer = SimpleNamespace(index=0, indexer=ix)
    step = dict(
        num_seqs=1,
        kv_lens_cuda=torch.tensor([4096], dtype=torch.int32),
        ix_pools={0: "k_pool"},
        indexer_k_cache_block_offsets=torch.zeros(1, 64, dtype=torch.int32),
        scheduler_metadata_buffer="sched",
        radix_aux_indices="aux_i",
        radix_aux_logits="aux_l",
    )
    keys_ready = _FakeEvent("keys_ready", trace)
    kw = torch.zeros(rows, idx_dim + idx_heads, dtype=torch.bfloat16)
    x = torch.zeros(rows, 6144, dtype=torch.bfloat16)
    pos = torch.arange(rows, dtype=torch.int32)
    topk_buf = torch.zeros(rows, idx_topk, dtype=torch.int32)

    tp4_modeling.DecodeTarget._indexer(
        target, step, x, pos, layer, topk_buf, False, None, kw=kw, keys_ready=keys_ready
    )

    assert keys_ready.wait_count == 1
    if expect_fused:
        assert "layer_norm" not in trace
        assert "linear_fp8" not in trace
        assert "ssmax_ix_q" not in trace
        assert trace.index("fused_ix_q:inline_wp") < trace.index("wait:keys_ready")
        assert trace.index("wait:keys_ready") < trace.index("mqa_logits")
    else:
        assert "layer_norm" in trace
        assert "linear_fp8" in trace
        if unfused_glue:
            assert "q_side_fused" in trace
            assert trace.index("wait:keys_ready") < trace.index("q_side_fused")
        else:
            assert "ssmax_ix_q" in trace
            assert trace.index("wait:keys_ready") < trace.index("ssmax_ix_q")


# =============================================================================
# 3. NUMA-local pinned allocation & runtime_hooks tests
# =============================================================================


def test_numa_local_pin_context_binds_and_restores_on_exception(monkeypatch):
    calls: list[tuple] = []
    cur_aff = {0, 1, 2, 3, 88, 89, 90, 91}

    monkeypatch.setattr(numa_pinned, "gpu_numa_node", lambda _d=None: 1)
    monkeypatch.setattr(numa_pinned, "numa_node_cpus", lambda _n: {88, 89, 90, 91})
    monkeypatch.setattr(numa_pinned.os, "sched_getaffinity", lambda _pid: set(cur_aff))

    def _set_aff(_pid: int, mask: set[int]) -> None:
        calls.append(("aff", sorted(mask)))
        cur_aff.clear()
        cur_aff.update(mask)

    def _set_mpol(node):
        calls.append(("mpol", node))
        return True

    monkeypatch.setattr(numa_pinned.os, "sched_setaffinity", _set_aff)
    monkeypatch.setattr(numa_pinned, "_set_mempolicy_node", _set_mpol)

    with pytest.raises(RuntimeError, match="boom"):
        with numa_pinned.numa_local_pin_context(0) as node:
            assert node == 1
            assert cur_aff == {88, 89, 90, 91}
            raise RuntimeError("boom")

    assert calls == [
        ("aff", [88, 89, 90, 91]),
        ("mpol", 1),
        ("mpol", None),
        ("aff", [0, 1, 2, 3, 88, 89, 90, 91]),
    ]

    # When no GPU NUMA node is discovered, context cleanly yields None without OS calls.
    calls.clear()
    monkeypatch.setattr(numa_pinned, "gpu_numa_node", lambda _d=None: None)
    with numa_pinned.numa_local_pin_context(0) as node:
        assert node is None
    assert calls == []


def test_ml4_runtime_hooks_enables_incremental_block_offsets_and_steady_dsa():
    hooks = ml4_hooks.make_ml4_runtime_hooks()
    assert hooks.incremental_block_offsets is True
    assert hooks.steady_dsa_prepare is True
    assert hooks.pack_spec_rng_uploads is True


# =============================================================================
# 4. Incremental H2DSkipCache.copy_block_offsets contract tests (CPU-simulated)
# =============================================================================


class _FakeCudaTensor:
    """Minimal CPU-backed stand-in that emulates a 4D contiguous CUDA tensor for contract tests."""

    def __init__(self, data: torch.Tensor) -> None:
        self._data = data
        self.is_cuda = True

    def untyped_storage(self):
        return self._data.untyped_storage()

    def storage_offset(self) -> int:
        return self._data.storage_offset()

    @property
    def shape(self):
        return self._data.shape

    def stride(self):
        return self._data.stride()

    def dim(self) -> int:
        return self._data.dim()

    def is_contiguous(self) -> bool:
        return self._data.is_contiguous()

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def device(self):
        return self._data.device

    def __getitem__(self, item):
        return self._data[item]


def test_h2d_cache_copy_block_offsets_skip_incremental_and_fallbacks(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev=None: None)
    staged_copies: list[tuple[torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(h2d_cache.step_inputs, "is_open", lambda: True)

    def _fake_stage(dst: torch.Tensor, src: torch.Tensor):
        staged_copies.append((dst, src.clone()))
        dst.copy_(src)
        return dst

    monkeypatch.setattr(h2d_cache.step_inputs, "copy_", _fake_stage)

    bad = -1
    scales = [72]
    kv_offs = [3]
    num_blocks = 64
    host_src = torch.full((1, 4, 2, num_blocks), bad, dtype=torch.int32)
    host_src[0, 2, 0, :10] = torch.arange(10, 20, dtype=torch.int32)
    dst_raw = torch.zeros((1, 2, 2, num_blocks), dtype=torch.int32)
    dst = _FakeCudaTensor(dst_raw)
    full_calls = [0]

    def _ref_full(slots):
        full_calls[0] += 1
        for s, sl in enumerate(slots):
            raw = host_src[0, sl, 0, :]
            k = torch.where(raw == bad, 0, raw * scales[0])
            v = torch.where(raw == bad, 0, raw * scales[0] + kv_offs[0])
            dst_raw[0, s, 0, :].copy_(k)
            dst_raw[0, s, 1, :].copy_(v)

    cache = h2d_cache.H2DSkipCache(changed_span=True, incremental_block_offsets=True)

    # 1. First call -> full copy.
    mode = cache.copy_block_offsets(
        dst, host_src, [2], scales, kv_offs, bad, lambda: _ref_full([2]), manager_id=("m", 0)
    )
    assert mode == "full" and full_calls[0] == 1

    # 2. Unchanged steady decode -> skip (0 GPU work).
    staged_copies.clear()
    mode = cache.copy_block_offsets(
        dst, host_src, [2], scales, kv_offs, bad, lambda: _ref_full([2]), manager_id=("m", 0)
    )
    assert mode == "skip" and full_calls[0] == 1 and len(staged_copies) == 0

    # 3. Page boundary crossing (+1 block at col 10) -> incremental (2 slices: K & V of len 1).
    host_src[0, 2, 0, 10] = 55
    mode = cache.copy_block_offsets(
        dst, host_src, [2], scales, kv_offs, bad, lambda: _ref_full([2]), manager_id=("m", 0)
    )
    assert mode == "incremental" and full_calls[0] == 1 and len(staged_copies) == 2
    assert int(dst_raw[0, 0, 0, 10]) == 55 * 72
    assert int(dst_raw[0, 0, 1, 10]) == 55 * 72 + 3

    # 4. Eviction / BAD_PAGE_INDEX reset at col 3 -> incremental maps BAD_PAGE_INDEX to 0.
    staged_copies.clear()
    host_src[0, 2, 0, 3] = bad
    mode = cache.copy_block_offsets(
        dst, host_src, [2], scales, kv_offs, bad, lambda: _ref_full([2]), manager_id=("m", 0)
    )
    assert mode == "incremental" and int(dst_raw[0, 0, 0, 3]) == 0 and int(dst_raw[0, 0, 1, 3]) == 0

    # 5. Large span (> max_span) -> clean fallback to full copy.
    host_src[0, 2, 0, 12:40] = 99
    mode = cache.copy_block_offsets(
        dst,
        host_src,
        [2],
        scales,
        kv_offs,
        bad,
        lambda: _ref_full([2]),
        manager_id=("m", 0),
        max_span=8,
    )
    assert mode == "full" and full_calls[0] == 2
    assert int(dst_raw[0, 0, 0, 39]) == 99 * 72

    # 6. Slot / request change -> full copy.
    host_src[0, 1, 0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    mode = cache.copy_block_offsets(
        dst, host_src, [1], scales, kv_offs, bad, lambda: _ref_full([1]), manager_id=("m", 0)
    )
    assert mode == "full" and full_calls[0] == 3

    # 7. Uncached generation bump -> full copy.
    mode = cache.copy_block_offsets(
        dst, host_src, [1], scales, kv_offs, bad, lambda: _ref_full([1]), manager_id=("m", 1)
    )
    assert mode == "full" and full_calls[0] == 4


def test_steady_mtp_prepare_dsa_rng_and_dspark_fast_paths(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import steady_spec_prepare as ssp

    monkeypatch.setattr(ssp, "prefer_pinned", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    staged: list[tuple[torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(ssp.step_inputs, "is_open", lambda: True)

    def _stage_copy(dst, src):
        staged.append((dst, src.clone()))
        dst.copy_(src)
        return dst

    monkeypatch.setattr(ssp.step_inputs, "copy_", _stage_copy)

    # Build minimal DSA-like TrtllmAttentionMetadata stub.
    class _FakeDsaMeta(ssp.TrtllmAttentionMetadata):
        def __init__(self):
            self.kv_cache_manager = None
            self.draft_kv_cache_manager = None
            self._num_contexts = 0
            self._num_generations = 1
            self._seq_lens = torch.tensor([4], dtype=torch.int32)
            self._seq_lens_cuda = self._seq_lens
            self.beam_width = 1
            self.enable_helix = False
            self.enable_flash_mla = False
            self.enable_context_mla_with_cached_kv = True
            self.is_ragged_verify = False
            self.use_expanded_buffers_for_mtp = False
            self.expand_for_dsl = False
            self.use_fp8_ds_mla = False
            self.fp4_mla_state = None
            self.device_windows_mode = False
            self.enable_indexer_skip = True
            self.num_sparse_topk = 2048
            self.skip_indexer_for_gen_reqs = False
            self._seq_lens_kv = torch.tensor([4], dtype=torch.int32)
            self._seq_lens_kv_cuda = self._seq_lens_kv
            self.kv_lens = torch.tensor([4100], dtype=torch.int32)
            self.kv_lens_cuda = torch.tensor([4100], dtype=torch.int32)
            self.prompt_lens_cpu = torch.tensor([4000], dtype=torch.int32)
            self.prompt_lens_cuda = torch.tensor([4000], dtype=torch.int32)
            self.host_total_kv_lens = [0, 4100]
            self.host_request_types = torch.ones(4, dtype=torch.int32)
            self.kv_cache_block_offsets = torch.zeros((1, 1, 2, 128), dtype=torch.int32)
            self.indexer_k_cache_block_offsets = torch.zeros((1, 128), dtype=torch.int32)
            self.host_indexer_k_cache_block_offsets = torch.zeros((1, 128), dtype=torch.int32)
            self.block_table = torch.full((1, 128), -1, dtype=torch.int32)
            self.full_prepare_calls = 0

        def _invalidate_pool_view_cache(self):
            pass

        def _bind_runtime_views(self, **_kw):
            pass

        def prepare(self):
            self.full_prepare_calls += 1

    host_bo = torch.full((1, 4, 2, 128), -1, dtype=torch.int32)
    host_bo[0, 1, 0, :65] = torch.arange(100, 165, dtype=torch.int32)
    bo_copy_calls = [0]
    cache_mgr = SimpleNamespace(
        host_kv_cache_block_offsets=host_bo,
        index_mapper=SimpleNamespace(
            get_copy_index=lambda req_ids, nc, bw: torch.tensor(
                [1] * len(req_ids), dtype=torch.int32
            )
        ),
        indexer_k_cache_page_scale=44,
        num_blocks=1000,
        tokens_per_block=64,
        max_seq_len=32768,
        _use_per_layer_page_tables=False,
        copy_batch_block_offsets=lambda *a, **kw: bo_copy_calls.__setitem__(
            0, bo_copy_calls[0] + 1
        ),
    )
    md = _FakeDsaMeta()
    md.kv_cache_manager = cache_mgr

    worker = SimpleNamespace(
        _win_inited=True,
        _req_to_slot={42: 0},
        _batch_to_slot=torch.zeros(4, dtype=torch.long),
        _scratch_slot=3,
        _ctx_len=[10, 0, 0, 99],
        _valid_len=[10, 0, 0, 88],
        _position_initialized=[True, False, False, True],
    )
    rng_calls = [0]

    def _take_offsets(reqs, seeded):
        rng_calls[0] += 1
        return [1000 * rng_calls[0]] * len(reqs)

    spec_md = SimpleNamespace(
        _dspark_worker=worker,
        batch_indices_cuda=torch.zeros(4, dtype=torch.int32),
        context_capture_lens=[4],
        enable_penalty=False,
        temperatures=torch.ones(16),
        request_seeds=torch.zeros(4, dtype=torch.int64),
        request_offsets=torch.zeros(4, dtype=torch.int64),
        seeds=torch.zeros(16, dtype=torch.int64),
        offsets=torch.zeros(16, dtype=torch.int64),
        _take_rng_window_offsets=_take_offsets,
    )
    req = SimpleNamespace(
        py_request_id=42,
        py_seq_slot=1,
        py_batch_idx=1,
        sampling_config=SimpleNamespace(seed=None),
    )

    prep = ssp.SteadyMtpPrepare(
        max_batch_size=4,
        max_num_tokens=16,
        max_tokens_per_step=4,
        use_mrope=False,
        has_static_inputs=False,
        stage_rng=True,
        steady_dsa=True,
        defer_dsa_derived=True,
    )
    # Force rng_span on CPU (zero_copy is False without CUDA) to unit-test RNG helper.
    prep.rng_span = 16
    prep.record_batch(
        requests=[req],
        prompt_lengths=[4000],
        sequence_lengths=[4],
        tokens_per_step=4,
        runtime_draft_len=3,
        num_extra_kv_tokens=3,
        use_full_generation_page_table=True,
        attn_metadata=md,
        spec_metadata=spec_md,
        static_tensors=None,
    )
    assert prep.record is not None
    assert prep.record.attn_dsa_steady is True
    assert prep.record.dspark_slots == [0]

    # Steady step within same page (4100 -> 4104 -> still 65 blocks): 0 block_table/indexer copies!
    staged.clear()
    prep.prepare_attention(md, [4104])
    assert md.full_prepare_calls == 0
    assert bo_copy_calls[0] == 1
    assert len(staged) == 0

    # Cross page boundary (65 -> 66 blocks at col 65 = 222): patches ONLY col 65!
    host_bo[0, 1, 0, 65] = 222
    prep.prepare_attention(md, [65 * 64 + 4])
    assert md.full_prepare_calls == 0
    assert len(staged) == 2
    assert int(md.indexer_k_cache_block_offsets[0, 65]) == 222 * 44
    assert int(md.block_table[0, 65]) == 222

    # Steady DSpark spec prepare resets scratch row without full prepare.
    assert prep.try_prepare_steady_spec([req], spec_md) is True
    assert spec_md.context_capture_lens is None
    assert worker._ctx_len[3] == 0 and worker._valid_len[3] == 0
    assert worker._position_initialized[3] is False

    # Steady RNG state advances Philox offset cleanly.
    rng = prep.try_take_steady_rng_state([req], spec_md)
    assert rng is not None and rng[1] == [1000]

    # When kv_len drops below skip_indexer threshold, falls back cleanly to full prepare().
    prep.prepare_attention(md, [1024])
    assert md.full_prepare_calls == 1
    assert prep.record.attn_dsa_steady is False
