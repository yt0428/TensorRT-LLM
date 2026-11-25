# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for pure-Triton ``step_inputs``, ``h2d_cache``, and ``steady_spec_prepare``."""

import inspect
import threading
import types

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import h2d_cache, step_inputs
from tensorrt_llm._torch.pyexecutor.h2d_cache import (
    H2DSkipCache,
    copy_h2d_if_changed,
    copy_h2d_pool_shared,
    fill_tail_if_changed,
    fill_tail_pool_shared,
)
from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack, _disjoint, _parallel_ok

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# step_inputs: pure-Python/Triton invariants (zero C++ scatter dependency)
# ---------------------------------------------------------------------------


def test_step_inputs_has_no_cpp_scatter_op_reference():
    """Verify step_inputs uses pure Triton kernels and never references trtllm::step_inputs_scatter."""
    src = inspect.getsource(step_inputs)
    assert "step_inputs_scatter" not in src
    assert "_serial_scatter_kernel" in src
    assert "_parallel_scatter_kernel" in src
    assert "_graph_scatter_kernel" in src


def test_disjoint_and_parallel_ok_rules():
    assert _disjoint([(100, 16), (116, 8), (200, 4)]) is True
    assert _disjoint([(100, 16), (115, 8)]) is False
    assert _disjoint([]) is True

    # Disjoint host copies + H2D feeding D2D is order-free (H2D finishes before D2D launches).
    h2d = [(1000, 0, 16), (2000, 16, 16)]
    d2d = [(3000, 1000, 16), (4000, 2000, 16)]
    assert _parallel_ok(h2d, d2d) is True

    # Overlapping H2D destinations -> must fall back to serial Triton scatter.
    assert _parallel_ok([(1000, 0, 16), (1008, 16, 16)], []) is False

    # Overlapping D2D destinations -> serial.
    assert _parallel_ok([], [(3000, 1000, 16), (3008, 2000, 16)]) is False

    # D2D read-after-write hazard (one D2D writes where another D2D reads) -> serial.
    assert _parallel_ok([], [(3000, 1000, 16), (4000, 3000, 16)]) is False


def test_step_inputs_contextvar_and_thread_isolation():
    assert not step_inputs.is_open()
    assert step_inputs.step_epoch() is None

    dummy_pack = types.SimpleNamespace(open=True, epoch="fake_epoch")
    seen = {}

    def _other():
        seen["open"] = step_inputs.is_open()
        seen["epoch"] = step_inputs.step_epoch()

    with step_inputs.active(dummy_pack):  # type: ignore[arg-type]
        assert step_inputs.is_open() is True
        assert step_inputs.is_active(dummy_pack) is True  # type: ignore[arg-type]
        assert step_inputs.step_epoch() == "fake_epoch"
        t = threading.Thread(target=_other)
        t.start()
        t.join()

    assert seen == {"open": False, "epoch": None}
    assert not step_inputs.is_open()


def test_step_input_pack_cpu_staging_alignment_and_empty_flush(monkeypatch):
    """Exercise StepInputPack state transitions on CPU using mocked CUDA flags."""
    monkeypatch.setattr(step_inputs, "prefer_pinned", lambda: False)
    pack = StepInputPack(torch.device("cpu"), parallel_scatter=False)
    assert pack.in_graph is False
    assert pack.early_copy is False
    assert pack.issue_after_prepare is False

    # Non-CUDA destination is declined by stage().
    dst = torch.zeros(4, dtype=torch.int32)
    src = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    pack.begin()
    assert pack.open is True
    assert pack.stage(dst, src) is False

    # Empty flush closes pack cleanly without issuing device work.
    pack.flush()
    assert pack.open is False
    assert pack.h2d == [] and pack.d2d == []

    # Discard closes and clears staged entries.
    pack.begin()
    pack.h2d.append((1, 2, 3))
    pack.discard()
    assert pack.open is False
    assert pack.h2d == []


def test_step_input_pack_in_graph_empty_flush_quiesces_window(monkeypatch):
    """An empty flush(in_graph=True) quiesces a non-zero header without copying 6KB padding."""
    pack = object.__new__(StepInputPack)
    pack.open = True
    pack.in_graph = True
    pack.early_copy = True
    pack.h2d = []
    pack.d2d = []
    pack.staged = []
    pack._graph_staged = []
    pack.data_start = 6160
    pack.used = 6160
    pack.ghost = torch.zeros(StepInputPack.GRAPH_WINDOW, dtype=torch.uint8)
    pack.ghost[:8].view(torch.int64).fill_(3)  # stale non-zero header
    uploaded = []
    pack._ghost_writable = lambda: None
    pack._upload_window = lambda nbytes: uploaded.append(nbytes)

    pack.flush(in_graph=True)
    assert pack.open is False
    assert not pack.ghost[: StepInputPack.HEADER_BYTES].any()
    assert uploaded == [StepInputPack.HEADER_BYTES]

    # Second empty flush with already-zero ghost header issues zero uploads.
    pack.open = True
    pack.flush(in_graph=True)
    assert uploaded == [StepInputPack.HEADER_BYTES]


# ---------------------------------------------------------------------------
# h2d_cache: CPU shadow logic, changed_span diffing, tail fills, pool-shared
# ---------------------------------------------------------------------------


class _FakeCudaTensor:
    """Thin CPU-backed wrapper presenting ``is_cuda=True`` for testing H2DSkipCache on CPU."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._t = tensor

    @property
    def is_cuda(self) -> bool:
        return True

    @property
    def shape(self) -> torch.Size:
        return self._t.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._t.dtype

    def dim(self) -> int:
        return self._t.dim()

    def numel(self) -> int:
        return self._t.numel()

    def storage_offset(self) -> int:
        return self._t.storage_offset()

    def stride(self, *args):
        return self._t.stride(*args)

    def untyped_storage(self):
        return self._t.untyped_storage()

    def copy_(self, src: torch.Tensor, non_blocking: bool = False):
        self._t.copy_(src)
        return self

    def __getitem__(self, idx):
        return _FakeCudaTensor(self._t[idx])

    def fill_(self, value):
        self._t.fill_(value)
        return self


def test_h2d_skip_cache_shadow_snapshot_and_invalidation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    cache = H2DSkipCache()
    backing = torch.zeros(4, dtype=torch.int32)
    dst = _FakeCudaTensor(backing)
    src = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

    with h2d_cache.active(cache):
        assert h2d_cache.enabled() is True
        assert copy_h2d_if_changed(dst, src) is True  # type: ignore[arg-type]
        assert backing.tolist() == [1, 2, 3, 4]
        # Identical values -> skipped.
        assert copy_h2d_if_changed(dst, src) is False  # type: ignore[arg-type]
        # Mutating src in place must NOT fool the snapshot.
        src[0] = 99
        assert copy_h2d_if_changed(dst, src) is True  # type: ignore[arg-type]
        assert backing.tolist() == [99, 2, 3, 4]
        # clear() forces reissue.
        cache.clear()
        assert copy_h2d_if_changed(dst, src) is True  # type: ignore[arg-type]
        # Pool-shared write drops shadow and always issues.
        assert copy_h2d_pool_shared(dst, torch.tensor([5, 6, 7, 8], dtype=torch.int32)) is True  # type: ignore[arg-type]
        assert backing.tolist() == [5, 6, 7, 8]
        assert copy_h2d_if_changed(dst, src) is True  # type: ignore[arg-type]
    assert h2d_cache.enabled() is False


def test_h2d_skip_cache_changed_span_stages_minimal_row_spans(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    staged_spans: list[tuple[tuple[int, ...], list[int]]] = []

    class _OpenPack:
        open = True

        def stage(self, dst, src) -> bool:
            staged_spans.append((tuple(dst.shape), src.tolist()))
            dst.copy_(src)
            return True

    cache = H2DSkipCache(changed_span=True)
    backing = torch.zeros((4, 16), dtype=torch.int32)
    dst = _FakeCudaTensor(backing)
    src = torch.arange(64, dtype=torch.int32).reshape(4, 16)

    with h2d_cache.active(cache), step_inputs.active(_OpenPack()):  # type: ignore[arg-type]
        # First upload stages full table.
        assert copy_h2d_if_changed(dst, src, pack=True) is True  # type: ignore[arg-type]
        assert len(staged_spans) == 1
        assert staged_spans[0][0] == (4, 16)

        staged_spans.clear()
        # Mutate a single column on row 2 and columns 3..5 on row 0.
        src2 = src.clone()
        src2[0, 3] = 777
        src2[0, 5] = 888
        src2[2, 11] = 999
        assert copy_h2d_if_changed(dst, src2, pack=True) is True  # type: ignore[arg-type]
        assert staged_spans == [
            ((3,), [777, 4, 888]),
            ((1,), [999]),
        ]
        assert torch.equal(backing, src2)


def test_h2d_fill_tail_if_changed_and_pool_shared(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    cache = H2DSkipCache()
    buf = torch.zeros(6, dtype=torch.int32)
    with h2d_cache.active(cache):
        fill_tail_if_changed(buf, 2, -1)
        assert buf.tolist() == [0, 0, -1, -1, -1, -1]
        buf[3] = 42
        # Repeated identical tail fill is skipped.
        fill_tail_if_changed(buf, 2, -1)
        assert buf[3].item() == 42
        # Pool-shared tail fill always writes and drops tail record.
        fill_tail_pool_shared(buf, 2, -1)
        assert buf[3].item() == -1
        buf[3] = 77
        fill_tail_if_changed(buf, 2, -1)
        assert buf[3].item() == -1


# ---------------------------------------------------------------------------
# steady_spec_prepare: CPU record/match/qualifies/ring invariants
# ---------------------------------------------------------------------------


def test_steady_mtp_prepare_cpu_qualifies_and_matches(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import steady_spec_prepare
    from tensorrt_llm._torch.pyexecutor.steady_spec_prepare import SteadyMtpPrepare
    from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateTensorsSpec

    monkeypatch.setattr(steady_spec_prepare, "prefer_pinned", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    prep = SteadyMtpPrepare(
        max_batch_size=4,
        max_num_tokens=16,
        max_tokens_per_step=4,
        use_mrope=False,
        has_static_inputs=True,
        stage_rng=False,
    )
    assert prep.record is None
    assert prep.block == 16

    req = types.SimpleNamespace(
        py_request_id=42,
        py_seq_slot=1,
        py_batch_idx=1,
        is_dummy=False,
        py_multimodal_data=None,
        py_needs_onehot_draft_probs=False,
        max_beam_num_tokens=128,
        py_num_compressed_tokens=0,
    )
    assert SteadyMtpPrepare.request_qualifies(req) is True
    req_dummy = types.SimpleNamespace(**{**req.__dict__, "is_dummy": True})
    assert SteadyMtpPrepare.request_qualifies(req_dummy) is False
    req_mm = types.SimpleNamespace(**{**req.__dict__, "py_multimodal_data": {"img": 1}})
    assert SteadyMtpPrepare.request_qualifies(req_mm) is False

    attn_meta = types.SimpleNamespace(
        is_cuda_graph=True,
        num_seqs=1,
        seq_lens_kv=torch.tensor([4], dtype=torch.int32),
        prepare=lambda: None,
    )
    spec_meta = types.SimpleNamespace()
    static_tensors = {
        "input_ids": torch.zeros(16, dtype=torch.int32),
        "position_ids": torch.zeros((1, 16), dtype=torch.int32),
    }
    prep.record_batch(
        requests=[req],
        prompt_lengths=[64],
        sequence_lengths=[4],
        tokens_per_step=4,
        runtime_draft_len=3,
        num_extra_kv_tokens=3,
        use_full_generation_page_table=False,
        attn_metadata=attn_meta,
        spec_metadata=spec_meta,
        static_tensors=static_tensors,
    )
    assert prep.record is not None
    assert prep.record.attn_steady is False
    assert prep.record.static_ids is static_tensors["input_ids"]
    assert prep.record.static_pos is static_tensors["position_ids"]

    prev_tensors = SampleStateTensorsSpec(
        new_tokens=torch.zeros((4, 4, 1), dtype=torch.int32),
        new_tokens_lens=torch.ones(4, dtype=torch.int32),
        next_draft_tokens=torch.zeros((4, 3), dtype=torch.int32),
    )
    assert prep.matches([req], attn_meta, spec_meta, prev_tensors, runtime_draft_len=3) is True
    assert prep.matches([req], attn_meta, spec_meta, prev_tensors, runtime_draft_len=2) is False
    req_changed_slot = types.SimpleNamespace(
        **{**req.__dict__, "py_seq_slot": 2, "py_batch_idx": 2}
    )
    assert (
        prep.matches([req_changed_slot], attn_meta, spec_meta, prev_tensors, runtime_draft_len=3)
        is False
    )
    prep.invalidate()
    assert prep.record is None
