# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for ``RuntimeHooks`` and engine/executor opt-in wiring."""

import ast
import contextlib
import dataclasses
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import h2d_cache, step_inputs
from tensorrt_llm._torch.pyexecutor.runtime_hooks import (
    DEFAULT_RUNTIME_HOOKS,
    ExecutorEvents,
    RuntimeHooks,
    runtime_hooks_of,
)

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[4]


class _PlainModel(torch.nn.Module):
    """A model that declares no hooks."""


class _ClassHooksModel(torch.nn.Module):
    runtime_hooks = RuntimeHooks(loop_on_execution_stream=True)


class _RecordingEvents:
    def __init__(self, log: list[str] | None = None) -> None:
        self.log = log if log is not None else []

    def after_warmup(self) -> None:
        self.log.append("after_warmup")

    def before_overlap_loop(self) -> None:
        self.log.append("before_overlap_loop")

    def at_shutdown(self) -> None:
        self.log.append("at_shutdown")


class _IterationEvents(_RecordingEvents):
    def on_iteration(self, iteration: int) -> None:
        self.log.append(f"on_iteration:{iteration}")


@pytest.mark.parametrize("model", [_PlainModel(), object(), None], ids=["module", "object", "none"])
def test_model_without_hooks_gets_all_off_default(model):
    hooks = runtime_hooks_of(model)
    assert hooks is DEFAULT_RUNTIME_HOOKS
    assert hooks == RuntimeHooks()
    assert not hooks.loop_on_execution_stream
    assert hooks.decode_prologue is None
    assert hooks.executor_events is None


def test_every_runtime_hooks_field_defaults_to_off():
    for field in dataclasses.fields(RuntimeHooks):
        value = getattr(DEFAULT_RUNTIME_HOOKS, field.name)
        if field.name == "first_content_emit_iters":
            assert value == 1
        else:
            assert value in (False, None), field.name
    assert DEFAULT_RUNTIME_HOOKS.skips_unchanged_metadata_uploads is False


def test_hooks_declared_as_none_mean_default():
    model = _PlainModel()
    model.runtime_hooks = None
    assert runtime_hooks_of(model) is DEFAULT_RUNTIME_HOOKS


def test_class_and_instance_hooks_precedence():
    assert runtime_hooks_of(_ClassHooksModel()) is _ClassHooksModel.runtime_hooks
    override = RuntimeHooks(kv_lens_update_once=True)
    model = _ClassHooksModel()
    model.runtime_hooks = override
    assert runtime_hooks_of(model) is override


@pytest.mark.parametrize(
    "declared",
    [True, {"loop_on_execution_stream": True}, "RuntimeHooks()"],
    ids=["bool", "dict", "str"],
)
def test_wrong_type_runtime_hooks_raises(declared):
    model = _PlainModel()
    model.runtime_hooks = declared
    with pytest.raises(TypeError, match=r"_PlainModel\.runtime_hooks must be a RuntimeHooks"):
        runtime_hooks_of(model)


def test_runtime_hooks_is_frozen():
    hooks = RuntimeHooks()
    with pytest.raises(dataclasses.FrozenInstanceError):
        hooks.loop_on_execution_stream = True  # type: ignore[misc]


def test_skips_unchanged_metadata_uploads_property():
    assert RuntimeHooks(skip_unchanged_h2d=True).skips_unchanged_metadata_uploads is True
    assert RuntimeHooks(skip_unchanged_h2d=False).skips_unchanged_metadata_uploads is False


def test_on_iteration_is_optional_not_a_protocol_member():
    assert "on_iteration" not in vars(ExecutorEvents)
    assert "before_overlap_loop" in vars(ExecutorEvents)
    assert "after_warmup" in vars(ExecutorEvents)
    assert "at_shutdown" in vars(ExecutorEvents)


class _RecordingPack:
    def __init__(self, issue_after_prepare: bool) -> None:
        self.open = True
        self.issue_after_prepare = issue_after_prepare
        self.staged_copies: list[tuple[torch.Tensor, torch.Tensor]] = []

    def stage(self, dst: torch.Tensor, src: torch.Tensor) -> bool:
        self.staged_copies.append((dst, src))
        return True


def test_copy_metadata_routes_only_when_issue_after_prepare(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    dst, src = torch.zeros(4, dtype=torch.int32), torch.arange(4, dtype=torch.int32)

    pack_replay = _RecordingPack(issue_after_prepare=False)
    with step_inputs.active(pack_replay):
        step_inputs.copy_metadata_(dst, src)
    assert torch.equal(dst, src)
    assert pack_replay.staged_copies == []

    dst2 = torch.zeros(4, dtype=torch.int32)
    pack_after_prep = _RecordingPack(issue_after_prepare=True)
    with step_inputs.active(pack_after_prep):
        step_inputs.copy_metadata_(dst2, src)
    assert len(pack_after_prep.staged_copies) == 1
    assert pack_after_prep.staged_copies[0][0] is dst2
    assert torch.equal(dst2, torch.zeros(4, dtype=torch.int32))


def test_pool_shared_upload_without_skip_cache_follows_pack(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for issue_after_prepare in (False, True):
        dst = torch.zeros(3, dtype=torch.int32)
        src = torch.tensor([5, 6, 7], dtype=torch.int32)
        pack = _RecordingPack(issue_after_prepare=issue_after_prepare)
        with h2d_cache.active(None), step_inputs.active(pack):
            assert h2d_cache.copy_h2d_pool_shared(dst, src, pack=True) is True
        assert len(pack.staged_copies) == (1 if issue_after_prepare else 0)
        expected = torch.zeros(3, dtype=torch.int32) if issue_after_prepare else src
        assert torch.equal(dst, expected)


def test_runner_and_inputs_ast_wiring():
    """Verify key runtime-hook AST invariants in DecoderRunner and DecoderInputPreparer."""
    runner_src = (
        _REPO_ROOT / "tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/runner.py"
    ).read_text()
    runner_tree = ast.parse(runner_src)
    runner_cls = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DecoderRunner"
    )
    init_fn = next(
        n for n in runner_cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    fwd_fn = next(
        n
        for n in runner_cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_forward_decoder"
    )
    assert any(
        isinstance(n, ast.Assign)
        and ast.unparse(n) == "pack_factory = self.runtime_hooks.step_input_pack_factory"
        for n in init_fn.body
    )
    prepare_lines = [
        n.lineno
        for n in ast.walk(fwd_fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "prepare_inputs"
    ]
    copy_event_lines = [
        n.lineno
        for n in ast.walk(fwd_fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "record_input_copy"
    ]
    issue_flushes = [
        n
        for n in ast.walk(fwd_fn)
        if isinstance(n, ast.If) and "issue_after_prepare" in ast.unparse(n.test)
    ]
    assert len(prepare_lines) == 1 and len(copy_event_lines) == 1 and len(issue_flushes) == 1
    assert prepare_lines[0] < issue_flushes[0].lineno < copy_event_lines[0]


def test_py_executor_runtime_hook_lifecycle(monkeypatch):
    """Exercise PyExecutor hook class defaults, stream scope, cleanup, and overlap loop events."""
    from tensorrt_llm._torch.pyexecutor import py_executor
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    assert PyExecutor._execution_stream_loop_armed is False
    assert PyExecutor._loop_on_execution_stream is False
    assert PyExecutor._executor_events is None
    assert PyExecutor.first_content_emit_iters == 1

    # _overlap_loop_stream_scope: skipped when unarmed or incompatible; entered when armed.
    exec_obj = object.__new__(PyExecutor)
    exec_obj._execution_stream_loop_armed = True
    exec_obj.drafter = None
    exec_obj.draft_model_engine = None
    exec_obj.guided_decoder = None
    exec_obj.kv_connector_manager = None
    exec_obj.is_encoder_decoder = False
    exec_obj._mm_encoder_item_scheduling_enabled = False
    exec_obj.execution_stream = object()
    entered_streams = []

    @contextlib.contextmanager
    def _fake_stream(stream):
        entered_streams.append(stream)
        yield

    monkeypatch.setattr(torch.cuda, "stream", _fake_stream)
    with exec_obj._overlap_loop_stream_scope():
        assert exec_obj._loop_on_execution_stream is True
    assert exec_obj._loop_on_execution_stream is False
    assert entered_streams == [exec_obj.execution_stream]

    # _executor_loop_overlap invokes before_overlap_loop and optional on_iteration(iter_counter).
    class _StopLoop(Exception):
        pass

    class _LoopHead(PyExecutor):
        @property
        def enable_iter_perf_stats(self):
            log.append("loop_body")
            raise _StopLoop

    class _Hang:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def checkpoint(self):
            log.append("checkpoint")

    monkeypatch.setattr(torch.cuda, "set_device", lambda _d: None)
    monkeypatch.setattr(py_executor, "CUASSERT", lambda _r: None)
    monkeypatch.setattr(py_executor.cudart, "cudaSetDevice", lambda _d: None)

    for mode in ("none", "without_on_iter", "with_on_iter"):
        log: list[str] = []
        events = {
            "none": None,
            "without_on_iter": _RecordingEvents(log),
            "with_on_iter": _IterationEvents(log),
        }[mode]
        ex = _LoopHead.__new__(_LoopHead)
        ex.device_id = 0
        ex.iter_counter = 11
        ex.is_benchmark_disagg = False
        ex._executor_events = events
        ex._overlap_loop_stream_scope = contextlib.nullcontext
        ex._profiler = lambda: contextlib.nullcontext(lambda: log.append("profile"))
        ex.hang_detector = _Hang()
        with pytest.raises(_StopLoop):
            ex._executor_loop_overlap()
        expected = {
            "none": ["checkpoint", "profile", "loop_body"],
            "without_on_iter": ["before_overlap_loop", "checkpoint", "profile", "loop_body"],
            "with_on_iter": [
                "before_overlap_loop",
                "checkpoint",
                "profile",
                "on_iteration:11",
                "loop_body",
            ],
        }[mode]
        assert log == expected

    # _executor_loop_cleanup calls at_shutdown only when _event_loop_completed is True.
    for completed in (False, True):
        log = []
        ev = _RecordingEvents(log)
        stub = object.__new__(PyExecutor)
        stub.response_cv = MagicMock()
        stub.shutdown_event = MagicMock()
        stub.num_micro_batches = 0
        stub._executor_events = ev
        stub._event_loop_completed = completed
        stub._executor_loop_cleanup()
        assert log == (["at_shutdown"] if completed else [])


def test_preprocess_inputs_kv_lens_update_once_and_spec_offsets_op():
    """Verify kv_lens_update_once eliminates the redundant pre-correction on_update_kv_lens call
    and spec_offsets_op replaces the two stock offset additions."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer

    def _run(kv_once: bool, use_fused_op: bool):
        preparer = object.__new__(DecoderInputPreparer)
        preparer._config = types.SimpleNamespace(disable_overlap_scheduler=False)
        preparer.mapping = types.SimpleNamespace(has_cp_helix=lambda: False)
        preparer.guided_decoder = None
        preparer.decode_prologue = None
        fused_calls = []

        def _fused(pos, pos_off, kv, kv_off):
            fused_calls.append((pos.tolist(), pos_off.tolist(), kv.tolist(), kv_off.tolist()))
            pos += pos_off
            kv += kv_off
            return True

        preparer.runtime_hooks = RuntimeHooks(
            kv_lens_update_once=kv_once,
            spec_offsets_op=_fused if use_fused_op else None,
        )
        preparer._buffers = types.SimpleNamespace(
            previous_pos_id_offsets_cuda=torch.tensor([2, 2, 2, 2], dtype=torch.int32),
            previous_kv_lens_offsets_cuda=torch.tensor([-2], dtype=torch.int32),
        )
        kv_update_calls = []
        attn_meta = types.SimpleNamespace(
            kv_cache_manager=object(),
            num_seqs=1,
            num_contexts=0,
            num_generations=1,
            num_ctx_tokens=0,
            num_chunked_ctx_requests=0,
            kv_lens_cuda=torch.tensor([100], dtype=torch.int32),
            on_update_kv_lens=lambda: kv_update_calls.append(int(attn_meta.kv_lens_cuda[0])),
        )
        inputs = {
            "attn_metadata": attn_meta,
            "input_ids": torch.zeros(4, dtype=torch.int32),
            "position_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
        }
        preparer.preprocess_inputs(inputs, enable_spec_decode=True, runtime_draft_len=3)
        return inputs, attn_meta, kv_update_calls, fused_calls

    inp_stock, meta_stock, calls_stock, fused_stock = _run(kv_once=False, use_fused_op=False)
    inp_opt, meta_opt, calls_opt, fused_opt = _run(kv_once=True, use_fused_op=True)

    assert calls_stock == [100, 98]
    assert calls_opt == [98]
    assert len(fused_stock) == 0 and len(fused_opt) == 1
    assert torch.equal(inp_opt["position_ids"], inp_stock["position_ids"])
    assert torch.equal(meta_opt.kv_lens_cuda, meta_stock.kv_lens_cuda)
    assert inp_opt["position_ids"].tolist() == [[12, 13, 14, 15]]
    assert meta_opt.kv_lens_cuda.tolist() == [98]
