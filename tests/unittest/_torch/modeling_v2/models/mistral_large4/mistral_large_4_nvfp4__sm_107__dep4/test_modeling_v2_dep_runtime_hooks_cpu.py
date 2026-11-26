# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit and contract tests for Mistral Large 4 SM107 DEP4/DEP8 & TEP4 runtime hooks
and single-launch draft-block embedding wiring.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import pathlib
import subprocess
import sys
import types

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[7]
_ML4_DIR = (
    _REPO_ROOT
    / "tensorrt_llm"
    / "_torch"
    / "_experimental"
    / "modeling_v2"
    / "models"
    / "mistral_large4"
)
_DEP4_DIR = _ML4_DIR / "mistral_large_4_nvfp4__sm_107__dep4"
_TEP4_DIR = _ML4_DIR / "mistral_large_4_nvfp4__sm_107__tep4"
_TP4_DIR = _ML4_DIR / "mistral_large_4_nvfp4__sm_107__tp4"
_PYEXEC_HOOKS_PATH = _REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "runtime_hooks.py"
_DSPARK_DIR = (
    _REPO_ROOT / "tensorrt_llm" / "_torch" / "_experimental" / "modeling_v2" / "catalog" / "dspark"
)


def _load_dep_hooks_modules():
    """Import ``runtime_hooks`` modules cleanly in CPU-only or sparse checkouts without the C++
    extension by seeding minimal package stubs for the shared ``RuntimeHooks`` module."""

    # Ensure lightweight package hierarchy exists so relative imports resolve identically.
    def _ensure_pkg(name: str, path: pathlib.Path | None = None) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is not None:
            return mod
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)] if path is not None else []
        sys.modules[name] = mod
        return mod

    def _load_file(name: str, path: pathlib.Path) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is not None:
            return mod
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Provide a minimal logger stub only if tensorrt_llm cannot be imported directly.
    if "tensorrt_llm" not in sys.modules:
        _ensure_pkg("tensorrt_llm", _REPO_ROOT / "tensorrt_llm")
    if "tensorrt_llm.logger" not in sys.modules:
        log_mod = types.ModuleType("tensorrt_llm.logger")

        class _NoopLogger:
            def info(self, *_a, **_kw):
                pass

            def warning(self, *_a, **_kw):
                pass

        log_mod.logger = _NoopLogger()
        sys.modules["tensorrt_llm.logger"] = log_mod
    _ensure_pkg("tensorrt_llm._torch", _REPO_ROOT / "tensorrt_llm" / "_torch")
    _ensure_pkg(
        "tensorrt_llm._torch.pyexecutor",
        _REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor",
    )
    pyexec_hooks = _load_file("tensorrt_llm._torch.pyexecutor.runtime_hooks", _PYEXEC_HOOKS_PATH)

    pkg_base = "_ml4_dep_hooks_cpu_test"
    _ensure_pkg(pkg_base, _ML4_DIR)
    _ensure_pkg(f"{pkg_base}.mistral_large_4_nvfp4__sm_107__tp4", _TP4_DIR)
    _ensure_pkg(f"{pkg_base}.mistral_large_4_nvfp4__sm_107__dep4", _DEP4_DIR)
    tp4_hooks = _load_file(
        f"{pkg_base}.mistral_large_4_nvfp4__sm_107__tp4.runtime_hooks",
        _TP4_DIR / "runtime_hooks.py",
    )
    dep_hooks = _load_file(
        f"{pkg_base}.mistral_large_4_nvfp4__sm_107__dep4.runtime_hooks",
        _DEP4_DIR / "runtime_hooks.py",
    )
    return pyexec_hooks, tp4_hooks, dep_hooks


def test_make_ml4_dep_runtime_hooks_gen_and_ctx_profiles() -> None:
    pyexec_hooks, tp4_hooks, dep_hooks = _load_dep_hooks_modules()

    # DEP8 GEN (CUDA graphs ON, attention DP ON): full graph packing + H2D skip + incremental
    # block offsets + GC disabled, while steady_* / pack_spec_rng_uploads stay False under ADP.
    dep8_mapping = types.SimpleNamespace(local_rank=2, enable_attention_dp=True)
    gen_hooks = dep_hooks.make_ml4_dep_runtime_hooks(
        dep8_mapping, use_cuda_graph=True, enable_attention_dp=True
    )
    assert isinstance(gen_hooks, pyexec_hooks.RuntimeHooks)
    assert gen_hooks.pack_step_inputs is True
    assert gen_hooks.pack_parallel_scatter is True
    assert gen_hooks.pack_in_graph is True
    assert gen_hooks.pack_early_copy is True
    assert gen_hooks.skip_unchanged_h2d is True
    assert gen_hooks.h2d_changed_span is True
    assert gen_hooks.incremental_block_offsets is True
    assert gen_hooks.steady_dsa_prepare is False
    assert gen_hooks.kv_lens_update_once is True
    assert gen_hooks.spec_offsets_op is tp4_hooks.fused_spec_offsets_op
    assert gen_hooks.loop_on_execution_stream is True
    assert gen_hooks.steady_spec_prepare is False
    assert gen_hooks.pack_spec_rng_uploads is False
    assert gen_hooks.iteration_timing_one_event is True
    assert gen_hooks.perf_metrics_timing_side_stream is True
    assert isinstance(gen_hooks.executor_events, dep_hooks.Ml4DepExecutorEvents)
    assert gen_hooks.executor_events.local_rank == 2
    assert gen_hooks.executor_events.disable_gc is True

    # DEP4 CTX (CUDA graphs OFF, attention DP ON): skips graph-only packing/H2D/incremental
    # shadows and keeps cyclic GC enabled for long-running prefill servers.
    ctx_hooks = dep_hooks.make_ml4_dep_runtime_hooks(
        dep8_mapping, use_cuda_graph=False, enable_attention_dp=True
    )
    assert ctx_hooks.pack_step_inputs is False
    assert ctx_hooks.pack_parallel_scatter is False
    assert ctx_hooks.pack_in_graph is False
    assert ctx_hooks.pack_early_copy is False
    assert ctx_hooks.skip_unchanged_h2d is False
    assert ctx_hooks.h2d_changed_span is False
    assert ctx_hooks.incremental_block_offsets is False
    assert ctx_hooks.steady_dsa_prepare is False
    assert ctx_hooks.steady_spec_prepare is False
    assert ctx_hooks.pack_spec_rng_uploads is False
    assert ctx_hooks.kv_lens_update_once is True
    assert ctx_hooks.spec_offsets_op is tp4_hooks.fused_spec_offsets_op
    assert ctx_hooks.loop_on_execution_stream is True
    assert ctx_hooks.iteration_timing_one_event is True
    assert ctx_hooks.perf_metrics_timing_side_stream is True
    assert isinstance(ctx_hooks.executor_events, dep_hooks.Ml4DepExecutorEvents)
    assert ctx_hooks.executor_events.disable_gc is False

    # TEP4 GEN (CUDA graphs ON, attention DP OFF): enables steady_spec_prepare,
    # steady_dsa_prepare, pack_spec_rng_uploads, and incremental_block_offsets.
    tep4_mapping = types.SimpleNamespace(local_rank=1, enable_attention_dp=False)
    tep_gen_hooks = dep_hooks.make_ml4_dep_runtime_hooks(tep4_mapping, use_cuda_graph=True)
    assert tep_gen_hooks.incremental_block_offsets is True
    assert tep_gen_hooks.steady_spec_prepare is True
    assert tep_gen_hooks.steady_dsa_prepare is True
    assert tep_gen_hooks.pack_spec_rng_uploads is True
    assert tep_gen_hooks.pack_step_inputs is True
    assert tep_gen_hooks.skip_unchanged_h2d is True
    assert tep_gen_hooks.h2d_changed_span is True
    assert tep_gen_hooks.executor_events.disable_gc is True

    # TEP4 CTX (CUDA graphs OFF, attention DP OFF): all graph/steady hooks and GC disable stay off.
    tep_ctx_hooks = dep_hooks.make_ml4_dep_runtime_hooks(tep4_mapping, use_cuda_graph=False)
    assert tep_ctx_hooks.incremental_block_offsets is False
    assert tep_ctx_hooks.steady_spec_prepare is False
    assert tep_ctx_hooks.steady_dsa_prepare is False
    assert tep_ctx_hooks.pack_spec_rng_uploads is False
    assert tep_ctx_hooks.pack_step_inputs is False
    assert tep_ctx_hooks.skip_unchanged_h2d is False
    assert tep_ctx_hooks.h2d_changed_span is False
    assert tep_ctx_hooks.executor_events.disable_gc is False

    # Explicit enable_attention_dp=True overrides a non-ADP mapping; steady_* must stay False.
    adp_override = dep_hooks.make_ml4_dep_runtime_hooks(
        tep4_mapping, use_cuda_graph=True, enable_attention_dp=True
    )
    assert adp_override.incremental_block_offsets is True
    assert adp_override.steady_spec_prepare is False
    assert adp_override.steady_dsa_prepare is False
    assert adp_override.pack_spec_rng_uploads is False

    # Default mapping=None defaults to ADP=True, graph_worker=True (safe conservative default).
    default_hooks = dep_hooks.make_ml4_dep_runtime_hooks()
    assert default_hooks.incremental_block_offsets is True
    assert default_hooks.steady_spec_prepare is False
    assert default_hooks.steady_dsa_prepare is False
    assert default_hooks.pack_spec_rng_uploads is False


def test_incremental_block_offsets_adp_stale_metadata_guards(monkeypatch) -> None:
    """Verify H2DSkipCache.copy_block_offsets guards against stale metadata across ADP dummy
    additions, slot reordering, span overflow, explicit invalidation, and disabled mode."""
    _load_dep_hooks_modules()

    pyexec_dir = _REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor"
    if "tensorrt_llm._utils" not in sys.modules:
        utils_mod = types.ModuleType("tensorrt_llm._utils")
        utils_mod.prefer_pinned = lambda: False
        sys.modules["tensorrt_llm._utils"] = utils_mod

    def _load_pyexec_sub(name: str, filename: str) -> types.ModuleType:
        full = f"tensorrt_llm._torch.pyexecutor.{name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, pyexec_dir / filename)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    _load_pyexec_sub("numa_pinned", "numa_pinned.py")
    _load_pyexec_sub("step_inputs", "step_inputs.py")
    h2d_cache = _load_pyexec_sub("h2d_cache", "h2d_cache.py")

    class _FakeCudaTensor:
        def __init__(self, backing: torch.Tensor) -> None:
            self._t = backing

        @property
        def is_cuda(self) -> bool:
            return True

        @property
        def dtype(self) -> torch.dtype:
            return self._t.dtype

        @property
        def shape(self) -> torch.Size:
            return self._t.shape

        @property
        def device(self) -> torch.device:
            return self._t.device

        def dim(self) -> int:
            return self._t.dim()

        def stride(self) -> tuple[int, ...]:
            return self._t.stride()

        def storage_offset(self) -> int:
            return self._t.storage_offset()

        def untyped_storage(self):
            return self._t.untyped_storage()

        def is_contiguous(self) -> bool:
            return self._t.is_contiguous()

        def __getitem__(self, item):
            return self._t[item]

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev=None: None)
    monkeypatch.setattr(h2d_cache.step_inputs, "is_open", lambda: True)
    staged_copies: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _fake_stage(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        staged_copies.append((dst, src.clone()))
        dst.copy_(src)
        return dst

    monkeypatch.setattr(h2d_cache.step_inputs, "copy_", _fake_stage)

    bad = -1
    scales = [72]
    kv_offs = [3]
    num_blocks = 32
    host_src = torch.full((1, 4, 2, num_blocks), bad, dtype=torch.int32)
    host_src[0, 0, 0, :4] = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    host_src[0, 1, 0, :3] = torch.tensor([20, 21, 22], dtype=torch.int32)
    host_src[0, 3, 0, :1] = torch.tensor([99], dtype=torch.int32)  # ADP dummy slot

    dst_raw = torch.zeros((1, 3, 2, num_blocks), dtype=torch.int32)
    dst = _FakeCudaTensor(dst_raw)
    full_calls = [0]

    def _ref_full(slots: list[int]) -> None:
        full_calls[0] += 1
        dst_raw.zero_()
        for s, sl in enumerate(slots):
            raw = host_src[0, sl, 0, :]
            dst_raw[0, s, 0, :].copy_(torch.where(raw == bad, 0, raw * scales[0]))
            dst_raw[0, s, 1, :].copy_(torch.where(raw == bad, 0, raw * scales[0] + kv_offs[0]))

    cache = h2d_cache.H2DSkipCache(changed_span=True, incremental_block_offsets=True)

    # 1. Initial call for [0, 1] -> full copy.
    assert (
        cache.copy_block_offsets(
            dst, host_src, [0, 1], scales, kv_offs, bad, lambda: _ref_full([0, 1]), manager_id=1
        )
        == "full"
    )
    assert full_calls[0] == 1

    # 2. Same slots & content -> skip.
    staged_copies.clear()
    assert (
        cache.copy_block_offsets(
            dst, host_src, [0, 1], scales, kv_offs, bad, lambda: _ref_full([0, 1]), manager_id=1
        )
        == "skip"
    )
    assert full_calls[0] == 1 and len(staged_copies) == 0

    # 3. Single block growth on slot 1 -> incremental patches K and V planes cleanly.
    host_src[0, 1, 0, 3] = 23
    assert (
        cache.copy_block_offsets(
            dst, host_src, [0, 1], scales, kv_offs, bad, lambda: _ref_full([0, 1]), manager_id=1
        )
        == "incremental"
    )
    assert int(dst_raw[0, 1, 0, 3]) == 23 * 72
    assert int(dst_raw[0, 1, 1, 3]) == 23 * 72 + 3

    # 4. ADP dummy slot appended ([0, 1] -> [0, 1, 3]) -> key mismatch forces full copy (no stale).
    assert (
        cache.copy_block_offsets(
            dst,
            host_src,
            [0, 1, 3],
            scales,
            kv_offs,
            bad,
            lambda: _ref_full([0, 1, 3]),
            manager_id=1,
        )
        == "full"
    )
    assert full_calls[0] == 2
    assert int(dst_raw[0, 2, 0, 0]) == 99 * 72

    # 5. Reordered slots ([0, 1, 3] -> [1, 0, 3]) -> key mismatch forces full copy.
    assert (
        cache.copy_block_offsets(
            dst,
            host_src,
            [1, 0, 3],
            scales,
            kv_offs,
            bad,
            lambda: _ref_full([1, 0, 3]),
            manager_id=1,
        )
        == "full"
    )
    assert full_calls[0] == 3
    assert int(dst_raw[0, 0, 0, 0]) == 20 * 72

    # 6. Cache clear() invalidates block-offset shadow -> next call runs full copy.
    cache.clear()
    assert (
        cache.copy_block_offsets(
            dst,
            host_src,
            [1, 0, 3],
            scales,
            kv_offs,
            bad,
            lambda: _ref_full([1, 0, 3]),
            manager_id=1,
        )
        == "full"
    )
    assert full_calls[0] == 4

    # 7. Disabled incremental_block_offsets (CTX profile) always takes full copy.
    disabled = h2d_cache.H2DSkipCache(changed_span=False, incremental_block_offsets=False)
    for _ in range(2):
        assert (
            disabled.copy_block_offsets(
                dst,
                host_src,
                [0, 1],
                scales,
                kv_offs,
                bad,
                lambda: _ref_full([0, 1]),
                manager_id=1,
            )
            == "full"
        )
    assert full_calls[0] == 6


def test_ml4_dep_executor_events_lifecycle_gc_and_gil_restore(monkeypatch) -> None:
    _, _, dep_hooks = _load_dep_hooks_modules()

    refresh_calls: list[int] = []

    class _FakePin:
        def maybe_refresh(self, iteration: int) -> None:
            refresh_calls.append(iteration)

    monkeypatch.setattr(
        dep_hooks.ExecutorThreadPin, "create", classmethod(lambda cls, **_kw: _FakePin())
    )

    orig_interval = sys.getswitchinterval()
    orig_gc = gc.isenabled()
    gc.enable()
    try:
        # GEN mode (disable_gc=True): disables cyclic GC and restores on shutdown.
        ev_gen = dep_hooks.Ml4DepExecutorEvents(local_rank=0, gil_switch_us=200.0, disable_gc=True)
        ev_gen.after_warmup()
        ev_gen.before_overlap_loop()
        assert not gc.isenabled()
        assert sys.getswitchinterval() == pytest.approx(200e-6)
        ev_gen.on_iteration(16)
        assert refresh_calls == [16]
        ev_gen.at_shutdown()
        assert gc.isenabled()
        assert sys.getswitchinterval() == pytest.approx(orig_interval)

        # CTX mode (disable_gc=False): leaves cyclic GC enabled throughout.
        ev_ctx = dep_hooks.Ml4DepExecutorEvents(local_rank=1, gil_switch_us=200.0, disable_gc=False)
        ev_ctx.before_overlap_loop()
        assert gc.isenabled()
        assert sys.getswitchinterval() == pytest.approx(200e-6)
        ev_ctx.at_shutdown()
        assert gc.isenabled()
        assert sys.getswitchinterval() == pytest.approx(orig_interval)
    finally:
        sys.setswitchinterval(orig_interval)
        if orig_gc:
            gc.enable()
        else:
            gc.disable()


def test_dep4_and_tep4_modeling_wiring_contracts() -> None:
    """Verify both sm_107__dep4 and sm_107__tep4 wire RuntimeHooks and triton_embed_draft_block."""
    dep4_src = (_DEP4_DIR / "modeling.py").read_text()
    tep4_src = (_TEP4_DIR / "modeling.py").read_text()
    for src in (dep4_src, tep4_src):
        assert "self.runtime_hooks = make_ml4_dep_runtime_hooks(" in src
        assert "triton_embed_draft_block(" in src
        assert "triton_embed_draft_block.reference(" in src


def _impl_embed_draft_block_interpret() -> None:
    pkg = "_ml4_dspark_embed_cpu_test"
    mod_pkg = types.ModuleType(pkg)
    mod_pkg.__path__ = [str(_DSPARK_DIR)]
    sys.modules[pkg] = mod_pkg

    k_spec = importlib.util.spec_from_file_location(f"{pkg}._kernels", _DSPARK_DIR / "_kernels.py")
    assert k_spec is not None and k_spec.loader is not None
    k_mod = importlib.util.module_from_spec(k_spec)
    sys.modules[f"{pkg}._kernels"] = k_mod
    k_spec.loader.exec_module(k_mod)

    vocab, hidden, block_size = 64, 128, 3
    gen = torch.Generator().manual_seed(123)
    embed = torch.randn((vocab, hidden), generator=gen, dtype=torch.bfloat16)
    masks = torch.randn((block_size - 1, hidden), generator=gen, dtype=torch.bfloat16)
    anchors = torch.tensor([0, 7, -1, vocab, 63, 12], dtype=torch.int64)

    out = torch.empty((anchors.shape[0], block_size, hidden), dtype=torch.bfloat16)
    k_mod.embed_draft_block_kernel[(anchors.shape[0] * block_size, 1)](
        anchors,
        embed,
        masks,
        out,
        vocab,
        BLOCK=block_size,
        HIDDEN=hidden,
        HID_BLOCK=hidden,
        num_warps=4,
    )

    valid = (anchors >= 0) & (anchors < vocab)
    emb = torch.nn.functional.embedding(anchors.clamp(0, vocab - 1), embed)
    emb = emb.masked_fill(~valid.unsqueeze(-1), 0)
    ref = torch.cat((emb.unsqueeze(1), masks.unsqueeze(0).expand(anchors.shape[0], -1, -1)), dim=1)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def test_triton_embed_draft_block_interpret_matches_reference() -> None:
    """Verify the shared single-launch Triton draft-block embedding op matches its reference
    bit-for-bit on valid and out-of-vocab sentinel anchors."""
    if os.environ.get("TRITON_INTERPRET") == "1":
        _impl_embed_draft_block_interpret()
        return
    env = dict(os.environ, TRITON_INTERPRET="1")
    runner = (
        "import importlib.util, sys; "
        "spec = importlib.util.spec_from_file_location('_m', sys.argv[1]); "
        "mod = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        "mod._impl_embed_draft_block_interpret()"
    )
    res = subprocess.run(
        [sys.executable, "-c", runner, str(pathlib.Path(__file__).resolve())],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if res.returncode != 0:
        raise AssertionError(
            f"TRITON_INTERPRET subprocess failed:\n--- stdout ---\n{res.stdout}\n"
            f"--- stderr ---\n{res.stderr}"
        )
