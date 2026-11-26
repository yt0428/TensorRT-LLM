# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit and contract tests for 8R/16R parallel wave Lamport AllReduce+Add+RMSNorm."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import struct
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[5]
_KERNEL_PATH = (
    _REPO_ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/comm/_tp4_allreduce_fused_rms_kernel.py"
)


def _load_ar_kernel_module():
    name = "_modeling_v2_lamport_ar_kernel_cpu_test"
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, _KERNEL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_AR = _load_ar_kernel_module()


@pytest.mark.parametrize(
    "row_elem_unit",
    (_AR._BF16_VEC_SIZE, _AR._SCALE_BLOCK),
    ids=("fused_bf16", "out_quant_nvfp4"),
)
def test_resolve_parallel_poll_config_defaults_by_world_size(row_elem_unit: int) -> None:
    """TP4 stays on its 4-CTA x 192-thread path; TP8/TP16 enable parallel wave polling + cpr8."""
    tp4 = _AR._resolve_parallel_poll_config(
        world_size=4,
        n=6144,
        split_rows=True,
        use_ready_flags=False,
        ctas_per_row=_AR._FUSED_CTAS_PER_ROW,
        threads_per_block=_AR._SPLIT_ROW_THREADS_PER_BLOCK,
        row_elem_unit=row_elem_unit,
    )
    assert tp4 == (False, 4, False, 4, 192)

    tp8 = _AR._resolve_parallel_poll_config(
        world_size=8,
        n=6144,
        split_rows=True,
        use_ready_flags=False,
        ctas_per_row=_AR._FUSED_CTAS_PER_ROW,
        threads_per_block=_AR._SPLIT_ROW_THREADS_PER_BLOCK,
        row_elem_unit=row_elem_unit,
    )
    assert tp8 == (True, 8, True, 8, 128)

    tp16 = _AR._resolve_parallel_poll_config(
        world_size=16,
        n=6144,
        split_rows=True,
        use_ready_flags=False,
        ctas_per_row=_AR._FUSED_CTAS_PER_ROW,
        threads_per_block=_AR._SPLIT_ROW_THREADS_PER_BLOCK,
        row_elem_unit=row_elem_unit,
    )
    assert tp16 == (True, 16, True, 8, 128)

    tp32 = _AR._resolve_parallel_poll_config(
        world_size=32,
        n=6144,
        split_rows=True,
        use_ready_flags=False,
        ctas_per_row=_AR._FUSED_CTAS_PER_ROW,
        threads_per_block=_AR._SPLIT_ROW_THREADS_PER_BLOCK,
        row_elem_unit=row_elem_unit,
    )
    assert tp32 == (True, 16, True, 8, 128)


def test_resolve_parallel_poll_config_respects_overrides_and_alignment() -> None:
    """Explicit caller overrides, batch/ready-mask polling, ready-flags, and unaligned N."""
    disabled = _AR._resolve_parallel_poll_config(
        world_size=16,
        n=6144,
        split_rows=True,
        use_ready_flags=False,
        ctas_per_row=4,
        threads_per_block=192,
        row_elem_unit=_AR._BF16_VEC_SIZE,
        parallel_poll_peers=False,
    )
    assert disabled == (False, 16, False, 4, 192)

    custom_threads = _AR._resolve_parallel_poll_config(
        world_size=8,
        n=6144,
        split_rows=True,
        use_ready_flags=False,
        ctas_per_row=4,
        threads_per_block=256,
        row_elem_unit=_AR._BF16_VEC_SIZE,
    )
    assert custom_threads == (True, 8, True, 4, 256)

    unaligned_out_quant = _AR._resolve_parallel_poll_config(
        world_size=8,
        n=64,
        split_rows=True,
        use_ready_flags=False,
        ctas_per_row=4,
        threads_per_block=192,
        row_elem_unit=_AR._SCALE_BLOCK,
    )
    assert unaligned_out_quant == (True, 8, True, 4, 192)

    batch_poll = _AR._resolve_parallel_poll_config(
        world_size=8,
        n=6144,
        split_rows=True,
        use_ready_flags=False,
        ctas_per_row=4,
        threads_per_block=192,
        row_elem_unit=_AR._BF16_VEC_SIZE,
        fused_batch_poll_peers=True,
    )
    assert batch_poll == (False, 8, False, 4, 192)

    ready_mask_poll = _AR._resolve_parallel_poll_config(
        world_size=16,
        n=6144,
        split_rows=True,
        use_ready_flags=False,
        ctas_per_row=4,
        threads_per_block=192,
        row_elem_unit=_AR._BF16_VEC_SIZE,
        fused_ready_mask_poll_peers=True,
    )
    assert ready_mask_poll == (False, 16, False, 4, 192)

    ready_flags_path = _AR._resolve_parallel_poll_config(
        world_size=8,
        n=6144,
        split_rows=True,
        use_ready_flags=True,
        ctas_per_row=4,
        threads_per_block=192,
        row_elem_unit=_AR._BF16_VEC_SIZE,
    )
    assert ready_flags_path == (False, 8, False, 4, 192)

    non_split = _AR._resolve_parallel_poll_config(
        world_size=8,
        n=6144,
        split_rows=False,
        use_ready_flags=False,
        ctas_per_row=4,
        threads_per_block=192,
        row_elem_unit=_AR._BF16_VEC_SIZE,
    )
    assert non_split == (False, 8, False, 4, 192)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(split_rows=False, parallel_poll_peers=True), "parallel_poll_peers requires"),
        (dict(use_ready_flags=True, parallel_poll_peers=True), "parallel_poll_peers requires"),
        (dict(parallel_poll_wave_size=0), "parallel_poll_wave_size must be >= 1"),
        (dict(use_ready_flags=True, fast_clear=True), "fast_clear requires"),
        (dict(split_rows=False, fast_clear=True), "fast_clear requires"),
    ],
)
def test_resolve_parallel_poll_config_rejects_invalid_combinations(
    kwargs: dict, match: str
) -> None:
    params = dict(
        world_size=8,
        n=6144,
        split_rows=True,
        use_ready_flags=False,
        ctas_per_row=4,
        threads_per_block=192,
        row_elem_unit=_AR._BF16_VEC_SIZE,
    )
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        _AR._resolve_parallel_poll_config(**params)


def test_packed_bf16x2_sentinel_check_and_sanitize_exhaustive() -> None:
    """Exhaustively verify the packed u32 sentinel bit-trick and -0.0 sanitization semantics."""

    def word_has_neg_zero(word: int) -> bool:
        x = word ^ 0x80008000
        t = ((x - 0x00010001) & 0xFFFFFFFF) & (~x & 0xFFFFFFFF) & 0x80008000
        return t != 0

    def sanitize_word(word: int) -> int:
        r = word
        if (r & 0xFFFF) == 0x8000:
            r ^= 0x8000
        if (r & 0xFFFF0000) == 0x80000000:
            r ^= 0x80000000
        return r

    borrow_neighbors = (0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0x3F80, 0xFFFF)
    for lo in range(65536):
        for hi in borrow_neighbors:
            word = lo | (hi << 16)
            expected_not_ready = (lo == 0x8000) or (hi == 0x8000)
            assert word_has_neg_zero(word) is expected_not_ready, f"lo={lo:#x} hi={hi:#x}"
            san = sanitize_word(word)
            assert (san & 0xFFFF) == (0 if lo == 0x8000 else lo)
            assert ((san >> 16) & 0xFFFF) == (0 if hi == 0x8000 else hi)


def test_packed_bf16x2_add_matches_torch_bf16_add() -> None:
    """Verify packed bf16x2 FP32-rounded add matches torch.add bit-for-bit."""
    edge_vals = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 1e-4, -1e-4, 123.5, -456.25, 65000.0, -65000.0],
        dtype=torch.bfloat16,
    )
    gen = torch.Generator().manual_seed(42)
    rand_a = torch.randn(256, generator=gen, dtype=torch.bfloat16)
    rand_b = torch.randn(256, generator=gen, dtype=torch.bfloat16)
    a = torch.cat([edge_vals, rand_a])
    b = torch.cat([edge_vals.flip(0), rand_b])
    expected = torch.add(a, b).view(torch.uint16).tolist()

    a_u16 = a.view(torch.uint16).tolist()
    b_u16 = b.view(torch.uint16).tolist()
    for i in range(0, len(a_u16), 2):
        wa = a_u16[i] | (a_u16[i + 1] << 16)
        wb = b_u16[i] | (b_u16[i + 1] << 16)
        # Mirror _add_bf16x2_word_to_u32: unpack bf16x2 to f32, add.rn.f32, cvt.rn.bf16x2.f32.
        al = struct.unpack("<f", struct.pack("<I", (wa << 16) & 0xFFFFFFFF))[0]
        ah = struct.unpack("<f", struct.pack("<I", wa & 0xFFFF0000))[0]
        bl = struct.unpack("<f", struct.pack("<I", (wb << 16) & 0xFFFFFFFF))[0]
        bh = struct.unpack("<f", struct.pack("<I", wb & 0xFFFF0000))[0]
        got_lo = torch.tensor(al + bl, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16)
        got_hi = torch.tensor(ah + bh, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16)
        assert int(got_lo) == expected[i]
        assert int(got_hi) == expected[i + 1]


def test_parallel_wave_kernel_contracts_and_clean_hygiene() -> None:
    """Verify kernel entrypoints, signatures, and absence of temporary A/B env toggles."""
    for symbol in (
        "_parallel_wave_poll_and_accumulate_f32",
        "_words4_not_ready_i32",
        "_sanitize_bf16x2_u32",
        "_add_bf16x2_word_to_u32",
        "_ld_volatile_v4_u32",
        "_ld_volatile_v4_u32_if",
        "_ld_acquire_gpu_v4_s32",
        "_st_evict_first_v4_b32",
        "_resolve_parallel_poll_config",
    ):
        assert hasattr(_AR, symbol), f"missing {symbol}"

    for fn in (_AR.compile_fused_for_rank, _AR.compile_fused_out_quant_for_rank):
        sig = inspect.signature(fn)
        for param in ("parallel_poll_peers", "parallel_poll_wave_size", "fast_clear"):
            assert param in sig.parameters
            assert sig.parameters[param].default is None
        src = inspect.getsource(fn)
        assert "_resolve_parallel_poll_config" in src

    source = _KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "os" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "os"
    assert "os.environ" not in source
    assert "TRTLLM_ML4_AR_" not in source
