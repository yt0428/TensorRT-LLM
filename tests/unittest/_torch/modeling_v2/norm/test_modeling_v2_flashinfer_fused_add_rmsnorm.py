# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_fused_add_rmsnorm catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and
`compare` are the gate, `is_valid` is the guard. This file owns only what a
cell cannot carry -- turning a spec into real tensors -- plus the inputs the
guard exists to refuse, which are claims about inputs no cell describes.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)

__extra_import_path__ = [".."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "flashinfer_fused_add_rmsnorm requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    x = torch.randn(shape, generator=g, device="cuda").to(dtype)
    residual = torch.randn(shape, generator=g, device="cuda").to(dtype)
    weight = torch.randn(shape[-1], generator=g, device="cuda").to(dtype)
    return x, residual, weight, spec["eps"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Every certified configuration, kernel against reference.

    Driven inside `validating`, so each cell also proves the guard admits the
    inputs a shipped target actually passes.
    """
    x, residual, weight, eps = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref_x, ref_residual = op.reference(x, residual, weight, eps)
    with validating(op):
        out_x, out_residual = op(x, residual, weight, eps)
        assert out_x is x and out_residual is residual, "the op writes in place"
    torch.testing.assert_close(residual, ref_residual)
    torch.testing.assert_close(x, ref_x)


@pytest.mark.parametrize("rows", [4, 16, 32, 64, 128])
def test_matches_the_separate_add_and_rmsnorm(rows) -> None:
    """The pair it replaces in a decode target, `residual = residual + x` then
    `rmsnorm(residual)`: the residual is bitwise the bf16 add (the fp32 sum
    rounded once); the normalized row reads that sum before the rounding, so it
    differs from normalizing the rounded residual by that rounding and by each
    side's own output rounding: three bf16 roundings of about 0.42 ulp RMS each,
    0.73 ulp RMS measured on VR200 at every row count here."""
    g = torch.Generator(device="cuda").manual_seed(rows)
    shape = (rows, 6144)
    # A residual well above the projection added into it, so the add rounds.
    x = (torch.randn(shape, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    residual = (torch.randn(shape, generator=g, device="cuda") * 8.0).to(torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn(6144, generator=g, device="cuda")).to(torch.bfloat16)
    expected_residual = torch.add(residual, x)
    expected_x = flashinfer_rmsnorm(expected_residual, weight, 1e-5)
    out_x, out_residual = op(x, residual, weight, 1e-5)
    assert torch.equal(out_residual, expected_residual)
    assert_within_ulp(out_x, expected_x, element_ulp=2.0, rms_ulp=1.0)


def _one(**over):
    g = torch.Generator(device="cuda").manual_seed(7)
    kw = dict(rows=4, hidden=2880, dtype=torch.bfloat16)
    kw.update(over)
    shape = (kw["rows"], kw["hidden"])
    return (
        torch.randn(shape, generator=g, device="cuda").to(kw["dtype"]),
        torch.randn(shape, generator=g, device="cuda").to(kw["dtype"]),
        torch.randn(shape[-1], generator=g, device="cuda").to(kw["dtype"]),
    )


def test_guard_refuses_fp8() -> None:
    """Accepted, and silently plausible: the case a caller cannot detect."""
    x, residual, weight = _one(dtype=torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="dequantize"):
        with validating(op):
            op(x, residual, weight, 1e-6)


def test_guard_refuses_a_strided_normalized_dim() -> None:
    """`stride(-1) != 1` reads the wrong elements rather than failing."""
    buf = torch.randn(8, 2880, 2, dtype=torch.bfloat16, device="cuda")
    x, residual = buf[..., 0], buf[..., 1]
    assert x.stride(-1) != 1
    weight = torch.randn(2880, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, residual, weight, 1e-6)


def test_guard_refuses_aliased_buffers() -> None:
    """The op writes both tensors; one buffer passed twice yields neither."""
    x, _, weight = _one()
    with pytest.raises(AssertionError, match="distinct buffers"):
        with validating(op):
            op(x, x, weight, 1e-6)


def test_the_guard_is_not_on_the_hot_path() -> None:
    """Outside `validating`, the wrapper goes straight to the kernel.

    Stated as a test because it is the property that lets the guard be as
    strict as it likes: a served engine never runs it.
    """
    x, residual, weight = _one(dtype=torch.float8_e4m3fn)
    # fp8 is what the guard refuses; unguarded, the kernel answers anyway.
    op(x, residual, weight, 1e-6)
