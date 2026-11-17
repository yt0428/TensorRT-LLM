# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the triton_widen_softcap catalog entry: every cell bitwise against its reference,
the fp32 upcast and tanh_softcap's three torch ops, with one draw of each carrying every finite
bf16 bit pattern; the captured cells also replayed in a CUDA graph on fresh logits, and the
guard's refusals."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.triton_widen_softcap import (
    triton_widen_softcap as op,
)

__extra_import_path__ = [".."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires VR200/SM107",
)

DRAWS = 3
GRAPH_REPLAYS = 10


def _logits(rows: int, cols: int, seed: int) -> torch.Tensor:
    """bf16 lm_head logits spread over the soft-cap's range, saturation included."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, cols, device="cuda", generator=gen) * 20).to(torch.bfloat16)


def _with_every_finite_bf16(x: torch.Tensor) -> torch.Tensor:
    """Overwrite the head of `x` with every finite bf16 bit pattern, subnormals and both zeros
    included."""
    bits = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda").to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[values.isfinite()]
    flat = x.view(-1)
    assert flat.numel() >= values.numel(), (flat.numel(), values.numel())
    flat[: values.numel()] = values
    return x


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    assert out.dtype == ref.dtype == torch.float32, (where, out.dtype, ref.dtype)
    assert out.shape == ref.shape, (where, out.shape, ref.shape)
    differ = out.view(torch.int32) != ref.view(torch.int32)
    if differ.any():
        first = differ.nonzero()[0].tolist()
        raise AssertionError(
            f"{where}: {int(differ.sum())} of {out.numel()} values differ, first at {first}: "
            f"{out[tuple(first)].item()!r} vs {ref[tuple(first)].item()!r}"
        )


@pytest.mark.parametrize("cell", op.CELLS, ids=[cell.why for cell in op.CELLS])
def test_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    for draw in range(DRAWS):
        x = _logits(spec["rows"], spec["cols"], cell.seed_base() + draw)
        if draw == 0:
            _with_every_finite_bf16(x)
        out = op(x, spec["cap"])
        _assert_bitwise(out, op.reference(x, spec["cap"]), f"draw {draw}")


_CAPTURED = [cell for cell in op.CELLS if cell.spec["graph"]]


@pytest.mark.parametrize("cell", _CAPTURED, ids=[cell.why for cell in _CAPTURED])
def test_graph_replay(cell) -> None:
    """The call captured once and replayed on fresh logits written into the captured input,
    as a decode graph replays it; the output tensor is the one the capture returned."""
    spec = cell.spec
    x = _logits(spec["rows"], spec["cols"], cell.seed_base())
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x, spec["cap"])
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, spec["cap"])
    for replay in range(GRAPH_REPLAYS):
        x.copy_(_logits(spec["rows"], spec["cols"], cell.seed_base() + 100 + replay))
        graph.replay()
        _assert_bitwise(out, op.reference(x, spec["cap"]), f"replay {replay}")


def test_nonfinite_logits_follow_the_torch_chain() -> None:
    """+-inf caps to +-cap and nan stays nan, as the torch chain answers them."""
    x = _logits(2, 8192, 5)
    x[0, :3] = torch.tensor([float("inf"), float("-inf"), float("nan")], dtype=torch.bfloat16)
    out = op(x, 30.0)
    ref = op.reference(x, 30.0)
    assert out[0, 0].item() == 30.0 and out[0, 1].item() == -30.0 and out[0, 2].isnan()
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0, equal_nan=True)


@pytest.mark.parametrize(
    "x, cap",
    [
        (torch.zeros(4, 64, dtype=torch.float32), 30.0),
        (torch.zeros(4, 128, dtype=torch.bfloat16)[:, ::2], 30.0),
        (torch.zeros(2, 4, 64, dtype=torch.bfloat16), 30.0),
        (torch.zeros(4, 64, dtype=torch.bfloat16), 0.0),
        (torch.zeros(4, 64, dtype=torch.bfloat16), float("inf")),
        (torch.zeros(4, 64, dtype=torch.bfloat16), float("nan")),
    ],
    ids=["fp32 input", "strided rows", "3-D input", "zero cap", "infinite cap", "nan cap"],
)
def test_is_valid_refuses(x, cap) -> None:
    with pytest.raises(ValueError):
        op.is_valid(x, cap)
