# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VR200 test for the sm_107 dep4 target's moe_finalize entry: bitwise against its reference,
the expert runner's finalize sum, eager and under graph replay."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__dep4.ops.moe.moe_finalize import (  # noqa: E501
    HIDDEN,
    TOPK,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__dep4.ops.moe.moe_finalize import (  # noqa: E501
    moe_finalize as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires a VR200/SM107 device",
)

#: Experts of all ranks; the runner's window is `local` of them.
EXPERTS = 128


def _inputs(rows: int, local: int, seed: int):
    """A window's runner outputs as finalize reads them: top-k ids over all experts, the
    expanded-to-permuted map (-1 outside the window, each in-window slot its own permuted row,
    the rows padded past the last), the permuted FC2 rows and the routing weights."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.argsort(torch.rand(rows, EXPERTS, generator=g, device="cuda"), dim=1)[:, :TOPK]
    # Every case holds tokens with no expert in the window: the first rows route outside it.
    ids[: max(rows // 8, 1)] = torch.arange(local, local + TOPK, device="cuda")
    inside = ids < local
    idx = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
    slots = int(inside.sum())
    order = torch.randperm(slots, generator=g, device="cuda").to(torch.int32)
    idx[inside] = order
    permuted = (torch.randn(slots + 128, HIDDEN, generator=g, device="cuda") * 4).to(torch.bfloat16)
    weights = torch.softmax(torch.randn(rows, TOPK, generator=g, device="cuda"), dim=1)
    return permuted, idx, weights.to(torch.bfloat16)


def _check_rows(rows: int, local: int, seed: int) -> None:
    permuted, idx, weights = _inputs(rows, local, seed)
    out = torch.full((rows, HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda")
    op(permuted, idx, weights, out)
    want = op.reference(permuted, idx, weights, torch.empty_like(out))
    assert torch.equal(out.view(torch.int16), want.view(torch.int16)), f"rows {rows}: not bitwise"
    empty = (idx < 0).all(dim=1)
    assert bool(empty.any()) and not bool(out[empty].view(torch.int16).any()), (
        f"rows {rows}: a token with no expert in the window is not +0.0"
    )


def _check_graph(rows: int, local: int, seed: int) -> None:
    """Capture at `rows` after one warm-up call, replay on new data, require the eager result."""
    static = _inputs(rows, local, seed)
    fresh = _inputs(rows, local, seed + 1)
    slots = min(static[0].shape[0], fresh[0].shape[0])
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    op(*static, out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        op(*static, out)
    torch.cuda.current_stream().wait_stream(stream)
    # New values and a new map over the slots both draws hold.
    static[0][:slots].copy_(fresh[0][:slots])
    static[1].copy_(torch.where(fresh[1] < slots, fresh[1], -1))
    static[2].copy_(fresh[2])
    graph.replay()
    want = op(*static, torch.empty_like(out))
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("cell", op.CELLS, ids=[cell.why[:48] for cell in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest fixture
    certifying(op)
    spec = cell.spec
    seed = cell.seed_base()
    for rows in spec["rows"]:
        _check_rows(rows, spec["local_experts"], seed + rows)
        _check_graph(rows, spec["local_experts"], seed + 1000 + rows)
