# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Return the attention-DP expert partials to the ranks that own the rows and sum them there,
through the NVLink one-sided alltoall."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Mistral Large 4 NVFP4 at dep4: the combine of every MoE layer (1..71) of a step whose largest
# rank holds more than the 512-row one-shot workspace, on the alltoall an engine of
# max_num_tokens 16384 builds: the runner's bf16 [world * max(counts), 6144] partial of its
# expert window over the rows `a2a_moe_dispatch` handed it, written into the combine payload.
# `routing` as in `a2a_moe_dispatch`'s cells.
_ML4_DEP4 = dict(world_size=4, hidden=6144, num_experts=128, top_k=4, max_rows=16384)


def a2a_moe_combine_payload(counts: Sequence[int], hidden: int, a2a: object) -> torch.Tensor:
    """The combine's input in `a2a`'s workspace, bf16 `[world * max(counts), hidden]`, the layout
    of the dispatch it follows: the expert runner writes its partial here, and `a2a_moe_combine`
    reads it in place. Valid between a dispatch and its combine."""
    return a2a.get_combine_payload_tensor_in_workspace(max(counts), hidden, torch.bfloat16)


class A2aMoeCombine(OpWrapper):
    """This rank's own rows' sum of the partials the ranks computed for them, freshly allocated
    bf16 `[counts[rank], hidden]`.

    `partial` is `a2a_moe_combine_payload` after the runner wrote into it: row `j` of block `r`
    is this rank's expert window's output for the row rank `r` sent to slot `j` of the dispatch
    this combine follows. Each rank's row comes back from every rank it went to and those
    partials are summed in fp32 and rounded once; a rank it did not go to adds nothing.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep4 prefill step at max_num_tokens on every rank",
            spec=dict(_ML4_DEP4, counts=(16384,) * 4, routing="uniform"),
        ),
        Cell(
            why="dep4 prefill step at max_num_tokens beside a rank holding the engine's one-row "
            "dummy",
            spec=dict(_ML4_DEP4, counts=(16384, 16384, 16384, 1), routing="uniform"),
        ),
        Cell(
            why="dep4 prefill step, ragged counts on both sides of the one-shot workspace",
            spec=dict(_ML4_DEP4, counts=(16384, 9001, 513, 2), routing="uniform"),
        ),
        Cell(
            why="dep4 the smallest step past the one-shot workspace, beside three one-row dummies",
            spec=dict(_ML4_DEP4, counts=(513, 1, 1, 1), routing="uniform"),
        ),
        Cell(
            why="dep4 every row on an expert of every rank: four partials per row",
            spec=dict(_ML4_DEP4, counts=(8192,) * 4, routing="spread"),
        ),
        Cell(
            why="dep4 every row on its own rank's experts: one partial per row, none crossing",
            spec=dict(_ML4_DEP4, counts=(4096, 2048, 1024, 513), routing="local"),
        ),
        Cell(
            why="dep4 every row on rank 0's experts: every partial from one rank",
            spec=dict(_ML4_DEP4, counts=(4096,) * 4, routing="hotspot"),
        ),
    )

    note = """
    `NVLinkOneSided.combine` (`torch.ops.trtllm.moe_a2a_combine`) with the payload in the
    workspace: each rank sends each row's partial back to the row's own rank, which sums the
    up to `world` partials of each of its rows in fp32 and rounds once to bf16.

    Numerics: on random partials at 13 count sets from 513 to 16384 rows per rank, the sum
    differed from the fp32 sum in rank order, rounded once, in 1 or 2 of 50 to 200 million
    elements -- the order of its terms is the kernel's -- and from NCCL's ring reduce-scatter
    (bf16 hop by hop) in 16% of the elements with each row on 69% of the ranks, 33% with each
    row on every rank. The cells' partials are small integers, whose every sum is exact, so
    they are gated bitwise against `reference`.

    Preconditions: collective, every rank of the alltoall calls it once after each dispatch.
    """

    def __call__(self, partial: torch.Tensor, a2a: object) -> torch.Tensor:
        return a2a.combine(partial)

    def reference(
        self,
        partials: Sequence[torch.Tensor],
        hits: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """`partials[p]` is what rank `p`'s window gives this rank's rows (bf16
        `[counts[rank], hidden]`) and `hits[p]` which of the rows went to `p`: their fp32 sum
        in rank order over the ranks each row went to, rounded once (the collective exception
        to mirroring `__call__`)."""
        acc = torch.zeros_like(partials[0], dtype=torch.float32)
        for p, hit in zip(partials, hits):
            acc += p.float() * hit[:, None]
        return acc.to(partials[0].dtype)

    def is_valid(self, partial: torch.Tensor, a2a: object) -> None:
        assert a2a is not None, "the engine built no alltoall"
        assert partial.dtype == torch.bfloat16 and partial.dim() == 2, (
            f"partial must be bf16 [world * max(counts), hidden], got {partial.dtype} "
            f"{tuple(partial.shape)}"
        )
        assert partial.is_contiguous() and partial.shape[0] % a2a.ep_size == 0, (
            "partial must be the combine payload (`a2a_moe_combine_payload`)"
        )


a2a_moe_combine = A2aMoeCombine()
