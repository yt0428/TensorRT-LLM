# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send each attention-DP rank's routed-expert inputs only to the ranks holding their experts,
through the NVLink one-sided alltoall."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._moe_workspace import build_moe_alltoall as build_moe_alltoall

# Mistral Large 4 NVFP4 at dep4: the dispatch of every MoE layer (1..71) of a step whose largest
# rank holds more than the 512-row one-shot workspace, on the alltoall an engine of
# max_num_tokens 16384 builds (`build_moe_alltoall`): each rank's own rows' e2m1 codes
# [rows, 3072] uint8, linear e4m3 scales [rows, 384] uint8, top-4 expert ids int32 and weights
# bf16, 32 of the 128 experts per rank. `routing` names how the cells' ids are drawn: 4 distinct
# experts uniformly ("uniform"), one on every rank ("spread"), all on the row's own rank
# ("local"), all on rank 0 ("hotspot").
_ML4_DEP4 = dict(world_size=4, hidden=6144, num_experts=128, top_k=4, max_rows=16384)


class A2aMoeDispatch(OpWrapper):
    """The rows every rank sent this rank: `(codes, scales, ids, weights)`, each
    `[world * max(counts), width]`, row block `r` holding rank `r`'s rows that have one of their
    experts in this rank's window, in no fixed order, then rows whose ids are all -1. Codes,
    scales and weights arrive bit for bit, and ids as the global expert ids they were sent with.

    `codes` is uint8 `[counts[rank], hidden // 2]`, `scales` uint8 `[counts[rank], hidden // 16]`,
    `ids` int32 and `weights` bf16 `[counts[rank], top_k]`, this rank's own rows. The results are
    views of `a2a`'s workspace, valid until its combine (`a2a_moe_combine`), which must follow
    before the next dispatch. Every rank calls with the same `counts`.
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
            why="dep4 every row on an expert of every rank: the alltoall's most traffic",
            spec=dict(_ML4_DEP4, counts=(8192,) * 4, routing="spread"),
        ),
        Cell(
            why="dep4 every row on its own rank's experts: nothing crosses ranks",
            spec=dict(_ML4_DEP4, counts=(4096, 2048, 1024, 513), routing="local"),
        ),
        Cell(
            why="dep4 every row on rank 0's experts: one rank receives every row",
            spec=dict(_ML4_DEP4, counts=(4096,) * 4, routing="hotspot"),
        ),
    )

    note = """
    `NVLinkOneSided.dispatch` (`torch.ops.trtllm.moe_a2a_dispatch` and, past the counted-write
    batch, `moe_a2a_sanitize_expert_ids`) over MNNVL memory, on the alltoall `build_moe_alltoall`
    makes. Each rank writes each of its rows once into the receive region of every rank holding
    one of the row's experts; a receiving rank's region per source holds those rows in arrival
    order, and every other row of it carries ids -1, which the runner skips. A row's four
    payloads move as bytes, so the results are its own codes, scales, ids and weights.

    Preconditions: collective; every rank of the alltoall calls with the same `counts`, no entry
    past the alltoall's max_rows, and pairs each dispatch with one combine before the next.

    Measured on four sm_107 ranks of one VR200 node with the Mistral Large 4 payload, with ids
    of 4 distinct experts drawn uniformly (each rank receiving about 69% of the rows): this
    dispatch and `a2a_moe_combine` moved 16384 rows per rank in 540 us against 1767 for
    `allgather_list` and `reducescatter` on the MPI-session communicator, 512 against 1705
    beside a one-row dummy, 503 against 1078 at (16384, 9001, 513, 2) and 40 against 132 at
    (513, 1, 1, 1); with every row on every rank, 770 against 1767 at 16384 per rank.
    """

    def __call__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        counts: Sequence[int],
        a2a: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return a2a.dispatch(codes, scales, ids, weights, list(counts))

    def reference(
        self,
        per_rank_inputs: Sequence[Sequence[torch.Tensor]],
        rank: int,
        num_experts: int,
    ) -> list[tuple[torch.Tensor, ...]]:
        """Per source rank, its rows with an expert in `rank`'s window, in the source's row
        order: what that source's block holds, up to row order and the -1 rows (the collective
        exception to mirroring `__call__`)."""
        per = num_experts // len(per_rank_inputs)
        lo, hi = rank * per, (rank + 1) * per
        out = []
        for codes, scales, ids, weights in per_rank_inputs:
            hit = ((ids >= lo) & (ids < hi)).any(dim=1)
            out.append((codes[hit], scales[hit], ids[hit], weights[hit]))
        return out

    def is_valid(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        counts: Sequence[int],
        a2a: object,
    ) -> None:
        assert a2a is not None, "the engine built no alltoall"
        assert len(counts) == a2a.ep_size, (len(counts), a2a.ep_size)
        assert max(counts) <= a2a.max_num_tokens_per_rank, (
            f"counts {tuple(counts)} pass the alltoall's {a2a.max_num_tokens_per_rank} rows"
        )
        n = counts[a2a.ep_rank]
        for name, t, dtype, width in (
            ("codes", codes, torch.uint8, None),
            ("scales", scales, torch.uint8, None),
            ("ids", ids, torch.int32, a2a.top_k),
            ("weights", weights, torch.bfloat16, a2a.top_k),
        ):
            assert t.dim() == 2 and t.shape[0] == n and t.is_contiguous(), (
                f"{name} {tuple(t.shape)} must be this rank's {n} rows, packed 2-D"
            )
            assert t.dtype == dtype and (width is None or t.shape[1] == width), (name, t.dtype)


a2a_moe_dispatch = A2aMoeDispatch()
