# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather several tensors' rows from every rank in one NCCL group, via the trtllm
allgather_list op."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._utils import mpi_rank

from .._op import Arch, Cell, OpWrapper

# Mistral Large 4 NVFP4 at dep4 and dep8: the routed experts' inputs of every MoE layer (1..71)
# above the one-shot workspace -- e2m1 codes [rows, 3072] uint8, linear e4m3 scales
# [rows, 384] uint8, top-4 ids int32 [rows, 4] and weights bf16 [rows, 4] of each rank's own
# rows -- on the MPI-session communicator. `sizes` is every rank's row count when they differ
# (ncclBroadcast per rank), None when they are equal (ncclAllGather).
_ML4_PAYLOAD = (
    ((3072,), torch.uint8),
    ((384,), torch.uint8),
    ((4,), torch.int32),
    ((4,), torch.bfloat16),
)
_DEP4 = dict(payload=_ML4_PAYLOAD, world_size=4, process_group=None)
_DEP8 = dict(payload=_ML4_PAYLOAD, world_size=8, nodes=2, process_group=None)


class AllgatherList(OpWrapper):
    """Gather each tensor of `input_list` from every rank in `group`, concatenated along dim 0.

    Every tensor of the list has the same `shape[0]` on a rank. `sizes` gives each rank's
    `shape[0]` in ascending rank order, or is None when every rank holds the same number of
    rows. Returns one freshly allocated tensor per input, in order; the inputs are untouched.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep4 MoE dispatch past the one-shot workspace, equal counts: one ncclAllGather "
            "per tensor in one group",
            spec=dict(_DEP4, counts=(513, 513, 513, 513), sizes=None),
        ),
        Cell(
            why="dep4 MoE dispatch beside a prefilling peer: ragged counts, one ncclBroadcast per "
            "rank and tensor in one group",
            spec=dict(_DEP4, counts=(16352, 1, 64, 16)),
        ),
        Cell(
            why="dep4 MoE dispatch at max_num_tokens on one rank, the engine's one-row dummy on "
            "the rest",
            spec=dict(_DEP4, counts=(32768, 1, 1, 1)),
        ),
        Cell(
            why="dep4 MoE dispatch at max_num_tokens on every rank, equal counts",
            spec=dict(_DEP4, counts=(32768,) * 4, sizes=None),
        ),
        Cell(
            why="dep4 MoE dispatch, ragged counts swept: every rank's count from 1 to 2048 at a "
            "stride of 97 against three ranks at 1, 513 and 4096",
            spec=dict(_DEP4, sweep=(1, 2048, 97), others=(1, 513, 4096)),
        ),
        Cell(
            why="dep8 MoE dispatch over two nodes beside a prefilling peer: ragged counts",
            spec=dict(_DEP8, counts=(16352, 1, 64, 16, 1, 600, 3, 17)),
        ),
        Cell(
            why="dep8 MoE dispatch over two nodes at max_num_tokens on every rank, equal counts",
            spec=dict(_DEP8, counts=(32768,) * 8, sizes=None),
        ),
    )

    note = """
    `torch.ops.trtllm.allgather_list` on the MPI-session communicator (`process_group=None`),
    the communicator `allgather` documents, or `allgather_list_pg` on a boxed c10d group. One
    ncclGroupStart/End around the list: with `sizes=None` (or every entry equal) one
    ncclAllGather per tensor, otherwise one ncclBroadcast per rank and tensor, the i-th rank
    sending `sizes[i]` rows of each tensor to row `sum(sizes[:i])` of every rank's result.
    The ragged form takes the row count from `sizes` alone and the trailing elements from the
    tensor's shape, so a 1-D tensor must be passed 2-D (`[rows, width]`): a flat one would
    broadcast `sizes[i]` elements, not rows.

    Preconditions are `allgather`'s on the same symbol: every rank makes the same calls in the
    same order at the same shapes, the group's first call is eager, `sizes[i]` is the i-th
    rank's `shape[0]` for every tensor of the list (`is_valid`).

    Measured on four sm_107 ranks of one VR200 node at equal counts, 71 calls back to back in
    one CUDA graph, the slowest rank's median per call, the Mistral Large 4 payload: 9.4 to
    15.8 us up to 16 rows per rank, 23.8 at 64, 25.0 at 128, 35.9 at 256 and 64.2 at 512.
    One bf16 all-gather of the `[rows, 6144]` activations takes 9.5 to 11.1, 28.1, 49.5, 65.6
    and 76.0 us there: below 64 rows per rank the four collectives' fixed cost outweighs the
    smaller payload. The world-8 cells were driven on two VR200 nodes of one NVL72 domain, bit
    for bit on all eight ranks.
    """

    def __call__(
        self,
        input_list: List[torch.Tensor],
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> List[torch.Tensor]:
        if process_group is None:
            return torch.ops.trtllm.allgather_list(input_list, sizes, group)
        return torch.ops.trtllm.allgather_list_pg(input_list, sizes, group, process_group)

    def reference(
        self,
        per_rank_inputs: Sequence[Sequence[torch.Tensor]],
        sizes: Optional[List[int]] = None,
        group: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Each tensor's concatenation over the group, from every rank's list in rank order
        (the collective exception to mirroring `__call__`)."""
        return [torch.cat(list(parts), dim=0) for parts in zip(*per_rank_inputs)]

    def is_valid(
        self,
        input_list: List[torch.Tensor],
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> None:
        rows = {t.shape[0] for t in input_list}
        assert len(rows) == 1, f"every tensor must hold the rank's rows; heights {sorted(rows)}"
        for t in input_list:
            assert t.dim() >= 1 and t.is_contiguous(), "inputs are read as packed rows"
            assert sizes is None or t.dim() >= 2, (
                "a 1-D tensor in the ragged form broadcasts sizes[i] elements rather than rows"
            )
        assert sizes is None or len(sizes) == len(group), (len(sizes), len(group))
        if process_group is None and sizes is not None and mpi_rank() in group:
            mine = sorted(set(group)).index(mpi_rank())
            assert sizes[mine] == next(iter(rows)), (
                f"sizes[{mine}]={sizes[mine]} must equal this rank's rows {next(iter(rows))}"
            )


allgather_list = AllgatherList()
