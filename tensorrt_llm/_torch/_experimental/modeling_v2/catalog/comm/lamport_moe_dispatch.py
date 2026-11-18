# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather every attention-DP rank's routed-expert inputs in one-shot Lamport stores."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import _moe_lamport_kernel as _k
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm._lamport import MoeCommWorkspace

# A target builds the workspace this entry is handed through the entry itself, so `_lamport`
# stays private to the catalog.
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm._lamport import (
    build_moe_comm_workspace as build_moe_comm_workspace,
)

#: The most ranks the kernel's row counts address.
_MAX_WORLD = 8
#: A gathered row travels zero-padded to 16 B, eight 16-bit words.
_ROW_ALIGN = 8

# Mistral Large 4 NVFP4 at dep4 and dep8: the routed experts' inputs of every MoE layer (1..71)
# -- e2m1 codes [rows, 3072], linear e4m3 scales [rows * 384], top-4 ids int32 and weights
# bf16 -- of each rank's own rows, with a 512-row workspace (`max_rows`), on the ranks the
# engine starts under mpirun.
_ML4_DEP4 = dict(world_size=4, hidden=6144, topk=4, max_rows=512)
_ML4_DEP8 = dict(world_size=8, nodes=2, hidden=6144, topk=4, max_rows=512)
# Rows per rank a decode graph holds: one row per request, or DSpark's four (the target's
# three drafts and the verified token) per request at batches 1..32.
_GRAPH_ROWS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
_MOE_LAYERS = 71


class LamportMoeDispatch(OpWrapper):
    """`codes`, `scales`, `ids` and `weights` of every rank, concatenated in rank order.

    Each rank passes its own `n = counts[rank]` rows: NVFP4 `codes` uint8 `[n, hidden / 2]`,
    their linear-layout e4m3 `scales` (uint8 or float8_e4m3fn) `[n * hidden / 16]`, the
    routing's `ids` int32 `[n, topk]` and `weights` bf16 `[n, topk]`. Returns the four
    gathered over `sum(counts)` rows, the same dtypes, rank `p`'s rows at
    `sum(counts[:p])`, freshly allocated. A 16-bit word 0x8000 comes back as 0x0000 in every
    gathered row, the rank's own included: an e2m1 -0 code in the high nibble of a byte whose
    other nibbles are +0 becomes +0, which no product or sum tells apart; no scale, id or
    weight holds the pattern. Every rank calls with the same `counts` and `workspace` geometry.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{g['world_size']}-rank decode graph at {rows} row(s) per rank: "
                f"{_MOE_LAYERS} dispatches captured and replayed on new rows",
                spec=dict(g, counts=(rows,) * g["world_size"], graph=True, sites=_MOE_LAYERS),
            )
            for g in (_ML4_DEP4, _ML4_DEP8)
            for rows in _GRAPH_ROWS
        ),
        Cell(
            why="dep4 eager step beside a prefilling peer: ragged counts, one rank at the workspace",
            spec=dict(_ML4_DEP4, counts=(512, 1, 16, 257)),
        ),
        Cell(
            why="dep4 eager step: every rank holds the engine's one-row dummy",
            spec=dict(_ML4_DEP4, counts=(1, 1, 1, 1)),
        ),
        Cell(
            why="dep4 eager steps: every ragged count pattern of 1..3 rows, each its own call",
            spec=dict(_ML4_DEP4, sweep_counts=(1, 3)),
        ),
        Cell(
            why="dep4 payload bits: every 16-bit word pattern in the codes, -0 codes in place",
            spec=dict(_ML4_DEP4, counts=(64, 64, 64, 64), payload="bits"),
        ),
        Cell(
            why="dep8 eager step over two nodes: ragged counts, one rank at the workspace",
            spec=dict(_ML4_DEP8, counts=(512, 1, 16, 257, 1, 40, 300, 3)),
        ),
        Cell(
            why="dep8 payload bits over two nodes: every 16-bit word pattern in the codes",
            spec=dict(_ML4_DEP8, counts=(64,) * 8, payload="bits"),
        ),
    )

    note = """
    The transport is `_moe_lamport_kernel.dispatch_kernel` over a `MoeCommWorkspace` built by
    `build_moe_comm_workspace(hidden, max_rows, rank, world)`, imported from this module: each
    rank stores its rows into every peer's slot region over NVLink, and polls its own regions
    until every peer's rows have landed. One launch, no NCCL. The rows travel as 16-bit words
    (3480 B per row at hidden 6144 and top-4, padded to 3488).

    Preconditions:
    - Collective, and the slot counter pairs calls by position: every rank of the workspace
      makes the same sequence of `lamport_moe_dispatch` and `lamport_moe_combine` calls with
      the same `counts`. A rank that skips one hangs the group; nothing raises.
    - `max(counts) <= workspace.max_rows`: a larger row count stores past every peer's region.
      The caller picks NCCL above it, from `counts`, which every rank shares.
    - At most as many programs as the device has SMs (`workspace.programs`), so each one is
      resident while it polls: a program spinning on a row would otherwise wait on a program
      of its own rank that never got an SM to send it.

    Measured on four sm_107 ranks of one VR200 node, every rank at the same rows, 71 calls
    back to back in one CUDA graph, the slowest rank's median per call: 4.7 us at 1 row per
    rank, 5.0 at 16, 5.7 at 64, 6.9 at 128 and 13.2 at 512. NCCL's bf16 all-gather of the
    `[rows, 6144]` activations the gather replaced takes 9.5, 11.1, 28.1, 49.5 and 76.0 us
    there. The world-8 cells were driven on two VR200 nodes of one NVL72 domain (the eight
    GPUs one MNNVL clique; IMEX channels in the container), bit for bit on all eight ranks; at
    world 8 the call takes 6.9 us at 1 row per rank, 7.2 at 16, 8.9 at 64, 11.7 at 128 and
    28.7 at 512, against 18.4, 21.0, 55.7, 102.4 and 160.3 us for the bf16 all-gather.
    """

    def __call__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        counts: Sequence[int],
        workspace: MoeCommWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = codes.shape[0]
        total, widest = sum(counts), max(counts)
        widths = (codes.shape[1] // 2, scales.numel() // n // 2, ids.shape[1] * 2, ids.shape[1])
        row = triton.cdiv(sum(widths), _ROW_ALIGN) * _ROW_ALIGN
        out_codes = codes.new_empty((total, codes.shape[1]))
        out_scales = scales.new_empty((total * widths[1] * 2,))
        out_ids = ids.new_empty((total, ids.shape[1]))
        out_weights = weights.new_empty((total, weights.shape[1]))
        padded = [int(c) for c in counts] + [0] * (_MAX_WORLD - len(counts))
        _k.dispatch_kernel[(min(workspace.world * widest, workspace.programs),)](
            codes.view(torch.int16), scales.view(torch.int16), ids.view(torch.int16),
            weights.view(torch.int16), out_codes.view(torch.int16), out_scales.view(torch.int16),
            out_ids.view(torch.int16), out_weights.view(torch.int16),
            workspace.buffer.view(torch.int16), workspace.peers, workspace.state, workspace.rank,
            *padded, workspace.max_rows * workspace.hidden,
            CODES=widths[0], SCALES=widths[1], IDS=widths[2], WEIGHTS=widths[3], ROW=row,
            WORLD=workspace.world, BLOCK=triton.next_power_of_2(row), num_warps=4,
        )  # fmt: skip
        return out_codes, out_scales, out_ids, out_weights

    def reference(
        self,
        per_rank: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        counts: Sequence[int],
        workspace: MoeCommWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The concatenations, every 16-bit word 0x8000 replaced by 0x0000.

        `per_rank` is every rank's `(codes, scales, ids, weights)` in rank order, the collective
        exception to mirroring `__call__`. `workspace` is ignored."""
        out = []
        for parts in zip(*per_rank):
            cat = torch.cat(list(parts), dim=0).clone()
            words = cat.view(torch.int16)
            words.masked_fill_(words == _k.SENTINEL.value, 0)
            out.append(cat)
        return tuple(out)

    def is_valid(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        counts: Sequence[int],
        workspace: MoeCommWorkspace,
    ) -> None:
        n = codes.shape[0]
        assert len(counts) == workspace.world <= _MAX_WORLD, (len(counts), workspace.world)
        assert counts[workspace.rank] == n == ids.shape[0] == weights.shape[0], (
            f"counts[{workspace.rank}]={counts[workspace.rank]} must be this rank's rows ({n}); "
            "the kernel sends that many rows whatever the inputs hold"
        )
        assert scales.numel() == n * codes.shape[1] // 8, (
            "scales must be the linear layout, one per 16 inputs, n * hidden / 16 bytes"
        )
        assert max(counts) <= workspace.max_rows, (
            f"{max(counts)} rows exceed the workspace's {workspace.max_rows}: the stores would "
            "run past every peer's slot region"
        )
        assert ids.dtype == torch.int32 and weights.dtype == torch.bfloat16, (
            ids.dtype,
            weights.dtype,
        )
        for t in (codes, scales, ids, weights):
            assert t.is_contiguous(), "the kernel reads every input as packed rows"


lamport_moe_dispatch = LamportMoeDispatch()
