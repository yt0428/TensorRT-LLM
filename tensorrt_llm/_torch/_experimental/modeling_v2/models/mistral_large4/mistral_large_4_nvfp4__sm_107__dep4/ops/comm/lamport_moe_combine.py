# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduce-scatter the attention-DP expert partials in one-shot Lamport stores, adding the
shared expert and, on request, the residual and the next layer's RMSNorm."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import triton

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _moe_lamport_kernel as _k
from ._moe_workspace import MoeCommWorkspace

#: The most ranks the kernel's row counts address.
_MAX_WORLD = 8

# Mistral Large 4 NVFP4 at dep4 and dep8: the combine of every MoE layer (1..71). Each rank
# passes the runner's finalized `[sum(counts), 6144]` bf16 partial of its expert window over
# every rank's rows, its own rows' shared expert output, and on a decode step its residual and
# the next layer's input norm (the final norm after layer 71), rms eps 1e-5; a 512-row workspace.
_ML4_DEP4 = dict(world_size=4, hidden=6144, eps=1e-5, max_rows=512)
_ML4_DEP8 = dict(world_size=8, nodes=2, hidden=6144, eps=1e-5, max_rows=512)
_GRAPH_ROWS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
_MOE_LAYERS = 71


class LamportMoeCombine(OpWrapper):
    """This rank's rows of the sum over ranks of `partial`, plus `shared`; with `residual`,
    also the residual add and the RMSNorm the next layer reads.

    `partial` is bf16 `[sum(counts), hidden]`, this rank's expert window's output over every
    rank's rows in rank order; `shared` bf16 `[counts[rank], hidden]`. The sum over ranks is
    fp32 in rank order, rounded once to bf16 (`routed`), and `mlp = bf16(routed + shared)`, the
    bits `torch.add` gives. Without `residual` (and `norm_weight`) returns `mlp`. With them,
    `residual` (bf16 `[counts[rank], hidden]`) becomes `bf16(residual + mlp)` in place and the
    return is `rmsnorm(residual) * norm_weight`, fp32 arithmetic rounded once.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{g['world_size']}-rank decode graph at {rows} row(s) per rank, the fused "
                f"residual and norm: {_MOE_LAYERS} combines captured and replayed on new rows",
                spec=dict(
                    g, counts=(rows,) * g["world_size"], norm=True, graph=True, sites=_MOE_LAYERS
                ),
            )
            for g in (_ML4_DEP4, _ML4_DEP8)
            for rows in _GRAPH_ROWS
        ),
        Cell(
            why="dep4 eager step beside a prefilling peer, the prefill target's form (no norm)",
            spec=dict(_ML4_DEP4, counts=(512, 1, 16, 257), norm=False),
        ),
        Cell(
            why="dep4 eager step beside a prefilling peer, the decode target's form (norm)",
            spec=dict(_ML4_DEP4, counts=(512, 1, 16, 257), norm=True),
        ),
        Cell(
            why="dep4 eager steps: every ragged count pattern of 1..3 rows, each its own call",
            spec=dict(_ML4_DEP4, sweep_counts=(1, 3), norm=True),
        ),
        Cell(
            why="dep4 partials of every bf16 bit pattern, -0.0 and both infinities included",
            spec=dict(_ML4_DEP4, counts=(64, 64, 64, 64), norm=False, payload="bits"),
        ),
        Cell(
            why="dep8 eager step over two nodes: ragged counts, one rank at the workspace",
            spec=dict(_ML4_DEP8, counts=(512, 1, 16, 257, 1, 40, 300, 3), norm=True),
        ),
    )

    note = """
    The transport is `_moe_lamport_kernel.combine_kernel` over the `MoeCommWorkspace` that
    `lamport_moe_dispatch` uses (built by `lamport_moe_dispatch.build_moe_comm_workspace`):
    each rank stores every peer's rows of its partial into that peer's slot region over
    NVLink, then reduces its own rows from its regions. One launch, no NCCL; each rank sends
    and receives `(world - 1) * rows * hidden` bf16, what a ring reduce-scatter moves.

    Arithmetic against NCCL's reduce-scatter: NCCL sums bf16 partials in bf16 along its ring,
    rounding after every hop; this sums in fp32 in rank order and rounds once. The two differ
    by rounding wherever a hop rounds, and this one is the nearer to the exact sum. The norm
    is fp32 with its own reduction order, so it is not flashinfer_rmsnorm's bits.

    Preconditions are `lamport_moe_dispatch`'s: every rank makes the same sequence of calls
    with the same `counts`, `max(counts) <= workspace.max_rows`, at most `workspace.programs`
    programs. `partial`, `shared`, `residual` contiguous; `residual` may alias nothing else
    the call reads.

    Measured as `lamport_moe_dispatch` is, with the residual and the norm: 7.4 us per call at
    1 row per rank, 8.0 at 16, 9.8 at 64, 12.2 at 128 and 33.0 at 512, against 9.3, 10.4,
    24.2, 42.1 and 71.5 us for NCCL's bf16 reduce-scatter of the same partial alone, before
    the adds and the norm. The world-8 cells were driven as `lamport_moe_dispatch`'s were; at
    world 8 the call takes 13.1 us at 1 row per rank, 15.1 at 16, 19.7 at 64, 24.9 at 128 and
    82.2 at 512, against 17.9, 19.5, 49.3, 92.1 and 154.9 us for the reduce-scatter alone.
    """

    def __call__(
        self,
        partial: torch.Tensor,
        shared: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        eps: float,
        counts: Sequence[int],
        workspace: MoeCommWorkspace,
    ) -> torch.Tensor:
        hidden = partial.shape[1]
        norm = residual is not None
        out = torch.empty_like(shared)
        padded = [int(c) for c in counts] + [0] * (_MAX_WORLD - len(counts))
        _k.combine_kernel[(min(workspace.world * max(counts), workspace.programs),)](
            partial, shared, residual if norm else shared, residual if norm else out,
            norm_weight if norm else shared, out, workspace.buffer.view(torch.int16),
            workspace.peers, workspace.state, workspace.rank, *padded,
            workspace.max_rows * workspace.hidden, eps, H=hidden, WORLD=workspace.world,
            BLOCK=triton.next_power_of_2(hidden), NORM=norm, num_warps=8,
        )  # fmt: skip
        return out

    def reference(
        self,
        partials: Sequence[torch.Tensor],
        shared: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        eps: float,
        counts: Sequence[int],
        workspace: MoeCommWorkspace,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """`(out, new_residual)`: `new_residual` None without `residual`, which is not modified.

        `partials` is every rank's partial in rank order (the collective exception to
        mirroring `__call__`). The sum is fp32 in rank order over this rank's rows, every
        -0.0 a peer sends taken as +0.0; the norm is fp64 and rounds once."""
        lo = sum(counts[: workspace.rank])
        hi = lo + counts[workspace.rank]
        acc = torch.zeros(shared.shape, dtype=torch.float32, device=shared.device)
        for p, part in enumerate(partials):
            block = part[lo:hi].float()
            if p != workspace.rank:
                block = torch.where(block == 0, torch.zeros_like(block), block)
            acc += block
        mlp = (acc.to(torch.bfloat16).float() + shared.float()).to(torch.bfloat16)
        if residual is None:
            return mlp, None
        new_residual = (residual.float() + mlp.float()).to(torch.bfloat16)
        x = new_residual.double()
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return (x * inv * norm_weight.double()).to(torch.bfloat16), new_residual

    def is_valid(
        self,
        partial: torch.Tensor,
        shared: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        eps: float,
        counts: Sequence[int],
        workspace: MoeCommWorkspace,
    ) -> None:
        assert len(counts) == workspace.world <= _MAX_WORLD, (len(counts), workspace.world)
        assert partial.shape == (sum(counts), workspace.hidden), (
            f"partial {tuple(partial.shape)} must be every rank's rows, "
            f"({sum(counts)}, {workspace.hidden})"
        )
        assert shared.shape == (counts[workspace.rank], workspace.hidden), (
            f"shared {tuple(shared.shape)} must be this rank's rows"
        )
        assert max(counts) <= workspace.max_rows, (
            f"{max(counts)} rows exceed the workspace's {workspace.max_rows}"
        )
        assert (residual is None) == (norm_weight is None), "residual and norm_weight go together"
        if residual is not None:
            assert residual.shape == shared.shape and residual.is_contiguous()
        for t in (partial, shared):
            assert t.dtype == torch.bfloat16 and t.is_contiguous()


lamport_moe_combine = LamportMoeCombine()
