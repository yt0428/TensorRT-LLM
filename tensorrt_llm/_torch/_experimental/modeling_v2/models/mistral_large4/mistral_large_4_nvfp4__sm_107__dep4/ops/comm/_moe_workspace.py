# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The slots the attention-DP expert path's one-shot gather and reduce-scatter are handed, and
the alltoall its larger steps take.

Not an entry: it holds what `lamport_moe_dispatch` and `lamport_moe_combine` are handed, built
once at load by `build_moe_comm_workspace`, and builds the alltoall `a2a_moe_dispatch` and
`a2a_moe_combine` are handed (`build_moe_alltoall`). The kernels (`_moe_lamport_kernel`) fix the
slot count and the not-ready sentinel; this module allocates to them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm._lamport import ensure_process_group
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm.mapping import Mapping

from . import _moe_lamport_kernel as _k

_SLOTS = int(_k.SLOTS.value)
#: Bitwise bf16 -0.0, the in-band "not arrived" word, as an int16 fill value.
_SENTINEL = int(_k.SENTINEL.value)


@dataclass(frozen=True)
class MoeCommWorkspace:
    """The slots the attention-DP expert path's one-shot gather and reduce-scatter move rows
    through (`lamport_moe_dispatch`, `lamport_moe_combine`; kernels in `_moe_lamport_kernel`).

    `buffer` is this rank's `[3, world, max_rows * hidden]` bf16 symmetric buffer, region `p`
    of a slot holding what rank `p` sent this rank, every word -0.0 until a row lands on it
    and again once its receiver has read it. `peers` holds every rank's buffer address, mapped
    into this rank, in rank order. `state` is the `[2]` int32 device slot counter and its
    arrival count: every launch of either kernel reads and advances it, so the tensors are
    shared state although the dataclass is frozen. One workspace per rank per process.
    """

    hidden: int
    max_rows: int
    rank: int
    world: int
    #: The most programs one launch takes: the device's SMs, so every program is resident.
    programs: int
    buffer: torch.Tensor
    handle: object
    peers: torch.Tensor
    state: torch.Tensor


def build_moe_comm_workspace(hidden: int, max_rows: int, rank: int, world: int) -> MoeCommWorkspace:
    """Bring up the group, pin and rendezvous the slots, fill them not-ready.

    Collective: every rank of the group calls it with the same `hidden` and `max_rows`, outside
    any graph capture. Allocates `3 * world * max_rows * hidden` bf16 of symmetric memory per
    rank and compiles nothing (the kernels are Triton, compiled on their first launch). Returns
    once every rank's slots hold the sentinel, so no rank's first row lands on unfilled memory.

    A group this call brought up (`ensure_process_group`) is torn down before it returns, and
    the rendezvous address it set is taken back out of the environment: nothing after the
    rendezvous uses the group (the kernels reach the peers through `peers` and synchronize
    in-band), and left up, its NCCL watchdog polls rank 0's store until the process exits, so
    a rank on another node that outlives rank 0 at shutdown logs the lost store. A group that
    was already up is left as it is.
    """
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    address = {key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT")}
    created = ensure_process_group(rank, world)
    dev = torch.device("cuda", torch.cuda.current_device())
    buffer = symm_mem.empty((_SLOTS, world, max_rows * hidden), dtype=torch.bfloat16, device=dev)
    buffer.view(torch.int16).fill_(_SENTINEL)
    group = dist.group.WORLD
    handle = symm_mem.rendezvous(buffer, group=getattr(group, "group_name", group))
    peers = torch.tensor(
        [handle.get_buffer(peer, buffer.shape, buffer.dtype).data_ptr() for peer in range(world)],
        dtype=torch.int64,
        device=dev,
    )
    state = torch.zeros(2, dtype=torch.int32, device=dev)
    torch.cuda.synchronize(dev)
    dist.barrier(group=group, device_ids=[dev.index])
    if created:
        dist.destroy_process_group()
        for key, value in address.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return MoeCommWorkspace(
        hidden=hidden,
        max_rows=max_rows,
        rank=rank,
        world=world,
        programs=torch.cuda.get_device_properties(dev).multi_processor_count,
        buffer=buffer,
        handle=handle,
        peers=peers,
        state=state,
    )


def build_moe_alltoall(
    mapping: Mapping, num_experts: int, top_k: int, max_rows: int, hidden: int
) -> NVLinkOneSided:
    """The expert path's alltoall over `mapping`'s expert-parallel ranks, for steps of up to
    `max_rows` rows per rank: `NVLinkOneSided`, whose dispatch sends each row only to the ranks
    holding one of its experts (`num_experts` split evenly, `top_k` per row) and whose combine
    sums each row's returned bf16 `hidden`-wide partials on its own rank. The combine reads its
    payload in place in the workspace (`payload_in_workspace`): the expert runner writes there.

    Collective over the expert-parallel ranks, outside any graph capture. The workspace is MNNVL
    memory exchanged over the MPI session, so no torch.distributed group stays up; at 4 ranks
    and 16384 rows it is 2.4 GB per rank (the layout that leaves room for counted writes, which
    this device supports; they run only up to 128 rows per rank, below the one-shot workspace).
    """
    assert mapping.moe_ep_size == mapping.tp_size and mapping.moe_tp_size == 1, (
        f"the alltoall spans the attention-DP ranks as experts: {mapping}"
    )
    return NVLinkOneSided(
        mapping=mapping,
        num_slots=num_experts,
        top_k=top_k,
        max_num_tokens_per_rank=max_rows,
        hidden_size=hidden,
        dtype=torch.bfloat16,
        payload_in_workspace=True,
    )
