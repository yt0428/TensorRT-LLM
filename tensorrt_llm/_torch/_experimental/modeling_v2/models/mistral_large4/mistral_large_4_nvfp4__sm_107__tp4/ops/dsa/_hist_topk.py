# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shard geometry, dynamic length threshold and MNNVL workspace for key-sharded DSA top-k (ISS-8/16).

The decode indexer is replicated over the tensor-parallel group: without sharding every rank
computes the same FP8 MQA logits over the whole context and the same top-2048. For long contexts
(``kv_len > L_thresh``), rank ``r`` scores only its contiguous shard of the request's indexer keys
(``glm_dsa_hist_shard_meta`` builds the shard's context length and block table), and
``glm_dsa_hist_topk`` finds every row's exact global top-2048 over all shards by exchanging
3-level radix histograms through a private MNNVL multicast workspace; every rank ends with
bit-identical rows. See ``cpp/tensorrt_llm/kernels/glmMoeDsa/glmDsaHistTopK.h``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import torch

    from tensorrt_llm.mapping import Mapping

# Constants of cpp/tensorrt_llm/kernels/glmMoeDsa/glmDsaHistTopK.h.
MAX_RANKS = 16
SHARD_ALIGN = 128
MAX_SHARD = 98304
MAX_ROWS = 8
TOPK = 2048
WORKSPACE_WORDS = 3 * (3 * 8 * 16 * 1088 + 8 * 2048 + 8 * 4096 * 2)
STATE_WORDS = 4 + 8 * 4

#: TP world sizes that enable ISS key-sharded DSA indexing by default on SM107.
ISS_WORLD_SIZES: tuple[int, ...] = (8, 16)
#: Default context-length threshold L_thresh above which ISS replaces local top-k.
_DEFAULT_ISS_MIN_CTX_LEN_BY_TP: dict[int, int] = {
    8: 262144,
    16: 131072,
}
_DEFAULT_ISS_MIN_CTX_LEN = 131072
#: DeepGEMM paged MQA logits schedule block granularity.
DG_SCHEDULE_BLOCK_KV = 64


def shard_size(kv_len: int, group_size: int) -> int:
    """Keys per rank of a request of ``kv_len`` indexer keys: ceil(kv_len / group_size), rounded
    up to SHARD_ALIGN. Rank ``r`` holds the keys [r * S, r * S + n_r), n_r = clamp(kv_len - r * S,
    0, S)."""
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    per_rank = -(-max(kv_len, 0) // group_size)
    return -(-per_rank // SHARD_ALIGN) * SHARD_ALIGN


def shard_capacity(max_seq_len: int, group_size: int) -> int:
    """Largest shard of a request of at most ``max_seq_len`` indexer keys."""
    return shard_size(max_seq_len, group_size)


def iss_enabled(env: Optional[AbcMapping[str, str]] = None) -> bool:
    """Whether ISS-8/16 key-sharded DSA indexing is enabled (TRTLLM_ML4_ENABLE_ISS != '0')."""
    source = os.environ if env is None else env
    return source.get("TRTLLM_ML4_ENABLE_ISS", "1") != "0"


def default_iss_min_ctx_len(tp_size: int, env: Optional[AbcMapping[str, str]] = None) -> int:
    """Dynamic context-length threshold L_thresh for switching from local top-k to ISS.

    Reads ``TRTLLM_ML4_ISS_MIN_CTX_LEN`` when set (0 activates ISS for any eligible length above
    the skip threshold; negative disables ISS). Defaults to 131072 at TP16 and 262144 at TP8.
    """
    source = os.environ if env is None else env
    raw = source.get("TRTLLM_ML4_ISS_MIN_CTX_LEN")
    if raw is not None and raw != "":
        return int(raw)
    return _DEFAULT_ISS_MIN_CTX_LEN_BY_TP.get(tp_size, _DEFAULT_ISS_MIN_CTX_LEN)


def should_use_iss(
    kv_len: int,
    max_ctx_len: int,
    num_seqs: int,
    rows: int,
    tp_size: int,
    tokens_per_block: int,
    l_thresh: int,
    *,
    capturing: bool,
    has_workspace: bool,
    skip_gen: bool = False,
    enabled: bool = True,
    supported_world_sizes: tuple[int, ...] = ISS_WORLD_SIZES,
) -> bool:
    """Decide whether this decode step takes the ISS key-sharded indexer path.

    In eager mode, switches dynamically on ``kv_len > l_thresh`` while keeping the shard within
    ``MAX_SHARD`` (98,304 keys/rank: up to 786,432 at TP8 and 1,572,864 at TP16). During CUDA
    graph capture, the captured graph must remain valid for every replay length up to
    ``max_ctx_len``, so ISS is captured only when ``max_ctx_len > l_thresh`` and
    ``shard_capacity(max_ctx_len, tp_size) <= MAX_SHARD``.
    """
    if (
        not enabled
        or not has_workspace
        or skip_gen
        or l_thresh < 0
        or tp_size not in supported_world_sizes
        or not (2 <= tp_size <= MAX_RANKS)
        or num_seqs != 1
        or not (1 <= rows <= MAX_ROWS)
        or tokens_per_block <= 0
        or SHARD_ALIGN % tokens_per_block != 0
    ):
        return False
    if capturing:
        return max_ctx_len > l_thresh and shard_capacity(max_ctx_len, tp_size) <= MAX_SHARD
    return kv_len > l_thresh and shard_size(kv_len, tp_size) <= MAX_SHARD


@dataclass
class HistTopkWorkspace:
    """The exchange workspace and device state of ``glm_dsa_hist_topk`` for one TP group.

    ``comm_buffer`` is this rank's unicast view of ``WORKSPACE_WORDS`` int32 words of a
    McastGPUBuffer, all -1 at creation; ``state`` is ``STATE_WORDS`` int32 words, zero at creation.
    Both are private to the op and live as long as the engine: captured CUDA graphs launch on
    them. ``handle`` keeps the multicast allocation alive.
    """

    group_size: int
    handle: Any
    comm_buffer: "torch.Tensor"
    state: "torch.Tensor"
    comm: Any


def hist_topk_supported(mapping: "Mapping") -> bool:
    """Whether the key-sharded top-k can serve this mapping: a pure tensor-parallel group of 2 to
    16 ranks without attention DP or context parallelism, on an MNNVL fabric."""
    import torch

    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    return (
        2 <= mapping.tp_size <= MAX_RANKS
        and mapping.world_size == mapping.tp_size
        and not mapping.has_cp()
        and not mapping.enable_attention_dp
        and MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16)
    )


def create_hist_topk_workspace(
    mapping: "Mapping", comm: Optional[Any] = None
) -> HistTopkWorkspace:
    """Allocate the workspace on every rank of ``mapping``'s TP group (a collective call).

    ``comm`` is the TP-group communicator the multicast handles are exchanged over: the MNNVL
    all-reduce's (an MPI split, or the TP ProcessGroup without MPI) when None. The outcome is
    converged over the group, so every rank raises or none does.
    """
    import torch

    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_device_index,
        _mnnvl_workspace_all_succeeded,
    )

    if comm is None:
        comm = _get_mnnvl_workspace_comm(mapping)
    error: Optional[Exception] = None
    workspace = None
    try:
        handle = _make_mnnvl_mcast_buffer(
            comm, WORKSPACE_WORDS * 4, mapping, use_fabric_handle=mapping.is_multi_node()
        )
        comm_buffer = handle.get_uc_buffer(mapping.tp_rank, (WORKSPACE_WORDS,), torch.int32, 0)
        with torch.inference_mode():
            comm_buffer.fill_(-1)
            state = torch.zeros(
                STATE_WORDS,
                dtype=torch.int32,
                device=torch.device("cuda", _mnnvl_device_index(mapping)),
            )
        torch.cuda.synchronize()
        workspace = HistTopkWorkspace(mapping.tp_size, handle, comm_buffer, state, comm)
    except Exception as e:
        error = e
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError(
            "DSA histogram top-k workspace construction failed on at least one rank"
        ) from error
    return workspace


_WORKSPACES: Dict["Mapping", HistTopkWorkspace] = {}


def get_hist_topk_workspace(mapping: "Mapping") -> HistTopkWorkspace:
    """The workspace of ``mapping``'s TP group, created on the first call (collective: every rank
    of the group makes its first call in the same place, outside CUDA graph capture)."""
    import torch

    workspace = _WORKSPACES.get(mapping)
    if workspace is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the DSA histogram top-k workspace is created outside capture")
        workspace = _WORKSPACES[mapping] = create_hist_topk_workspace(mapping)
    return workspace
