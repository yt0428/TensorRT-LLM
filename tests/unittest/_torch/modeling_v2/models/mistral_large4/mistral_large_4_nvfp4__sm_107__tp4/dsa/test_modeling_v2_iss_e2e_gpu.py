# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 GPU lifecycle and dynamic threshold tests for ML4 ISS-8/16 step preparation."""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.modeling import (  # noqa: E501
    MistralLarge4Tp4ModelingV2,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.dsa._hist_topk import (  # noqa: E501
    MAX_SHARD,
    shard_capacity,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.dsa.hist_shard_meta import (  # noqa: E501
    hist_shard_meta,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="requires VR200/SM107",
)


def _core(tp_size: int, tp_rank: int, max_ctx_len: int) -> MistralLarge4Tp4ModelingV2:
    core = MistralLarge4Tp4ModelingV2.__new__(MistralLarge4Tp4ModelingV2)
    core.tp_size = tp_size
    core.tp_rank = tp_rank
    core._max_ctx_len = max_ctx_len
    core._hist_workspace = object()
    core._hist_buffers = None
    return core


def _step(kv_len: int, max_seq_len: int, tokens_per_block: int = 32, num_seqs: int = 1) -> dict:
    pages = -(-max_seq_len // tokens_per_block)
    return dict(
        num_seqs=num_seqs,
        tokens_per_block=tokens_per_block,
        kv_lens_host=torch.tensor([kv_len] * num_seqs, dtype=torch.int32),
        kv_lens_cuda=torch.tensor([kv_len] * num_seqs, dtype=torch.int32, device="cuda"),
        indexer_k_cache_block_offsets=(torch.arange(pages, dtype=torch.int32, device="cuda") + 1000)
        .view(1, -1)
        .expand(num_seqs, -1)
        .contiguous(),
    )


def test_dynamic_threshold_switches_between_local_and_iss_on_gpu(monkeypatch) -> None:
    """Eager decode dynamically switches at L > L_thresh and preserves buffer pointers."""
    monkeypatch.setenv("TRTLLM_ML4_ISS_MIN_CTX_LEN", "131072")
    core = _core(tp_size=16, tp_rank=15, max_ctx_len=1_048_584)

    # Below threshold -> None (local top-k).
    assert core._prepare_iss_step(_step(65_536, 1_048_584), rows=4, skip_gen=False) is None
    assert core._prepare_iss_step(_step(131_072, 1_048_584), rows=4, skip_gen=False) is None

    # Above threshold -> ISS active with exact shard_ctx and shard_block_table.
    step = _step(530_000, 1_048_584, tokens_per_block=32)
    buffers = core._prepare_iss_step(step, rows=4, skip_gen=False)
    assert buffers is not None
    expected_ctx, expected_table = hist_shard_meta.reference(
        step["kv_lens_cuda"][:1],
        step["indexer_k_cache_block_offsets"][0],
        buffers["shard_ctx"],
        buffers["shard_block_table"],
        16,
        15,
        32,
    )
    torch.cuda.synchronize()
    assert torch.equal(buffers["shard_ctx"], expected_ctx)
    assert torch.equal(buffers["shard_block_table"], expected_table)

    # CUDA graph capture replays on updated kv_lens with identical buffer addresses.
    ptrs = {k: v.data_ptr() for k, v in buffers.items() if torch.is_tensor(v)}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = core._prepare_iss_step(step, rows=4, skip_gen=False)
        assert captured is buffers
    for kv_len in (2048, 262_145, 1_048_584):
        step["kv_lens_cuda"].fill_(kv_len)
        graph.replay()
        expected_ctx, expected_table = hist_shard_meta.reference(
            step["kv_lens_cuda"][:1],
            step["indexer_k_cache_block_offsets"][0],
            buffers["shard_ctx"],
            buffers["shard_block_table"],
            16,
            15,
            32,
        )
        torch.cuda.synchronize()
        assert all(buffers[k].data_ptr() == ptr for k, ptr in ptrs.items())
        assert torch.equal(buffers["shard_ctx"], expected_ctx)
        assert torch.equal(buffers["shard_block_table"], expected_table)


def test_iss8_respects_max_shard_bound_on_gpu(monkeypatch) -> None:
    """ISS-8 engages up to 786,432 keys and safely declines past 8 * MAX_SHARD."""
    monkeypatch.setenv("TRTLLM_ML4_ISS_MIN_CTX_LEN", "262144")
    core = _core(tp_size=8, tp_rank=7, max_ctx_len=1_048_584)
    assert core._prepare_iss_step(_step(780_000, 1_048_584), rows=4, skip_gen=False) is not None
    assert core._hist_buffers["capacity"] == MAX_SHARD
    assert (
        core._prepare_iss_step(_step(8 * MAX_SHARD + 1, 1_048_584), rows=4, skip_gen=False) is None
    )
    assert shard_capacity(786_432, 8) == MAX_SHARD
