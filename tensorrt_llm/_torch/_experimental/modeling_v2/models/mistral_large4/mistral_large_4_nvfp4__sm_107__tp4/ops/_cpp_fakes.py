# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake implementations for C++ Torch operators used by the Mistral Large 4 SM107 TP target."""

from __future__ import annotations

import torch

if hasattr(torch.ops.trtllm, "glm_dsa_hist_topk"):

    @torch.library.register_fake("trtllm::glm_dsa_hist_topk")
    def _glm_dsa_hist_topk(
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        out: torch.Tensor,
        comm_buffer: torch.Tensor,
        state: torch.Tensor,
        next_n: int,
        group_size: int,
    ) -> None:
        return None


if hasattr(torch.ops.trtllm, "glm_dsa_hist_shard_meta"):

    @torch.library.register_fake("trtllm::glm_dsa_hist_shard_meta")
    def _glm_dsa_hist_shard_meta(
        kv_lens: torch.Tensor,
        block_table: torch.Tensor,
        shard_ctx: torch.Tensor,
        shard_block_table: torch.Tensor,
        group_size: int,
        rank: int,
        tokens_per_block: int,
    ) -> None:
        return None
