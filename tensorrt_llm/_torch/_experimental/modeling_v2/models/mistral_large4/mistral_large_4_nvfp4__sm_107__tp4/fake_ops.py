# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake implementations for the Mistral Large 4 SM107 megakernel C++ Torch operators."""

from __future__ import annotations

import torch

_DEFAULT_NS = "ml4_megakernel"


def _resolve_namespace() -> str:
    try:
        from tensorrt_llm._torch.modeling_v2._native import namespace_of

        return namespace_of(__name__)
    except (ImportError, ValueError):
        return _DEFAULT_NS


_NS = _resolve_namespace()


def _host_query(qualname: str):
    """Setup queries have no tensors and must run outside tracing."""

    def fake(*args, **kwargs):
        raise NotImplementedError(f"{qualname} is a setup-time query; call it outside tracing")

    return torch.library.register_fake(qualname)(fake)


def register_fakes(namespace: str = _NS) -> None:
    """Register Torch fake kernels for the ML4 megakernel native ops."""
    _host_query(f"{namespace}::ml4_megakernel_info")

    @torch.library.register_fake(f"{namespace}::ml4_megakernel_indexer_tmap")
    def _ml4_megakernel_indexer_tmap(pool: torch.Tensor, rows: int, tokens_per_block: int):
        return torch.empty((128,), dtype=torch.uint8, device="cpu")

    @torch.library.register_fake(f"{namespace}::ml4_megakernel_weight_table")
    def _ml4_megakernel_weight_table(params: torch.Tensor, offset: int):
        return None

    @torch.library.register_fake(f"{namespace}::ml4_megakernel_sm_die_probe")
    def _ml4_megakernel_sm_die_probe(die0_buf: torch.Tensor, die1_buf: torch.Tensor):
        return torch.empty((212, 3), dtype=torch.int64, device=die0_buf.device)

    @torch.library.register_fake(f"{namespace}::ml4_megakernel")
    def _ml4_megakernel(
        params,
        x,
        residual,
        topk,
        block_table,
        kv_lens,
        positions,
        state,
        kv_cache,
        pdl,
        is_leader=False,
    ):
        return torch.empty_like(x), torch.empty_like(residual)


_REGISTERED = False


def ensure_fakes_registered(namespace: str = _NS) -> None:
    """Idempotently register fake ops once the native library has defined the schemas."""
    global _REGISTERED
    if _REGISTERED:
        return
    register_fakes(namespace)
    _REGISTERED = True
