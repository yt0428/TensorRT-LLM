# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady-state one-model speculative decode input preparation.

A decode step whose batch is the previous step's (same requests and slots, each verifying the
previous step's device drafts, same CUDA graph metadata, no padding) needs only a few new integers
per request. The preparer records a full ``DecoderInputPreparer.prepare_inputs`` pass of such a
step (``record_batch``); later matching steps write those integers to a pinned host slab (3-slot
ring, fenced by events) and one Triton kernel stages every device input from it, including the
CUDA graph's static inputs. The attention metadata's ``prepare_steady_decode_step`` stands in for
its ``prepare()`` where it supports it.

Whether a full pass qualifies is decided when it is recorded
(``DecoderInputPreparer._record_steady_mtp``); a later step matches when it runs the recorded
requests on the recorded slots and metadata. Any other step (batch or slot change, context rows,
dummies, multimodal, LoRA, guided decoding, beams, attention DP, helix CP, warmup) takes the full
pass, which records again when it qualifies.

With ``stage_rng`` (``RuntimeHooks.pack_spec_rng_uploads``) and zero-copy slabs, the same kernel
also writes the step's Philox state for sampling (``SpecMetadata.request_seeds`` /
``request_offsets`` / ``seeds`` / ``offsets``) from a second pinned ring.
"""

from __future__ import annotations

import types
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned

from ..attention.backends.interface import AttentionMetadata
from ..attention.backends.trtllm import TrtllmAttentionMetadata
from ..speculative.spec_sampler_base import SampleStateTensorsSpec
from . import step_inputs
from .numa_pinned import numa_pinned_zeros

if TYPE_CHECKING:
    from ..speculative.interface import SpecMetadata
    from .llm_request import LlmRequest

__all__ = ["SteadyMtpPrepare", "SteadyMtpRecord", "stage_steady_inputs"]

# Depth of the host slab ring.
_RING = 3

# Integer args unspecialized, pointers not specialized on alignment, one BLOCK for every batch size:
# one variant per engine, compiled during warmup instead of at each new batch size.
_STAGE_INT_ARGS = [
    "nnt_tok_stride",
    "nnt_slot_stride",
    "nd_slot_stride",
    "mrope_axis_stride",
    "static_pos_axis_stride",
    "rng_span",
    "B",
]
_STAGE_PTR_ARGS = [
    "slab",
    "next_new_tokens",
    "next_drafts",
    "new_lens",
    "input_ids",
    "draft_tokens",
    "prev_batch_idx",
    "prev_pos_idx",
    "prev_pos_offsets",
    "prev_kv_offsets",
    "positions",
    "mrope_positions",
    "gather_ids",
    "kv_lens_out",
    "static_ids",
    "static_pos",
    "rng_slab",
    "req_seeds_out",
    "req_offsets_out",
    "seeds_out",
    "offsets_out",
]


def _stage_jit(**kwargs):
    try:
        return triton.jit(**kwargs)
    except TypeError:  # a Triton without do_not_specialize_on_alignment
        kwargs.pop("do_not_specialize_on_alignment", None)
        return triton.jit(**kwargs)


@_stage_jit(do_not_specialize=_STAGE_INT_ARGS, do_not_specialize_on_alignment=_STAGE_PTR_ARGS)
def _steady_stage_kernel(
    slab,  # int32 [slots (B) | positions (B * T) | kv lens (B, HAS_KV)]
    next_new_tokens,  # [>= T, num_slots, 1] previous step's next-step input tokens
    nnt_tok_stride,
    nnt_slot_stride,
    next_drafts,  # [num_slots, >= D]
    nd_slot_stride,
    new_lens,  # [num_slots] previous step's accepted-token counts
    input_ids,
    draft_tokens,
    prev_batch_idx,
    prev_pos_idx,
    prev_pos_offsets,
    prev_kv_offsets,
    positions,
    mrope_positions,  # [3, 1, max_tokens] or a dummy when HAS_MROPE is False
    mrope_axis_stride,
    gather_ids,
    kv_lens_out,  # [B] kv_lens_cuda (HAS_KV): slab[B + B*T + r]
    static_ids,  # graph static input ids (HAS_STATIC)
    static_pos,  # graph static positions, [3, 1, M] (mrope) or [1, M]
    static_pos_axis_stride,
    rng_slab,  # int64 [2 * B] (HAS_RNG): each request's Philox seed, then its offset-window base
    req_seeds_out,  # int64 [B] SpecMetadata.request_seeds (HAS_RNG)
    req_offsets_out,  # int64 [B] SpecMetadata.request_offsets (HAS_RNG)
    seeds_out,  # int64 [B * T] SpecMetadata.seeds, one per logits row (HAS_RNG)
    offsets_out,  # int64 [B * T] SpecMetadata.offsets (HAS_RNG)
    rng_span,  # Philox offsets between a request's consecutive rows (_RNG_SLOT_SPAN)
    B,
    T: tl.constexpr,
    D: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    HAS_KV: tl.constexpr,
    HAS_STATIC: tl.constexpr,
    HAS_RNG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    # token rows: request r = i // T, position j = i % T
    tok = i < B * T
    r = i // T
    j = i % T
    slot = tl.load(slab + r, mask=tok, other=0)
    pos = tl.load(slab + B + i, mask=tok, other=0)
    ids = tl.load(next_new_tokens + j * nnt_tok_stride + slot * nnt_slot_stride, mask=tok, other=0)
    tl.store(input_ids + i, ids, mask=tok)
    tl.store(positions + i, pos, mask=tok)
    if HAS_STATIC:
        tl.store(static_ids + i, ids, mask=tok)
        tl.store(static_pos + i, pos, mask=tok)
        if HAS_MROPE:
            tl.store(static_pos + static_pos_axis_stride + i, pos, mask=tok)
            tl.store(static_pos + 2 * static_pos_axis_stride + i, pos, mask=tok)
    if HAS_MROPE:
        tl.store(mrope_positions + i, pos, mask=tok)
        tl.store(mrope_positions + mrope_axis_stride + i, pos, mask=tok)
        tl.store(mrope_positions + 2 * mrope_axis_stride + i, pos, mask=tok)
    tl.store(gather_ids + i, i, mask=tok)
    tl.store(prev_pos_idx + i, slot, mask=tok)
    lens = tl.load(new_lens + slot, mask=tok, other=0)
    tl.store(prev_pos_offsets + i, lens, mask=tok)
    # request rows
    req = i < B
    slot_r = tl.load(slab + i, mask=req, other=0)
    tl.store(prev_batch_idx + i, slot_r, mask=req)
    lens_r = tl.load(new_lens + slot_r, mask=req, other=0)
    tl.store(prev_kv_offsets + i, lens_r - T, mask=req)
    if HAS_KV:
        tl.store(kv_lens_out + i, tl.load(slab + B + B * T + i, mask=req, other=0), mask=req)
    if HAS_RNG:
        tl.store(req_seeds_out + i, tl.load(rng_slab + i, mask=req, other=0), mask=req)
        tl.store(req_offsets_out + i, tl.load(rng_slab + B + i, mask=req, other=0), mask=req)
        row_seed = tl.load(rng_slab + r, mask=tok, other=0)
        row_off = tl.load(rng_slab + B + r, mask=tok, other=0)
        tl.store(seeds_out + i, row_seed, mask=tok)
        tl.store(offsets_out + i, row_off + j.to(tl.int64) * rng_span, mask=tok)
    # draft rows: request i // D, draft i % D
    drf = i < B * D
    slot_d = tl.load(slab + i // D, mask=drf, other=0)
    dval = tl.load(next_drafts + slot_d * nd_slot_stride + i % D, mask=drf, other=0)
    tl.store(draft_tokens + i, dval, mask=drf)


def stage_steady_inputs(
    slab: torch.Tensor,
    b: int,
    t: int,
    new_tensors_device: Any,
    *,
    input_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    prev_batch_idx: torch.Tensor,
    prev_pos_idx: torch.Tensor,
    prev_pos_offsets: torch.Tensor,
    prev_kv_offsets: torch.Tensor,
    positions: torch.Tensor,
    mrope_positions: torch.Tensor | None,
    gather_ids: torch.Tensor,
    kv_lens_cuda: torch.Tensor | None = None,
    static_input_ids: torch.Tensor | None = None,
    static_position_ids: torch.Tensor | None = None,
    block: int | None = None,
    rng_slab: torch.Tensor | None = None,
    rng_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    rng_span: int = 0,
) -> None:
    """One launch of the steady staging kernel."""
    d = t - 1
    n = b * t
    need = max(n, b * d, b, 1)
    if block is None or block < need:
        block = triton.next_power_of_2(need)
    new_tokens = new_tensors_device.new_tokens
    next_drafts = new_tensors_device.next_draft_tokens
    has_mrope = mrope_positions is not None
    has_rng = rng_slab is not None
    if has_rng:
        assert rng_buffers is not None
        req_seeds_out, req_offsets_out, seeds_out, offsets_out = rng_buffers
    else:
        # Untouched without HAS_RNG; one dtype for every launch keeps a single variant.
        rng_slab = req_seeds_out = req_offsets_out = seeds_out = offsets_out = gather_ids
    _steady_stage_kernel[(1,)](
        slab,
        new_tokens,
        new_tokens.stride(0),
        new_tokens.stride(1),
        next_drafts,
        next_drafts.stride(0),
        new_tensors_device.new_tokens_lens,
        input_ids,
        draft_tokens,
        prev_batch_idx,
        prev_pos_idx,
        prev_pos_offsets,
        prev_kv_offsets,
        positions,
        mrope_positions if has_mrope else positions,
        mrope_positions.stride(0) if has_mrope else 0,
        gather_ids,
        kv_lens_cuda if kv_lens_cuda is not None else gather_ids,
        static_input_ids if static_input_ids is not None else gather_ids,
        static_position_ids if static_position_ids is not None else gather_ids,
        static_position_ids.stride(0) if (static_position_ids is not None and has_mrope) else 0,
        rng_slab,
        req_seeds_out,
        req_offsets_out,
        seeds_out,
        offsets_out,
        rng_span,
        b,
        T=t,
        D=d,
        HAS_MROPE=has_mrope,
        HAS_KV=kv_lens_cuda is not None,
        HAS_STATIC=static_input_ids is not None,
        HAS_RNG=has_rng,
        BLOCK=block,
    )


@dataclass
class SteadyMtpRecord:
    """What the last qualifying full pass left, keyed by the batch it ran."""

    request_ids: list[int]
    slots: list[int]
    prompt_lengths: list[int]
    sequence_lengths: list[int]
    tokens_per_step: int  # T = runtime draft length + 1
    runtime_draft_len: int
    num_extra_kv_tokens: int
    use_full_generation_page_table: bool
    attn_metadata: AttentionMetadata
    spec_metadata: "SpecMetadata"
    seq_lens_kv: list[int]
    # Whether attn_metadata.prepare_steady_decode_step replaces its prepare().
    attn_steady: bool
    # Whether the pure-decode DSA steady fast path replaces attn_metadata.prepare().
    attn_dsa_steady: bool
    # Snapshot of raw block-offset rows [B, max_blocks_per_seq] and per-seq block counts for DSA.
    dsa_raw_blocks: np.ndarray | None
    dsa_num_blocks: list[int] | None
    dsa_max_blocks: int
    # Recorded DSpark rolling-window slot list matching request_ids (or None).
    dspark_slots: list[int] | None
    # The CUDA graph's static input ids / positions the staging kernel writes, or None.
    static_ids: torch.Tensor | None
    static_pos: torch.Tensor | None


class SteadyMtpPrepare:
    """Record, host slab ring and staging-kernel launch of steady one-model speculative decode."""

    rng_span: int = 0
    host_rng_slabs: tuple[torch.Tensor, ...] = ()
    host_rng_views: tuple[np.ndarray, ...] = ()
    DSA_BLOCK_PATCH_MAX_SPAN: int = 256

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        max_tokens_per_step: int,
        use_mrope: bool,
        has_static_inputs: bool,
        stage_rng: bool = False,
        steady_dsa: bool = False,
        defer_dsa_derived: bool = False,
    ) -> None:
        self.record: SteadyMtpRecord | None = None
        self.use_mrope = use_mrope
        self.steady_dsa = bool(steady_dsa)
        self.defer_dsa_derived = bool(defer_dsa_derived)
        max_requests = max(1, int(max_batch_size))
        self.slab_capacity = 3 * max_requests + int(max_num_tokens)
        # With pinned memory the kernel reads the host slab through its device mapping (no H2D copy);
        # otherwise the slab goes through a device copy.
        self.zero_copy = prefer_pinned()
        self.host_slabs = [
            numa_pinned_zeros(self.slab_capacity, dtype=torch.int32, pin_memory=self.zero_copy)
            for _ in range(_RING)
        ]
        self.host_views = [slab.numpy() for slab in self.host_slabs]
        self.slab_events: list[torch.cuda.Event | None] = [None] * _RING
        self.device_slab = (
            None
            if self.zero_copy or not torch.cuda.is_available()
            else torch.zeros(self.slab_capacity, dtype=torch.int32, device="cuda")
        )
        self.ring_pos = 0
        if stage_rng and self.zero_copy and max_tokens_per_step >= 2:
            from ..speculative.interface import _RNG_SLOT_SPAN

            self.rng_span = int(_RNG_SLOT_SPAN)
            self.host_rng_slabs = tuple(
                numa_pinned_zeros(2 * max_requests, dtype=torch.int64, pin_memory=self.zero_copy)
                for _ in range(_RING)
            )
            self.host_rng_views = tuple(slab.numpy() for slab in self.host_rng_slabs)
        self.block: int | None = None
        if max_tokens_per_step >= 2:
            self.block = triton.next_power_of_2(max_requests * max_tokens_per_step)
            self._warm_stage_kernel(max_tokens_per_step, has_static_inputs)

    def _warm_stage_kernel(self, t: int, has_static_inputs: bool) -> None:
        """Compile the staging kernel variant with a B = 0 launch (reads and writes nothing)."""
        if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
            return
        dummy = torch.zeros(1, dtype=torch.int32, device="cuda")
        variants: list[dict[str, Any]] = [{}]
        if self.rng_span:
            dummy64 = torch.zeros(1, dtype=torch.int64, device="cuda")
            variants.append(
                dict(
                    rng_slab=dummy64,
                    rng_buffers=(dummy64, dummy64, dummy64, dummy64),
                    rng_span=self.rng_span,
                )
            )
        for rng_kwargs in variants:
            stage_steady_inputs(
                dummy,
                0,
                t,
                types.SimpleNamespace(
                    new_tokens=dummy.view(1, 1, 1),
                    next_draft_tokens=dummy.view(1, 1),
                    new_tokens_lens=dummy,
                ),
                input_ids=dummy,
                draft_tokens=dummy,
                prev_batch_idx=dummy,
                prev_pos_idx=dummy,
                prev_pos_offsets=dummy,
                prev_kv_offsets=dummy,
                positions=dummy,
                mrope_positions=dummy.view(1, 1, 1) if self.use_mrope else None,
                gather_ids=dummy,
                kv_lens_cuda=dummy,
                static_input_ids=dummy if has_static_inputs else None,
                static_position_ids=(
                    (dummy.view(1, 1, 1) if self.use_mrope else dummy.view(1, 1))
                    if has_static_inputs
                    else None
                ),
                block=self.block,
                **rng_kwargs,
            )

    def rng_rows(self) -> int | None:
        """Logits rows per request when this stager writes the step's Philox state, else None."""
        if not self.rng_span or self.record is None:
            return None
        return self.record.tokens_per_step

    def invalidate(self) -> None:
        """Drop the record (every full pass does, before it may record again)."""
        self.record = None

    def record_batch(
        self,
        *,
        requests: list["LlmRequest"],
        prompt_lengths: list[int],
        sequence_lengths: list[int],
        tokens_per_step: int,
        runtime_draft_len: int,
        num_extra_kv_tokens: int,
        use_full_generation_page_table: bool,
        attn_metadata: AttentionMetadata,
        spec_metadata: "SpecMetadata",
        static_tensors: dict[str, torch.Tensor] | None,
    ) -> None:
        """Record a full pass that prepared ``requests``, all verifying their previous drafts."""
        b = len(requests)
        n_tokens = b * tokens_per_step
        if 3 * b + n_tokens > self.slab_capacity:
            return
        attn_steady = (
            isinstance(attn_metadata, TrtllmAttentionMetadata)
            and getattr(attn_metadata, "supports_steady_decode_step", False)
            and attn_metadata.num_seqs == b
        )
        attn_dsa_steady = False
        dsa_raw_blocks: np.ndarray | None = None
        dsa_num_blocks: list[int] | None = None
        dsa_max_blocks = 0
        if self.steady_dsa and not attn_steady and self._qualifies_dsa_steady(attn_metadata, b):
            cache_mgr = attn_metadata.kv_cache_manager
            req_ids = [request.py_request_id for request in requests]
            copy_idx = cache_mgr.index_mapper.get_copy_index(req_ids, 0, 1)
            slots = copy_idx.tolist()
            src_np = cache_mgr.host_kv_cache_block_offsets.numpy()
            dsa_raw_blocks = np.stack([src_np[0, sl, 0, :] for sl in slots], axis=0).copy()
            tpb = int(cache_mgr.tokens_per_block)
            kv_list = attn_metadata.kv_lens[:b].tolist()
            dsa_num_blocks = [
                max(1, (int(kl) - int(num_extra_kv_tokens) + tpb - 1) // tpb) for kl in kv_list
            ]
            dsa_max_blocks = max(dsa_num_blocks) if dsa_num_blocks else 1
            attn_dsa_steady = True
        dspark_slots = self._snapshot_dspark_slots(
            spec_metadata, [r.py_request_id for r in requests]
        )
        static_ids = static_pos = None
        static = static_tensors or {}
        if (
            "input_ids" in static
            and "position_ids" in static
            and static["input_ids"].shape[0] >= n_tokens
            and static["position_ids"].shape[-1] >= n_tokens
            and static["position_ids"].dim() == (3 if self.use_mrope else 2)
            and static["input_ids"].dtype == torch.int32
            and static["position_ids"].dtype == torch.int32
        ):
            static_ids, static_pos = static["input_ids"], static["position_ids"]
        if attn_steady:
            attn_metadata.key_prepared_block_offsets()
        self.record = SteadyMtpRecord(
            request_ids=[request.py_request_id for request in requests],
            slots=[request.py_seq_slot for request in requests],
            prompt_lengths=list(prompt_lengths),
            sequence_lengths=list(sequence_lengths),
            tokens_per_step=tokens_per_step,
            runtime_draft_len=runtime_draft_len,
            num_extra_kv_tokens=num_extra_kv_tokens,
            use_full_generation_page_table=use_full_generation_page_table,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            seq_lens_kv=attn_metadata.seq_lens_kv[:b].tolist(),
            attn_steady=attn_steady,
            attn_dsa_steady=attn_dsa_steady,
            dsa_raw_blocks=dsa_raw_blocks,
            dsa_num_blocks=dsa_num_blocks,
            dsa_max_blocks=dsa_max_blocks,
            dspark_slots=dspark_slots,
            static_ids=static_ids,
            static_pos=static_pos,
        )

    @staticmethod
    def _qualifies_dsa_steady(attn_metadata: AttentionMetadata, b: int) -> bool:
        """Whether ``attn_metadata`` is a pure-decode DSA metadata eligible for steady updates."""
        if not isinstance(attn_metadata, TrtllmAttentionMetadata):
            return False
        cache_mgr = attn_metadata.kv_cache_manager
        if (
            cache_mgr is None
            or attn_metadata.num_contexts != 0
            or attn_metadata.num_generations != b
            or attn_metadata.num_seqs != b
            or attn_metadata.beam_width != 1
            or getattr(attn_metadata, "enable_helix", False)
            or getattr(attn_metadata, "enable_flash_mla", False)
            or getattr(attn_metadata, "is_ragged_verify", False)
            or getattr(attn_metadata, "use_expanded_buffers_for_mtp", False)
            or getattr(attn_metadata, "expand_for_dsl", False)
            or getattr(attn_metadata, "use_fp8_ds_mla", False)
            or getattr(attn_metadata, "fp4_mla_state", None) is not None
            or getattr(attn_metadata, "device_windows_mode", False)
            or getattr(attn_metadata, "skip_indexer_for_gen_reqs", False)
        ):
            return False
        return (
            hasattr(attn_metadata, "indexer_k_cache_block_offsets")
            and hasattr(attn_metadata, "host_indexer_k_cache_block_offsets")
            and hasattr(attn_metadata, "block_table")
            and hasattr(cache_mgr, "host_kv_cache_block_offsets")
            and hasattr(cache_mgr, "index_mapper")
            and hasattr(cache_mgr, "indexer_k_cache_page_scale")
            and hasattr(cache_mgr, "num_blocks")
            and bool(getattr(cache_mgr, "tokens_per_block", 0))
            and not getattr(cache_mgr, "_use_per_layer_page_tables", False)
        )

    @staticmethod
    def _snapshot_dspark_slots(
        spec_metadata: "SpecMetadata", request_ids: list[int]
    ) -> list[int] | None:
        worker = getattr(spec_metadata, "_dspark_worker", None)
        if (
            worker is None
            or not getattr(worker, "_win_inited", False)
            or not hasattr(spec_metadata, "batch_indices_cuda")
            or not hasattr(worker, "_req_to_slot")
            or not hasattr(worker, "_batch_to_slot")
        ):
            return None
        if not all(rid in worker._req_to_slot for rid in request_ids):
            return None
        return [int(worker._req_to_slot[rid]) for rid in request_ids]

    def matches(
        self,
        requests: list["LlmRequest"],
        attn_metadata: AttentionMetadata,
        spec_metadata: "SpecMetadata | None",
        new_tensors_device: Any,
        runtime_draft_len: int,
    ) -> bool:
        """Whether the generation-only batch ``requests`` is the recorded one, one step later."""
        rec = self.record
        if (
            rec is None
            or attn_metadata is not rec.attn_metadata
            or spec_metadata is not rec.spec_metadata
            or not attn_metadata.is_cuda_graph
            or runtime_draft_len != rec.runtime_draft_len
            or not isinstance(new_tensors_device, SampleStateTensorsSpec)
            or new_tensors_device.next_draft_tokens is None
            or new_tensors_device.new_tokens_lens is None
            or len(requests) != len(rec.request_ids)
        ):
            return False
        return all(
            request.py_request_id == rid
            and request.py_seq_slot == slot
            and request.py_batch_idx == slot
            and self.request_qualifies(request)
            for request, rid, slot in zip(requests, rec.request_ids, rec.slots)
        )

    @staticmethod
    def request_qualifies(request: "LlmRequest") -> bool:
        """Whether a request's step can be staged: real, text-only, no one-hot draft probs."""
        return (
            not request.is_dummy
            and request.py_multimodal_data is None
            and not getattr(request, "py_needs_onehot_draft_probs", False)
        )

    def _next_slab(self) -> int:
        k = self.ring_pos
        self.ring_pos = (k + 1) % _RING
        event = self.slab_events[k]
        if event is not None and not event.query():
            event.synchronize()
        return k

    def stage(
        self,
        requests: list["LlmRequest"],
        new_tensors_device: Any,
        buffers: Any,
        attn_metadata: AttentionMetadata,
        rng_state: tuple[list[int], list[int]] | None = None,
        rng_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[int], list[int], int]:
        """Write this step's device inputs (one slab write, one kernel launch)."""
        rec = self.record
        assert rec is not None
        b = len(requests)
        t = rec.tokens_per_step
        n = b * t
        k = self._next_slab()
        view = self.host_views[k]
        num_cached = []
        kv_lens = []
        for i, request in enumerate(requests):
            past_seen = request.max_beam_num_tokens - 1
            view[i] = rec.slots[i]
            base = b + i * t
            view[base : base + t] = np.arange(past_seen, past_seen + t, dtype=np.int32)
            num_cached.append(past_seen + t - request.py_num_compressed_tokens)
            kv_lens.append(num_cached[-1] + rec.seq_lens_kv[i])
        used = b + n
        stage_kv = rec.attn_steady or rec.attn_dsa_steady
        if stage_kv:
            view[used : used + b] = kv_lens
            used += b
        if self.zero_copy:
            slab = self.host_slabs[k]
        else:
            assert self.device_slab is not None
            self.device_slab[:used].copy_(self.host_slabs[k][:used], non_blocking=True)
            slab = self.device_slab
        rng_kwargs: dict[str, Any] = {}
        if rng_state is not None and self.rng_span:
            seeds, offsets = rng_state
            assert len(seeds) == b and len(offsets) == b and rng_buffers is not None
            rng_view = self.host_rng_views[k]
            rng_view[:b] = seeds
            rng_view[b : 2 * b] = offsets
            rng_kwargs = dict(
                rng_slab=self.host_rng_slabs[k], rng_buffers=rng_buffers, rng_span=self.rng_span
            )

        stage_steady_inputs(
            slab,
            b,
            t,
            new_tensors_device,
            input_ids=buffers.input_ids_cuda,
            draft_tokens=buffers.draft_tokens_cuda,
            prev_batch_idx=buffers.previous_batch_indices_cuda,
            prev_pos_idx=buffers.previous_pos_indices_cuda,
            prev_pos_offsets=buffers.previous_pos_id_offsets_cuda,
            prev_kv_offsets=buffers.previous_kv_lens_offsets_cuda,
            positions=buffers.position_ids_cuda,
            mrope_positions=buffers.mrope_position_ids_cuda if self.use_mrope else None,
            gather_ids=buffers.gather_ids_cuda,
            kv_lens_cuda=attn_metadata.kv_lens_cuda if stage_kv else None,
            static_input_ids=rec.static_ids,
            static_position_ids=rec.static_pos,
            block=self.block,
            **rng_kwargs,
        )
        event = self.slab_events[k]
        if event is None:
            event = self.slab_events[k] = torch.cuda.Event()
        event.record()
        return num_cached, kv_lens, n if rec.static_ids is not None else 0

    def try_take_steady_rng_state(
        self, requests: list["LlmRequest"], spec_metadata: "SpecMetadata"
    ) -> tuple[list[int], list[int]] | None:
        """Advance Philox RNG offsets for a steady decode batch so ``_steady_stage_kernel`` writes
        all four RNG device buffers in the same launch. Returns None to fall back to the full
        ``populate_sampling_params_for_one_model`` call."""
        rec = self.record
        if (
            rec is None
            or not self.rng_span
            or getattr(spec_metadata, "enable_penalty", False)
            or getattr(spec_metadata, "temperatures", None) is None
            or getattr(spec_metadata, "request_seeds", None) is None
            or getattr(spec_metadata, "request_offsets", None) is None
            or getattr(spec_metadata, "seeds", None) is None
            or getattr(spec_metadata, "offsets", None) is None
            or not hasattr(spec_metadata, "_take_rng_window_offsets")
        ):
            return None
        b = len(requests)
        n_rows = b * rec.tokens_per_step
        if (
            spec_metadata.request_seeds.numel() < b
            or spec_metadata.request_offsets.numel() < b
            or spec_metadata.seeds.numel() < n_rows
            or spec_metadata.offsets.numel() < n_rows
        ):
            return None
        from ..speculative.interface import DEFAULT_SAMPLING_SEED
        from .sampler.sampler_common import request_random_seed

        user_seeds = [request_random_seed(req) for req in requests]
        req_seeds = [DEFAULT_SAMPLING_SEED if s is None else s for s in user_seeds]
        req_offsets = spec_metadata._take_rng_window_offsets(
            requests, [s is not None for s in user_seeds]
        )
        return req_seeds, req_offsets

    def try_prepare_steady_spec(
        self, requests: list["LlmRequest"], spec_metadata: "SpecMetadata"
    ) -> bool:
        """Fast-path steady ``DSparkSpecMetadata.prepare()`` when batch & slot mapping match."""
        rec = self.record
        if rec is None or not self.steady_dsa or rec.dspark_slots is None:
            return False
        worker = getattr(spec_metadata, "_dspark_worker", None)
        if worker is None or not getattr(worker, "_win_inited", False):
            return False
        if len(worker._req_to_slot) != len(rec.request_ids):
            return False
        cur_slots = [worker._req_to_slot.get(rid, -1) for rid in rec.request_ids]
        if cur_slots != rec.dspark_slots:
            return False
        spec_metadata.context_capture_lens = None
        scratch = worker._scratch_slot
        worker._ctx_len[scratch] = 0
        worker._valid_len[scratch] = 0
        worker._position_initialized[scratch] = False
        return True

    def prepare_attention(self, attn_metadata: AttentionMetadata, kv_lens: list[int]) -> None:
        """The attention metadata's prepare() of a staged steady step."""
        assert self.record is not None
        if self.record.attn_steady:
            attn_metadata.prepare_steady_decode_step(kv_lens)
            return
        if self.record.attn_dsa_steady and self._prepare_dsa_steady_decode_step(
            attn_metadata, kv_lens
        ):
            return
        if self.record.attn_dsa_steady:
            # Fallback occurred (e.g. large block-table change): run full prepare and drop stale snapshot.
            self.record.attn_dsa_steady = False
        attn_metadata.prepare()

    def _prepare_dsa_steady_decode_step(
        self, attn_metadata: AttentionMetadata, kv_lens: list[int]
    ) -> bool:
        """Steady pure-decode update for ``DSATrtllmAttentionMetadata``: skips unchanged block
        tables / host-to-device metadata copies and packs tiny page-boundary patches into
        ``step_inputs``."""
        rec = self.record
        assert rec is not None and rec.dsa_raw_blocks is not None and rec.dsa_num_blocks is not None
        cache_mgr = attn_metadata.kv_cache_manager
        if cache_mgr is None:
            return False
        b = len(kv_lens)
        max_kv_len = max(kv_lens) if b > 0 else 0
        if max_kv_len + rec.num_extra_kv_tokens > cache_mgr.max_seq_len:
            return False
        if getattr(attn_metadata, "enable_indexer_skip", False):
            if max_kv_len <= attn_metadata.num_sparse_topk - rec.num_extra_kv_tokens:
                return False

        from tensorrt_llm._torch.utils import get_global_attrs, get_model_extra_attrs

        if get_model_extra_attrs() is None:
            get_global_attrs().attention_metadata = weakref.ref(attn_metadata)
        attn_metadata._invalidate_pool_view_cache()

        kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device="cpu")
        attn_metadata.kv_lens[:b].copy_(kv_lens_t + rec.num_extra_kv_tokens)
        attn_metadata.host_total_kv_lens[0] = 0
        attn_metadata.host_total_kv_lens[1] = int(sum(kv_lens))
        attn_metadata.host_request_types[:b].fill_(1)
        if attn_metadata.enable_context_mla_with_cached_kv:
            attn_metadata.num_ctx_cached_tokens = 0
            attn_metadata.max_ctx_cached_token_len = 0
            attn_metadata.max_ctx_kv_len = 0
            attn_metadata.max_ctx_seq_len = 0

        cache_mgr.copy_batch_block_offsets(
            attn_metadata.kv_cache_block_offsets,
            rec.request_ids,
            1,
            0,
            b,
            max_blocks=None,
        )
        if attn_metadata.draft_kv_cache_manager is not None:
            attn_metadata.draft_kv_cache_manager.copy_batch_block_offsets(
                attn_metadata.draft_kv_cache_block_offsets,
                rec.request_ids,
                1,
                0,
                b,
                max_blocks=None,
            )
        attn_metadata._bind_runtime_views(
            kv_lens_cuda=attn_metadata.kv_lens_cuda[:b],
            kv_lens=kv_lens_t,
            prompt_lens_cuda=attn_metadata.prompt_lens_cuda[:b],
            prompt_lens_cpu=attn_metadata.prompt_lens_cpu[:b],
            host_request_types=attn_metadata.host_request_types[:b],
        )

        attn_metadata.num_ctx_cached_tokens = 0
        attn_metadata.num_ctx_mla_kv_tokens = 0
        attn_metadata.max_ctx_kv_len = 0
        attn_metadata.max_ctx_seq_len = 0
        attn_metadata.max_gen_seq_len = rec.tokens_per_step
        attn_metadata.skip_indexer_for_ctx_reqs = False
        attn_metadata.skip_indexer_for_gen_reqs = False
        attn_metadata.num_ctx_kv_tokens = 0

        # Incremental indexer_k_cache_block_offsets + block_table update.
        copy_idx = cache_mgr.index_mapper.get_copy_index(list(rec.request_ids), 0, 1)
        slots = copy_idx.tolist()
        src_np = cache_mgr.host_kv_cache_block_offsets.numpy()
        tpb = int(cache_mgr.tokens_per_block)
        cur_nb = [max(1, (int(kl) + tpb - 1) // tpb) for kl in kv_lens]
        new_max_nb = max(cur_nb) if cur_nb else 1
        old_max_nb = rec.dsa_max_blocks
        max_blk_id = int(cache_mgr.num_blocks) - 1
        page_scale = int(cache_mgr.indexer_k_cache_page_scale)

        ix_patches: list[tuple[int, int, int, np.ndarray]] = []
        bt_patches: list[tuple[int, int, int, np.ndarray]] = []
        total_span = 0
        for s in range(b):
            cur_row = src_np[0, slots[s], 0, :]
            prev_row = rec.dsa_raw_blocks[s, :]
            changed = not np.array_equal(cur_row, prev_row)
            old_nb_s = rec.dsa_num_blocks[s]
            new_nb_s = cur_nb[s]
            lo_chg = hi_chg = -1
            if changed:
                diff = np.flatnonzero(cur_row != prev_row)
                lo_chg = int(diff[0])
                hi_chg = int(diff[-1]) + 1
                total_span += hi_chg - lo_chg
                if total_span > self.DSA_BLOCK_PATCH_MAX_SPAN:
                    return False
                ix_patches.append((s, lo_chg, hi_chg, cur_row))
            # Determine block_table columns requiring update (changed active cols or new tail cols).
            bt_lo = new_max_nb
            bt_hi = 0
            if changed and lo_chg < new_nb_s:
                bt_lo = min(bt_lo, lo_chg)
                bt_hi = max(bt_hi, min(hi_chg, new_nb_s))
            if new_nb_s != old_nb_s:
                bt_lo = min(bt_lo, min(old_nb_s, new_nb_s))
                bt_hi = max(bt_hi, max(old_nb_s, new_nb_s))
            if new_max_nb > old_max_nb and new_nb_s < new_max_nb:
                bt_lo = min(bt_lo, old_max_nb)
                bt_hi = max(bt_hi, new_max_nb)
            if bt_hi > bt_lo:
                total_span += bt_hi - bt_lo
                if total_span > self.DSA_BLOCK_PATCH_MAX_SPAN:
                    return False
                bt_patches.append((s, bt_lo, bt_hi, cur_row))

        for s, lo, hi, cur_row in ix_patches:
            clamped = np.clip(cur_row[lo:hi], 0, max_blk_id).astype(np.int32, copy=False)
            page_vals = (clamped * page_scale).astype(np.int32, copy=False)
            host_slice = attn_metadata.host_indexer_k_cache_block_offsets[s, lo:hi]
            host_slice.copy_(torch.from_numpy(page_vals))
            step_inputs.copy_(attn_metadata.indexer_k_cache_block_offsets[s, lo:hi], host_slice)
            rec.dsa_raw_blocks[s, lo:hi] = cur_row[lo:hi]

        for s, bt_lo, bt_hi, cur_row in bt_patches:
            cols = np.arange(bt_lo, bt_hi, dtype=np.int32)
            clamped = np.clip(cur_row[bt_lo:bt_hi], 0, max_blk_id).astype(np.int32, copy=False)
            bt_vals = np.where(cols < cur_nb[s], clamped, -1).astype(np.int32, copy=False)
            bt_src = torch.from_numpy(bt_vals)
            if not step_inputs.is_open():
                bt_src = maybe_pin_memory(bt_src)
            step_inputs.copy_(attn_metadata.block_table[s, bt_lo:bt_hi], bt_src)

        rec.dsa_num_blocks = cur_nb
        rec.dsa_max_blocks = max(old_max_nb, new_max_nb)

        if not self.defer_dsa_derived:
            cached_t = torch.tensor(
                attn_metadata.kv_cache_params.num_cached_tokens_per_seq,
                dtype=torch.int32,
                device="cpu",
            )
            attn_metadata.prepare_for_mla_rope_append(cached_t, kv_lens_t)
            attn_metadata.prepare_for_indices_conversion()
            from ..attention.backends.sparse.dsa.indexer import Indexer

            Indexer.prepare(metadata=attn_metadata)
        return True
