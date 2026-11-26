# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-loop behaviour a model may opt into, read once by the engine.

A model declares its hooks as ``runtime_hooks`` (an instance attribute or a class
attribute). Every field defaults to off, and a model that declares nothing gets
``RuntimeHooks()``: the step loop, the samplers and the attention metadata then
run exactly as they do without this module.

The hooks change *where* and *how* the step loop moves data, or how the engine
sizes and manages its resources (device memory, streams, timing events, thread
affinity), never what a forward computes. Each one is armed only by a model
that has opted into it; none is a general default.

A modeling-v2 target declares them on the registered (outer) model class, the
class the loader resolves and the engine holds as ``model_engine.model``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

if TYPE_CHECKING:
    import torch

__all__ = [
    "DEFAULT_RUNTIME_HOOKS",
    "DecodePrologue",
    "ExecutorEvents",
    "RuntimeHooks",
    "SpecStepFold",
    "runtime_hooks_of",
]


class SpecStepFold(Protocol):
    """Speculative step-input staging folded into the previous step's forward.

    A one-model speculative step ends by storing its accepted tokens, draft
    tokens and lengths into the sampler's per-slot store, and the next step
    begins by gathering those values back into the engine's flat input buffers.
    A model that implements this protocol does both inside its last draft pass:
    it writes the sampler store and stages the next step's inputs directly, and
    the engine and the sampler skip their copies on the steps where it did.

    The object is owned by the model; the engine and the sampler only call it.
    Every method is host-side bookkeeping or a single kernel launch on the
    current stream.
    """

    def bind_engine_buffers(
        self,
        input_ids: "torch.Tensor",
        draft_tokens: "torch.Tensor",
        position_id_offsets: "torch.Tensor",
        kv_lens_offsets: "torch.Tensor",
        max_batch_size: int,
    ) -> None:
        """Receive the engine's persistent speculative input buffers, once, at construction."""

    def bind_sampler_store(self, store: object) -> None:
        """Receive the speculative sampler's persistent per-slot store, once, at construction."""

    def begin_step(self, seq_slots: Optional[list[int]], num_rows: int) -> bool:
        """Start a step of ``num_rows`` generation rows.

        ``seq_slots`` are those rows' sequence slots on a decode-only step whose
        rows all have one, else None: the forward then stores nothing for them.
        Returns whether the previous step's forward staged its inputs for
        ``seq_slots``; the engine additionally checks that this step adds no
        inputs of its own before it skips its gathers.
        """

    def gather(
        self,
        new_tokens: "torch.Tensor",
        next_draft_tokens: "torch.Tensor",
        new_tokens_lens: "torch.Tensor",
        previous_slots: "torch.Tensor",
        previous_pos_indices: "torch.Tensor",
        previous_batch_len: int,
        tokens_per_gen_step: int,
        draft_token_buffer_width: int,
        num_tokens: int,
        num_draft_tokens: int,
        first_previous_row: int,
    ) -> bool:
        """Do the engine's previous-batch gathers in one launch.

        Writes the engine buffers exactly as the engine's own gathers would,
        including zeroing both offset buffers outside the gathered range.
        Returns False, having written nothing, when it cannot take the step.
        """

    def sampler_rows_stored(self, outputs: dict, num_skip: int, seq_slots: list[int]) -> bool:
        """Whether the forward already stored this step's outputs for ``seq_slots``."""

    def scatter(
        self,
        seq_slots: "torch.Tensor",
        new_tokens: "torch.Tensor",
        next_new_tokens: "torch.Tensor",
        new_tokens_lens: "torch.Tensor",
        next_draft_tokens: "torch.Tensor",
    ) -> bool:
        """Store the step's outputs into the sampler store in one launch.

        Returns False, having written nothing, when it cannot take the step.
        """


class DecodePrologue(Protocol):
    """Replaces the decode step's input prologue with one launch.

    Called first in the step's input preprocessing, inside CUDA graph capture
    as well. It decides on the host before any device work; returning False
    keeps the native prologue, and returning True means every engine-visible
    tensor the native prologue writes holds the value it would have written.
    """

    def __call__(
        self, preparer: Any, inputs: Any, *, enable_spec_decode: bool, runtime_draft_len: int
    ) -> bool:
        """``preparer`` is the ``DecoderInputPreparer`` running the step's preprocessing."""


class ExecutorEvents(Protocol):
    """Points in the executor's life a model may act at.

    Optional, so not a protocol member: ``on_iteration(iteration)``, called by
    the overlap executor loop at the top of every iteration with its iteration
    counter (host bookkeeping only; must not raise).
    """

    def after_warmup(self) -> None:
        """After warmup and CUDA graph capture, before the first request is served."""

    def before_overlap_loop(self) -> None:
        """Right before the overlap executor loop starts, on every rank; must not raise."""

    def at_shutdown(self) -> None:
        """When the executor loop has completed, during cleanup; must not raise."""


@dataclass(frozen=True)
class RuntimeHooks:
    """The step-loop hooks a model opts into. All off by default."""

    # Pack a decode step's small host-to-device input copies into one upload and
    # one scatter launch, issued just before the CUDA graph replay.
    pack_step_inputs: bool = False
    # Issue the step-input pack's copies with one parallel scatter (a program per copy)
    # instead of the serial one-CTA scatter, when their destinations do not overlap.
    pack_parallel_scatter: bool = False
    # Issue the step-input pack inside the replayed graph (a prologue copies a pinned window
    # and scatters it), so no operation is left between the decode graphs.
    pack_in_graph: bool = False
    # With pack_in_graph: a side copy stream uploads the pinned window into the prologue's own
    # device buffer while the previous graph still runs, so the prologue only waits on that copy
    # and scatters device to device.
    pack_early_copy: bool = False
    # With pack_in_graph: keep one int64 device counter incremented once per replay by the
    # graph prologue, never reset, for per-step device flags.
    pack_step_epoch: bool = False
    # With pack_early_copy: the model's one-launch prologue, called inside the capture as
    # ``op(window, epoch_or_None, max_desc)``.
    pack_prologue_op: Optional[Callable[[torch.Tensor, Optional[torch.Tensor], int], None]] = None
    # With pack_step_inputs: the model's own step-input pack factory, built once per engine as
    # ``factory(device)`` in place of the stock StepInputPack.
    step_input_pack_factory: Optional[Callable[[torch.device], Any]] = None
    decode_prologue: Optional[DecodePrologue] = None
    executor_events: Optional[ExecutorEvents] = None
    # Skip host-to-device metadata copies whose values the destination already holds.
    skip_unchanged_h2d: bool = False
    # With skip_unchanged_h2d: a changed packed 2-D upload stages only each row's changed
    # column span so the pack keeps fitting the in-graph window.
    h2d_changed_span: bool = False
    # A speculative step under the overlap scheduler re-derives its KV-length metadata
    # (on_update_kv_lens) once, after the forward's KV-length correction, instead of also
    # before it: the first pass's device writes are all rewritten by the second.
    kv_lens_update_once: bool = False
    # The overlap scheduler's speculative step correction (position_ids += position offsets,
    # kv_lens += KV-length offsets) as the model's one launch, called as
    # ``op(positions, position_offsets, kv_lens, kv_lens_offsets)`` on the generation rows;
    # returns False to decline so the two in-tree adds run.
    spec_offsets_op: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], bool]
    ] = None
    # Run the overlap executor loop with the execution stream current, so the loop's own
    # stream work and the forward are ordered on one stream.
    loop_on_execution_stream: bool = False
    # Streaming: also emit a response on decoding iterations 2..N. 1 keeps the stock schedule.
    first_content_emit_iters: int = 1
    # Enable the steady one-model speculative decode input stager (steady_spec_prepare.py)
    # for unchanged decode batches.
    steady_spec_prepare: bool = False
    # Stage the one-model speculative sampler's per-step Philox seed and offset uploads into
    # the steady stager's slab or the step-input pack when supported.
    pack_spec_rng_uploads: bool = False
    # Speculative step-input staging folded into the forward (see SpecStepFold).
    spec_step_fold: Optional[SpecStepFold] = None
    # Time the iteration log's device step with one timing event per iteration boundary,
    # recorded on a side stream that waits for the current stream.
    iteration_timing_one_event: bool = False
    # Record per-request perf-metrics timing events on a side stream that waits for the
    # current stream.
    perf_metrics_timing_side_stream: bool = False
    # Device memory, in bytes, left outside the PyTorch cap for driver allocations made after
    # setup, up to the memory free at that point. None keeps the engine's default margin.
    torch_mem_safety_margin: Optional[int] = None

    @property
    def skips_unchanged_metadata_uploads(self) -> bool:
        """Whether a hook skips uploads into graph metadata buffers that already hold the values."""
        return self.skip_unchanged_h2d


DEFAULT_RUNTIME_HOOKS = RuntimeHooks()


def runtime_hooks_of(model: object) -> RuntimeHooks:
    """The hooks ``model`` declares, or the all-off default when it declares none.

    Declaring hooks is optional for every model class, so absence is the common
    case and means the default.
    """
    hooks = getattr(model, "runtime_hooks", None)
    if hooks is None:
        return DEFAULT_RUNTIME_HOOKS
    if not isinstance(hooks, RuntimeHooks):
        raise TypeError(
            f"{type(model).__name__}.runtime_hooks must be a RuntimeHooks, got {type(hooks).__name__}"
        )
    return hooks
