# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime hooks for Mistral Large 4 SM107 attention-DP (DEP4/DEP8) and TEP4 targets.

Reuses the NUMA-aware :class:`ExecutorThreadPin` and single-launch Triton
:func:`fused_spec_offsets_op` from the SM107 TP4 target without modifying that module,
while tailoring CUDA-graph step-input packing, H2D skip caching, steady speculative
preparation, and cyclic GC control to each worker's execution mode (CUDA-graph decode
vs. eager context prefill, attention-DP vs. non-ADP).
"""

from __future__ import annotations

import gc
import sys
from typing import Optional

from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm.logger import logger

from ..mistral_large_4_nvfp4__sm_107__tp4.runtime_hooks import (
    GIL_SWITCH_US,
    PIN_REFRESH_ITERS,
    ExecutorThreadPin,
    fused_spec_offsets_op,
)

__all__ = [
    "Ml4DepExecutorEvents",
    "make_ml4_dep_runtime_hooks",
]

_LOG = "ML4 SM107 DEP runtime"


class Ml4DepExecutorEvents:
    """Executor lifecycle events for Mistral Large 4 SM107 DEP4/DEP8 and TEP4 serving.

    Pins the executor loop thread to an exclusive NUMA-local physical core (isolating it
    from background NIXL/disk-I/O threads) and lowers the Python GIL switch interval.
    Periodic cyclic GC is disabled during the overlap loop only when ``disable_gc`` is
    True (CUDA-graph decode / aggregated serving), and restored on clean shutdown; pure
    eager context workers keep cyclic GC active so long-running high-churn prefill
    servers never accumulate reference cycles.
    """

    def __init__(
        self,
        *,
        pin_refresh_iters: int = PIN_REFRESH_ITERS,
        gil_switch_us: float = GIL_SWITCH_US,
        local_rank: Optional[int] = None,
        disable_gc: bool = True,
    ) -> None:
        self.pin_refresh_iters = max(1, int(pin_refresh_iters))
        self.gil_switch_us = float(gil_switch_us)
        self.local_rank = local_rank
        self.disable_gc = bool(disable_gc)
        self.pin: Optional[ExecutorThreadPin] = None
        self._prev_switch_interval: Optional[float] = None
        self._gc_was_enabled: bool = False

    def after_warmup(self) -> None:
        return

    def before_overlap_loop(self) -> None:
        self.pin = ExecutorThreadPin.create(
            fallback_local_rank=self.local_rank,
            refresh_iters=self.pin_refresh_iters,
        )
        self._prev_switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(self.gil_switch_us / 1_000_000)
        logger.info(f"{_LOG}: thread switch interval {sys.getswitchinterval() * 1e6:.0f} us")
        if self.disable_gc:
            self._gc_was_enabled = gc.isenabled()
            if self._gc_was_enabled:
                gc.collect()
                gc.disable()
                logger.info(f"{_LOG}: Python cyclic GC disabled for overlap serving loop")

    def on_iteration(self, iteration: int) -> None:
        if self.pin is not None:
            self.pin.maybe_refresh(iteration)

    def at_shutdown(self) -> None:
        if self._prev_switch_interval is not None:
            sys.setswitchinterval(self._prev_switch_interval)
            self._prev_switch_interval = None
        if self._gc_was_enabled:
            gc.enable()
            self._gc_was_enabled = False


def make_ml4_dep_runtime_hooks(
    mapping: object = None,
    *,
    use_cuda_graph: bool = True,
    enable_attention_dp: Optional[bool] = None,
) -> RuntimeHooks:
    """Build ML4 SM107 :class:`RuntimeHooks` for DEP4/DEP8 and TEP4 workers.

    Args:
        mapping: Parallelism mapping providing ``local_rank`` and ``enable_attention_dp``.
        use_cuda_graph: Whether the worker replays CUDA graphs. When False (e.g. eager
            disaggregated context prefill workers), step-input packing, H2D skip shadow
            caches, incremental block-offset patching, steady speculative preparation,
            and cyclic-GC suppression stay off while NUMA thread pinning, fused
            spec-offset correction, single-stream loop ordering, and side-stream timing
            remain active.
        enable_attention_dp: Explicit override for attention-DP mode; defaults to
            ``bool(getattr(mapping, "enable_attention_dp", True))``. Steady speculative
            fast paths (``steady_spec_prepare``, ``steady_dsa_prepare``, and
            ``pack_spec_rng_uploads``) are armed only for non-ADP CUDA-graph workers
            (TEP4 decode), because ``SteadyMtpPrepare`` disables itself under ADP.
    """
    local_rank = getattr(mapping, "local_rank", None)
    if enable_attention_dp is None:
        enable_attention_dp = bool(getattr(mapping, "enable_attention_dp", True))
    graph_worker = bool(use_cuda_graph)
    steady_non_adp = graph_worker and (not enable_attention_dp)
    events = Ml4DepExecutorEvents(
        local_rank=local_rank if isinstance(local_rank, int) else None,
        disable_gc=graph_worker,
    )
    return RuntimeHooks(
        pack_step_inputs=graph_worker,
        pack_parallel_scatter=graph_worker,
        pack_in_graph=graph_worker,
        pack_early_copy=graph_worker,
        executor_events=events,
        skip_unchanged_h2d=graph_worker,
        h2d_changed_span=graph_worker,
        incremental_block_offsets=graph_worker,
        steady_dsa_prepare=steady_non_adp,
        kv_lens_update_once=True,
        spec_offsets_op=fused_spec_offsets_op,
        loop_on_execution_stream=True,
        steady_spec_prepare=steady_non_adp,
        pack_spec_rng_uploads=steady_non_adp,
        iteration_timing_one_event=True,
        perf_metrics_timing_side_stream=True,
    )
