# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Onboarding of reused disk KV blocks before a context request is scheduled.

Covers the KVCacheManagerV2 bookkeeping (start, poll, cancel), the executor hook that gates
requests with ``py_kv_onboard_pending``, and the scheduler check that honors the flag. The KV
cache itself is faked; the disk path is tested with real caches in kv_cache_manager_v2_tests.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import RequestScheduler
from tensorrt_llm.runtime.kv_cache_manager_v2 import CACHE_LEVEL1, DiskIoStatus

pytestmark = pytest.mark.cpu_only


class _FakeKvCache:
    """Answers poll_disk_reads() from a script; the last status repeats."""

    def __init__(self, statuses, prefetch_result=True):
        self.statuses = list(statuses)
        self.prefetch_result = prefetch_result
        self.prefetch_targets = []
        self.closed = False

    def prefetch(self, target):
        self.prefetch_targets.append(target)
        return self.prefetch_result

    def poll_disk_reads(self, wait=False):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def close(self):
        assert not self.closed
        self.closed = True


def _manager(*kv_caches, block_reuse=True):
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = block_reuse
    manager._context_onboards = {}
    manager._iter_onboarded_requests = 0
    manager._iter_onboard_failures = 0
    manager._iter_onboard_wait_s = 0.0
    manager._context_reuse_tokens = lambda req: [1, 2, 3]
    manager._derive_reuse_salt = lambda cache_salt: None
    manager.impl = SimpleNamespace(create_kv_cache=Mock(side_effect=list(kv_caches)))
    return manager


def _request(request_id):
    return SimpleNamespace(py_request_id=request_id, lora_task_id=None, cache_salt=None)


def test_start_keeps_the_cache_open_while_reads_are_pending():
    kv_cache = _FakeKvCache([DiskIoStatus.PENDING])
    manager = _manager(kv_cache)
    assert manager.start_context_onboard(_request(7))
    assert kv_cache.prefetch_targets == [CACHE_LEVEL1]
    assert not kv_cache.closed
    assert manager.num_context_onboards == 1
    # A second start for the same request does not match again.
    assert not manager.start_context_onboard(_request(7))
    assert manager.impl.create_kv_cache.call_count == 1


@pytest.mark.parametrize(
    ("statuses", "prefetch_result"),
    [
        ([DiskIoStatus.OK], True),  # Nothing matched on disk.
        ([DiskIoStatus.PENDING], False),  # The host tier could not take the blocks.
    ],
)
def test_start_closes_the_cache_when_there_is_nothing_to_wait_for(statuses, prefetch_result):
    kv_cache = _FakeKvCache(statuses, prefetch_result)
    manager = _manager(kv_cache)
    assert not manager.start_context_onboard(_request(7))
    assert kv_cache.closed
    assert manager.num_context_onboards == 0


def test_start_is_a_no_op_without_block_reuse():
    manager = _manager(block_reuse=False)
    assert not manager.start_context_onboard(_request(7))
    manager.impl.create_kv_cache.assert_not_called()


@pytest.mark.parametrize("final_status", [DiskIoStatus.OK, DiskIoStatus.FAILED])
def test_poll_reports_each_request_once_and_closes_its_cache(final_status):
    kv_cache = _FakeKvCache([DiskIoStatus.PENDING, DiskIoStatus.PENDING, final_status])
    manager = _manager(kv_cache)
    request = _request(7)
    assert manager.start_context_onboard(request)
    assert manager.poll_context_onboards() == []
    assert not kv_cache.closed
    assert manager.poll_context_onboards() == [request]
    assert kv_cache.closed
    assert manager.num_context_onboards == 0
    assert manager.poll_context_onboards() == []
    assert manager._iter_onboarded_requests == 1
    assert manager._iter_onboard_failures == (1 if final_status == DiskIoStatus.FAILED else 0)


def test_cancel_closes_the_cache():
    kv_cache = _FakeKvCache([DiskIoStatus.PENDING])
    manager = _manager(kv_cache)
    assert manager.start_context_onboard(_request(7))
    manager.cancel_context_onboard(7)
    assert kv_cache.closed
    assert manager.num_context_onboards == 0
    manager.cancel_context_onboard(7)
    manager.cancel_context_onboard(8)


@pytest.mark.parametrize(
    ("enable_attention_dp", "tp_size", "pp_size", "cp_size", "expected"),
    [
        (True, 4, 1, 1, True),
        (False, 1, 1, 1, True),
        (False, 2, 1, 1, False),
        (True, 4, 2, 1, False),
        (True, 4, 1, 2, False),
    ],
)
def test_gating_needs_each_rank_to_schedule_its_own_requests(
    enable_attention_dp, tp_size, pp_size, cp_size, expected
):
    manager = object.__new__(KVCacheManagerV2)
    manager.mapping = SimpleNamespace(
        enable_attention_dp=enable_attention_dp, tp_size=tp_size, pp_size=pp_size, cp_size=cp_size
    )
    assert manager.can_gate_context_onboards is expected


class _OnboardingKvManager:
    """Stands in for KVCacheManagerV2: requests in `on_disk` onboard until finish() is called."""

    def __init__(self, on_disk, limit=1, can_gate=True):
        self.enable_block_reuse = True
        self.disk_prefetch_num_reqs = limit
        self.can_gate_context_onboards = can_gate
        self.kv_cache_map = {}
        self.on_disk = set(on_disk)
        self.onboarding = {}
        self.probed = []
        self._finished = []

    @property
    def num_context_onboards(self):
        return len(self.onboarding)

    def start_context_onboard(self, request):
        self.probed.append(request.py_request_id)
        if request.py_request_id not in self.on_disk:
            return False
        self.onboarding[request.py_request_id] = request
        return True

    def poll_context_onboards(self):
        finished, self._finished = self._finished, []
        return finished

    def finish(self, request_id):
        self._finished.append(self.onboarding.pop(request_id))


def _context_request(request_id, **overrides):
    fields = dict(
        py_request_id=request_id,
        is_first_context_chunk=True,
        is_dummy=False,
        is_disagg_generation_init_state=False,
        py_kv_onboard_pending=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _executor(kv_cache_manager, requests):
    return SimpleNamespace(
        _is_kv_manager_v2=True,
        kv_cache_manager=kv_cache_manager,
        active_requests=requests,
        _prefetched_request_ids=set(),
    )


def test_executor_gates_disk_hits_up_to_the_limit():
    kv_manager = _OnboardingKvManager(on_disk={1, 3}, limit=1)
    requests = [_context_request(i) for i in (1, 2, 3)]
    executor = _executor(kv_manager, requests)

    PyExecutor._prefetch_for_context_requests(executor)
    assert kv_manager.probed == [1]
    assert [r.py_kv_onboard_pending for r in requests] == [True, False, False]

    # Still reading: nothing new starts and nobody is released.
    PyExecutor._prefetch_for_context_requests(executor)
    assert kv_manager.probed == [1]
    assert requests[0].py_kv_onboard_pending

    kv_manager.finish(1)
    PyExecutor._prefetch_for_context_requests(executor)
    assert kv_manager.probed == [1, 2, 3]
    assert [r.py_kv_onboard_pending for r in requests] == [False, False, True]


def test_executor_onboards_without_gating_where_it_cannot_gate():
    kv_manager = _OnboardingKvManager(on_disk={1}, can_gate=False)
    request = _context_request(1)
    PyExecutor._prefetch_for_context_requests(_executor(kv_manager, [request]))
    assert kv_manager.probed == [1]
    assert not request.py_kv_onboard_pending


def test_executor_skips_requests_that_cannot_onboard():
    kv_manager = _OnboardingKvManager(on_disk={1, 2, 3, 4, 5}, limit=8)
    kv_manager.kv_cache_map[4] = object()
    requests = [
        _context_request(1, is_first_context_chunk=False),
        _context_request(2, is_dummy=True),
        _context_request(3, is_disagg_generation_init_state=True),
        _context_request(4),
        _context_request(5),
    ]
    PyExecutor._prefetch_for_context_requests(_executor(kv_manager, requests))
    assert kv_manager.probed == [5]
    assert [r.py_kv_onboard_pending for r in requests] == [False] * 4 + [True]


def test_executor_onboarding_is_off_with_a_zero_limit():
    kv_manager = _OnboardingKvManager(on_disk={1}, limit=0)
    kv_manager.poll_context_onboards = Mock()
    PyExecutor._prefetch_for_context_requests(_executor(kv_manager, [_context_request(1)]))
    kv_manager.poll_context_onboards.assert_not_called()
    assert kv_manager.probed == []


def test_scheduler_leaves_onboarding_requests_out():
    scheduler = SimpleNamespace(
        scheduling_state_range=(
            LlmRequestState.CONTEXT_INIT,
            LlmRequestState.GENERATION_TO_COMPLETE,
        )
    )
    request = SimpleNamespace(
        state=LlmRequestState.CONTEXT_INIT,
        state_value=LlmRequestState.CONTEXT_INIT.value,
        is_context_init_state=True,
        py_kv_onboard_pending=True,
    )
    assert not RequestScheduler.is_request_in_schedulable_state(scheduler, request)
    request.py_kv_onboard_pending = False
    assert RequestScheduler.is_request_in_schedulable_state(scheduler, request)
    # An object that answers every attribute name is not mistaken for a gated request.
    assert RequestScheduler.is_request_in_schedulable_state(
        scheduler, Mock(state=request.state, state_value=request.state_value)
    )


@pytest.mark.parametrize("waiting_for_peer", [False, True])
@pytest.mark.parametrize("onboarding", [False, True])
def test_gen_first_context_waits_for_both_its_peer_and_its_reads(waiting_for_peer, onboarding):
    """A generation-first context request may onboard while it waits for its peer; either wait holds it."""
    scheduler = SimpleNamespace(
        scheduling_state_range=(
            LlmRequestState.CONTEXT_INIT,
            LlmRequestState.GENERATION_TO_COMPLETE,
        )
    )
    state = (
        LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
        if waiting_for_peer
        else LlmRequestState.CONTEXT_INIT
    )
    request = SimpleNamespace(
        state=state,
        state_value=state.value,
        is_context_init_state=state == LlmRequestState.CONTEXT_INIT,
        py_kv_onboard_pending=onboarding,
    )
    expected = not waiting_for_peer and not onboarding
    assert RequestScheduler.is_request_in_schedulable_state(scheduler, request) is expected
