# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A standalone drafter that reads only its last N context rows keeps only those pages.

The drafter declares ``draft_kv_read_window`` on its config. Without block reuse the
separate draft pool then gets a sliding window of that many tokens plus two pages, and
is funded for one window per sequence the managers admit. The paged context it reads
stays addressable:

* a disaggregated generation slot starts on the page of its first token, so its
  columns land on pages past the freed prompt pages;
* the draft mirror of a disaggregated generation-init request takes its prompt as
  history, so it allocates only the window's pages rather than the prompt's;
* a row's page count is one past its last held page, so the freed pages before the
  window do not shorten it.
"""

import types
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    get_max_admissible_sequences,
)
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

READ_WINDOW = 2048
PAGE = 32
NUM_DRAFT_LAYERS = 3


def _creator(draft_pretrained_config, **attrs):
    creator = object.__new__(KvCacheCreator)
    creator._draft_config = SimpleNamespace(pretrained_config=draft_pretrained_config)
    creator._speculative_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(
            is_dflash=lambda: False, is_dspark=lambda: True, is_external_drafter=lambda: True
        )
    )
    creator._tokens_per_block = 128
    for name, value in attrs.items():
        setattr(creator, name, value)
    return creator


def _draft_config(read_window=READ_WINDOW):
    cfg = SimpleNamespace(num_hidden_layers=NUM_DRAFT_LAYERS)
    if read_window is not None:
        cfg.draft_kv_read_window = read_window
    return cfg


def test_retention_window_is_the_read_window_plus_two_pages():
    creator = _creator(_draft_config())
    no_reuse = KvCacheConfig(enable_block_reuse=False)
    with patch.object(KvCacheCreator, "_get_num_draft_layers", return_value=NUM_DRAFT_LAYERS):
        assert creator._get_draft_kv_retention_window(no_reuse) == READ_WINDOW + 2 * PAGE
        # Uniform across the drafter's layers, so one window, not a VSWA layout.
        assert creator._get_draft_max_attention_window(1 << 20, no_reuse) == [
            READ_WINDOW + 2 * PAGE
        ]


def test_retention_window_stays_off_under_block_reuse():
    creator = _creator(_draft_config())
    reuse = KvCacheConfig(enable_block_reuse=True)
    assert creator._get_draft_kv_retention_window(reuse) is None
    with (
        patch.object(KvCacheCreator, "_get_num_draft_layers", return_value=NUM_DRAFT_LAYERS),
        patch(
            "tensorrt_llm._torch.pyexecutor._util._derive_draft_max_attention_window",
            return_value=None,
        ) as derive,
    ):
        assert creator._get_draft_max_attention_window(1 << 20, reuse) is None
    derive.assert_called_once()


def test_drafter_without_a_read_window_keeps_the_derived_windows():
    creator = _creator(_draft_config(read_window=None))
    no_reuse = KvCacheConfig(enable_block_reuse=False)
    assert creator._get_draft_kv_retention_window(no_reuse) is None
    with (
        patch.object(KvCacheCreator, "_get_num_draft_layers", return_value=NUM_DRAFT_LAYERS),
        patch(
            "tensorrt_llm._torch.pyexecutor._util._derive_draft_max_attention_window",
            return_value=[4096],
        ),
    ):
        assert creator._get_draft_max_attention_window(1 << 20, no_reuse) == [4096]


@pytest.mark.parametrize(
    ("pp_size", "is_disagg", "disable_overlap_scheduler", "expected"),
    [
        (1, True, True, 32),  # in-transfer requests hold slots
        (1, False, False, 32),  # a retiring request holds its slot one more step
        (1, False, True, 16),
        (2, False, False, 32),  # microbatches in flight, no overlap lease under PP
        (2, True, False, 64),
    ],
)
def test_max_admissible_sequences(pp_size, is_disagg, disable_overlap_scheduler, expected):
    mapping = Mapping(world_size=pp_size, pp_size=pp_size)
    assert (
        get_max_admissible_sequences(
            16,
            mapping,
            is_disagg=is_disagg,
            disable_overlap_scheduler=disable_overlap_scheduler,
        )
        == expected
    )


@pytest.mark.parametrize(("enable_block_reuse", "expected"), [(False, 2 * 16 + 1), (True, None)])
def test_windowed_draft_pool_is_funded_for_every_admitted_sequence(enable_block_reuse, expected):
    creator = _creator(
        _draft_config(),
        _mapping=Mapping(),
        _draft_model_engine=None,
        _max_seq_len=1 << 20,
        _max_batch_size=16,
        _is_disagg=True,
        _disable_overlap_scheduler=False,
        _cache_transceiver_config=None,
    )
    kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse)
    with (
        patch.object(KvCacheCreator, "_get_num_draft_layers", return_value=NUM_DRAFT_LAYERS),
        patch.object(
            KvCacheCreator,
            "_get_draft_kv_model_config",
            return_value=creator._draft_config,
        ),
        patch.object(
            KvCacheCreator,
            "_validate_or_fallback_kv_cache_manager_v2",
            return_value=KVCacheManagerV2,
        ),
        patch(
            "tensorrt_llm._torch.pyexecutor._util.get_kv_cache_manager_cls",
            return_value=KVCacheManagerV2,
        ),
        patch(
            "tensorrt_llm._torch.pyexecutor._util._derive_draft_max_attention_window",
            return_value=None,
        ),
        patch.object(
            KvCacheCreator, "_per_manager_cache_cost", return_value=CacheCost(slope=0)
        ) as cost,
    ):
        creator._get_draft_cache_cost(kv_cache_config, use_separate_draft_kv_cache=True)
    assert cost.call_args.kwargs["max_batch_size"] == expected


def _draft_cost(max_attention_window, max_batch_size):
    """Static cost of the ML4 drafter's pool: 3 layers of one bf16 576-wide latent."""
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_attention_heads=64, num_key_value_heads=1, kv_lora_rank=512, qk_rope_head_dim=64
        ),
        quant_config=None,
    )
    return KVCacheManagerV2.get_cache_size_per_token(
        model_config,
        Mapping(),
        num_layers=NUM_DRAFT_LAYERS,
        tokens_per_block=PAGE,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, max_attention_window=max_attention_window
        ),
        max_seq_len=1 << 20,
        max_batch_size=max_batch_size,
        max_num_tokens=128,
        is_draft=True,
    )


def test_windowed_draft_pool_has_no_slope_and_an_intercept_per_sequence():
    bytes_per_token = NUM_DRAFT_LAYERS * 576 * 2
    slope, intercept = _draft_cost([READ_WINDOW + 2 * PAGE], 33)
    assert slope == 0
    # At least one window of pages per sequence, plus one step's context tokens.
    window_pages = (READ_WINDOW + 2 * PAGE) // PAGE
    assert intercept >= 33 * window_pages * PAGE * bytes_per_token + 128 * bytes_per_token
    _, twice = _draft_cost([READ_WINDOW + 2 * PAGE], 66)
    assert twice > intercept


def test_unwindowed_draft_pool_pays_per_token():
    slope, intercept = _draft_cost(None, 33)
    assert slope == NUM_DRAFT_LAYERS * 576 * 2
    assert intercept == 0


def _stale_end(history: int, window: int) -> int:
    """First page AttnLifeCycle::getStaleRange keeps (it frees every page before it)."""
    return max(0, (history + 1 - window) // PAGE)


def _first_unheld_read(prompt_len, accepted, window, shift_pages):
    """Simulate a disaggregated generation request; return the first read off its pages.

    The draft mirror's history starts at the prompt (prepare_disagg_gen_init) and moves
    to tokens - 1 after each step (update_resources). The slot's column c is on page
    shift + c // PAGE; it holds one column per token since the first generated one,
    and reads the last READ_WINDOW of them before and after this step's accepted
    tokens are stored.
    """
    num_tokens = prompt_len + 1
    history = prompt_len
    ctx_len = 0
    for a in accepted:
        for length in (ctx_len, ctx_len + a):
            first_read = max(0, length - READ_WINDOW)
            if shift_pages + first_read // PAGE < _stale_end(history, window):
                return length
        ctx_len += a
        num_tokens += a
        history = num_tokens - 1
    return None


@pytest.mark.parametrize("prompt_len", [1, 31, 32, 33, 2047, 2048, 2049, 4113, 100005])
@pytest.mark.parametrize("pattern", ["ones", "full", "mixed"])
def test_two_pages_of_slack_keep_every_generation_read_on_held_pages(prompt_len, pattern):
    accepted = {
        "ones": [1] * 2600,
        "full": [4] * 700,
        "mixed": [1, 4, 2, 3] * 300,
    }[pattern]
    creator = _creator(_draft_config())
    window = creator._get_draft_kv_retention_window(KvCacheConfig(enable_block_reuse=False))
    # The slot starts on the page of position prompt_len: DFlashSpecMetadata.prepare.
    shift_pages = prompt_len // PAGE
    assert _first_unheld_read(prompt_len, accepted, window, shift_pages) is None


def test_the_slack_is_what_keeps_generation_reads_on_held_pages():
    """With the read window alone, a slot starting a page early reads a freed page."""
    prompt_len = 4096 + PAGE - 1
    assert _first_unheld_read(prompt_len, [1] * 2600, READ_WINDOW, prompt_len // PAGE) is not None


def _refresh(encoded_rows, shifts, batch_to_slot, divisor=2):
    num_seqs, width = len(encoded_rows), len(encoded_rows[0])
    src = torch.tensor(encoded_rows, dtype=torch.int32).reshape(1, num_seqs, 1, width)
    src = src.expand(1, num_seqs, 2, width).contiguous()
    worker = SimpleNamespace(
        _ctx_block_tables=torch.zeros((num_seqs + 1, width), dtype=torch.int32),
        _ctx_block_counts=torch.zeros(num_seqs + 1, dtype=torch.long),
        _ctx_block_cols=torch.arange(width, dtype=torch.long),
        _slot_block_shift=torch.tensor(shifts, dtype=torch.long),
        _batch_to_slot=torch.tensor(batch_to_slot, dtype=torch.long),
        _ctx_pool_idx=0,
        _ctx_block_divisor=divisor,
    )
    before = src.clone()
    assert DFlashWorker._refresh_ctx_block_tables(
        worker, SimpleNamespace(draft_kv_cache_block_offsets=src), num_seqs
    )
    assert torch.equal(src, before), "attention metadata's offsets must not be edited"
    return worker._ctx_block_tables[:num_seqs].tolist(), worker._ctx_block_counts[
        :num_seqs
    ].tolist()


def test_block_table_rows_start_on_their_slot_page():
    bad = -1
    # Row 0 (slot 2, prefill): pages 10..15 held. Row 1 (slot 0, generation from page 3):
    # pages 0..2 freed by the window, 3..5 held, the rest unallocated.
    rows = [
        [20, 22, 24, 26, 28, 30, bad, bad],
        [bad, bad, bad, 70, 72, 74, bad, bad],
    ]
    tables, counts = _refresh(rows, shifts=[3, 0, 0], batch_to_slot=[2, 0])
    assert tables[0] == [10, 11, 12, 13, 14, 15, 0, 0]
    assert counts[0] == 6
    # Column 0 of the generation slot is page 3 of its row; past the row reads as unallocated.
    assert tables[1] == [35, 36, 37, 0, 0, 0, 0, 0]
    assert counts[1] == 3


def test_block_count_reaches_past_pages_a_window_freed():
    bad = -1
    # A prefill row whose first three pages fell out of the window still holds up to page 5.
    tables, counts = _refresh(
        [[bad, bad, bad, 70, 72, 74, bad, bad]], shifts=[0], batch_to_slot=[0]
    )
    assert tables[0][3:6] == [35, 36, 37]
    assert counts[0] == 6


def test_cpp_backend_rows_count_to_the_table_edge_less_the_shift():
    # The cpp backend maps every unheld page to 0, so only the row width bounds a slot.
    tables, counts = _refresh([[0, 0, 0, 70, 72, 74, 0, 0]], shifts=[3], batch_to_slot=[0])
    assert tables[0] == [35, 36, 37, 0, 0, 0, 0, 0]
    assert counts[0] == 8 - 3


@pytest.mark.parametrize("prompt_len", [4096 + PAGE - 1, 115000, 115000 + PAGE - 1])
@pytest.mark.parametrize("generated", [0, 1, 2111, 2112, 3000, 5000])
@pytest.mark.parametrize("accepted", [1, 4])
def test_generation_rows_read_and_write_held_pages(prompt_len, generated, accepted):
    """A long prompt on a generation worker, through outputs longer than the window.

    The row holds the pages the window has not freed (python V2 backend: the freed ones
    read as BAD_PAGE_INDEX). Every column the drafter reads or writes this step must
    land on a held page of the request's own row.
    """
    block_width = 4
    window = READ_WINDOW + 2 * PAGE
    num_tokens = prompt_len + 1 + generated
    # prepare_disagg_gen_init's history before the first step, update_resources' after.
    history = prompt_len if generated == 0 else num_tokens - 1
    first_held = _stale_end(history, window)
    width = (num_tokens + 2 * READ_WINDOW) // PAGE
    row = [-1 if block < first_held else 1000 + block for block in range(width)]
    shift = prompt_len // PAGE
    assert shift * PAGE <= prompt_len  # never past where a prefill slot would write
    tables, counts = _refresh([row], shifts=[shift], batch_to_slot=[0], divisor=1)
    stored = generated + accepted  # this step's accepted tokens join the context
    columns = set(range(max(0, generated - READ_WINDOW), generated))
    columns |= set(range(max(0, stored - READ_WINDOW), stored + block_width))
    for column in sorted(columns):
        assert tables[0][column // PAGE] == 1000 + shift + column // PAGE, column
    assert counts[0] * PAGE >= stored + block_width


def _gen_worker(free_slots, req_to_slot=None, shift_host=None):
    shifts_written = []
    worker = SimpleNamespace(
        _ctx_buf_inited=True,
        _req_to_slot=dict(req_to_slot or {}),
        _req_ctx_pos={},
        _free_slots=deque(free_slots),
        _graph_dummy_id_floor=1 << 40,
        _dummy_slot=7,
        _ctx_page_size=PAGE,
        _batch_to_slot=torch.zeros(8, dtype=torch.long),
        _slot_block_shift_host=dict(shift_host or {}),
        _write_ctx_len=lambda updates: None,
        _write_slot_block_shift=shifts_written.append,
    )
    worker._assign_slot = types.MethodType(DFlashWorker._assign_slot, worker)
    return worker, shifts_written


def test_generation_slots_start_on_their_first_token_page():
    worker, shifts_written = _gen_worker(free_slots=[4, 5], req_to_slot={30: 1})
    # One prefill row (5 tokens), then generation rows of 1 + 3 drafts each: request 30
    # already has a slot; 31 and 32 are new on this disaggregated generation worker.
    seq_lens = [5, 4, 4, 4]
    positions = (
        list(range(100, 105)) + list(range(900, 904)) + list(range(1000, 1004)) + [64, 65, 66, 67]
    )
    meta = SimpleNamespace(
        request_ids=[20, 30, 31, 32],
        num_generations=3,
        seq_lens=seq_lens,
        host_position_ids=torch.tensor(positions, dtype=torch.int32),
        batch_indices_cuda=torch.zeros(8, dtype=torch.int),
        _dflash_worker=worker,
    )
    meta._first_position = types.MethodType(DFlashSpecMetadata._first_position, meta)
    DFlashSpecMetadata.prepare(meta)
    assert worker._req_to_slot == {30: 1, 31: 4, 32: 5}
    assert shifts_written == [{4: 1000 // PAGE, 5: 64 // PAGE}]
    # The prefill row is not a generation-assigned slot and maps to the dummy slot here.
    assert worker._batch_to_slot[:4].tolist() == [7, 1, 4, 5]


def test_a_recycled_slot_starts_on_page_zero_again():
    worker, shifts_written = _gen_worker(free_slots=[4], shift_host={4: 31})
    assert worker._assign_slot(40) == 4
    assert shifts_written == [{4: 0}]


class _FakeDraftCache:
    def __init__(self, history_length=0, capacity=0):
        self.history_length = history_length
        self.capacity = capacity
        self.resizes = []

    def resize(self, capacity, history_length=None):
        self.resizes.append((capacity, history_length))
        return True


def _draft_manager(cache):
    manager = SimpleNamespace(
        enable_joint_kv_cache_reuse=True,
        num_extra_kv_tokens=0,
        _mirror_draft_kv_cache=lambda req: cache,
        _resume_and_restore=lambda rid, kv_cache: True,
        _after_shrink=lambda rid, kv_cache: None,
    )
    return manager


def _context_request(gen_init, prompt_len=5000, chunk=5000):
    return SimpleNamespace(
        py_request_id=9,
        is_disagg_generation_init_state=gen_init,
        is_last_context_chunk=True,
        prompt_len=prompt_len,
        context_current_position=0,
        context_chunk_size=chunk,
        py_draft_tokens=[1, 2, 3],
    )


def test_generation_init_draft_mirror_takes_its_prompt_as_history():
    cache = _FakeDraftCache()
    batch = SimpleNamespace(context_requests=[_context_request(True)], generation_requests=[])
    KVCacheManagerV2._prepare_draft_resources(_draft_manager(cache), batch)
    assert cache.resizes == [(5000 + 3, 5000)]


def test_prefill_draft_mirror_keeps_its_history():
    cache = _FakeDraftCache()
    batch = SimpleNamespace(
        context_requests=[_context_request(False, chunk=1024)], generation_requests=[]
    )
    KVCacheManagerV2._prepare_draft_resources(_draft_manager(cache), batch)
    assert cache.resizes == [(1024 + 3, None)]
