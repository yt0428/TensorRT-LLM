# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for Round 5 GEN decode step prototypes.

Runnable both via ``pytest -m cpu_only`` inside a TRT-LLM environment and directly via
``python3 -m unittest`` on a plain CPU host without PyTorch or CuTe DSL installed.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import random
import struct
import unittest
from pathlib import Path

try:
    import pytest

    pytestmark = pytest.mark.cpu_only
except ImportError:  # pragma: no cover - plain stdlib unittest host
    pytest = None

REPO_ROOT = Path(__file__).resolve().parents[5]
DYNAMIC_TOPK_PATH = (
    REPO_ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/attention/_dynamic_radix_topk_decode.py"
)
MODELING_PATH = (
    REPO_ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/mistral_large4/"
    / "mistral_large_4_nvfp4__sm_107__dep4/modeling.py"
)
WEIGHTS_PATH = (
    REPO_ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/mistral_large4/"
    / "mistral_large_4_nvfp4__sm_107__dep4/weights.py"
)
CLUSTER_TOPK_PATH = (
    REPO_ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/attention/_cluster_radix_topk_decode.py"
)
CUTE_TOPK_PATH = (
    REPO_ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/attention/cute_dsl_radix_topk_decode.py"
)
DRAFT_KERNELS_PATH = (
    REPO_ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/dspark/_kernels.py"
)
ADP_ROUTER_PATH = REPO_ROOT / "tensorrt_llm/_torch/pyexecutor/scheduler/adp_router.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so dataclass/from __future__ annotations resolve cleanly.
    import sys

    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DYN = _load_module(DYNAMIC_TOPK_PATH, "_dynamic_radix_topk_decode_test_mod")


def _reference_row_keys(row_bits: list[int], end: int, top_k: int) -> list[int]:
    """Exact expected sorted key multiset for finite / non-negative-NaN rows."""
    pad = DYN.op_pad_count(end, DYN.PARTITIONS, top_k)
    if end <= top_k:
        valid = [
            DYN.ordered_key_from_bits(row_bits[c])
            for c in range(end)
            if DYN.ordered_key_from_bits(row_bits[c]) <= DYN.PAD_KEY
        ]
        return sorted(valid + [DYN.PAD_KEY] * (top_k - len(valid)))
    keys = [DYN.ordered_key_from_bits(row_bits[c]) for c in range(end)]
    # Columns beat pads on exact key(-FLT_MAX) ties.
    ranked = sorted([(k, 0) for k in keys] + [(DYN.PAD_KEY, 1)] * pad)[:top_k]
    return [k for k, _ in ranked]


class TestDynamicCtaAllocationPlanner(unittest.TestCase):
    """Verify length-proportional CTA allocation removes single-row long-tail skew."""

    def test_agentx_skewed_b16_eliminates_straggler_load(self) -> None:
        # AgentX d0-like skewed rank: 1x 721k, 2x 300k, 9x 80k-160k, 4x padding dummies (len=4).
        seq_lens = [721_684, 310_000, 290_000, 160_000, 140_000, 130_000, 120_000, 110_000]
        seq_lens += [100_000, 95_000, 85_000, 80_000, 4, 4, 4, 4]
        plan = DYN.plan_dynamic_cta_allocation(seq_lens, next_n=4, total_ctas=424)

        self.assertEqual(len(plan.row_ends), 64)
        self.assertEqual(plan.total_ctas_used, 424)
        # Padding dummy rows (last 16 rows) consume only 1 CTA each instead of 8.
        self.assertEqual(plan.ctas_per_row[-16:], (1,) * 16)
        # Longest row (721k) gets >= 24 CTAs (vs fixed 8 in cluster C=8 -> >=3x fewer keys/CTA!).
        self.assertGreaterEqual(plan.ctas_per_row[0], 24)
        # Fixed C=8 gave 721_684 / 8 = 90_211 keys/CTA; dynamic plan cuts peak keys/CTA by >2.5x.
        self.assertLess(plan.max_keys_per_cta, 32_000)

    def test_c360a_b24_with_padding_dummies(self) -> None:
        # C360A steady-state b24: 15 real requests (1x 700k, 14x 100k-250k) + 9 dummies.
        seq_lens = [700_000] + [100_000 + 10_000 * i for i in range(14)] + [4] * 9
        plan = DYN.plan_dynamic_cta_allocation(seq_lens, next_n=4, total_ctas=424)

        self.assertEqual(len(plan.row_ends), 96)
        self.assertEqual(plan.total_ctas_used, 424)
        self.assertEqual(plan.ctas_per_row[-36:], (1,) * 36)
        # Longest row gets >= 20 CTAs -> <= 35k keys/CTA even at b24.
        self.assertGreaterEqual(plan.ctas_per_row[0], 20)
        self.assertLess(plan.max_keys_per_cta, 36_000)

    def test_all_dummy_rows_and_b32_full_capacity(self) -> None:
        plan_dummy = DYN.plan_dynamic_cta_allocation([4] * 32, next_n=4, total_ctas=424)
        self.assertEqual(plan_dummy.ctas_per_row, (1,) * 128)
        self.assertEqual(plan_dummy.total_ctas_used, 128)

        # Single giant 1M request + 31 dummies at b32: hits max_ctas_per_row cap cleanly.
        plan_1m = DYN.plan_dynamic_cta_allocation(
            [1_048_584] + [4] * 31, next_n=4, total_ctas=424, max_ctas_per_row=32
        )
        self.assertEqual(plan_1m.ctas_per_row[:4], (32, 32, 32, 32))
        self.assertEqual(plan_1m.ctas_per_row[4:], (1,) * 124)
        self.assertEqual(plan_1m.total_ctas_used, 4 * 32 + 124)

    def test_slices_partition_causal_window_without_gaps_or_overlap(self) -> None:
        seq_lens = [4, 2048, 2049, 20_481, 131_077, 500_003]
        plan = DYN.plan_dynamic_cta_allocation(seq_lens, next_n=4, width=600_000, total_ctas=212)
        for row, end in enumerate(plan.row_ends):
            c = plan.ctas_per_row[row]
            covered = []
            prev_hi = 0
            for rk in range(c):
                lo, hi = plan.slice_bounds(row, rk)
                self.assertEqual(lo, prev_hi)
                self.assertLessEqual(lo, hi)
                if rk < c - 1 and lo < hi:
                    self.assertEqual((hi - lo) % DYN.SLICE_ALIGN, 0)
                prev_hi = hi
                covered.append((lo, hi))
            self.assertEqual(prev_hi, end)


class TestDynamicRadixTopkEmulation(unittest.TestCase):
    """Bit-exact verification of multi-CTA radix selection + Pass-2 empty-slice skip."""

    def _run_and_verify(
        self,
        logits_bits: list[list[int]],
        seq_lens: list[int],
        *,
        top_k: int = 32,
        total_ctas: int = 32,
        next_n: int = 4,
    ):
        outs, plan, stats = DYN.emulate_dynamic_radix_topk_decode(
            logits_bits,
            seq_lens,
            next_n=next_n,
            total_ctas=total_ctas,
            max_ctas_per_row=8,
            top_k=top_k,
        )
        for row, out in enumerate(outs):
            end = plan.row_ends[row]
            self.assertEqual(len(out), top_k)
            valid = [c for c in out if c >= 0]
            # C2/C3: valid columns form a contiguous prefix, -1 fills suffix.
            self.assertEqual(out[: len(valid)], valid)
            self.assertEqual(out[len(valid) :], [-1] * (top_k - len(valid)))
            # C4: distinct and within causal window [0, end).
            self.assertEqual(len(set(valid)), len(valid))
            self.assertTrue(all(0 <= c < end for c in valid))

            has_negnan = any(
                DYN.ordered_key_from_bits(logits_bits[row][c]) > DYN.NEG_INF_KEY for c in range(end)
            )
            if not has_negnan:
                got_keys = sorted(
                    [DYN.ordered_key_from_bits(logits_bits[row][c]) for c in valid]
                    + [DYN.PAD_KEY] * (top_k - len(valid))
                )
                want_keys = _reference_row_keys(logits_bits[row], end, top_k)
                self.assertEqual(got_keys, want_keys, f"mismatch on row {row} (end={end})")
        return outs, plan, stats

    def test_ordinary_ties_crowded_and_padding_dummies(self) -> None:
        rng = random.Random(2026)
        width = 256
        top_k = 32
        seq_lens = [4, 32, 33, 256]  # dummy, exact top_k, top_k+1, full
        rows = len(seq_lens) * 4

        # 1. Ordinary random floats + poison (+1e30) past causal end
        poison = DYN.float32_to_bits(1.0e30)
        logits_bits = []
        ends = DYN.row_causal_ends(seq_lens, 4, width)
        for r in range(rows):
            row = [DYN.float32_to_bits(rng.uniform(-5.0, 5.0)) for _ in range(ends[r])]
            row += [poison] * (width - ends[r])
            logits_bits.append(row)
        self._run_and_verify(logits_bits, seq_lens, top_k=top_k, total_ctas=32)

        # 2. Heavy ties (all zeros / few levels -> forces DIRECT mode across all 3 digits)
        logits_ties = []
        for r in range(rows):
            row = [
                DYN.float32_to_bits(float(rng.choice([0.0, 0.0, 0.0, 1.0, -0.0, 2.0])))
                for _ in range(ends[r])
            ]
            row += [poison] * (width - ends[r])
            logits_ties.append(row)
        self._run_and_verify(logits_ties, seq_lens, top_k=top_k, total_ctas=32)

        # 3. Crowded top-12-bit bin (forces multi-round narrowing across level 1 and 2)
        logits_dense = []
        for r in range(rows):
            row = [0x3F800000 + rng.randrange(0, 1 << 18) for _ in range(ends[r])]
            row += [poison] * (width - ends[r])
            logits_dense.append(row)
        self._run_and_verify(logits_dense, seq_lens, top_k=top_k, total_ctas=32)

    def test_empty_slice_pass2_skip_triggers_and_stays_exact(self) -> None:
        # Concentrate all top-32 values in the first slice [0..31]; all other CTA slices must skip Pass 2.
        width = 512
        top_k = 32
        seq_lens = [512, 4]
        low = DYN.float32_to_bits(-5.0)
        logits_bits = []
        for r in range(4):
            row = [DYN.float32_to_bits(100.0 - c) for c in range(top_k)] + [low] * (width - top_k)
            logits_bits.append(row)
        for r in range(4):
            logits_bits.append([DYN.float32_to_bits(float(c)) for c in range(width)])
        outs, plan, stats = self._run_and_verify(logits_bits, seq_lens, top_k=top_k, total_ctas=36)
        # 4 long rows x 8 CTAs = 32 long CTAs; 7 of 8 slices per row have zero hits -> 28 skipped!
        self.assertEqual(stats.total_long_ctas, 32)
        self.assertEqual(stats.pass2_skipped_ctas, 28)
        for r in range(4):
            self.assertEqual(sorted(outs[r]), list(range(top_k)))

    def test_special_values_and_negnan_fallback(self) -> None:
        width = 160
        top_k = 16
        seq_lens = [160, 12]
        specials = [
            0x7FC00000,  # +NaN
            0x7F800000,  # +inf
            0xFF800000,  # -inf
            0xFF7FFFFF,  # -FLT_MAX (ties with pad, columns win)
            0x80000000,  # -0.0
            0x00000000,  # +0.0
            0x3F800000,  # +1.0
        ]
        # No -NaN first:
        logits_no_negnan = [
            [specials[(r * 17 + c) % len(specials)] for c in range(width)] for r in range(8)
        ]
        self._run_and_verify(logits_no_negnan, seq_lens, top_k=top_k, total_ctas=24)

        # Case A: -NaN (0xFFC00000) among finite scores -> fp16 first pass bin 0 has <= top_k
        # items so fp16 pass settles it and -NaN is selected (matches native op).
        logits_negnan_settled = [
            [DYN.float32_to_bits(float((r * 13 + c) % 31) - 10.0) for c in range(width)]
            for r in range(8)
        ]
        for r in range(4):
            logits_negnan_settled[r][7] = 0xFFC00000
        outs_a, _, stats_a = self._run_and_verify(
            logits_negnan_settled, seq_lens, top_k=top_k, total_ctas=24
        )
        self.assertEqual(stats_a.fallback_rows, 4)
        for r in range(4):
            self.assertIn(7, outs_a[r])

        # Case B: -NaN among > top_k +NaNs -> fp16 bin 0 exceeds top_k so key pass settles it
        # and -NaN ranks last (not selected).
        logits_negnan_crowded = [list(row) for row in logits_no_negnan]
        for r in range(4):
            logits_negnan_crowded[r][7] = 0xFFC00000
        outs_b, _, stats_b = self._run_and_verify(
            logits_negnan_crowded, seq_lens, top_k=top_k, total_ctas=24
        )
        self.assertEqual(stats_b.fallback_rows, 4)
        for r in range(4):
            self.assertNotIn(7, outs_b[r])


class TestDraftAttentionEmptyTileSkipEquivalence(unittest.TestCase):
    """Verify empty-tile loop bound in draft_attention_partial_kernel is bitwise identical."""

    @staticmethod
    def _sim_partial(length: int, window: int, nsplit: int, block_t: int, skip_empty: bool):
        start = max(length - window, 0)
        keys = window // nsplit
        results = []
        for split in range(nsplit):
            m_i = -1.0e30
            l_i = 0.0
            acc = 0.0
            split_start = split * keys
            split_end = (
                min(split_start + keys, max(length - start, 0))
                if skip_empty
                else split_start + keys
            )
            tiles_run = 0
            for t0 in range(split_start, split_end, block_t):
                tiles_run += 1
                tile_scores = []
                for dt in range(block_t):
                    pos = start + t0 + dt
                    if pos < length:
                        tile_scores.append((pos % 17) * 0.1 - 0.8)
                    else:
                        tile_scores.append(float("-inf"))
                m_new = max(m_i, max(tile_scores))
                alpha = math.exp(m_i - m_new)
                ps = [math.exp(s - m_new) if s != float("-inf") else 0.0 for s in tile_scores]
                l_i = l_i * alpha + sum(ps)
                acc = acc * alpha + sum(p * ((start + t0 + dt) % 7) for dt, p in enumerate(ps))
                m_i = m_new
            results.append((m_i, l_i, acc, tiles_run))
        return results

    def test_bitwise_identical_across_all_edge_lengths(self) -> None:
        for nsplit, block_t in ((2, 128), (8, 64), (16, 128)):
            for length in (0, 1, 3, 4, 127, 128, 129, 1023, 1024, 1025, 2047, 2048, 2049, 134000):
                ref = self._sim_partial(length, 2048, nsplit, block_t, skip_empty=False)
                opt = self._sim_partial(length, 2048, nsplit, block_t, skip_empty=True)
                for (rm, rl, ra, ref_tiles), (om, ol, oa, opt_tiles) in zip(ref, opt):
                    self.assertEqual(struct.pack("<f", rm), struct.pack("<f", om))
                    self.assertEqual(struct.pack("<f", rl), struct.pack("<f", ol))
                    self.assertEqual(struct.pack("<f", ra), struct.pack("<f", oa))
                    self.assertLessEqual(opt_tiles, ref_tiles)
        # Padding dummy (length=1) at 64-head b24 config (nsplit=2, block_t=128) skips 15/16 tiles!
        dummy_tiles = sum(r[3] for r in self._sim_partial(1, 2048, 2, 128, skip_empty=True))
        self.assertEqual(dummy_tiles, 1)


class TestStaticSyntaxAndContracts(unittest.TestCase):
    """AST/syntax checks on all prototype-modified files plus pure-Python bucket-guard logic."""

    def test_all_modified_files_parse_cleanly_and_have_no_temp_ab_env_flags(self) -> None:
        paths = (
            MODELING_PATH,
            WEIGHTS_PATH,
            CLUSTER_TOPK_PATH,
            CUTE_TOPK_PATH,
            DYNAMIC_TOPK_PATH,
            DRAFT_KERNELS_PATH,
            ADP_ROUTER_PATH,
        )
        for path in paths:
            src = path.read_text(encoding="utf-8")
            ast.parse(src, filename=str(path))
            # Verify copyright header on every modified/added file.
            self.assertIn("NVIDIA CORPORATION", src[:300], f"missing copyright in {path}")

        # Verify modeling.py enables b20/b24/b32 rows (80, 96, 128) for cluster_radix_topk_decode.
        modeling_src = MODELING_PATH.read_text(encoding="utf-8")
        self.assertIn("_TOPK_VERIFY_ROWS = tuple(range(4, 129, 4))", modeling_src)
        self.assertIn("layer.w13", modeling_src)
        self.assertNotIn("layer.gate", modeling_src)
        self.assertNotIn("layer.up", modeling_src)

        # Verify weights.py merges draft_w1 + draft_w3 into draft_w13.
        weights_src = WEIGHTS_PATH.read_text(encoding="utf-8")
        self.assertIn('"draft_w13"', weights_src)
        self.assertNotIn('"draft_w1"', weights_src)
        self.assertNotIn('"draft_w3"', weights_src)

    def test_bucket_guard_heap_ordering_logic(self) -> None:
        # Standalone check mirroring DefaultADPRouter._graph_bucket + HeapVal ordering.
        buckets = (1, 2, 4, 8, 16, 24, 32)

        def graph_bucket(n: int) -> int:
            for b in buckets:
                if n <= b:
                    return b
            return n

        # Rank 0 at 16 reqs (low tokens) vs Rank 1 at 13 reqs (higher tokens):
        # next bucket for Rank 0 is 24, next bucket for Rank 1 is 16 -> Rank 1 must win!
        k0 = (graph_bucket(16 + 1), 640_000, 16, 0)
        k1 = (graph_bucket(13 + 1), 1_500_000, 13, 1)
        self.assertLess(k1, k0)
        # Within same bucket (both <= 16), lower tokens wins:
        k2 = (graph_bucket(14 + 1), 900_000, 14, 2)
        self.assertLess(k2, k1)


if __name__ == "__main__":
    unittest.main()
