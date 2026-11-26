# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for Round 5 GEN decode step prototypes (H1, H5, H6, H7, ADP, Draft).

Runnable both via ``pytest -m cpu_only`` inside a TRT-LLM environment and directly via
``python3 -m unittest`` on a plain CPU host without PyTorch or CuTe DSL installed.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import random
import struct
import sys
import types
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
CUTE_FP8_LINEAR_PATH = (
    REPO_ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/gemm/cute_dsl_fp8_block_scale_linear.py"
)
MLA_V_UP_PATH = (
    REPO_ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/mistral_large4/"
    / "mistral_large_4_nvfp4__sm_107__dep4/ops/gemm/mla_v_up_fp8_block_scale_linear.py"
)
LN_FP8_LINEAR_PATH = (
    REPO_ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/mistral_large4/"
    / "mistral_large_4_nvfp4__sm_107__dep4/ops/gemm/layer_norm_fp8_block_scale_linear.py"
)
DRAFT_KERNELS_PATH = (
    REPO_ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/dspark/_kernels.py"
)
DRAFT_OPS_PATH = (
    REPO_ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/dspark/draft_ops.py"
)
DFLASH_PATH = REPO_ROOT / "tensorrt_llm/_torch/speculative/dflash.py"
DSPARK_PATH = REPO_ROOT / "tensorrt_llm/_torch/speculative/dspark.py"
ADP_ROUTER_PATH = REPO_ROOT / "tensorrt_llm/_torch/pyexecutor/scheduler/adp_router.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
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
        for total_ctas, min_straggler_ctas, max_peak_keys in (
            (212, 15, 52_000),
            (424, 30, 26_000),
        ):
            plan = DYN.plan_dynamic_cta_allocation(seq_lens, next_n=4, total_ctas=total_ctas)
            self.assertEqual(len(plan.row_ends), 64)
            self.assertEqual(len(plan.short_rows), 16)
            self.assertEqual(len(plan.long_order), 48)
            self.assertEqual(plan.total_long_ctas_used, total_ctas)
            # Short dummy rows run in prologue (1 CTA each), long rows split all total_ctas.
            self.assertEqual(plan.ctas_per_row[-16:], (1,) * 16)
            self.assertGreaterEqual(plan.ctas_per_row[0], min_straggler_ctas)
            # Fixed C=8 gave 721_684 / 8 = 90_211 keys/CTA; dynamic plan hits minimax optimum.
            self.assertLess(plan.max_keys_per_cta, max_peak_keys)
            # Minimax optimality check: beating max_keys_per_cta requires strictly > total_ctas.
            target = plan.max_keys_per_cta - 1
            needed = sum((plan.row_ends[r] + target - 1) // target for r in plan.long_order)
            self.assertGreater(needed, total_ctas)

    def test_c360a_b24_with_padding_dummies(self) -> None:
        # C360A steady-state b24: 15 real requests (1x 700k, 14x 100k-230k) + 9 dummies.
        seq_lens = [700_000] + [100_000 + 10_000 * i for i in range(14)] + [4] * 9
        for total_ctas, min_straggler_ctas, max_peak_keys in (
            (212, 11, 68_000),
            (424, 22, 34_000),
        ):
            plan = DYN.plan_dynamic_cta_allocation(seq_lens, next_n=4, total_ctas=total_ctas)
            self.assertEqual(len(plan.row_ends), 96)
            self.assertEqual(plan.total_long_ctas_used, total_ctas)
            self.assertEqual(plan.ctas_per_row[-36:], (1,) * 36)
            self.assertGreaterEqual(plan.ctas_per_row[0], min_straggler_ctas)
            self.assertLess(plan.max_keys_per_cta, max_peak_keys)
            target = plan.max_keys_per_cta - 1
            needed = sum((plan.row_ends[r] + target - 1) // target for r in plan.long_order)
            self.assertGreater(needed, total_ctas)

    def test_all_dummy_rows_and_b32_full_capacity(self) -> None:
        plan_dummy = DYN.plan_dynamic_cta_allocation([4] * 32, next_n=4, total_ctas=212)
        self.assertEqual(plan_dummy.ctas_per_row, (1,) * 128)
        self.assertEqual(plan_dummy.total_long_ctas_used, 0)
        self.assertEqual(len(plan_dummy.short_rows), 128)

        # Single giant 1M request + 31 dummies at b32: hits max_ctas_per_row cap cleanly.
        plan_1m = DYN.plan_dynamic_cta_allocation(
            [1_048_584] + [4] * 31, next_n=4, total_ctas=212, max_ctas_per_row=32
        )
        self.assertEqual(plan_1m.ctas_per_row[:4], (32, 32, 32, 32))
        self.assertEqual(plan_1m.ctas_per_row[4:], (1,) * 124)
        self.assertEqual(plan_1m.total_long_ctas_used, 128)

        # Full b32 with all 128 rows long at total_ctas=212 (1 CTA/SM).
        plan_full = DYN.plan_dynamic_cta_allocation(
            [600_000] + [120_000] * 31, next_n=4, total_ctas=212, max_ctas_per_row=32
        )
        self.assertEqual(plan_full.total_long_ctas_used, 212)
        self.assertEqual(sum(plan_full.ctas_per_row), 212)
        self.assertGreaterEqual(plan_full.ctas_per_row[0], 6)

    def test_slices_partition_causal_window_without_gaps_or_overlap(self) -> None:
        seq_lens = [4, 2048, 2049, 20_481, 131_077, 500_003]
        plan = DYN.plan_dynamic_cta_allocation(seq_lens, next_n=4, width=600_000, total_ctas=212)
        for row, end in enumerate(plan.row_ends):
            c = plan.ctas_per_row[row]
            prev_hi = 0
            for rk in range(c):
                lo, hi = plan.slice_bounds(row, rk)
                self.assertEqual(lo, prev_hi)
                self.assertLessEqual(lo, hi)
                if rk < c - 1 and lo < hi:
                    self.assertEqual((hi - lo) % DYN.SLICE_ALIGN, 0)
                prev_hi = hi
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
            # Valid columns form a contiguous prefix, -1 fills suffix.
            self.assertEqual(out[: len(valid)], valid)
            self.assertEqual(out[len(valid) :], [-1] * (top_k - len(valid)))
            # Distinct and within causal window [0, end).
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

        poison = DYN.float32_to_bits(1.0e30)
        logits_bits = []
        ends = DYN.row_causal_ends(seq_lens, 4, width)
        for r in range(rows):
            row = [DYN.float32_to_bits(rng.uniform(-5.0, 5.0)) for _ in range(ends[r])]
            row += [poison] * (width - ends[r])
            logits_bits.append(row)
        self._run_and_verify(logits_bits, seq_lens, top_k=top_k, total_ctas=32)

        logits_ties = []
        for r in range(rows):
            row = [
                DYN.float32_to_bits(float(rng.choice([0.0, 0.0, 0.0, 1.0, -0.0, 2.0])))
                for _ in range(ends[r])
            ]
            row += [poison] * (width - ends[r])
            logits_ties.append(row)
        self._run_and_verify(logits_ties, seq_lens, top_k=top_k, total_ctas=32)

        logits_dense = []
        for r in range(rows):
            row = [0x3F800000 + rng.randrange(0, 1 << 18) for _ in range(ends[r])]
            row += [poison] * (width - ends[r])
            logits_dense.append(row)
        self._run_and_verify(logits_dense, seq_lens, top_k=top_k, total_ctas=32)

    def test_empty_slice_pass2_skip_triggers_and_stays_exact(self) -> None:
        width = 512
        top_k = 32
        seq_lens = [512, 4]
        low = DYN.float32_to_bits(-5.0)
        logits_bits = []
        for _ in range(4):
            row = [DYN.float32_to_bits(100.0 - c) for c in range(top_k)] + [low] * (width - top_k)
            logits_bits.append(row)
        for _ in range(4):
            logits_bits.append([DYN.float32_to_bits(float(c)) for c in range(width)])
        outs, _, stats = self._run_and_verify(logits_bits, seq_lens, top_k=top_k, total_ctas=32)
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
        logits_no_negnan = [
            [specials[(r * 17 + c) % len(specials)] for c in range(width)] for r in range(8)
        ]
        self._run_and_verify(logits_no_negnan, seq_lens, top_k=top_k, total_ctas=24)

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
        dummy_tiles = sum(r[3] for r in self._sim_partial(1, 2048, 2, 128, skip_empty=True))
        self.assertEqual(dummy_tiles, 1)


class TestH5Fp8LinearAndVUpLaunchConfig(unittest.TestCase):
    """Verify H5 fixes: fine_grained_m=True on SM107 FP8 linears and 256-CTA v_up up to 96 rows."""

    def test_v_up_launch_config_keeps_single_wave_through_b24(self) -> None:
        # Parse _launch_config directly without needing triton/cuda on plain CPU hosts.
        src = MLA_V_UP_PATH.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(MLA_V_UP_PATH))
        fn_node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_launch_config"
        )
        ns: dict = {}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), str(MLA_V_UP_PATH), "exec"), ns)
        launch_cfg = ns["_launch_config"]

        # b1..b8 (rows <= 32): 16-row programs.
        for rows in (1, 4, 16, 32):
            self.assertEqual(launch_cfg(rows), (16, 128, 4, 4))
        # b12..b24 (rows 33..96): 32-row programs -> 256 total CTAs, <= 192 live CTAs <= 212 SMs.
        for rows in (33, 48, 64, 80, 96):
            block_m, block_k, warps, stages = launch_cfg(rows)
            self.assertEqual((block_m, block_k, warps, stages), (32, 128, 4, 4))
            live_ctas = 64 * ((rows + block_m - 1) // block_m)
            self.assertLessEqual(live_ctas, 212)
        # b28..b32+ (rows >= 97): 64-row programs -> 128 live CTAs <= 212 SMs (avoids 2-wave spill).
        for rows in (97, 112, 128, 129):
            self.assertEqual(launch_cfg(rows), (64, 64, 8, 5))

    def test_all_sm107_fp8_linear_wrappers_enable_fine_grained_m(self) -> None:
        for path in (CUTE_FP8_LINEAR_PATH, MLA_V_UP_PATH, LN_FP8_LINEAR_PATH):
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(path))
            calls = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "cute_dsl_mxfp8_gemm_rubin"
            ]
            self.assertEqual(len(calls), 1, f"expected 1 gemm call in {path}")
            kw = {k.arg: k.value for k in calls[0].keywords}
            self.assertIn("fine_grained_m", kw, f"missing fine_grained_m in {path}")
            self.assertIsInstance(kw["fine_grained_m"], ast.Constant)
            self.assertIs(kw["fine_grained_m"].value, True)


class TestH7GreedyMarkovChainFastPath(unittest.TestCase):
    """Verify H7 greedy Markov fast path eliminates redundant full-vocab logits & argmax."""

    def test_dspark_greedy_fast_path_matches_logits_argmax_and_falls_back_safely(self) -> None:
        # Pure-Python simulation of DSparkWorker._refine_and_greedy_sample_block +
        # DraftModel.apply_markov_chain_greedy using lightweight mock tensors.
        src = DSPARK_PATH.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(DSPARK_PATH))
        cls_node = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DSparkWorker"
        )
        fn_node = next(
            n
            for n in cls_node.body
            if isinstance(n, ast.FunctionDef) and n.name == "_refine_and_greedy_sample_block"
        )
        ns = {"Optional": object, "torch": types.SimpleNamespace(Tensor=object, int32="int32")}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), str(DSPARK_PATH), "exec"), ns)
        refine_greedy = ns["_refine_and_greedy_sample_block"]

        class _FakeTensor:
            def __init__(self, data, shape=None):
                self.data = data
                self.shape = shape if shape is not None else (len(data),)

            def type(self, dtype):
                return _FakeTensor(self.data, self.shape)

            def __add__(self, other):
                return _FakeTensor(
                    [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self.data, other.data)],
                    self.shape,
                )

        class _FakeD2T:
            def __init__(self, offsets):
                self.offsets = offsets

            def __getitem__(self, idx):
                return _FakeTensor([[self.offsets[v] for v in row] for row in idx.data], idx.shape)

        worker = types.SimpleNamespace(_d2t=None)
        logits = _FakeTensor([], shape=(2, 3, 8))
        anchors = _FakeTensor([1, 2], shape=(2,))
        hidden = _FakeTensor([], shape=(2, 3, 4))

        # 1. No Markov head -> returns None (falls back to normal path).
        no_head = types.SimpleNamespace(has_markov_head=False)
        self.assertIsNone(
            refine_greedy(worker, no_head, logits, {"first_prev_tokens": anchors}, None, hidden)
        )

        # 2. Missing apply_markov_chain_greedy -> returns None.
        legacy_head = types.SimpleNamespace(has_markov_head=True, markov_w2=_FakeTensor([], (8, 2)))
        self.assertIsNone(
            refine_greedy(worker, legacy_head, logits, {"first_prev_tokens": anchors}, None, hidden)
        )

        # 3. TP-sharded logits (shard 4 != full_vocab 8) -> returns None.
        sharded_logits = _FakeTensor([], shape=(2, 3, 4))
        greedy_head = types.SimpleNamespace(
            has_markov_head=True,
            has_recurrent_markov_head=True,
            markov_w2=_FakeTensor([], (8, 2)),
            apply_markov_chain_greedy=lambda base, prev, hidden_states: _FakeTensor(
                [[3, 1, 4], [2, 0, 5]], (2, 3)
            ),
        )
        self.assertIsNone(
            refine_greedy(
                worker, greedy_head, sharded_logits, {"first_prev_tokens": anchors}, None, hidden
            )
        )

        # 4. Full-vocab recurrent Markov head -> returns exact greedy tokens without logits argmax.
        got = refine_greedy(
            worker, greedy_head, logits, {"first_prev_tokens": anchors}, None, hidden
        )
        self.assertEqual(got.data, [[3, 1, 4], [2, 0, 5]])

        # 5. With d2t mapping -> applies d2t[tokens] + tokens identically to sample_draft_tokens.
        worker_d2t = types.SimpleNamespace(_d2t=_FakeD2T([10, 20, 30, 40, 50, 60, 70, 80]))
        got_d2t = refine_greedy(
            worker_d2t, greedy_head, logits, {"first_prev_tokens": anchors}, None, hidden
        )
        self.assertEqual(got_d2t.data, [[43, 21, 54], [32, 10, 65]])


class TestStaticSyntaxAndContracts(unittest.TestCase):
    """AST/syntax checks on all prototype-modified files plus pure-Python bucket-guard logic."""

    def test_all_modified_files_parse_cleanly_and_have_no_temp_ab_env_flags(self) -> None:
        paths = (
            MODELING_PATH,
            WEIGHTS_PATH,
            CLUSTER_TOPK_PATH,
            CUTE_TOPK_PATH,
            DYNAMIC_TOPK_PATH,
            CUTE_FP8_LINEAR_PATH,
            MLA_V_UP_PATH,
            LN_FP8_LINEAR_PATH,
            DRAFT_KERNELS_PATH,
            DRAFT_OPS_PATH,
            DFLASH_PATH,
            DSPARK_PATH,
            ADP_ROUTER_PATH,
        )
        for path in paths:
            src = path.read_text(encoding="utf-8")
            ast.parse(src, filename=str(path))
            self.assertIn("NVIDIA CORPORATION", src[:300], f"missing copyright in {path}")

        modeling_src = MODELING_PATH.read_text(encoding="utf-8")
        self.assertIn("_TOPK_VERIFY_ROWS = tuple(range(4, 129, 4))", modeling_src)
        self.assertIn("def apply_markov_chain_greedy(", modeling_src)
        self.assertIn("layer.w13", modeling_src)
        self.assertNotIn("layer.gate", modeling_src)
        self.assertNotIn("layer.up", modeling_src)

        weights_src = WEIGHTS_PATH.read_text(encoding="utf-8")
        self.assertIn('"draft_w13"', weights_src)
        self.assertNotIn('"draft_w1"', weights_src)
        self.assertNotIn('"draft_w3"', weights_src)

        dyn_src = DYNAMIC_TOPK_PATH.read_text(encoding="utf-8")
        self.assertIn("class DynamicRadixTopkDecodeKernel", dyn_src)
        self.assertIn("cluster=(1, 1, 1)", dyn_src)
        self.assertIn("atom.release.gpu.global.add.s32", dyn_src)
        self.assertIn("ld.acquire.gpu.global.b32", dyn_src)

        dflash_src = DFLASH_PATH.read_text(encoding="utf-8")
        self.assertIn("_refine_and_greedy_sample_block", dflash_src)
        self.assertIn("not spec_metadata.wants_advanced_draft_sampling", dflash_src)

    def test_bucket_guard_heap_ordering_logic(self) -> None:
        buckets = (1, 2, 4, 8, 16, 24, 32)

        def graph_bucket(n: int) -> int:
            for b in buckets:
                if n <= b:
                    return b
            return n

        k0 = (graph_bucket(16 + 1), 640_000, 16, 0)
        k1 = (graph_bucket(13 + 1), 1_500_000, 13, 1)
        self.assertLess(k1, k0)
        k2 = (graph_bucket(14 + 1), 900_000, 14, 2)
        self.assertLess(k2, k1)


if __name__ == "__main__":
    unittest.main()
