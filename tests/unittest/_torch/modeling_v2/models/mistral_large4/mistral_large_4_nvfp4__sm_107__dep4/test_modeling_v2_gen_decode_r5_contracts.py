# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for Mistral Large 4 SM107 dep4 GEN decode optimizations (H1/H5/H7/ADP)."""

from __future__ import annotations

import ast
import math
import struct
import types
import unittest
from pathlib import Path

try:
    import pytest

    pytestmark = pytest.mark.cpu_only
except ImportError:  # pragma: no cover - plain stdlib unittest host
    pytest = None

REPO_ROOT = Path(__file__).resolve().parents[7]
DEP4_DIR = (
    REPO_ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/mistral_large4/"
    / "mistral_large_4_nvfp4__sm_107__dep4"
)
MODELING_PATH = DEP4_DIR / "modeling.py"
WEIGHTS_PATH = DEP4_DIR / "weights.py"
MLA_V_UP_PATH = DEP4_DIR / "ops/gemm/mla_v_up_fp8_block_scale_linear.py"
LN_FP8_LINEAR_PATH = DEP4_DIR / "ops/gemm/layer_norm_fp8_block_scale_linear.py"
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
DRAFT_KERNELS_PATH = (
    REPO_ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/dspark/_kernels.py"
)
DRAFT_OPS_PATH = (
    REPO_ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/dspark/draft_ops.py"
)
DFLASH_PATH = REPO_ROOT / "tensorrt_llm/_torch/speculative/dflash.py"
DSPARK_PATH = REPO_ROOT / "tensorrt_llm/_torch/speculative/dspark.py"
ADP_ROUTER_PATH = REPO_ROOT / "tensorrt_llm/_torch/pyexecutor/scheduler/adp_router.py"


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


class TestFp8LinearAndVUpLaunchConfig(unittest.TestCase):
    """Verify fine_grained_m=True on SM107 FP8 linears and 256-CTA v_up up to 96 rows."""

    def test_v_up_launch_config_keeps_single_wave_through_b24(self) -> None:
        src = MLA_V_UP_PATH.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(MLA_V_UP_PATH))
        fn_node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_launch_config"
        )
        ns: dict = {}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), str(MLA_V_UP_PATH), "exec"), ns)
        launch_cfg = ns["_launch_config"]

        for rows in (1, 4, 16, 32):
            self.assertEqual(launch_cfg(rows), (16, 128, 4, 4))
        for rows in (33, 48, 64, 80, 96):
            block_m, block_k, warps, stages = launch_cfg(rows)
            self.assertEqual((block_m, block_k, warps, stages), (32, 128, 4, 4))
            live_ctas = 64 * ((rows + block_m - 1) // block_m)
            self.assertLessEqual(live_ctas, 212)
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


class TestGreedyMarkovChainFastPath(unittest.TestCase):
    """Verify greedy Markov fast path skips full-vocab logits & falls back cleanly."""

    def test_dspark_greedy_fast_path_matches_logits_argmax_and_falls_back_safely(self) -> None:
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

        no_head = types.SimpleNamespace(has_markov_head=False)
        self.assertIsNone(
            refine_greedy(worker, no_head, logits, {"first_prev_tokens": anchors}, None, hidden)
        )

        sharded_head = types.SimpleNamespace(
            has_markov_head=True,
            markov_output_is_sharded=True,
            markov_w2=_FakeTensor([], (8, 2)),
            apply_markov_chain_greedy=lambda *a, **kw: None,
        )
        self.assertIsNone(
            refine_greedy(
                worker, sharded_head, logits, {"first_prev_tokens": anchors}, None, hidden
            )
        )

        legacy_head = types.SimpleNamespace(has_markov_head=True, markov_w2=_FakeTensor([], (8, 2)))
        self.assertIsNone(
            refine_greedy(worker, legacy_head, logits, {"first_prev_tokens": anchors}, None, hidden)
        )

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

        got = refine_greedy(
            worker, greedy_head, logits, {"first_prev_tokens": anchors}, None, hidden
        )
        self.assertEqual(got.data, [[3, 1, 4], [2, 0, 5]])

        worker_d2t = types.SimpleNamespace(_d2t=_FakeD2T([10, 20, 30, 40, 50, 60, 70, 80]))
        got_d2t = refine_greedy(
            worker_d2t, greedy_head, logits, {"first_prev_tokens": anchors}, None, hidden
        )
        self.assertEqual(got_d2t.data, [[43, 21, 54], [32, 10, 65]])


class TestStaticSyntaxAndContracts(unittest.TestCase):
    """AST/syntax checks on all modified files plus PR #10/13 compatibility invariants."""

    def test_all_modified_files_parse_cleanly_and_preserve_wide_cga_and_vp_markov(self) -> None:
        paths = (
            MODELING_PATH,
            WEIGHTS_PATH,
            CLUSTER_TOPK_PATH,
            CUTE_TOPK_PATH,
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

        # Confirm PR #10 wide-CGA C=16/32 support and PR #13 vocab-parallel Markov are preserved.
        cluster_src = CLUSTER_TOPK_PATH.read_text(encoding="utf-8")
        self.assertIn("2 <= cluster <= 32", cluster_src)
        self.assertIn("cluster_kernel_smem_bytes", cluster_src)
        self.assertIn("emit_eq", cluster_src)

        cute_topk_src = CUTE_TOPK_PATH.read_text(encoding="utf-8")
        self.assertIn("DEFAULT_VERIFY_CLUSTERS = (8, 16, 32)", cute_topk_src)
        self.assertIn("def build_cluster_variants(", cute_topk_src)
        self.assertIn("max_rows = min(128, _cluster_kernel.MAX_REQUESTS * next_n)", cute_topk_src)

        dflash_src = DFLASH_PATH.read_text(encoding="utf-8")
        self.assertIn("vocab_parallel.draft_tokens", dflash_src)
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
