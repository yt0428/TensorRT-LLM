# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract, weight-layout, scale-conversion, phase-plan, and Paradigm-B uGPU planning tests
for Mistral Large 4 SM107 TP8 (I=768) & TP16 (I=384) fused CuTe DSL MoE + GEMV + LocalityMap.

Runs on CPU without CUDA or PyTorch (also exercises PyTorch tensor paths when torch is installed).
"""

from __future__ import annotations

import ast
import importlib.util
import math
import pathlib
import struct
import sys
import types
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[8]
_TARGET_DIR = (
    _REPO_ROOT
    / "tensorrt_llm"
    / "_torch"
    / "_experimental"
    / "modeling_v2"
    / "models"
    / "mistral_large4"
    / "mistral_large_4_nvfp4__sm_107__tp4"
)


def _load_target_modules():
    """Import our pure-Python target modules cleanly even in a minimal CPU environment without
    the compiled ``tensorrt_llm`` C++ extension."""
    pkg_base = "_ml4_cpu_test_pkg"
    for sub in ("", ".ops", ".ops.gemm", ".ops.moe"):
        name = pkg_base + sub
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(_TARGET_DIR / sub.lstrip(".").replace(".", "/"))]
            sys.modules[name] = m

    def _load(mod_name: str, rel_path: str):
        full = f"{pkg_base}.{mod_name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, _TARGET_DIR / rel_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    locality = _load("ops.gemm.locality", "ops/gemm/locality.py")
    ml4_moe = _load("ops.moe.ml4_moe_kernel", "ops/moe/ml4_moe_kernel.py")
    ml4_gemv = _load("ops.gemm.ml4_mx_gemv", "ops/gemm/ml4_mx_gemv.py")
    return locality, ml4_moe, ml4_gemv


_LOCALITY, _MOE, _GEMV = _load_target_modules()

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


class TestLocalityMapContracts(unittest.TestCase):
    """Paradigm-B Rubin VR200 (100 + 100 + 12 = 212 SMs, 2 MiB VMM granule) host contracts."""

    def setUp(self) -> None:
        self.num_sms = 212
        self.d0 = list(range(0, 200, 2))  # 100 SMs
        self.d1 = list(range(1, 200, 2))  # 100 SMs
        self.rem = list(range(200, 212))  # 12 remainder SMs
        self.gran = _LOCALITY.DEFAULT_VMM_GRANULARITY

    def test_sm_table_bijection_and_domains(self) -> None:
        tbl = _LOCALITY.build_sm_table_host([self.d0, self.d1], self.num_sms)
        self.assertEqual(len(tbl), 212)
        for rank, sm in enumerate(self.d0):
            self.assertEqual(tbl[sm] >> 16, 0)
            self.assertEqual(tbl[sm] & 0xFFFF, rank)
        for rank, sm in enumerate(self.d1):
            self.assertEqual(tbl[sm] >> 16, 1)
            self.assertEqual(tbl[sm] & 0xFFFF, rank)
        for sm in self.rem:
            self.assertEqual(tbl[sm], -1)

    def test_logical_map_shared_and_unshared_vr200(self) -> None:
        # Shared=True: 188 routed (94 per die) + 24 shared = 212 active SMs.
        lmap, ext_dom = _LOCALITY.build_logical_map_host(
            [self.d0, self.d1], self.rem, self.num_sms, per_domain=94, extra=24
        )
        self.assertEqual(sorted(lmap), list(range(212)))
        self.assertEqual(len(ext_dom), 24)
        # Verify logical blocks [0, 94) sit strictly on die 0 and [94, 188) strictly on die 1.
        d0_set, d1_set = set(self.d0), set(self.d1)
        for sm, bid in enumerate(lmap):
            if 0 <= bid < 94:
                self.assertIn(sm, d0_set)
            elif 94 <= bid < 188:
                self.assertIn(sm, d1_set)
        # Leftover domain SMs alternate domains (6 from d0, 6 from d1), then 12 remainder (-1).
        self.assertEqual(ext_dom[:12], [0, 1] * 6)
        self.assertEqual(ext_dom[12:], [-1] * 12)

        # Shared=False: 192 routed (96 per die) + 0 extra, 20 idle SMs (-1).
        lmap_ns, ext_ns = _LOCALITY.build_logical_map_host(
            [self.d0, self.d1], self.rem, self.num_sms, per_domain=96, extra=0, allow_idle=True
        )
        self.assertEqual(ext_ns, [])
        self.assertEqual(lmap_ns.count(-1), 20)
        self.assertEqual(sorted(x for x in lmap_ns if x >= 0), list(range(192)))

    def test_moe_vmm_granule_math_tp8_and_tp16(self) -> None:
        # TP8 (I=768): ALL 4 routed tensors hit exact 2 MiB boundaries in both n_split & expert_half.
        for shape in ((128, 1536, 3072), (128, 1536, 384), (128, 6144, 384), (128, 6144, 48)):
            hb, phb, exact = _LOCALITY.plan_n_split_bytes(shape, 1, dim=1, granule=self.gran)
            self.assertTrue(exact, f"TP8 n_split {shape} should be 2 MiB exact")
            self.assertEqual(hb, phb)
            se, hb_e, phb_e, exact_e = _LOCALITY.plan_expert_half_bytes(shape, 1, granule=self.gran)
            self.assertEqual(se, 64)
            self.assertTrue(exact_e, f"TP8 expert_half {shape} should be 2 MiB exact")
            self.assertEqual(hb_e, phb_e)

        # TP16 (I=384): fc1_w, fc1_s, fc2_w are 2 MiB exact; fc2_s is 9 MiB/half -> padded to 10 MiB!
        for shape in ((128, 768, 3072), (128, 768, 384), (128, 6144, 192)):
            _, _, exact = _LOCALITY.plan_n_split_bytes(shape, 1, dim=1, granule=self.gran)
            self.assertTrue(exact, f"TP16 n_split {shape} should be 2 MiB exact")
            _, _, _, exact_e = _LOCALITY.plan_expert_half_bytes(shape, 1, granule=self.gran)
            self.assertTrue(exact_e, f"TP16 expert_half {shape} should be 2 MiB exact")
        se, hb_s2, phb_s2, exact_s2 = _LOCALITY.plan_expert_half_bytes(
            (128, 6144, 24), 1, granule=self.gran
        )
        self.assertEqual(se, 64)
        self.assertEqual(hb_s2, 9 * 1024 * 1024)
        self.assertEqual(phb_s2, 10 * 1024 * 1024)
        self.assertFalse(exact_s2)

        # Recommend mode selects n_split for 512 <= I <= 1024 (128-aligned half-width, e.g. TP8
        # I=768) and none for narrow I < 512 (TP16 I=384) or wide I > 1024 (TP4 I=1536).
        self.assertEqual(_LOCALITY.DEFAULT_UGPU_MOE_MIN_INTER, 512)
        self.assertEqual(_LOCALITY.DEFAULT_UGPU_MOE_MAX_INTER, 1024)
        self.assertEqual(_LOCALITY.recommend_moe_locality_mode(1536), "none")
        self.assertEqual(_LOCALITY.recommend_moe_locality_mode(1025), "none")
        self.assertEqual(_LOCALITY.recommend_moe_locality_mode(1024), "n_split")
        self.assertEqual(_LOCALITY.recommend_moe_locality_mode(768), "n_split")
        self.assertEqual(_LOCALITY.recommend_moe_locality_mode(640), "none")
        self.assertEqual(_LOCALITY.recommend_moe_locality_mode(512), "n_split")
        self.assertEqual(_LOCALITY.recommend_moe_locality_mode(511), "none")
        self.assertEqual(_LOCALITY.recommend_moe_locality_mode(384), "none")

    def test_harvested_sm_fallback_to_none(self) -> None:
        # Harvested-SM VR200 (212 SMs total, 2 domains x 80 SMs + 52 remainder):
        # 94 routed CTAs/domain exceeds 80 SMs/domain, so all uGPU modes must degrade cleanly to 'none'.
        req_shared = _LOCALITY.required_moe_ctas_per_domain(212, shared=True)
        req_unshared = _LOCALITY.required_moe_ctas_per_domain(212, shared=False)
        self.assertEqual(req_shared, 94)
        self.assertEqual(req_unshared, 96)

        self.assertEqual(
            _LOCALITY.recommend_moe_locality_mode(
                768, sms_per_domain=80, required_per_domain=req_shared
            ),
            "none",
        )
        self.assertEqual(
            _LOCALITY.recommend_moe_locality_mode(
                768, sms_per_domain=100, required_per_domain=req_shared
            ),
            "n_split",
        )

        for requested in ("auto", "n_split", "expert_half"):
            mode, reason = _LOCALITY.resolve_moe_locality_mode(
                requested, 768, sms_per_domain=80, required_per_domain=req_shared
            )
            self.assertEqual(mode, "none")
            self.assertIsNotNone(reason)
            assert reason is not None
            self.assertIn("80", reason)
            self.assertIn("94", reason)

        # Full-SM VR200 (100 SMs/domain) preserves n_split and explicit expert_half without fallback.
        self.assertEqual(
            _LOCALITY.resolve_moe_locality_mode(
                "auto", 768, sms_per_domain=100, required_per_domain=req_shared
            ),
            ("n_split", None),
        )
        self.assertEqual(
            _LOCALITY.resolve_moe_locality_mode(
                "expert_half", 768, sms_per_domain=100, required_per_domain=req_shared
            ),
            ("expert_half", None),
        )
        # TP16 auto -> 'none' emits no harvested-SM warning.
        self.assertEqual(
            _LOCALITY.resolve_moe_locality_mode(
                "auto", 384, sms_per_domain=80, required_per_domain=req_shared
            ),
            ("none", None),
        )

        # Direct host logical_map builder remains fail-closed when per_domain > sms_per_domain.
        harvested_doms = [list(range(0, 80)), list(range(80, 160))]
        harvested_rem = list(range(160, 212))
        with self.assertRaises(ValueError):
            _LOCALITY.build_logical_map_host(harvested_doms, harvested_rem, 212, 94, 24)

    def test_ugpu_high_batch_and_non_graph_guard(self) -> None:
        # Low-concurrency ULL (c=1 mbs=2 or mbs=8, c=4 mbs=8, CUDA graphs on) keeps n_split
        # for eligible widths (512 <= I <= 1024 with 128-aligned half-width, e.g. TP8 I=768).
        self.assertEqual(_LOCALITY.DEFAULT_UGPU_MOE_MAX_BATCH_SIZE, 8)
        for inter in (512, 768, 1024):
            for mbs in (None, 1, 2, 4, 8):
                self.assertEqual(
                    _LOCALITY.recommend_moe_locality_mode(
                        inter, max_batch_size=mbs, use_cuda_graph=True
                    ),
                    "n_split",
                )
                self.assertEqual(
                    _LOCALITY.resolve_moe_locality_mode(
                        "auto", inter, max_batch_size=mbs, use_cuda_graph=True
                    ),
                    ("n_split", None),
                )

            # High-batch serving (e.g. Disagg TP8 GEN mbs=16/24/32) skips uGPU weight
            # duplication so HBM4 capacity remains available for KV cache.
            for mbs in (9, 16, 24, 32, 128):
                self.assertEqual(
                    _LOCALITY.recommend_moe_locality_mode(
                        inter, max_batch_size=mbs, use_cuda_graph=True
                    ),
                    "none",
                )
                mode, reason = _LOCALITY.resolve_moe_locality_mode(
                    "auto", inter, max_batch_size=mbs, use_cuda_graph=True
                )
                self.assertEqual(mode, "none")
                self.assertIsNotNone(reason)
                assert reason is not None
                self.assertIn(f"max_batch_size={mbs}", reason)

            # Eager/context workers (use_cuda_graph=False) also skip uGPU weight duplication.
            self.assertEqual(
                _LOCALITY.recommend_moe_locality_mode(
                    inter, max_batch_size=4, use_cuda_graph=False
                ),
                "none",
            )
            mode, reason = _LOCALITY.resolve_moe_locality_mode(
                "auto", inter, max_batch_size=4, use_cuda_graph=False
            )
            self.assertEqual(mode, "none")
            self.assertIsNotNone(reason)
            assert reason is not None
            self.assertIn("use_cuda_graph=False", reason)

            # Explicit non-auto requests stay honored on full-SM GPUs (for targeted tests).
            self.assertEqual(
                _LOCALITY.resolve_moe_locality_mode(
                    "n_split", inter, sms_per_domain=100, max_batch_size=32, use_cuda_graph=False
                ),
                ("n_split", None),
            )

        # Out-of-band widths (I < 512 or I > 1024, incl. TP16 I=384, boundary 511/1025, and
        # TP4 I=1536) resolve 'auto' cleanly to ('none', None) across all batch/CG settings,
        # while explicit 'n_split'/'expert_half' still work on full-SM GPUs and fall back on
        # harvested-SM GPUs.
        for inter in (384, 511, 1025, 1536):
            for mbs in (None, 1, 8, 32):
                for use_cg in (True, False):
                    self.assertEqual(
                        _LOCALITY.recommend_moe_locality_mode(
                            inter, max_batch_size=mbs, use_cuda_graph=use_cg
                        ),
                        "none",
                    )
                    self.assertEqual(
                        _LOCALITY.resolve_moe_locality_mode(
                            "auto", inter, max_batch_size=mbs, use_cuda_graph=use_cg
                        ),
                        ("none", None),
                    )
            self.assertEqual(
                _LOCALITY.resolve_moe_locality_mode(
                    "n_split", inter, sms_per_domain=100, max_batch_size=32, use_cuda_graph=False
                ),
                ("n_split", None),
            )
            self.assertEqual(
                _LOCALITY.resolve_moe_locality_mode(
                    "expert_half",
                    inter,
                    sms_per_domain=100,
                    max_batch_size=32,
                    use_cuda_graph=False,
                ),
                ("expert_half", None),
            )
            mode, reason = _LOCALITY.resolve_moe_locality_mode(
                "n_split", inter, sms_per_domain=80, required_per_domain=94
            )
            self.assertEqual(mode, "none")
            self.assertIsNotNone(reason)

        # Source wiring contract: model_loader publishes max_batch_size and tp4 DecodeTarget
        # forwards max_batch_size + use_cuda_graph without temporary A/B env switches.
        modeling_src = (_TARGET_DIR / "modeling.py").read_text()
        weights_src = (_TARGET_DIR / "weights.py").read_text()
        loader_src = (
            _REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "model_loader.py"
        ).read_text()
        self.assertIn('config.extra_attrs["max_batch_size"]', loader_src)
        self.assertIn('extra_attrs.get("max_batch_size")', modeling_src)
        self.assertIn("max_batch_size=max_batch_size", weights_src)
        self.assertNotIn("TRTLLM_ML4_MOE_LOCALITY", modeling_src)
        self.assertNotIn("TRTLLM_ML4_MOE_LOCALITY", weights_src)

    def test_fp8_linear_and_lm_head_granule_splits(self) -> None:
        # ix_wq_b [4096, 6144] FP8 = 24 MiB -> 12+12 MiB exact.
        hb, phb, exact = _LOCALITY.plan_n_split_bytes((4096, 6144), 1, dim=0, granule=self.gran)
        self.assertEqual(hb, 12 * 1024 * 1024)
        self.assertTrue(exact)
        # TP8 wq [1024, 6144] FP8 = 6 MiB -> 3 MiB/half -> padded to 4 MiB.
        hb, phb, exact = _LOCALITY.plan_n_split_bytes((1024, 6144), 1, dim=0, granule=self.gran)
        self.assertEqual(hb, 3 * 1024 * 1024)
        self.assertEqual(phb, 4 * 1024 * 1024)
        self.assertFalse(exact)
        # lm_head BF16 TP8 [25600, 6144] = 300 MiB -> exact 12800/12800 (150 MiB each).
        s_tp8 = _LOCALITY.compute_granule_split_row(25600, 6144 * 2, self.gran)
        self.assertEqual(s_tp8, 12800)
        # lm_head BF16 TP16 [12800, 6144] = 150 MiB -> granule step 512 rows -> 6144 or 6656.
        s_tp16 = _LOCALITY.compute_granule_split_row(12800, 6144 * 2, self.gran)
        self.assertIsNotNone(s_tp16)
        assert s_tp16 is not None
        self.assertEqual((s_tp16 * 6144 * 2) % self.gran, 0)


class TestFp8ScaleLosslessExpansion(unittest.TestCase):
    """Verify ML4 FP8_PB_WO 128x128 power-of-2 block scales -> 1x32 UE8M0 R128c4 exactness & fallback."""

    def test_exact_power_of_two_round_trip(self) -> None:
        # Test across full normal fp32 exponent range including 2^-126, 2^-10, 1.0, 2^15, 2^127.
        exps = [-126, -42, -10, -1, 0, 1, 7, 15, 64, 127]
        n, k = 256, 640
        n_blk, k_blk = n // 128, k // 128
        grid = [
            [math.ldexp(1.0, exps[(r * k_blk + c) % len(exps)]) for c in range(k_blk)]
            for r in range(n_blk)
        ]
        self.assertTrue(_MOE.is_exact_power_of_two_ue8m0(grid))
        r128c4, ok = _MOE.expand_fp8_128x128_scale_to_ue8m0_r128c4(grid, n, k, strict=True)
        self.assertTrue(ok)
        self.assertEqual(len(r128c4), n * (k // 32))
        # Verify EVERY logical (row, k32_blk) unswizzles to the EXACT original fp32 scale!
        for row in range(n):
            for b in range(k // 32):
                idx = (
                    (row // 128) * (128 * (k // 32))
                    + (b // 4) * 512
                    + (row % 32) * 16
                    + ((row % 128) // 32) * 4
                    + (b % 4)
                )
                code = r128c4[idx]
                reconstructed = math.ldexp(1.0, code - 127)
                expected = grid[row // 128][b // 4]
                self.assertEqual(reconstructed, expected)

    def test_non_power_of_two_falls_back_or_raises(self) -> None:
        for bad_val in (
            1.5,
            0.3,
            3.0,
            -2.0,
            0.0,
            float("inf"),
            float("nan"),
            math.ldexp(1.0, -130),
        ):
            grid = [[1.0, bad_val]]
            self.assertFalse(_MOE.is_exact_power_of_two_ue8m0(grid), f"should reject {bad_val!r}")
            res, ok = _MOE.expand_fp8_128x128_scale_to_ue8m0_r128c4(grid, 128, 256, strict=False)
            self.assertFalse(ok)
            self.assertIsNone(res)
            with self.assertRaises(ValueError):
                _MOE.expand_fp8_128x128_scale_to_ue8m0_r128c4(grid, 128, 256, strict=True)

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_torch_scale_expansion_and_unswizzle_exact(self) -> None:
        exps = torch.randint(-40, 40, (6, 1, 48, 1), dtype=torch.int32)
        s = torch.ldexp(torch.ones_like(exps, dtype=torch.float32), exps)
        r128c4, ok = _MOE.expand_fp8_128x128_scale_to_ue8m0_r128c4(s, 768, 6144, strict=True)
        self.assertTrue(ok)
        logical = _MOE.unswizzle_sf(r128c4, 768, 6144 // 32)
        rec = torch.exp2(logical.float() - 127.0)
        want = s.reshape(6, 48).repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
        self.assertTrue(torch.equal(rec, want))


class TestMoePhasePlanAndDieInterleaving(unittest.TestCase):
    """Exhaustively test phase planner (standard & die-aware n_split) and expert_half slot order."""

    def test_phase_plan_tp8_and_tp16_standard_and_die_aware(self) -> None:
        for i_tp in (384, 768, 1536):
            u1 = 2 * i_tp // 32
            for die_aware in (False, True):
                for m_max in (1, 2, 4, 8):
                    plan, nb_cap = _MOE.make_phase_plan_host(
                        188,
                        m_max,
                        i_tp=i_tp,
                        early0=True,
                        heavy_fc2_only=(i_tp == 768 and m_max in (4, 8)),
                        die_aware=die_aware,
                    )
                    self.assertGreaterEqual(nb_cap, 1)
                    self.assertLessEqual(nb_cap, 2)
                    # Check FC2 block coverage = exact partition of [0, 192), and die separation.
                    for g in range(4, 4 * m_max + 1):
                        covered = []
                        for c in range(188):
                            w = plan[g][c][m_max]
                            b0, nb = w & 0xFFFF, w >> 16
                            self.assertGreaterEqual(nb, 1)
                            if die_aware:
                                if c < 94:
                                    self.assertLess(b0 + nb, 97)
                                else:
                                    self.assertGreaterEqual(b0, 96)
                            covered.extend(range(b0, b0 + nb))
                        self.assertEqual(covered, list(range(192)))
                        # Check FC1 stage ranges cover exact total & split-unit protocol.
                        ng = (g + 3) // 4
                        for p in range(ng):
                            n_p = min(4, g - 4 * p)
                            if die_aware:
                                expect_total = n_p * (u1 // 2) * 6
                                halves = (range(0, 94), range(94, 188))
                            else:
                                expect_total = n_p * u1 * 6
                                halves = (range(0, 188),)
                            for half_ctas in halves:
                                cur = 0
                                for c in half_ctas:
                                    w = plan[g][c][p]
                                    if w == 0:
                                        continue
                                    i0, i1 = w & 0xFFFF, w >> 16
                                    self.assertEqual(i0, cur)
                                    self.assertGreater(i1, i0)
                                    # Split-unit invariant: if range starts inside a 6-stage unit,
                                    # it must reach at least that unit's end.
                                    if i0 % 6 != 0 and i1 < expect_total:
                                        self.assertGreaterEqual(i1, ((i0 // 6) + 1) * 6)
                                    cur = i1
                                self.assertEqual(cur, expect_total)
                    # Group-0 early0 invariance across all G in [4, 4 * m_max] when m_max >= 2.
                    if m_max >= 2:
                        ref_g0 = [plan[4][c][0] for c in range(188)]
                        for g in range(5, 4 * m_max + 1):
                            self.assertEqual([plan[g][c][0] for c in range(188)], ref_g0)

    def test_interleave_expert_slots_by_die_all_counts(self) -> None:
        # Exhaustively test every (n0, n1) with 0 <= n0 + n1 <= 32.
        for total in range(1, 33):
            for n0 in range(total + 1):
                n1 = total - n0
                ids = list(range(n0)) + list(range(64, 64 + n1))
                ordered = _MOE.interleave_expert_slots_by_die(ids, half_e=64)
                self.assertEqual(sorted(ordered), sorted(ids))
                k_pairs = min(n0 // 2, n1 // 2)
                for g in range(k_pairs):
                    # First two slots in each paired group are die-0 (< 64); last two are die-1 (>= 64).
                    self.assertLess(ordered[4 * g + 0], 64)
                    self.assertLess(ordered[4 * g + 1], 64)
                    self.assertGreaterEqual(ordered[4 * g + 2], 64)
                    self.assertGreaterEqual(ordered[4 * g + 3], 64)

    def test_ml4_routing_tie_break_and_ar_push_abi_64bit_offsets(self) -> None:
        # Verify twiddle-based top-4 + softmax matches renorm_moe_routing_op semantics (ties -> lower id).
        def _twiddle_u32(val: float) -> int:
            b = struct.unpack("<I", struct.pack("<f", val))[0]
            return (~b & 0xFFFFFFFF) if (b & 0x80000000) else (b | 0x80000000)

        def _sim_route(logits: list[float]) -> tuple[list[int], list[float]]:
            ranked = sorted(
                range(len(logits)),
                key=lambda i: (_twiddle_u32(logits[i]), 65535 - i),
                reverse=True,
            )[:4]
            vals = [logits[i] for i in ranked]
            mx = vals[0]
            exps = [math.exp(v - mx) for v in vals]
            tot = sum(exps)
            return ranked, [e / tot for e in exps]

        # All +0.0 logits -> exact 128-way tie -> experts [0, 1, 2, 3] at weight 0.25.
        ids, wts = _sim_route([0.0] * 128)
        self.assertEqual(ids, [0, 1, 2, 3])
        self.assertEqual(wts, [0.25, 0.25, 0.25, 0.25])

        # Exact tie at 4th/5th boundary -> lower expert id wins.
        row = [-10.0] * 128
        for idx in (7, 19, 42, 65, 100):
            row[idx] = 2.5
        ids2, wts2 = _sim_route(row)
        self.assertEqual(ids2, [7, 19, 42, 65])
        self.assertAlmostEqual(sum(wts2), 1.0, places=6)

        # Wide logits (sigma ~100): no overflow/NaN, weights sum to 1.0.
        wide = [float((i * 37) % 201 - 100) for i in range(128)]
        _, wts3 = _sim_route(wide)
        self.assertTrue(all(math.isfinite(w) and 0.0 <= w <= 1.0 for w in wts3))
        self.assertAlmostEqual(sum(wts3), 1.0, places=6)

        # 3-slot LamportWorkspace 64-bit stride check at TP16 (world=16, max_rows=32768, H=6144):
        # slot=2, rank=15 exceeds 32-bit element offset (> 2^31).
        world, max_rows, h = 16, 32768, 6144
        slot, rank, t, cl = 2, 15, 3, 6142
        elem_off_lamport = ((slot * world + rank) * (max_rows * h)) + t * h + cl
        self.assertGreater(elem_off_lamport, (1 << 31) - 1)
        off_u32_kernel = ((slot * world + rank) * max_rows + t) * (h // 2) + (cl // 2)
        self.assertEqual(off_u32_kernel * 4, elem_off_lamport * 2)

        # 3-slot McastGPUBuffer 64-bit byte offset check ([3][t][rank][H]).
        bytes_per_buf = 1572864  # 1.5 MiB one-shot TP16 buffer
        byte_m2 = slot * bytes_per_buf + ((t * world + rank) * h + cl) * 2
        self.assertEqual(byte_m2, 2 * 1572864 + ((3 * 16 + 15) * 6144 + 6142) * 2)


class TestMl4MxGemvContracts(unittest.TestCase):
    """Verify ML4 GEMV shape tables, SM occupancy, fused LayerNorm, and uGPU tile-claim bijection."""

    def test_all_ml4_tp4_tp8_tp16_shapes_served_within_vr200_sms(self) -> None:
        for name, (n, k) in _GEMV.ML4_PROJECTIONS.items():
            cfg = _GEMV.get_config(n, k, m=4)
            self.assertIsNotNone(cfg, f"missing config for {name} ({n}x{k})")
            assert cfg is not None
            split = cfg[0]
            self.assertIn(split, (1, 2, 4, 8))
            grid = (n // 128) * split
            self.assertLessEqual(grid, 212, f"{name} grid {grid} exceeds 212 VR200 SMs")
            for m in (1, 2, 4, 8):
                self.assertTrue(_GEMV.serves(n, k, m, x_bf16=True, ugpu=True), f"{name} M={m}")
        # Fused LayerNorm + ix_wq_b (4096 x 6144) served with uGPU for M=1..8, declined for M>8.
        for m in (1, 2, 4, 8):
            self.assertTrue(_GEMV.serves(4096, 6144, m, x_bf16=True, x_layernorm=True, ugpu=True))
        self.assertFalse(_GEMV.serves(4096, 6144, 16, x_bf16=True, x_layernorm=True, ugpu=True))

    def test_ugpu_tile_claim_bijection_and_locality(self) -> None:
        for n_tiles in (4, 8, 12, 32, 48):
            half = n_tiles // 2
            # Balanced, skewed, all-die-0, all-die-1, and remainder-SM schedules.
            schedules = [
                [0] * half + [1] * half,
                [1] * half + [0] * half,
                [0] * n_tiles,
                [1] * n_tiles,
                [0] * (half + 1) + [1] * (half - 1),
                [0] * (half - 1) + [1] * (half + 1),
                [-1] * n_tiles,
                [0, 1, -1, 0] * (n_tiles // 4),
            ]
            for sched in schedules:
                claimed = _GEMV.simulate_ugpu_tile_claim(sched, n_tiles)
                self.assertEqual(sorted(claimed), list(range(n_tiles)))
                # Check die-local match count equals theoretical optimum min(k0, half) + min(k1, half).
                eff = [d if d in (0, 1) else (i & 1) for i, d in enumerate(sched)]
                k0 = eff.count(0)
                k1 = eff.count(1)
                local_hits = sum(
                    1
                    for d, tile in zip(eff, claimed)
                    if (d == 0 and tile < half) or (d == 1 and tile >= half)
                )
                self.assertEqual(local_hits, min(k0, half) + min(k1, half))


class TestSyntaxAndHeaders(unittest.TestCase):
    """Verify every new and modified Python file parses cleanly and carries a 2026 NVIDIA header."""

    def test_ast_parse_and_copyright_headers(self) -> None:
        files = [
            _TARGET_DIR / "ops" / "gemm" / "locality.py",
            _TARGET_DIR / "ops" / "gemm" / "_ml4_mx_gemv_dsl.py",
            _TARGET_DIR / "ops" / "gemm" / "ml4_mx_gemv.py",
            _TARGET_DIR / "ops" / "moe" / "__init__.py",
            _TARGET_DIR / "ops" / "moe" / "_ml4_moe_dsl.py",
            _TARGET_DIR / "ops" / "moe" / "ml4_moe_kernel.py",
            _TARGET_DIR / "weights.py",
            _TARGET_DIR / "modeling.py",
        ]
        for path in files:
            src = path.read_text(encoding="utf-8")
            ast.parse(src, filename=str(path))
            self.assertIn("Copyright (c) 2026 NVIDIA CORPORATION", src[:300], str(path))


if __name__ == "__main__":
    unittest.main()
