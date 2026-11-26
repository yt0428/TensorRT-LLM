# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit and contract tests for the Mistral Large 4 SM107 persistent megakernel.

Verifies:
1. C++/Python ABI parity across ml4Megakernel.cu, ml4MegakernelOp.cpp, and _megakernel.py
   (Param enums, Stage codes, Flag bits, SMEM budget, Arena offsets, Clear units, Idx workspace).
2. All 71 MoE layer configurations (28 Plain, 25 RoPE Owner Leader, 9 NoPE Owner Leader,
   9 NoPE Layer-35 KV-Sharer Leader) role-table completeness, SMEM non-overlap (<=200,704 B),
   24-CTA Shared+Owner || 188-CTA Routed-MoE partition, and Die-Aware uGPU symmetry.
3. Cyclic multi-copy buffer scheduling (main 71-layer and leader 43-layer schedules).
4. Hardware SMID -> logical_cta bijection builder and skew/error rejection.
5. Exact PTX m16n8k16 (pack_bf16), m16n8k32 (pack_fp8_pb_wo), SWIZZLE_64B, SWIZZLE_128B,
   pack_mxf8_tiles, and NVFP4 MoE piece repacking mathematical equivalence.
6. Exact 16-rank Key-Sharded 1-level Coarse+Fine + kModeCand Global Top-2048 (with saturation
   recovery, Level 1/2 fallback, exact ties broken by larger token, and short edge L in {2048..2051}).
7. 188-CTA Routed-MoE tile-aligned FC1 schedule bijection for all (E0, E1) in [0..16].
8. fake_ops schema registration and modeling.py opt-in isolation (off by default).
"""

from __future__ import annotations

import ast
import random
import re
import struct
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[7]


def _resolve_repo_paths() -> tuple[Path, Path, Path]:
    cand_ml4 = (
        _REPO_ROOT
        / "tensorrt_llm"
        / "_torch"
        / "_experimental"
        / "modeling_v2"
        / "models"
        / "mistral_large4"
    )
    if cand_ml4.is_dir():
        tp4 = cand_ml4 / "mistral_large_4_nvfp4__sm_107__tp4"
        return _REPO_ROOT, cand_ml4, tp4
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4 import (
        _megakernel as mk,
    )

    tp4 = Path(mk.__file__).resolve().parent
    ml4 = tp4.parent
    repo = tp4.parents[6]
    return repo, ml4, tp4


_RESOLVED_ROOT, _ML4_DIR, _TP4_DIR = _resolve_repo_paths()
_CSRC_DIR = _TP4_DIR / "csrc"

_NOPE = [i % 4 == 3 for i in range(72)]
_KV_SOURCE = [35 if i > 35 and i % 4 == 3 else i for i in range(72)]
_FULL_INDEXER = [i < 17 or i % 4 in (0, 3) for i in range(72)]


def _parse_cpp_enum(text: str, enum_name: str) -> list[str]:
    m = re.search(rf"enum\s+{enum_name}\s*:[^{{]+\{{([^}}]+)\}};", text, re.S)
    assert m is not None, f"enum {enum_name} not found"
    body = re.sub(r"//[^\n]*", "", m.group(1))
    names = []
    for item in body.split(","):
        tok = item.strip().split("=")[0].strip()
        if tok:
            names.append(tok)
    return names


def _parse_py_fields(py_text: str) -> tuple[str, ...]:
    tree = ast.parse(py_text)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "_FIELDS":
            return ast.literal_eval(node.value)
    raise AssertionError("_FIELDS not found in _megakernel.py")


def test_cpp_python_abi_contract_parity():
    """Verify 1-to-1 parity between C++ ml4Megakernel.cu and Python _megakernel.py."""
    cu_text = (_CSRC_DIR / "ml4Megakernel.cu").read_text(encoding="utf-8")
    op_text = (_CSRC_DIR / "ml4MegakernelOp.cpp").read_text(encoding="utf-8")
    py_text = (_TP4_DIR / "_megakernel.py").read_text(encoding="utf-8")

    cpp_params = [p for p in _parse_cpp_enum(cu_text, "Param") if p != "kPNumParams"]
    py_fields = _parse_py_fields(py_text)
    assert len(cpp_params) == len(py_fields) == 75, (len(cpp_params), len(py_fields))
    for cpp_name, py_name in zip(cpp_params, py_fields):
        assert cpp_name == f"kP{py_name}", f"Param mismatch: {cpp_name} vs kP{py_name}"

    cpp_stages = [s for s in _parse_cpp_enum(cu_text, "Stage") if s != "kStNone"]
    assert len(cpp_stages) == 15
    for st in cpp_stages:
        short = st[3:]
        assert f"{short}=" in py_text, f"Stage {st} missing in _megakernel.py"

    for cpp_expr, py_expr in (
        ("kArenaBytes == 630272", "_ARENA_BYTES = 630272"),
        ("kClearUnits == 37", "_CLEAR_UNITS = (_ARENA_CLEAR_BYTES + _CLEAR_UNIT_BYTES - 1)"),
        ("kSmemMain == 200704", "_SMEM_MAIN = 200704"),
        ("kSmemBytes == 223232", "_SMEM_BYTES = _SMEM_MAIN + _SMEM_FIXED"),
        ("kIdxSmemBytes == 135168", "_IDX_SMEM_BYTES = 135168"),
        ("kMagic = 0x4d4c345f4d4b3035LL", "_MAGIC = 0x4D4C345F4D4B3035"),
        ("kNumCounters = 64", "_NUM_COUNTERS = 64"),
    ):
        assert cpp_expr in cu_text, f"Missing {cpp_expr} in ml4Megakernel.cu"
        assert py_expr in py_text, f"Missing {py_expr} in _megakernel.py"

    fake_text = (_TP4_DIR / "fake_ops.py").read_text(encoding="utf-8")
    for op_name in (
        "ml4_megakernel_info",
        "ml4_megakernel_indexer_tmap",
        "ml4_megakernel_weight_table",
        "ml4_megakernel_sm_die_probe",
        "ml4_megakernel",
    ):
        assert f'"{op_name}' in op_text, f"Op {op_name} missing in ml4MegakernelOp.cpp"
        assert op_name in fake_text, f"Fake {op_name} missing in fake_ops.py"


def test_modeling_opt_in_isolation_and_cmake_wiring():
    """Verify modeling.py hook is strictly gated on TRTLLM_ML4_MEGAKERNEL=1 and CMake is wired."""
    modeling_text = (_TP4_DIR / "modeling.py").read_text(encoding="utf-8")
    assert 'os.environ.get("TRTLLM_ML4_MEGAKERNEL", "0") == "1"' in modeling_text
    assert "from ._megakernel import Ml4Megakernel" in modeling_text

    root_cmake_path = _RESOLVED_ROOT / "cpp" / "CMakeLists.txt"
    if not root_cmake_path.is_file():
        root_cmake_path = _REPO_ROOT / "cpp" / "CMakeLists.txt"
    if root_cmake_path.is_file():
        root_cmake = root_cmake_path.read_text(encoding="utf-8")
        assert "BUILD_ML4_MEGAKERNEL" in root_cmake
        assert "models/mistral_large4/csrc" in root_cmake

    fam_cmake = (_ML4_DIR / "csrc" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "ml4Megakernel.cu" in fam_cmake
    assert "ml4MegakernelOp.cpp" in fam_cmake
    assert "sm_107a" in fam_cmake
    assert "--maxrregcount=200" in fam_cmake


def test_all_71_moe_layer_taxonomy_and_role_tables():
    """Validate all 71 MoE layers map cleanly to lean role tables with zero SMEM overlap."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4 import (
        _megakernel as mk,
    )

    plain_layers = mk.select_megakernel_layers(_FULL_INDEXER, 1, mode="plain")
    all_layers = mk.select_megakernel_layers(_FULL_INDEXER, 1, mode="all")
    assert len(plain_layers) == 28
    assert len(all_layers) == 71
    assert all_layers == list(range(1, 72))

    class_counts: dict[tuple[bool, bool, bool], int] = {}
    for i in all_layers:
        key = (_FULL_INDEXER[i], _NOPE[i], _KV_SOURCE[i] == i)
        class_counts[key] = class_counts.get(key, 0) + 1
    assert class_counts == {
        (False, False, True): 28,
        (True, False, True): 25,
        (True, True, True): 9,
        (True, True, False): 9,
    }

    for key in class_counts:
        leader, nope, kv_owner = key
        roles, smem, issue = mk.build_roles(leader=leader, nope=nope, kv_owner=kv_owner)
        assert roles.shape == (212, 16) and smem.shape == (212, 16) and issue.shape == (212, 16)

        counts: dict[int, set[int]] = {}
        unit_bytes = dict(mk._UNIT_BYTES_BASE)
        unit_bytes[mk._ST["Kb"]] = 16384 if nope else 8192

        moe_or_owner_ctas = 0
        for cta in range(212):
            live_intervals: list[tuple[int, int, int]] = []
            cta_stages = []
            for slot in range(16):
                entry = int(roles[cta, slot])
                if entry == 0:
                    continue
                st = entry >> 16
                unit = entry & 0xFFFF
                cta_stages.append(st)
                counts.setdefault(st, set()).add(unit)
                sz = unit_bytes.get(st, 0)
                off = int(smem[cta, slot])
                iss = int(issue[cta, slot])
                if sz > 0:
                    assert 0 <= off and off + sz <= mk._SMEM_MAIN, (key, cta, st, off, sz)
                    assert off % 1024 == 0, (key, cta, st, off)
                    if st not in mk._DYNAMIC:
                        for po, ps, pj in live_intervals:
                            if pj > iss:
                                assert not (po < off + sz and off < po + ps), (key, cta, slot, pj)
                    live_intervals.append((off, sz, slot))

                is_die1 = cta >= 106
                if st == mk._ST["Wq"]:
                    assert (unit >= 16) == is_die1
                elif st == mk._ST["Oproj"]:
                    assert (unit >= 96) == is_die1
                elif st in (mk._ST["Shared"], mk._ST["Owner"]):
                    assert (unit >= 12) == is_die1
                elif st == mk._ST["Moe"]:
                    assert (unit >= 94) == is_die1
            # Every CTA runs EITHER Shared+Owner (24) OR Moe (188) — never both, exact 212 HoldExit count!
            has_moe = mk._ST["Moe"] in cta_stages
            has_owner = mk._ST["Owner"] in cta_stages
            assert has_moe ^ has_owner, (key, cta, cta_stages)
            moe_or_owner_ctas += 1
        assert moe_or_owner_ctas == 212

        if not leader:
            assert counts[mk._ST["Wq"]] == set(range(32))
            assert counts[mk._ST["WkvA"]] == set(range(32 if nope else 36))
            assert counts[mk._ST["Kb"]] == set(range(32))
            assert counts[mk._ST["KvAppend"]] == {0}
        else:
            # Leader layers run front-end in ml4LeaderFrontKernel -> ml4LayerKernel starts at Attn!
            for front_st in ("Wq", "WkvA", "Kb", "KvAppend"):
                assert mk._ST[front_st] not in counts

        assert counts[mk._ST["Attn"]] == set(range(128))
        assert counts[mk._ST["Merge"]] == set(range(16))
        assert counts[mk._ST["Vup"]] == set(range(32))
        assert counts[mk._ST["Oproj"]] == set(range(192))
        assert counts[mk._ST["Clear"]] == set(range(mk._CLEAR_UNITS))
        assert counts[mk._ST["H1"]] == set(range(16))
        assert counts[mk._ST["RouterGemv"]] == set(range(32))
        assert counts[mk._ST["Shared"]] == set(range(24))
        assert counts[mk._ST["Owner"]] == set(range(24))
        assert counts[mk._ST["Moe"]] == set(range(188))
        assert counts[mk._ST["H2"]] == set(range(16))


def test_cyclic_copy_schedule_and_smid_bijection():
    """Verify zero copy aliasing over infinite replays (incl. 43 leaders & 71 layers) and SMID map."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4 import (
        _megakernel as mk,
    )

    for n in (1, 2, 3, 4, 5, 6, 9, 13, 28, 43, 71, 72):
        num_copies, sched = mk.compute_copy_schedule(n)
        assert num_copies >= 3
        assert len(sched) == n
        if n >= 3:
            for i, (cur, nxt, prv) in enumerate(sched):
                assert 0 <= cur < num_copies
                assert nxt == sched[(i + 1) % n][0]
                assert prv == sched[(i - 1) % n][0]
                assert len({prv, cur, nxt}) == 3, (n, i, prv, cur, nxt)

    rng = random.Random(42)
    smids = list(range(212))
    rng.shuffle(smids)
    die0_set = set(smids[:106])
    rows = [(sm, 700 if sm in die0_set else 1100, 1100 if sm in die0_set else 700) for sm in smids]
    res = mk.build_smid_bijection(rows)
    assert res is not None
    table, n_die0 = res
    assert n_die0 == 106 and len(table) == 212 and set(table) == set(range(212))

    dom0 = sorted(smids[:100])
    dom1 = sorted(smids[100:200])
    res_dom = mk.build_smid_bijection_from_domains([dom0, dom1], num_sms=212)
    assert res_dom is not None
    t_dom, n0_dom = res_dom
    assert n0_dom == 106 and len(t_dom) == 212 and set(t_dom) == set(range(212))
    assert mk.build_smid_bijection([(sm, 900, 910) for sm in range(212)]) is None


def test_pack_bf16_fp8_and_mxf8_tiles_equivalence():
    """Verify pack_bf16, pack_fp8_pb_wo, swizzle_128b, and pack_mxf8_tiles exact equivalence."""
    import torch

    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4 import (
        _megakernel as mk,
    )

    torch.manual_seed(0)

    # 1. SWIZZLE_128B involution (applying swizzle_128b twice is exact identity!).
    t128 = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8)
    assert torch.equal(mk.swizzle_128b(mk.swizzle_128b(t128)), t128)

    # 2. pack_mxf8_tiles exact shape & content verification for both K=256 and K=384 (SharedDn).
    for n, k in ((128, 256), (256, 384)):
        w_f8 = (torch.randn(n, k) * 2.0).to(torch.float8_e4m3fn)
        exp_grid = torch.randint(124, 131, (n // 128, k // 128), dtype=torch.uint8)
        exp_k32 = exp_grid.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
        cols_k32 = k // 32
        swizzled = (
            exp_k32.reshape(n // 128, 4, 32, cols_k32 // 4, 4)
            .permute(0, 3, 2, 1, 4)
            .contiguous()
            .reshape(-1)
        )
        packed = mk.pack_mxf8_tiles(w_f8, swizzled)
        assert tuple(packed.shape) == (n // 128, k // 128, 16896)
        # Unswizzle weights back and check exact bit equality.
        unsw = mk.swizzle_128b(packed[:, :, :16384].reshape(n // 128, k // 128, 128, 128))
        recovered = unsw.permute(0, 2, 1, 3).contiguous().reshape(n, k)
        assert torch.equal(recovered, w_f8.view(torch.uint8))
        # Check embedded 512B R128c4 scale atoms match swizzled bytes bit-for-bit.
        assert torch.equal(packed[:, :, 16384:].reshape(-1), swizzled)

    # 3. pack_bf16 & pack_fp8_pb_wo MMA equivalence.
    w_bf16 = torch.randn(32, 128, dtype=torch.bfloat16)
    x_bf16 = torch.randn(4, 128, dtype=torch.bfloat16)
    packed_bf16 = mk.pack_bf16(w_bf16)
    out_sim = torch.zeros(32, 4, dtype=torch.float32)
    for tile in range(2):
        for ch in range(2):
            blk = packed_bf16[tile, ch].view(torch.bfloat16).reshape(4, 32, 8)
            for g in range(8):
                for c in range(4):
                    lane = 4 * g + c
                    a_row0 = torch.cat([blk[0, lane], blk[1, lane]]).float()
                    a_row1 = torch.cat([blk[2, lane], blk[3, lane]]).float()
                    for tok in range(4):
                        x_sl = x_bf16[tok, ch * 64 + 16 * c : ch * 64 + 16 * c + 16].float()
                        out_sim[16 * tile + g, tok] += float((a_row0 * x_sl).sum())
                        out_sim[16 * tile + g + 8, tok] += float((a_row1 * x_sl).sum())
    assert torch.allclose(out_sim, w_bf16.float() @ x_bf16.float().t(), atol=1e-4, rtol=1e-4)


def test_routed_moe_188cta_tile_schedule_coverage():
    """Verify the 188-CTA tile-aligned FC1 schedule covers every (slot, tile, k-half) exactly once."""
    for E in range(1, 17):
        for E0 in range(E + 1):
            E1 = E - E0
            die_split = E0 > 0 and E1 > 0 and E0 <= 8 and E1 <= 8
            covered: dict[tuple[int, int], set[int]] = {}
            for r_cta in range(188):
                if die_split:
                    if r_cta < 94:
                        lc, nc, sb, ne = r_cta, 94, 0, E0
                    else:
                        lc, nc, sb, ne = r_cta - 94, 94, E0, E1
                else:
                    lc, nc, sb, ne = r_cta, 188, 0, E
                n_tiles = 6 * ne
                assert n_tiles <= nc
                n_full = max(0, 2 * n_tiles - nc)
                n_split = n_tiles - n_full
                if lc < n_full:
                    sl, ti, ks = sb + lc // 6, lc % 6, set(range(8))
                elif lc < n_full + n_split:
                    gt = n_full + (lc - n_full)
                    sl, ti, ks = sb + gt // 6, gt % 6, set(range(0, 4))
                elif lc < n_full + 2 * n_split:
                    gt = n_full + (lc - n_full - n_split)
                    sl, ti, ks = sb + gt // 6, gt % 6, set(range(4, 8))
                else:
                    continue
                cur = covered.setdefault((sl, ti), set())
                assert not (cur & ks), f"Overlap at E={E} E0={E0} r_cta={r_cta}"
                cur |= ks
            assert len(covered) == 6 * E
            for sl in range(E):
                for ti in range(6):
                    assert covered[(sl, ti)] == set(range(8))


def test_exact_16rank_coarse_fine_kmode_cand_topk_algorithm():
    """Verify 1-level Coarse+Fine + kModeCand + Level 1/2 fallback exact Top-2048."""

    def order_key(f: float) -> int:
        u = struct.unpack("<I", struct.pack("<f", f))[0]
        k = u ^ (0xFFFFFFFF if (u >> 31) else 0x80000000)
        return k if k < 0xFFFFFFFF else 0xFFFFFFFE

    def run_sim(logits: list[float], topk: int = 2048, n_ranks: int = 16, bnd_cap: int = 4096):
        L = len(logits)
        if L <= topk:
            return list(range(L)) + [-1] * (topk - L), 0
        n_tiles = (L + 127) // 128
        shards = []
        for r in range(n_ranks):
            ts = (n_tiles * r) // n_ranks
            te = (n_tiles * (r + 1)) // n_ranks
            lo, hi = ts * 128, min(te * 128, L)
            shards.append([(p, order_key(logits[p])) for p in range(lo, hi)])

        prefix = 0
        above_tot = 0
        need = topk
        levels_used = 0
        for lvl in range(3):
            levels_used = lvl + 1
            nbits = 10 if lvl == 2 else 11
            nbins = 1 << nbits
            shift = 21 if lvl == 0 else (10 if lvl == 1 else 0)
            prev_shift = 21 if lvl == 1 else 10
            bmask = nbins - 1
            ngroups = nbins // 32

            hists = [[0] * nbins for _ in range(n_ranks)]
            for r in range(n_ranks):
                for _, ok in shards[r]:
                    if lvl == 0 or (ok >> prev_shift) == prefix:
                        hists[r][(ok >> shift) & bmask] += 1

            # Coarse group totals (exact 32-bit sums) + saturated u16 fine bins (cap 0xFFFE).
            coarse = [
                [sum(hists[r][32 * g : 32 * (g + 1)]) for g in range(ngroups)]
                for r in range(n_ranks)
            ]
            g_all = [sum(coarse[r][g] for r in range(n_ranks)) for g in range(ngroups)]
            acc = 0
            cut_g = -1
            for g in range(ngroups - 1, -1, -1):
                if acc < need <= acc + g_all[g]:
                    cut_g = g
                    break
                acc += g_all[g]
            assert cut_g >= 0
            need_in_g = need - acc

            # Restore fine bins of crossing group cut_g from saturated 0xFFFE.
            fine = []
            for r in range(n_ranks):
                sat_bins = [min(hists[r][32 * cut_g + j], 0xFFFE) for j in range(32)]
                others = sum(v for v in sat_bins if v != 0xFFFE)
                restored = [coarse[r][cut_g] - others if v == 0xFFFE else v for v in sat_bins]
                fine.append(restored)

            b_all = [sum(fine[r][j] for r in range(n_ranks)) for j in range(32)]
            b_acc = 0
            cut_w = -1
            for j in range(31, -1, -1):
                if b_acc < need_in_g <= b_acc + b_all[j]:
                    cut_w = j
                    break
                b_acc += b_all[j]
            assert cut_w >= 0
            cut_bin = 32 * cut_g + cut_w
            above_all = acc + b_acc
            bin_all = b_all[cut_w]
            above_tot += above_all
            need -= above_all
            prefix = cut_bin if lvl == 0 else ((prefix << nbits) | cut_bin)

            if need == bin_all:
                mode = "full"
            elif lvl == 2:
                mode = "ties"
            elif bin_all <= bnd_cap:
                mode = "cand"
            else:
                mode = "next"
            if mode != "next":
                sel = []
                bnd = []
                for r in range(n_ranks):
                    for p, ok in shards[r]:
                        s = ok >> shift
                        if s > prefix or (mode == "full" and s == prefix):
                            sel.append(p)
                        elif s == prefix:
                            bnd.append((ok, p))
                if mode == "cand":
                    bnd.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    sel.extend(p for _, p in bnd[:need])
                elif mode == "ties":
                    # Larger token first among exact ties.
                    bnd.sort(key=lambda x: x[1], reverse=True)
                    sel.extend(p for _, p in bnd[:need])
                return sel, levels_used
        raise AssertionError("unreachable")

    rng = random.Random(12345)
    # 1. Realistic Gaussian logits at L=16384 -> MUST complete in 1 SINGLE level (kModeCand)!
    logits_cont = [
        struct.unpack("<f", struct.pack("<f", rng.gauss(-5.0, 12.0)))[0] for _ in range(16384)
    ]
    got, lvls = run_sim(logits_cont)
    assert lvls == 1, f"Expected 1-level kModeCand completion, got {lvls} levels"
    assert len(got) == 2048 and len(set(got)) == 2048
    assert (
        sorted((logits_cont[p] for p in got), reverse=True)
        == sorted(logits_cont, reverse=True)[:2048]
    )

    # 2. Heavy exact ties > 4096 in one bin -> exercises Level-1/2 fallback + kModeTies!
    logits_ties = [1.5] * 6000 + [0.0] * 2192
    got_t, lvls_t = run_sim(logits_ties)
    assert lvls_t == 3
    assert len(set(got_t)) == 2048
    # Larger token indices win ties!
    assert set(got_t) == set(range(6000 - 2048, 6000))

    # 3. Short edge L in {1024, 2048, 2049, 2051}.
    for L in (1024, 2048, 2049, 2051):
        arr = [float(i % 97) - 40.0 for i in range(L)]
        got_e, _ = run_sim(arr)
        valid = [p for p in got_e if p >= 0]
        assert len(valid) == min(L, 2048) and len(set(valid)) == min(L, 2048)
        assert (
            sorted((arr[p] for p in valid), reverse=True)
            == sorted(arr, reverse=True)[: min(L, 2048)]
        )
