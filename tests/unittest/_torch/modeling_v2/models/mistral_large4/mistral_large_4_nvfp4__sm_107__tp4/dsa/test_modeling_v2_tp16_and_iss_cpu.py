# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for ML4 SM107 TP16 routing, geometry, 4-head math, FP8 tactics and ISS-8/16."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[8]
_ML4_ROOT = (
    _REPO_ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/mistral_large4"
)
_TP4_ROOT = _ML4_ROOT / "mistral_large_4_nvfp4__sm_107__tp4"
_CATALOG_ROOT = _REPO_ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_HIST = _load_module(_TP4_ROOT / "ops/dsa/_hist_topk.py", "_ml4_hist_topk_cpu")


def _extract_constant(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def test_tp16_weight_geometry_and_swizzle_alignment():
    """Every TP16 per-rank dimension is cleanly divisible by FP8 128x128 and NVFP4 128x4 tiles."""
    shape = _extract_constant(_ML4_ROOT / "routing.py", "_SHAPE")
    (
        _layers,
        hidden,
        heads,
        _q_lora,
        kv_lora,
        qk_nope,
        qk_rope,
        v_dim,
        num_experts,
        _topk,
        n_shared,
        moe_inter_full,
        dense_inter_full,
        _dense_layers,
        idx_heads,
        idx_dim,
        idx_topk,
        vocab,
    ) = shape

    tp = 16
    assert hidden == 6144
    assert heads % tp == 0 and heads // tp == 4
    assert (heads // tp) * (qk_nope + qk_rope) == 512
    assert (heads // tp) * v_dim == 512
    assert moe_inter_full % tp == 0 and moe_inter_full // tp == 384
    assert (moe_inter_full * n_shared) % tp == 0 and (moe_inter_full * n_shared) // tp == 384
    assert dense_inter_full % tp == 0 and dense_inter_full // tp == 1536
    assert vocab % tp == 0 and vocab // tp == 12800

    moe_inter = moe_inter_full // tp
    # NVFP4 FC1/FC2 32-row shuffle and 128x4 block-scale swizzle require zero padding at TP16.
    assert (2 * moe_inter) % 128 == 0 and (2 * moe_inter) % 32 == 0
    assert (hidden // 16) % 4 == 0
    assert hidden % 128 == 0 and hidden % 32 == 0
    assert (moe_inter // 2) == 192
    assert (moe_inter // 16) == 24 and (moe_inter // 16) % 4 == 0
    assert kv_lora == 512 and num_experts == 128 and idx_heads == 32 and idx_dim == 128
    assert idx_topk == 2048


def test_four_head_sparse_mla_kernel_constants_and_bounds():
    """Verify 4-head padding and cluster-combine arithmetic in _ds_mla_sparse_decode_kernel.py."""
    kernel_path = _CATALOG_ROOT / "attention/_ds_mla_sparse_decode_kernel.py"
    supported_heads = _extract_constant(kernel_path, "SUPPORTED_HEADS")
    supported_splits = _extract_constant(kernel_path, "SUPPORTED_SPLITS")
    n_mma = _extract_constant(kernel_path, "N_MMA")
    d_nope = _extract_constant(kernel_path, "D_NOPE")

    assert supported_heads == (4, 8, 16)
    assert n_mma == 16
    for h in supported_heads:
        # Zero-fill loop in W9 covers (N_MMA - h) * 128 bytes cleanly in 512-byte warp steps.
        assert ((n_mma - h) * 128) % (16 * 32) == 0
        # num_splits == 1 epilogue packs pairs of heads.
        assert h % 2 == 0
        # Cluster combine splits total4 evenly over every supported split count.
        total4 = h * d_nope // 4
        for ns in supported_splits:
            assert total4 % ns == 0


def test_tp16_fp8_gemv_tactics_satisfy_kernel_invariants():
    """All TP16 FP8 linear shapes have valid tactics meeting FusedFp8QuantGemv constraints."""
    tactics = _extract_constant(
        _TP4_ROOT / "ops/gemm/cute_dsl_fused_fp8_block_scale_linear.py", "_TACTICS"
    )
    tp16_shapes = (
        (512, 6144),  # wq
        (6144, 512),  # wo
        (768, 6144),  # shared_gu
        (6144, 384),  # shared_dn
        (3072, 6144),  # dense0_gu
        (6144, 1536),  # dense0_dn
        (4096, 6144),  # indexer wq_b
    )
    for n, k in tp16_shapes:
        for m in (1, 2, 4, 8):
            assert (m, n, k) in tactics, f"missing TP16 tactic for {(m, n, k)}"
    for (m, n, k), (split_k, warps) in tactics.items():
        assert 1 <= m <= 16
        assert warps in (4, 8)
        assert k % 128 == 0
        assert n % (warps * 8) == 0
        assert (k // 128) % split_k == 0


def test_lamport_tp16_uses_int64_workspace_extent_and_fits_oneshot():
    """16-rank Lamport compiles with sym_int64 workspace extent; c=1 verify fits 1.5 MiB."""
    src = (_CATALOG_ROOT / "comm/_tp4_allreduce_fused_rms_kernel.py").read_text()
    assert "workspace_numel = cute.sym_int32(" not in src
    assert src.count("workspace_numel = cute.sym_int64(divisibility=_BF16_VEC_SIZE)") == 4
    # 4 verify rows x 6144 hidden x 2 bytes x 16 ranks = 768 KiB < 1.5 MiB MNNVL 1-shot bound.
    assert 4 * 6144 * 2 * 16 == 786_432 < 3 * 512 * 1024
    # Mode-0 stride at max_rows=32768 exceeds INT32_MAX, confirming Int64 extent is required.
    assert 16 * 32768 * 6144 == 3_221_225_472 > (1 << 31) - 1


def test_hist_topk_constants_match_cpp_header():
    """Python _hist_topk constants match cpp/tensorrt_llm/kernels/glmMoeDsa/glmDsaHistTopK.h."""
    assert _HIST.MAX_RANKS == 16
    assert _HIST.SHARD_ALIGN == 128
    assert _HIST.MAX_SHARD == 98304
    assert _HIST.MAX_ROWS == 8
    assert _HIST.TOPK == 2048
    assert _HIST.WORKSPACE_WORDS == 1_499_136
    assert _HIST.STATE_WORDS == 36
    header = (_REPO_ROOT / "cpp/tensorrt_llm/kernels/glmMoeDsa/glmDsaHistTopK.h").read_text()
    assert "kTopK = 2048" in header
    assert "kMaxRanks = 16" in header
    assert "kMaxRows = 8" in header
    assert "kShardAlign = 128" in header
    assert "kMaxShard = 98304" in header


@pytest.mark.parametrize("group_size", (4, 8, 16))
@pytest.mark.parametrize("tokens_per_block", (32, 64))
@pytest.mark.parametrize("kv_len", (0, 1, 2048, 2049, 131_072, 262_145, 530_000, 780_000, 1_048_584))
def test_shard_geometry_tiles_request_on_page_boundaries(group_size, tokens_per_block, kv_len):
    """Shards partition [0, kv_len) disjointly and every shard base lands on a page boundary."""
    s = _HIST.shard_size(kv_len, group_size)
    assert s % _HIST.SHARD_ALIGN == 0
    assert s % tokens_per_block == 0
    covered = 0
    for rank in range(group_size):
        base = rank * s
        assert base % tokens_per_block == 0
        n = min(max(kv_len - base, 0), s)
        if n > 0:
            assert base == covered
            covered += n
    assert covered == kv_len
    if kv_len <= group_size * _HIST.MAX_SHARD:
        assert s <= _HIST.MAX_SHARD
        assert _HIST.shard_capacity(kv_len, group_size) == s


def test_dynamic_iss_threshold_logic_eager_and_captured():
    """Verify dynamic L > L_thresh switching for TP4, TP8 and TP16 in eager and capture modes."""
    assert _HIST.default_iss_min_ctx_len(16, env={}) == 131072
    assert _HIST.default_iss_min_ctx_len(8, env={}) == 262144
    assert _HIST.default_iss_min_ctx_len(16, env={"TRTLLM_ML4_ISS_MIN_CTX_LEN": "65536"}) == 65536
    assert _HIST.iss_enabled(env={}) is True
    assert _HIST.iss_enabled(env={"TRTLLM_ML4_ENABLE_ISS": "0"}) is False

    # TP16 eager: switches cleanly at L > 131072 up to 1.57M.
    assert not _HIST.should_use_iss(
        131072, 1_048_584, 1, 4, 16, 32, 131072, capturing=False, has_workspace=True
    )
    assert _HIST.should_use_iss(
        131073, 1_048_584, 1, 4, 16, 32, 131072, capturing=False, has_workspace=True
    )
    assert _HIST.should_use_iss(
        1_572_864, 1_572_864, 1, 4, 16, 64, 131072, capturing=False, has_workspace=True
    )
    assert not _HIST.should_use_iss(
        1_572_865, 1_572_865, 1, 4, 16, 64, 131072, capturing=False, has_workspace=True
    )

    # TP16 capture: off for 32K server, on for 1M server.
    assert not _HIST.should_use_iss(
        4096, 32768, 1, 4, 16, 32, 131072, capturing=True, has_workspace=True
    )
    assert _HIST.should_use_iss(
        4096, 1_048_584, 1, 4, 16, 32, 131072, capturing=True, has_workspace=True
    )

    # TP8 eager: switches at L > 262144 up to 786432 (8 * 98304); falls back past 786432.
    assert not _HIST.should_use_iss(
        262144, 1_048_584, 1, 4, 8, 32, 262144, capturing=False, has_workspace=True
    )
    assert _HIST.should_use_iss(
        780_000, 1_048_584, 1, 4, 8, 32, 262144, capturing=False, has_workspace=True
    )
    assert not _HIST.should_use_iss(
        786_433, 1_048_584, 1, 4, 8, 32, 262144, capturing=False, has_workspace=True
    )

    # TP8 capture: safe when max_ctx_len <= 786432, refused when 1M could overflow MAX_SHARD.
    assert _HIST.should_use_iss(
        4096, 786_432, 1, 4, 8, 32, 262144, capturing=True, has_workspace=True
    )
    assert not _HIST.should_use_iss(
        4096, 1_048_584, 1, 4, 8, 32, 262144, capturing=True, has_workspace=True
    )

    # Structural guards: TP4, skip_gen, num_seqs > 1, rows > 8, non-dividing page size, disabled.
    assert not _HIST.should_use_iss(
        300_000, 300_000, 1, 4, 4, 32, 131072, capturing=False, has_workspace=True
    )
    assert not _HIST.should_use_iss(
        300_000, 1_048_584, 1, 4, 16, 32, 131072, capturing=False, has_workspace=True, skip_gen=True
    )
    assert not _HIST.should_use_iss(
        300_000, 1_048_584, 2, 8, 16, 32, 131072, capturing=False, has_workspace=True
    )
    assert not _HIST.should_use_iss(
        300_000, 1_048_584, 1, 9, 16, 32, 131072, capturing=False, has_workspace=True
    )
    assert not _HIST.should_use_iss(
        300_000, 1_048_584, 1, 4, 16, 48, 131072, capturing=False, has_workspace=True
    )
    assert not _HIST.should_use_iss(
        300_000, 1_048_584, 1, 4, 16, 32, -1, capturing=False, has_workspace=True
    )
    assert not _HIST.should_use_iss(
        300_000, 1_048_584, 1, 4, 16, 32, 131072, capturing=False, has_workspace=True, enabled=False
    )


def test_routing_source_allows_sm107_tp16_only():
    """Verify routing.py's pure decision logic permits TP16 on sm_107 and rejects sm_103/tep/dep."""
    routing_src = (_ML4_ROOT / "routing.py").read_text()
    tree = ast.parse(routing_src)
    assert "sm107_tp16_world = compute_capability == (10, 7) and mapping.world_size == 16" in (
        routing_src
    )
    # Build a synthetic config/mapping and evaluate routing end-to-end with stub target modules.
    import types

    sentinel_sm107_tp4 = type("ModelingV2MistralLarge4Nvfp4Sm107Tp4", (), {})
    pkg_name = "_ml4_routing_cpu_stub"
    stub_pkg = types.ModuleType(pkg_name)
    stub_pkg.__path__ = []
    stub_target = types.ModuleType(f"{pkg_name}.mistral_large_4_nvfp4__sm_107__tp4")
    stub_target.__path__ = []
    stub_modeling = types.ModuleType(f"{pkg_name}.mistral_large_4_nvfp4__sm_107__tp4.modeling")
    stub_modeling.ModelingV2MistralLarge4Nvfp4Sm107Tp4 = sentinel_sm107_tp4
    injected = {
        pkg_name: stub_pkg,
        stub_target.__name__: stub_target,
        stub_modeling.__name__: stub_modeling,
    }
    ns = {
        "__name__": f"{pkg_name}.routing",
        "__package__": pkg_name,
        "NoTarget": type("NoTarget", (Exception,), {}),
    }
    # Execute only the helper defs & constants (skip relative import of _router_index).
    body = [n for n in tree.body if not isinstance(n, ast.ImportFrom)]
    exec(compile(ast.Module(body=body, type_ignores=[]), "routing.py", "exec"), ns)

    cfg = SimpleNamespace(
        vision_config=None,
        num_hidden_layers=72,
        hidden_size=6144,
        num_attention_heads=64,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=128,
        num_experts_per_tok=4,
        n_shared_experts=1,
        moe_intermediate_size=6144,
        intermediate_size=24576,
        first_k_dense_replace=1,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        vocab_size=204800,
        nope_layer_flags=ns["_NOPE"],
        indexer_nope_layer_flags=ns["_NOPE"],
        kv_source_layers=ns["_KV_SOURCE"],
        index_sharing=ns["_INDEX_SHARING"],
        ssmax_scaling=True,
        indexer_ssmax_scaling=True,
        lerope=True,
        softmax_tanh_gating=30.0,
        scoring_func="softmax",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        topk_method=None,
        rope_parameters={"rope_type": "yarn", "attention_factor": 1.0, "factor": 1.0},
        fp8_kv_cache=True,
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "group_size": 16,
            "quantized_layers": ns["_MIXED_PRECISION_LAYERS"],
        },
    )
    assert ns["_per_layer_structure"](cfg)
    assert ns["_math_switches"](cfg)
    assert ns["_nvfp4_mixed_precision"](cfg)
    assert ns["_native_dspark"](None)

    def _mapping(world, *, moe_ep=1, moe_tp=None, adp=False):
        return SimpleNamespace(
            world_size=world,
            tp_size=world,
            pp_size=1,
            moe_ep_size=moe_ep,
            moe_tp_size=world if moe_tp is None else moe_tp,
            enable_attention_dp=adp,
            enable_lm_head_tp_in_adp=False,
        )

    sys.modules.update(injected)
    try:
        for cc, m in (
            ((10, 3), _mapping(16)),
            ((10, 7), _mapping(16, moe_ep=16, moe_tp=1, adp=False)),
            ((10, 7), _mapping(16, moe_ep=16, moe_tp=1, adp=True)),
            ((10, 7), _mapping(2)),
        ):
            with pytest.raises(ns["NoTarget"]):
                ns["route"](cfg, m, cc, None)

        for world in (4, 8, 16):
            assert ns["route"](cfg, _mapping(world), (10, 7), None) is sentinel_sm107_tp4
    finally:
        for k in injected:
            sys.modules.pop(k, None)
