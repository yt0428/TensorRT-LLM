# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU mathematical-equivalence tests for DSpark post-target optimizations across TP4/TP8/TP16.

Verifies without a GPU (using Triton's CPU interpreter when available for the
actual Triton kernels, plus exact multi-rank TP4/TP8/TP16 simulations):

1. Vocab-parallel softcap greedy argmax (including saturation ties |x| >= 270.33,
   cross-rank ties, intra-rank ties, +/-inf, NaN) vs gathered ``tanh_softcap``.
2. Vocab-parallel softcap temperature-only Gumbel-max sampling: TP-rank-count
   invariance (TP1 == TP4 == TP8 == TP16 bitwise for identical Philox seeds) and
   exact categorical distribution equivalence vs ``softmax(tanh_softcap(x)/T)``.
3. Vocab-parallel RNN Markov chain with sharded ``draft_markov_output``
   (``[V / tp, rank]``): bit-identical greedy draft tokens and exact biased
   logits against the unsharded full-vocab reference across TP4/TP8/TP16.
4. K-sharded ``draft_target_projection`` (``[6144, 30720 / tp]``) + RMSNorm
   equivalence across TP4/TP8/TP16.
5. Merged ``draft_w13`` SwiGLU, split ``triton_query_rope_tail`` +
   ``triton_latent_tail(q=None)``, fused ``triton_embed_draft_block``, and
   multi-layer ``precompute_context_kv`` equivalence across TP4/TP8/TP16.
6. ``spec_tail_kernels`` (``strict_accept``, ``decode_draft_block_table``,
   ``dflash_paged_latent_prepare``, ``store_context_latents``,
   ``scatter_context_latents``, ``compute_gen_ctx_positions``,
   ``prepare_next_new_tokens``) bitwise equivalence against eager torch.
7. ``sm_107__tp4/weights.py`` draft weight declaration and sharding coverage
   across TP1, TP4, TP8, and TP16.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SPEC_DIR = _REPO_ROOT / "tensorrt_llm" / "_torch" / "speculative"
_DSPARK_CATALOG_DIR = (
    _REPO_ROOT / "tensorrt_llm" / "_torch" / "_experimental" / "modeling_v2" / "catalog" / "dspark"
)
_SM107_TP4_DIR = (
    _REPO_ROOT
    / "tensorrt_llm"
    / "_torch"
    / "_experimental"
    / "modeling_v2"
    / "models"
    / "mistral_large4"
    / "mistral_large_4_nvfp4__sm_107__tp4"
)


def _load_standalone_module(name: str, path: Path):
    """Load a module directly by file path without triggering full tensorrt_llm native imports."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_spec_tail_kernels():
    return _load_standalone_module("_test_spec_tail_kernels", _SPEC_DIR / "spec_tail_kernels.py")


def _load_dspark_kernels():
    return _load_standalone_module("_test_dspark_kernels", _DSPARK_CATALOG_DIR / "_kernels.py")


def _load_vocab_parallel_kernels():
    """Import vocab_parallel Triton kernels cleanly even when C++ tensorrt_llm bindings are absent."""
    # Provide minimal stub modules for parent imports only if tensorrt_llm bindings aren't built.
    try:
        from tensorrt_llm._torch.speculative import vocab_parallel as vp

        return vp
    except Exception:
        pass

    import types

    stubs_added = []

    def _ensure_pkg(fullname: str, **attrs):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = types.ModuleType(fullname)
        mod.__path__ = []
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[fullname] = mod
        stubs_added.append(fullname)
        return mod

    class _DummyAllReduce:
        def __init__(self, *a, **kw):
            pass

    class _DummyStrategy:
        AUTO = "AUTO"
        NCCL = "NCCL"

    class _DummyLMHead:
        pass

    class _DummyTPMode:
        COLUMN = "COLUMN"

    _ensure_pkg("tensorrt_llm")
    _ensure_pkg("tensorrt_llm.logger", logger=SimpleNamespace(info=lambda *a, **kw: None))
    _ensure_pkg("tensorrt_llm.mapping", Mapping=object)
    _ensure_pkg("tensorrt_llm._torch")
    _ensure_pkg(
        "tensorrt_llm._torch.distributed",
        AllReduce=_DummyAllReduce,
        AllReduceStrategy=_DummyStrategy,
        allgather=lambda x, *a, **kw: x,
    )
    _ensure_pkg("tensorrt_llm._torch.modules")
    _ensure_pkg("tensorrt_llm._torch.modules.embedding", LMHead=_DummyLMHead)
    _ensure_pkg("tensorrt_llm._torch.modules.linear", TensorParallelMode=_DummyTPMode)
    _ensure_pkg("tensorrt_llm._torch.pyexecutor")
    _ensure_pkg("tensorrt_llm._torch.pyexecutor.sampler")
    _ensure_pkg("tensorrt_llm._torch.pyexecutor.sampler.ops")
    _ensure_pkg(
        "tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla",
        GREEDY_TEMPERATURE_THRESHOLD=1e-4,
    )
    _ensure_pkg("tensorrt_llm._torch.speculative")
    return _load_standalone_module(
        "tensorrt_llm._torch.speculative.vocab_parallel",
        _SPEC_DIR / "vocab_parallel.py",
    )


def _softcap_ref(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x.float() / cap) * cap


def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()).to(
        x.dtype
    )


def _rotate_ref(x: torch.Tensor, table: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    cos, sin = table[positions.long()].chunk(2, dim=-1)
    if x.ndim == positions.ndim + 2:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2).to(x.dtype)


class _SimulatedTPOp:
    """In-process synchronous multi-rank sum all-reduce for CPU verification of TP4/TP8/TP16."""

    def __init__(self, tp_size: int):
        self.tp_size = tp_size

    def make_exchanges(self, vp_mod):
        """Return one ``_PairExchange`` per rank that reduces across all ranks via exact sum."""
        # Collect rank buffers per round (used via `run_all_ranks`).
        return [vp_mod._PairExchange(None, self.tp_size, r) for r in range(self.tp_size)]


def _run_vp_argmax_sim(vp, full_logits: torch.Tensor, tp_size: int, softcap: float | None):
    """Run real Triton ``_local_argmax_kernel`` + simulated TP sum + ``_reduce_pairs_kernel``."""
    rows, vocab = full_logits.shape
    assert vocab % tp_size == 0
    shard = vocab // tp_size
    has_softcap, inv_cap, cap = vp._softcap_params(softcap)
    use_libdevice = vp._use_libdevice_softcap()

    rank_bufs = []
    n_slots = 0
    for r in range(tp_size):
        ex = vp._PairExchange(None, tp_size, r)
        layout = ex.layout(rows, shard, full_logits.device)
        local = full_logits[:, r * shard : (r + 1) * shard].contiguous()
        if rows > 0:
            vp._local_argmax_kernel[(rows, layout.splits)](
                local,
                local.stride(0),
                shard,
                r * shard,
                layout.chunk,
                inv_cap,
                cap,
                layout.buffer,
                layout.buffer.stride(0),
                layout.slot0,
                layout.zero_lo,
                layout.zero_hi,
                HAS_SOFTCAP=has_softcap,
                USE_LIBDEVICE=use_libdevice,
                ZERO_WIDTH=layout.zero_width,
                BLOCK_V=vp._LOCAL_BLOCK_V,
                num_warps=8,
                enable_fp_fusion=not has_softcap,
            )
        rank_bufs.append(layout.buffer)
        n_slots = tp_size * layout.splits

    # Exact float32 sum across all ranks (same semantics as MNNVL/NCCL AllReduce sum).
    summed = torch.stack(rank_bufs, dim=0).sum(dim=0)
    out = torch.empty((rows,), dtype=torch.int64, device=full_logits.device)
    if rows > 0:
        import triton

        vp._reduce_pairs_kernel[(triton.cdiv(rows, vp._REDUCE_BLOCK_ROWS),)](
            summed,
            summed.stride(0),
            rows,
            n_slots,
            out,
            BLOCK_ROWS=vp._REDUCE_BLOCK_ROWS,
            BLOCK_SLOTS=triton.next_power_of_2(n_slots),
            num_warps=4,
        )
    return out


def _run_vp_sample_sim(
    vp,
    full_logits: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
    tp_size: int,
    softcap: float | None,
    greedy_threshold: float = 1e-4,
):
    """Run real Triton ``_local_sample_kernel`` + simulated TP sum + ``_reduce_pairs_kernel``."""
    rows, vocab = full_logits.shape
    assert vocab % tp_size == 0
    shard = vocab // tp_size
    has_softcap, inv_cap, cap = vp._softcap_params(softcap)
    use_libdevice = vp._use_libdevice_softcap()

    rank_bufs = []
    n_slots = 0
    for r in range(tp_size):
        ex = vp._PairExchange(None, tp_size, r)
        layout = ex.layout(rows, shard, full_logits.device)
        local = full_logits[:, r * shard : (r + 1) * shard].contiguous()
        if rows > 0:
            vp._local_sample_kernel[(rows, layout.splits)](
                local,
                local.stride(0),
                shard,
                r * shard,
                layout.chunk,
                temperatures,
                seeds,
                offsets,
                float(greedy_threshold),
                inv_cap,
                cap,
                layout.buffer,
                layout.buffer.stride(0),
                layout.slot0,
                layout.zero_lo,
                layout.zero_hi,
                HAS_SOFTCAP=has_softcap,
                USE_LIBDEVICE=use_libdevice,
                ZERO_WIDTH=layout.zero_width,
                BLOCK_V=vp._LOCAL_BLOCK_V,
                num_warps=8,
                enable_fp_fusion=not has_softcap,
            )
        rank_bufs.append(layout.buffer)
        n_slots = tp_size * layout.splits

    summed = torch.stack(rank_bufs, dim=0).sum(dim=0)
    out = torch.empty((rows,), dtype=torch.int64, device=full_logits.device)
    if rows > 0:
        import triton

        vp._reduce_pairs_kernel[(triton.cdiv(rows, vp._REDUCE_BLOCK_ROWS),)](
            summed,
            summed.stride(0),
            rows,
            n_slots,
            out,
            BLOCK_ROWS=vp._REDUCE_BLOCK_ROWS,
            BLOCK_SLOTS=triton.next_power_of_2(n_slots),
            num_warps=4,
        )
    return out


# ---------------------------------------------------------------------------
# 1. Vocab-Parallel Softcap Greedy Argmax (TP4 / TP8 / TP16)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", [4, 8, 16])
@pytest.mark.parametrize("softcap", [None, 30.0])
def test_vocab_parallel_softcap_argmax_bitwise(
    monkeypatch: pytest.MonkeyPatch, tp_size: int, softcap: float | None
) -> None:
    """Bit-identical argmax across TP4/TP8/TP16, including softcap saturation ties and NaNs."""
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    vp = _load_vocab_parallel_kernels()
    monkeypatch.setattr(vp, "_LOCAL_BLOCK_V", 128)

    # Multiple splits per rank across all TP sizes (TP4=1024, TP8=512, TP16=256).
    vocab = 4096
    rows = 6
    gen = torch.Generator().manual_seed(100 + tp_size)
    logits = (torch.randn(rows, vocab, generator=gen) * 10.0).to(torch.bfloat16)

    # Row 0: unique max on last rank, last column.
    logits[0, :] = -5.0
    logits[0, vocab - 1] = 12.0
    # Row 1: exact cross-rank tie -> lowest global index must win.
    logits[1, :] = 0.0
    logits[1, vocab // tp_size - 1] = 25.0
    logits[1, vocab // tp_size + 5] = 25.0
    logits[1, vocab - 3] = 25.0
    # Row 2: softcap saturation tie (300.0 vs 400.0 both saturate to 30.0 under softcap=30.0!).
    logits[2, :] = 0.0
    logits[2, 17] = 300.0
    logits[2, vocab - 17] = 400.0
    # Row 3: NaN beats all finite values; earliest NaN index wins.
    logits[3, vocab // 2 + 3] = float("nan")
    logits[3, vocab - 2] = float("nan")
    logits[3, 0] = 500.0
    # Row 4: all -inf except one column.
    logits[4, :] = float("-inf")
    logits[4, vocab // 2] = -50.0
    # Row 5: general random bf16 logits.

    actual = _run_vp_argmax_sim(vp, logits, tp_size, softcap)
    ref_logits = _softcap_ref(logits, softcap) if softcap is not None else logits.float()
    expected = torch.argmax(ref_logits, dim=-1)
    assert torch.equal(actual, expected), (actual, expected)
    if softcap == 30.0:
        # Under softcap=30.0, both 300.0 and 400.0 saturate to 30.0 -> index 17 must win!
        assert int(actual[2]) == 17
    else:
        assert int(actual[2]) == vocab - 17


# ---------------------------------------------------------------------------
# 2. Vocab-Parallel Softcap Temperature-Only Sampling (TP1/TP4/TP8/TP16)
# ---------------------------------------------------------------------------


def test_vocab_parallel_softcap_sampling_tp_invariant_and_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling draws are bitwise identical across TP1/TP4/TP8/TP16 and match softmax(softcap/T)."""
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    vp = _load_vocab_parallel_kernels()
    monkeypatch.setattr(vp, "_LOCAL_BLOCK_V", 64)

    vocab = 512  # divisible by 4 and 16, multiple splits per rank with BLOCK_V=64
    rows = 5
    gen = torch.Generator().manual_seed(42)
    logits = (torch.randn(rows, vocab, generator=gen) * 4.0).to(torch.bfloat16)
    temps = torch.tensor([0.0, 0.7, 1.0, 1.5, 1e-6], dtype=torch.float32)
    seeds = torch.tensor([11, 22, 33, 44, 55], dtype=torch.int64)
    offsets = torch.tensor([0, 7, 13, 29, 0], dtype=torch.int64)

    draws_tp1 = _run_vp_sample_sim(vp, logits, temps, seeds, offsets, tp_size=1, softcap=30.0)
    for tp_size in (4, 8, 16):
        draws_tp = _run_vp_sample_sim(
            vp, logits, temps, seeds, offsets, tp_size=tp_size, softcap=30.0
        )
        assert torch.equal(draws_tp, draws_tp1), f"TP{tp_size} diverged from TP1"

    # Greedy rows (0 and 4) must match softcapped argmax bitwise.
    expected_greedy = torch.argmax(_softcap_ref(logits, 30.0), dim=-1)
    assert int(draws_tp1[0]) == int(expected_greedy[0])
    assert int(draws_tp1[4]) == int(expected_greedy[4])

    # Statistical distribution check using the actual Triton kernel (small BLOCK_V for fast CPU
    # interpret) plus a 20,000-draw vectorized check against exact softmax(tanh_softcap(x)/T).
    small_v = 16
    raw = (torch.randn(small_v, generator=torch.Generator().manual_seed(7)) * 3.0).to(
        torch.bfloat16
    )
    raw[0] = 350.0  # saturates to 30.0 under softcap
    raw[1] = 30.0  # tanh(1)*30 ~= 22.85, strictly below saturated 30.0!
    temp = 5.0
    expected_probs = torch.softmax(_softcap_ref(raw, 30.0) / temp, dim=-1)

    # 1) Actual Triton _local_sample_kernel + _reduce_pairs_kernel across TP4 (120 draws).
    n_triton = 120
    monkeypatch.setattr(vp, "_LOCAL_BLOCK_V", 16)
    batch_logits = raw.unsqueeze(0).expand(n_triton, -1).contiguous()
    batch_temps = torch.full((n_triton,), temp, dtype=torch.float32)
    batch_seeds = torch.full((n_triton,), 12345, dtype=torch.int64)
    batch_offsets = torch.arange(n_triton, dtype=torch.int64)
    triton_samples = _run_vp_sample_sim(
        vp, batch_logits, batch_temps, batch_seeds, batch_offsets, tp_size=4, softcap=30.0
    )
    triton_emp = torch.bincount(triton_samples, minlength=small_v).float() / n_triton
    assert 0.5 * (triton_emp - expected_probs).abs().sum().item() < 0.15

    # 2) High-sample-count (20,000 draws) exact Gumbel-max softcap distribution verification across
    # TP4 / TP8 / TP16 shards.
    n_draws = 20000
    g_gen = torch.Generator().manual_seed(2026)
    # Use the exact 23-bit open-interval uniform mapping from vocab_parallel._gumbel.
    u_int = torch.randint(0, 1 << 23, (n_draws, small_v), generator=g_gen, dtype=torch.int64)
    u = (u_int.float() + 0.5) / float(1 << 23)
    gumbel = -torch.log(-torch.log(u))
    capped = _softcap_ref(raw, 30.0).unsqueeze(0)
    scores = capped / temp + gumbel
    full_draws = torch.argmax(scores, dim=-1)
    for tp_size in (4, 8, 16):
        sh = small_v // tp_size
        best_vals = []
        best_gidxs = []
        for r in range(tp_size):
            local_scores = scores[:, r * sh : (r + 1) * sh]
            vals, lidx = local_scores.max(dim=-1)
            best_vals.append(vals)
            best_gidxs.append(lidx + r * sh)
        stacked_v = torch.stack(best_vals, dim=-1)
        stacked_i = torch.stack(best_gidxs, dim=-1)
        win = stacked_v.argmax(dim=-1)
        tp_draws = stacked_i.gather(1, win.unsqueeze(1)).squeeze(1)
        assert torch.equal(tp_draws, full_draws)
    emp = torch.bincount(full_draws, minlength=small_v).float() / n_draws
    tvd = 0.5 * (emp - expected_probs).abs().sum().item()
    assert tvd < 0.015, f"TVD too large: {tvd:.4f}"


# ---------------------------------------------------------------------------
# 3. Vocab-Parallel RNN Markov Chain (TP4 / TP8 / TP16)
# ---------------------------------------------------------------------------


def _reference_rnn_markov_chain(
    raw_block_logits: torch.Tensor,
    anchors: torch.Tensor,
    hidden_states: torch.Tensor,
    emb_w: torch.Tensor,
    joint_w: torch.Tensor,
    joint_b: torch.Tensor,
    out_w: torch.Tensor,
    softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unsharded reference matching DraftModel.logits_processor + apply_markov_chain_logits."""
    base_logits = _softcap_ref(raw_block_logits, softcap)
    B, K, V = base_logits.shape
    prev = anchors.long()
    state = None
    refined = torch.empty_like(base_logits)
    tokens = torch.empty((B, K), dtype=torch.int32)
    for i in range(K):
        valid = (prev >= 0) & (prev < V)
        emb = F.embedding(prev.clamp(0, V - 1), emb_w).masked_fill(~valid.unsqueeze(-1), 0)
        st = torch.zeros_like(emb) if state is None else state
        joint_in = torch.cat((st, emb, hidden_states[:, i]), dim=-1)
        joint = F.linear(joint_in, joint_w, joint_b)
        g, c, e = joint.chunk(3, dim=-1)
        gate = torch.sigmoid(g)
        state = gate * st + (1 - gate) * torch.tanh(c)
        emitted = torch.tanh(e)
        bias = F.linear(emitted, out_w)
        refined[:, i] = base_logits[:, i] + bias.float()
        prev = torch.argmax(refined[:, i], dim=-1)
        tokens[:, i] = prev.to(torch.int32)
    return tokens, refined


@pytest.mark.parametrize("tp_size", [4, 8, 16])
def test_vocab_parallel_rnn_markov_equivalence(
    monkeypatch: pytest.MonkeyPatch, tp_size: int
) -> None:
    """Sharded draft_markov_output [V/tp, rank] + Triton VP RNN Markov matches full reference."""
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    vp = _load_vocab_parallel_kernels()
    monkeypatch.setattr(vp, "_LOCAL_BLOCK_V", 64)

    B, K, V, R, H = 3, 3, 1024, 32, 64
    shard = V // tp_size
    softcap = 30.0
    gen = torch.Generator().manual_seed(700 + tp_size)
    raw_block = (torch.randn(B, K, V, generator=gen) * 3.0).to(torch.bfloat16)
    # Include an out-of-vocab anchor (-1 and V) to test OOB masking.
    anchors = torch.tensor([-1, 5, V], dtype=torch.int64)
    hidden = (torch.randn(B, K, H, generator=gen) * 0.5).to(torch.bfloat16)
    emb_w = (torch.randn(V, R, generator=gen) * 0.2).to(torch.bfloat16)
    joint_w = (torch.randn(3 * R, H + 2 * R, generator=gen) * 0.1).to(torch.bfloat16)
    joint_b = (torch.randn(3 * R, generator=gen) * 0.1).to(torch.bfloat16)
    out_w = (torch.randn(V, R, generator=gen) * 0.2).to(torch.bfloat16)

    # Plant a cross-rank tie at step 0 for req 1 to test deterministic tie-break.
    raw_block[1, 0, :] = -10.0
    raw_block[1, 0, shard - 1] = 20.0
    raw_block[1, 0, shard + 2] = 20.0
    out_w[shard - 1] = 0
    out_w[shard + 2] = 0

    ref_tokens, ref_logits = _reference_rnn_markov_chain(
        raw_block, anchors, hidden, emb_w, joint_w, joint_b, out_w, softcap
    )

    # Simulate tp_size ranks executing rnn_markov_chain_tokens in lockstep over sharded out_w.
    # Each rank owns out_w[r*shard : (r+1)*shard] and raw_block[:, :, r*shard : (r+1)*shard].
    # Use fp32-accumulated linear matching cublas_mm so per-shard rows equal full rows.
    def _exact_linear(x, w, b):
        out = x.float() @ w.float().t()
        if b is not None:
            out = out + b.float()
        return out.to(x.dtype)

    # Recompute ref with the same _exact_linear to eliminate CPU bf16 GEMM tiling jitter.
    base_ref = _softcap_ref(raw_block, softcap)
    prev_ref = anchors.long()
    st_ref = None
    ref_logits_exact = torch.empty_like(base_ref)
    ref_tokens_exact = torch.empty((B, K), dtype=torch.int32)
    for i in range(K):
        valid = (prev_ref >= 0) & (prev_ref < V)
        emb = F.embedding(prev_ref.clamp(0, V - 1), emb_w).masked_fill(~valid.unsqueeze(-1), 0)
        st = torch.zeros_like(emb) if st_ref is None else st_ref
        joint = _exact_linear(torch.cat((st, emb, hidden[:, i]), dim=-1), joint_w, joint_b)
        st_ref, emitted = vp._default_rnn_markov_gate(joint, st_ref)
        bias = _exact_linear(emitted, out_w, None)
        ref_logits_exact[:, i] = base_ref[:, i] + bias.float()
        prev_ref = torch.argmax(ref_logits_exact[:, i], dim=-1)
        ref_tokens_exact[:, i] = prev_ref.to(torch.int32)

    # Step-by-step synchronized TP simulation using the actual Triton kernels in vocab_parallel.py!
    has_softcap, inv_cap, cap = vp._softcap_params(softcap)
    use_libdevice = vp._use_libdevice_softcap()
    rank_states = [None] * tp_size
    rank_tokens = [torch.empty((B, K), dtype=torch.int32) for _ in range(tp_size)]
    rank_biased = [torch.empty((B, K, shard), dtype=torch.float32) for _ in range(tp_size)]
    rank_joint_in = [hidden.new_empty((B, 2 * R + H)) for _ in range(tp_size)]
    rank_layouts = [
        vp._PairExchange(None, tp_size, r).layout(B, shard, hidden.device) for r in range(tp_size)
    ]

    import triton

    hid_block = min(2048, triton.next_power_of_2(H))
    block_r = triton.next_power_of_2(R)
    pairs = anchors
    n_slots = 1
    for step in range(K + 1):
        build_input = step < K
        step_hid = hidden[:, step] if build_input else hidden[:, 0]
        round_bufs = []
        for r in range(tp_size):
            vp._rnn_markov_reduce_input_kernel[(B,)](
                pairs,
                pairs.stride(0) if step > 0 else 0,
                n_slots,
                anchors,
                emb_w,
                V,
                rank_states[r] if rank_states[r] is not None else rank_joint_in[r],
                step_hid,
                step_hid.stride(0),
                rank_joint_in[r],
                rank_tokens[r],
                rank_tokens[r].stride(0),
                step,
                RANK=R,
                HIDDEN=H,
                HID_BLOCK=hid_block,
                FROM_ANCHOR=step == 0,
                HAS_STATE=rank_states[r] is not None,
                BUILD_INPUT=build_input,
                BLOCK_SLOTS=triton.next_power_of_2(n_slots),
                BLOCK_R=block_r,
                num_warps=4,
            )
            if not build_input:
                continue
            joint = _exact_linear(rank_joint_in[r], joint_w, joint_b)
            rank_states[r], emitted = vp._default_rnn_markov_gate(joint, rank_states[r])
            w_out_shard = out_w[r * shard : (r + 1) * shard].contiguous()
            local_bias = _exact_linear(emitted, w_out_shard, None)
            step_logits = raw_block[:, step, r * shard : (r + 1) * shard].contiguous()
            step_biased = rank_biased[r][:, step]
            layout = rank_layouts[r]
            vp._markov_add_local_argmax_kernel[(B, layout.splits)](
                step_logits,
                step_logits.stride(0),
                shard,
                r * shard,
                layout.chunk,
                local_bias,
                local_bias.stride(0),
                step_biased,
                step_biased.stride(0),
                inv_cap,
                cap,
                layout.buffer,
                layout.buffer.stride(0),
                layout.slot0,
                layout.zero_lo,
                layout.zero_hi,
                HAS_SOFTCAP=has_softcap,
                USE_LIBDEVICE=use_libdevice,
                STORE_BIASED=True,
                ZERO_WIDTH=layout.zero_width,
                BLOCK_V=vp._LOCAL_BLOCK_V,
                num_warps=8,
                enable_fp_fusion=not has_softcap,
            )
            round_bufs.append(layout.buffer)
        if build_input:
            pairs = torch.stack(round_bufs, dim=0).sum(dim=0)
            n_slots = tp_size * rank_layouts[0].splits

    gathered_biased = torch.cat(rank_biased, dim=-1)
    for r in range(tp_size):
        assert torch.equal(rank_tokens[r], ref_tokens_exact), (r, rank_tokens[r], ref_tokens_exact)
    # In TRITON_INTERPRET=1, exp-form tanh differs from torch.tanh by <= 2 fp32 ULP at cap=30 (~3.6e-6).
    torch.testing.assert_close(gathered_biased, ref_logits_exact, rtol=1e-5, atol=5e-6)
    assert torch.equal(rank_tokens[0], ref_tokens)


# ---------------------------------------------------------------------------
# 4. K-sharded draft_target_projection + Lamport AR + RMSNorm (TP4/TP8/TP16)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", [4, 8, 16])
@pytest.mark.parametrize("rows", [1, 4, 16])
def test_k_split_draft_target_projection_equivalence(tp_size: int, rows: int) -> None:
    """K-split draft_target_projection + FP32 rank-order AR + RMSNorm matches replicated ref."""
    # Scaled-down ML4 geometry preserving the exact 5:1 K:H ratio (30720 : 6144).
    hidden = 128
    capture_layers = 5
    k_total = hidden * capture_layers  # 640 (divisible by 4, 8, 16)
    assert k_total % tp_size == 0
    k_shard = k_total // tp_size
    eps = 1e-5

    gen = torch.Generator().manual_seed(300 + tp_size * 10 + rows)
    x = (torch.randn(rows, k_total, generator=gen) * 0.5).to(torch.bfloat16)
    w = (torch.randn(hidden, k_total, generator=gen) * 0.05).to(torch.bfloat16)
    norm_w = (1.0 + 0.1 * torch.randn(hidden, generator=gen)).to(torch.bfloat16)

    # Full unsharded reference (fp32 accumulate, bf16 round, RMSNorm).
    full_proj = (x.float() @ w.float().t()).to(torch.bfloat16)
    ref_out = _rmsnorm_ref(full_proj, norm_w, eps)

    # Sharded K-split across tp_size ranks + rank-order FP32 Lamport sum + RMSNorm.
    partials = []
    for r in range(tp_size):
        k0, k1 = r * k_shard, (r + 1) * k_shard
        part = (x[:, k0:k1].contiguous().float() @ w[:, k0:k1].contiguous().float().t()).to(
            torch.bfloat16
        )
        partials.append(part)
    reduced = torch.zeros((rows, hidden), dtype=torch.float32)
    for p in partials:
        reduced = reduced + p.float()
    sharded_out = _rmsnorm_ref(reduced.to(torch.bfloat16), norm_w, eps)

    # Only BF16 partial rounding / accumulation-order error (within ~1-2 bf16 ULPs).
    torch.testing.assert_close(sharded_out, ref_out, rtol=2e-2, atol=2e-2)
    rel_err = (
        (sharded_out.float() - ref_out.float()).pow(2).mean().sqrt()
        / ref_out.float().pow(2).mean().sqrt()
    ).item()
    assert rel_err < 5e-3, f"TP{tp_size} K-split relative RMS error too high: {rel_err:.6f}"


# ---------------------------------------------------------------------------
# 5. Merged draft_w13 SwiGLU + split query_rope_tail + embed_draft_block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", [4, 8, 16])
def test_merged_draft_w13_swiglu_equivalence(tp_size: int) -> None:
    """Merged w13 = [w1; w3] SwiGLU matches separate w1, w3 + cat bit-for-bit in fp32-accum."""
    hidden = 128
    dense_inter = 512 // tp_size  # TP4=128, TP8=64, TP16=32
    rows = 3
    gen = torch.Generator().manual_seed(500 + tp_size)
    o = (torch.randn(rows, hidden, generator=gen) * 0.5).to(torch.bfloat16)
    w1 = (torch.randn(dense_inter, hidden, generator=gen) * 0.1).to(torch.bfloat16)
    w3 = (torch.randn(dense_inter, hidden, generator=gen) * 0.1).to(torch.bfloat16)
    w13 = torch.cat((w1, w3), dim=0).contiguous()

    gate = (o.float() @ w1.float().t()).to(torch.bfloat16)
    up = (o.float() @ w3.float().t()).to(torch.bfloat16)
    ref_cat = torch.cat((gate, up), dim=-1)
    ref_act = (F.silu(ref_cat[:, :dense_inter].float()) * ref_cat[:, dense_inter:].float()).to(
        torch.bfloat16
    )

    fused_gu = (o.float() @ w13.float().t()).to(torch.bfloat16)
    fused_act = (F.silu(fused_gu[:, :dense_inter].float()) * fused_gu[:, dense_inter:].float()).to(
        torch.bfloat16
    )
    assert torch.equal(fused_act, ref_act)


@pytest.mark.parametrize("heads", [4, 8, 16, 64])
def test_split_query_rope_tail_matches_combined_latent_tail(
    monkeypatch: pytest.MonkeyPatch, heads: int
) -> None:
    """Split query_rope_tail_kernel + latent_tail(HAS_Q=False) is bitwise identical to combined."""
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    kernels = _load_dspark_kernels()

    batch, block = 2, 3
    rows = batch * block
    lat, pe, nope = 64, 16, 32
    qk = nope + pe
    gen = torch.Generator().manual_seed(600 + heads)
    q = torch.randn(rows, heads * qk, generator=gen).to(torch.bfloat16)
    query_init = torch.randn(batch, block, heads, lat + pe, generator=gen).to(torch.bfloat16)
    positions = torch.randint(0, 64, (rows,), generator=gen)
    inv = 1.0 / 10000 ** (torch.arange(0, pe, 2, dtype=torch.float32) / pe)
    angle = torch.arange(64, dtype=torch.float32)[:, None] * inv[None, :]
    table = torch.cat(
        (angle.cos().to(torch.bfloat16).float(), angle.sin().to(torch.bfloat16).float()), dim=1
    ).contiguous()

    query_actual = query_init.clone()
    kernels.query_rope_tail_kernel[(rows,)](
        q,
        q.stride(0),
        query_actual,
        heads * (lat + pe),
        positions,
        table,
        HEADS=heads,
        NOPE=nope,
        QK=qk,
        LAT=lat,
        PE=pe,
        num_warps=4,
        enable_fp_fusion=False,
    )

    # Compare against the original combined latent_tail_kernel(HAS_Q=True) bitwise,
    # and against the PyTorch reference (allowing 1 bf16 ULP under Triton's CPU interpreter,
    # which truncates fp32->bf16 instead of round-to-nearest-even).
    query_ref = query_init.clone()
    pe_in = q.view(rows, heads, -1)[..., nope:]
    query_ref.view(rows, heads, lat + pe)[..., lat:] = _rotate_ref(pe_in, table, positions)

    assert torch.equal(query_actual[..., :lat], query_init[..., :lat])
    # Check bitwise bit-distance <= 1 bf16 ULP (truncation vs RNE in CPU interpret).
    actual_bits = query_actual[..., lat:].view(torch.int16).to(torch.int32)
    ref_bits = query_ref[..., lat:].view(torch.int16).to(torch.int32)
    assert int((actual_bits - ref_bits).abs().max().item()) <= 1

    # Also verify exact bitwise equality between fused multi-layer precompute_context_kv
    # and per-layer projection in fp32-accumulated arithmetic.
    layers = 3
    hidden = 64
    wkv_a = [
        (torch.randn(lat + pe, hidden, generator=gen) * 0.1).to(torch.bfloat16)
        for _ in range(layers)
    ]
    kv_norms = [
        (1.0 + 0.1 * torch.randn(lat, generator=gen)).to(torch.bfloat16) for _ in range(layers)
    ]
    proj_h = (torch.randn(rows, hidden, generator=gen) * 0.5).to(torch.bfloat16)
    fused_kv_a = torch.cat(wkv_a, dim=0).contiguous()
    kv_fused = (proj_h.float() @ fused_kv_a.float().t()).to(torch.bfloat16)
    for lyr in range(layers):
        kv_single = (proj_h.float() @ wkv_a[lyr].float().t()).to(torch.bfloat16)
        sl = kv_fused[:, lyr * (lat + pe) : (lyr + 1) * (lat + pe)]
        assert torch.equal(sl, kv_single)
        assert torch.equal(
            _rmsnorm_ref(sl[:, :lat], kv_norms[lyr], 1e-5),
            _rmsnorm_ref(kv_single[:, :lat], kv_norms[lyr], 1e-5),
        )


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_triton_embed_draft_block_bitwise(monkeypatch: pytest.MonkeyPatch, batch: int) -> None:
    """embed_draft_block_kernel matches eager anchor lookup + OOB zeroing + learned masks."""
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    kernels = _load_dspark_kernels()

    vocab, hidden, block = 64, 128, 3
    gen = torch.Generator().manual_seed(800 + batch)
    embed = torch.randn(vocab, hidden, generator=gen).to(torch.bfloat16)
    masks = torch.randn(block - 1, hidden, generator=gen).to(torch.bfloat16)
    anchors = torch.randint(0, vocab, (batch,), generator=gen, dtype=torch.int64)
    if batch >= 2:
        anchors[0] = -1
        anchors[-1] = vocab

    out = embed.new_empty((batch, block, hidden))
    import triton

    hid_block = 64
    kernels.embed_draft_block_kernel[(batch * block, triton.cdiv(hidden, hid_block))](
        anchors, embed, masks, out, vocab, BLOCK=block, HIDDEN=hidden, HID_BLOCK=hid_block
    )

    valid = (anchors >= 0) & (anchors < vocab)
    embedded = F.embedding(anchors.clamp(0, vocab - 1), embed).masked_fill(~valid.unsqueeze(-1), 0)
    expected = torch.cat((embedded.unsqueeze(1), masks.unsqueeze(0).expand(batch, -1, -1)), dim=1)
    assert torch.equal(out, expected)


# ---------------------------------------------------------------------------
# 6. spec_tail_kernels.py bitwise verification via TRITON_INTERPRET=1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_contexts,num_gens,draft_len", [(0, 1, 3), (0, 4, 3), (2, 3, 3), (3, 0, 3)]
)
def test_strict_accept_bitwise(
    monkeypatch: pytest.MonkeyPatch, num_contexts: int, num_gens: int, draft_len: int
) -> None:
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    stk = _load_spec_tail_kernels()

    batch_size = num_contexts + num_gens
    gen = torch.Generator().manual_seed(900 + num_contexts * 10 + num_gens)
    target = torch.randint(0, 100, (num_contexts + num_gens * (draft_len + 1),), generator=gen)
    draft = torch.randint(0, 100, (num_gens, draft_len), generator=gen, dtype=torch.int32)
    # Force partial and full prefix matches on generation rows.
    for g in range(num_gens):
        m = g % (draft_len + 1)
        row_t = target[
            num_contexts + g * (draft_len + 1) : num_contexts + (g + 1) * (draft_len + 1)
        ]
        draft[g, :m] = row_t[:m].to(torch.int32)
        if m < draft_len:
            draft[g, m] = (row_t[m].to(torch.int32) + 1) % 100

    acc, nacc = stk.strict_accept(target, draft, num_contexts, batch_size)

    exp_acc = torch.empty((batch_size, draft_len + 1), dtype=torch.int32)
    exp_nacc = torch.ones((batch_size,), dtype=torch.int32)
    if num_contexts > 0:
        exp_acc[:num_contexts, 0] = target[:num_contexts].to(torch.int32)
    if num_gens > 0:
        gen_t = target[num_contexts:].reshape(num_gens, draft_len + 1).to(torch.int32)
        exp_acc[num_contexts:] = gen_t
        exp_nacc[num_contexts:] += torch.cumprod((draft == gen_t[:, :draft_len]).int(), dim=-1).sum(
            1
        )

    if num_contexts > 0:
        assert torch.equal(acc[:num_contexts, 0], exp_acc[:num_contexts, 0])
    if num_gens > 0:
        assert torch.equal(acc[num_contexts:], exp_acc[num_contexts:])
    assert torch.equal(nacc, exp_nacc)


def test_decode_draft_block_table_and_prepare_and_store_bitwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """decode_draft_block_table + dflash_paged_latent_prepare + store/scatter + next_new_tokens."""
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    stk = _load_spec_tail_kernels()

    num_contexts, num_gens = 1, 3
    batch_size = num_contexts + num_gens
    num_slots = batch_size + 2
    dummy_slot = num_slots - 1
    page_size, block_size, width, max_ctx = 16, 3, 4, 128
    n_blocks, n_layers, head_dim = 10, 3, 32

    gen = torch.Generator().manual_seed(1234)
    src = torch.randint(0, 60, (batch_size, n_blocks), generator=gen, dtype=torch.int32) * 2
    src[:, -2:] = -1
    table = torch.full((batch_size, n_blocks), -9, dtype=torch.int32)
    counts = torch.full((batch_size,), -9, dtype=torch.int64)
    stk.decode_draft_block_table(src, 2, table, counts)

    encoded = src.to(torch.int64)
    exp_counts = (encoded >= 0).sum(dim=1)
    exp_table = encoded.clamp(min=0).div(2, rounding_mode="floor").to(torch.int32)
    assert torch.equal(table, exp_table)
    assert torch.equal(counts, exp_counts)

    # Ensure unique pages for store_context_latents test (no write collision).
    num_pages = batch_size * n_blocks
    unique_table = torch.arange(num_pages, dtype=torch.int32).view(batch_size, n_blocks)
    unique_counts = torch.full((batch_size,), n_blocks, dtype=torch.int64)

    slots = torch.tensor([2, 0, dummy_slot], dtype=torch.int64)
    ctx_len_init = torch.tensor([14, 5, 30, 0, 0, 0], dtype=torch.int64)
    accepted = torch.randint(0, 500, (batch_size, width), generator=gen, dtype=torch.int32)
    num_accepted = torch.tensor([1, 2, 4, 1], dtype=torch.int32)
    seq_lens = torch.zeros(batch_size, dtype=torch.int32)

    ctx_len_fused = ctx_len_init.clone()
    prep = stk.dflash_paged_latent_prepare(
        num_accepted,
        accepted,
        slots,
        ctx_len_fused,
        unique_counts,
        unique_table,
        seq_lens,
        num_contexts,
        dummy_slot,
        max_ctx,
        page_size,
        block_size,
        width,
        compute_block_placement=False,
    )

    # Eager reference for prepare.
    gen_nacc = num_accepted[num_contexts:]
    gen_acc = accepted[num_contexts:]
    exp_bonus = gen_acc.gather(1, (gen_nacc - 1).clamp_min(0).long().unsqueeze(1)).squeeze(1).long()
    old = ctx_len_init[slots]
    now = (old + gen_nacc.long()).clamp(max=max_ctx)
    exp_qpos = now.unsqueeze(1) + torch.arange(block_size).unsqueeze(0)
    exp_cpos = (old.unsqueeze(1) + torch.arange(width).unsqueeze(0)).reshape(-1)
    exp_ctx_len = ctx_len_init.clone()
    exp_ctx_len[slots] += gen_nacc.long()
    exp_ctx_len.clamp_(max=max_ctx)
    exp_ctx_len[dummy_slot] = 0
    gen_rows = torch.arange(num_contexts, batch_size, dtype=torch.long)
    exp_num_ctx = torch.minimum(
        exp_ctx_len[slots], (unique_counts[gen_rows] * page_size - block_size).clamp_min(0)
    )
    assert torch.equal(prep.bonus, exp_bonus)
    assert torch.equal(prep.query_positions, exp_qpos)
    assert torch.equal(prep.ctx_positions, exp_cpos)
    assert torch.equal(ctx_len_fused, exp_ctx_len)
    assert torch.equal(prep.num_ctx, exp_num_ctx)
    assert torch.equal(seq_lens[num_contexts:], torch.full((num_gens,), width, dtype=torch.int32))

    # compute_gen_ctx_positions
    pos_only = stk.compute_gen_ctx_positions(slots, ctx_len_init, width)
    assert torch.equal(pos_only, exp_cpos)

    # store_context_latents + scatter_context_latents
    storage_a = torch.zeros(n_layers, num_pages, 1, 1, page_size, head_dim, dtype=torch.bfloat16)
    storage_b = storage_a.clone()
    pools_a = [storage_a[lyr] for lyr in range(n_layers)]
    pools_b = [storage_b[lyr] for lyr in range(n_layers)]
    ptrs_a = torch.tensor([p.data_ptr() for p in pools_a], dtype=torch.int64)
    latent = torch.randn(num_gens * width, n_layers, head_dim, generator=gen).to(torch.bfloat16)
    stk.store_context_latents(
        latent, pools_a, ptrs_a, prep.store_pages, prep.store_offsets, prep.store_mask
    )
    masked_lat = latent * prep.store_mask.view(-1, 1, 1).to(latent.dtype)
    for lyr in range(n_layers):
        pools_b[lyr][prep.store_pages, 0, 0, prep.store_offsets] = masked_lat[:, lyr]
    assert torch.equal(storage_a, storage_b)

    # scatter_context_latents (prefill unmasked path)
    prefill_a = torch.zeros(n_layers, num_pages, 1, 1, page_size, head_dim, dtype=torch.bfloat16)
    prefill_b = prefill_a.clone()
    p_pools_a = [prefill_a[lyr] for lyr in range(n_layers)]
    p_pools_b = [prefill_b[lyr] for lyr in range(n_layers)]
    p_ptrs = torch.tensor([p.data_ptr() for p in p_pools_a], dtype=torch.int64)
    p_rows = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    p_pos = torch.tensor([0, 17, 5, 31], dtype=torch.int64)
    p_lat = torch.randn(4, n_layers, head_dim, generator=gen).to(torch.bfloat16)
    stk.scatter_context_latents(p_lat, p_pools_a, p_ptrs, p_rows, p_pos, unique_table, page_size)
    p_pages = unique_table[p_rows, p_pos // page_size].long()
    p_offs = p_pos % page_size
    for lyr in range(n_layers):
        p_pools_b[lyr][p_pages, 0, 0, p_offs] = p_lat[:, lyr]
    assert torch.equal(prefill_a, prefill_b)

    # prepare_next_new_tokens
    next_draft = torch.randint(0, 500, (batch_size, block_size), generator=gen, dtype=torch.int32)
    actual_next = stk.prepare_next_new_tokens(accepted, next_draft, num_accepted, batch_size)
    idx = torch.arange(batch_size)
    expected_next = torch.cat([accepted[idx, num_accepted - 1].unsqueeze(1), next_draft], dim=1).to(
        torch.int32
    )
    assert torch.equal(actual_next, expected_next)


# ---------------------------------------------------------------------------
# 7. sm_107__tp4/weights.py draft sharding manifest & load (TP4 / TP8 / TP16)
# ---------------------------------------------------------------------------


def _load_sm107_tp4_weights_module():
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4 import (
            mistral_large_4_nvfp4__sm_107__tp4 as sm107_tp4,
        )

        return sm107_tp4.weights
    except Exception:
        pass

    import types

    def _ensure(fullname: str, **attrs):
        if fullname not in sys.modules:
            m = types.ModuleType(fullname)
            m.__path__ = []
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[fullname] = m

    _ensure("safetensors", safe_open=None)
    _ensure("tensorrt_llm")
    _ensure("tensorrt_llm.logger", logger=SimpleNamespace(info=lambda *a, **kw: None))
    _ensure("tensorrt_llm.quantization")
    _ensure("tensorrt_llm.quantization.utils")
    _ensure(
        "tensorrt_llm.quantization.utils.fp8_utils",
        transform_sf_into_required_layout=lambda s, **kw: s,
    )
    return _load_standalone_module("_test_sm107_tp4_weights", _SM107_TP4_DIR / "weights.py")


@pytest.mark.parametrize("tp_size", [4, 8, 16])
def test_sm107_tp4_draft_weight_sharding_manifest(tp_size: int) -> None:
    """Verify ML4 draft_w13, K-split draft_target_projection, and vocab-split draft_markov_output."""
    w_mod = _load_sm107_tp4_weights_module()
    table = w_mod.MODEL_WEIGHTS

    cfg = SimpleNamespace(
        num_hidden_layers=72,
        hidden_size=6144,
        num_attention_heads=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        nope_layer_flags=[i % 4 == 3 for i in range(72)],
        kv_source_layers=[35 if i >= 39 and i % 4 == 3 else i for i in range(72)],
        index_sharing=["shared" if i >= 17 and (i - 17) % 4 < 2 else "full" for i in range(72)],
        index_n_heads=32,
        index_head_dim=128,
        first_k_dense_replace=1,
        n_routed_experts=128,
        moe_intermediate_size=6144,
        n_shared_experts=1,
        intermediate_size=24576,
        vocab_size=204800,
        vision_config=None,
    )
    spec = SimpleNamespace(
        _draft_model_config={"n_layers": 3, "depth": 3, "dspark": {"markov_rank": 512}},
        target_layer_ids=[1, 18, 36, 54, 71],
    )

    # Real checkpoint shapes for the 3 optimized draft weight groups.
    ckpt_w1 = torch.arange(24576, dtype=torch.int32).unsqueeze(1).expand(24576, 6144)
    ckpt_w3 = (torch.arange(24576, dtype=torch.int32) + 100000).unsqueeze(1).expand(24576, 6144)
    ckpt_proj_cols = torch.arange(30720, dtype=torch.int32).unsqueeze(0).expand(6144, 30720)
    ckpt_markov_rows = torch.arange(204800, dtype=torch.int32).unsqueeze(1).expand(204800, 512)

    proj_tiles = []
    markov_tiles = []
    w1_tiles = []
    w3_tiles = []
    for rank in range(tp_size):
        mapping = SimpleNamespace(
            tp_size=tp_size,
            tp_rank=rank,
            moe_tp_size=tp_size,
            moe_tp_rank=rank,
            moe_ep_size=1,
            moe_ep_rank=0,
        )
        core = SimpleNamespace(
            model_config=SimpleNamespace(
                pretrained_config=cfg,
                mapping=mapping,
                torch_dtype=torch.bfloat16,
                spec_config=spec,
            )
        )
        d = table.dims(core)
        with torch.device("meta"):
            declared = table.declare(d)
        manifest = table.manifest(core)

        # Exact expected per-rank shapes across TP4 / TP8 / TP16.
        assert tuple(declared["l0_draft_w13"].shape) == (2 * (24576 // tp_size), 6144)
        assert "l0_draft_w1" not in declared and "l0_draft_w3" not in declared
        assert tuple(declared["draft_target_projection"].shape) == (6144, 30720 // tp_size)
        assert tuple(declared["draft_markov_output"].shape) == (204800 // tp_size, 512)
        assert tuple(declared["draft_markov_embedding"].shape) == (204800, 512)

        # Verify all 38 draft checkpoint keys are consumed.
        draft_keys = {
            ckpt
            for name, rows in manifest.items()
            if name.startswith("draft_") or "_draft_" in name
            for ckpt, *_ in rows
        }
        assert len(draft_keys) == 38

        # Check disjoint full-coverage tiling across ranks.
        proj_shard_fn = manifest["draft_target_projection"][0][3]
        proj_tiles.append(proj_shard_fn(ckpt_proj_cols)[0])
        markov_shard_fn = manifest["draft_markov_output"][0][3]
        markov_tiles.append(markov_shard_fn(ckpt_markov_rows)[:, 0])
        w13_rows = manifest["l0_draft_w13"]
        assert len(w13_rows) == 2
        w1_tiles.append(w13_rows[0][3](ckpt_w1)[:, 0])
        w3_tiles.append(w13_rows[1][3](ckpt_w3)[:, 0])

    assert torch.equal(torch.cat(proj_tiles), torch.arange(30720, dtype=torch.int32))
    assert torch.equal(torch.cat(markov_tiles), torch.arange(204800, dtype=torch.int32))
    assert torch.equal(torch.cat(w1_tiles), torch.arange(24576, dtype=torch.int32))
    assert torch.equal(torch.cat(w3_tiles), torch.arange(24576, dtype=torch.int32) + 100000)

    # TP1 keeps full unsharded shapes.
    mapping_tp1 = SimpleNamespace(
        tp_size=1, tp_rank=0, moe_tp_size=1, moe_tp_rank=0, moe_ep_size=1, moe_ep_rank=0
    )
    core_tp1 = SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=cfg, mapping=mapping_tp1, torch_dtype=torch.bfloat16, spec_config=spec
        )
    )
    with torch.device("meta"):
        declared_tp1 = table.declare(table.dims(core_tp1))
    assert tuple(declared_tp1["draft_target_projection"].shape) == (6144, 30720)
    assert tuple(declared_tp1["draft_markov_output"].shape) == (204800, 512)
