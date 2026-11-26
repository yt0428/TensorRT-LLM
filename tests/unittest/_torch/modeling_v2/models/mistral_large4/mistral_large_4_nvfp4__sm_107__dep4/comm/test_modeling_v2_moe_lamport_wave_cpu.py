# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit, Triton-interpreter, concurrency, and PTX contract tests for the SM107 DEP4/DEP8
one-shot Lamport MoE dispatch/combine kernels (parallel wave peer-polling, rank-rotated sends,
and deferred non-polluting streaming sentinel clears).
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import pytest
import torch
import triton

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[8]
_COMM_DIR = (
    _REPO_ROOT
    / "tensorrt_llm"
    / "_torch"
    / "_experimental"
    / "modeling_v2"
    / "models"
    / "mistral_large4"
    / "mistral_large_4_nvfp4__sm_107__dep4"
    / "ops"
    / "comm"
)
_KERNEL_PATH = _COMM_DIR / "_moe_lamport_kernel.py"


def _load_kernel_module():
    name = "_ml4_dep_moe_lamport_kernel_cpu_test"
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, _KERNEL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_interpret_helper(func_name: str, payload: dict) -> None:
    """Run an interpreter helper in a subprocess with ``TRITON_INTERPRET=1`` set before
    ``import triton`` so ``triton.language.standard`` initializes in interpreter mode."""
    if os.environ.get("TRITON_INTERPRET") == "1":
        globals()[func_name](**payload)
        return
    env = dict(os.environ, TRITON_INTERPRET="1")
    runner = (
        "import importlib.util, json, sys; "
        "spec = importlib.util.spec_from_file_location('_m', sys.argv[1]); "
        "mod = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        "getattr(mod, sys.argv[2])(**json.loads(sys.argv[3]))"
    )
    res = subprocess.run(
        [
            sys.executable,
            "-c",
            runner,
            str(pathlib.Path(__file__).resolve()),
            func_name,
            json.dumps(payload),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0:
        raise AssertionError(
            f"TRITON_INTERPRET subprocess failed for {func_name}({payload}):\n"
            f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
        )


def test_resolve_combine_wave_size_bounds() -> None:
    k = _load_kernel_module()
    assert k.COMBINE_POLL_WAVE_SIZE == 4
    assert k.resolve_combine_wave_size(1) == 1
    assert k.resolve_combine_wave_size(2) == 2
    assert k.resolve_combine_wave_size(4) == 4
    assert k.resolve_combine_wave_size(6) == 4
    assert k.resolve_combine_wave_size(8) == 4
    for bad in (0, -1, 9, 16):
        with pytest.raises(ValueError):
            k.resolve_combine_wave_size(bad)


def test_source_contracts_no_ab_env_toggles() -> None:
    src = _KERNEL_PATH.read_text()
    assert "os.environ" not in src
    assert 'cache_modifier=".cs"' in src
    assert "def _poll_wave4(" in src
    assert "def _or_combine(" in src


def _impl_dispatch_interpret(world: int, counts: list[int]) -> None:
    k = _load_kernel_module()
    codes_w, scales_w, ids_w, weights_w = 32, 4, 8, 4
    row_w = ((codes_w + scales_w + ids_w + weights_w + 7) // 8) * 8
    block = triton.next_power_of_2(row_w)
    max_rows = max(counts)
    region = max_rows * row_w
    padded = list(counts) + [0] * (8 - len(counts))
    total = sum(counts)

    bufs = [torch.full((3, world, region), -32768, dtype=torch.int16) for _ in range(world)]
    states = [torch.zeros(2, dtype=torch.int32) for _ in range(world)]
    gen = torch.Generator().manual_seed(17 + world)

    for rnd in range(4):
        slot = rnd % 3
        per_rank = []
        for r in range(world):
            n = counts[r]
            codes = torch.randint(-32768, 32767, (n, codes_w), generator=gen, dtype=torch.int16)
            scales = torch.randint(-32768, 32767, (n, scales_w), generator=gen, dtype=torch.int16)
            ids = torch.randint(0, 128, (n, ids_w), generator=gen, dtype=torch.int16)
            weights = torch.randint(0, 16000, (n, weights_w), generator=gen, dtype=torch.int16)
            codes[0, 0] = -32768
            per_rank.append((codes, scales, ids, weights))

        for r in range(world):
            for q in range(world):
                if q == r:
                    continue
                c, s, idx, wt = per_rank[q]
                n_q = counts[q]
                packed = torch.zeros((n_q, row_w), dtype=torch.int16)
                packed[:, :codes_w] = c
                packed[:, codes_w : codes_w + scales_w] = s
                packed[:, codes_w + scales_w : codes_w + scales_w + ids_w] = idx
                packed[:, codes_w + scales_w + ids_w : codes_w + scales_w + ids_w + weights_w] = wt
                packed[packed == -32768] = 0
                bufs[r][slot, q, : n_q * row_w] = packed.reshape(-1)

        send_targets = [
            torch.full((3, world, region), -32768, dtype=torch.int16) for _ in range(world)
        ]
        send_peers = torch.tensor([b.data_ptr() for b in send_targets], dtype=torch.int64)

        exp_parts = []
        for part_idx in range(4):
            cat = torch.cat([per_rank[q][part_idx] for q in range(world)], dim=0).clone()
            cat[cat == -32768] = 0
            exp_parts.append(cat)

        for r in range(world):
            c, s, idx, wt = per_rank[r]
            out_c = torch.empty((total, codes_w), dtype=torch.int16)
            out_s = torch.empty((total, scales_w), dtype=torch.int16)
            out_i = torch.empty((total, ids_w), dtype=torch.int16)
            out_w = torch.empty((total, weights_w), dtype=torch.int16)
            k.dispatch_kernel[(min(world * max_rows, 8),)](
                c,
                s,
                idx,
                wt,
                out_c,
                out_s,
                out_i,
                out_w,
                bufs[r],
                send_peers,
                states[r],
                r,
                *padded,
                region,
                CODES=codes_w,
                SCALES=scales_w,
                IDS=ids_w,
                WEIGHTS=weights_w,
                ROW=row_w,
                WORLD=world,
                BLOCK=block,
                num_warps=4,
            )
            assert int(states[r][0].item()) == (slot + 1) % 3
            assert int(states[r][1].item()) == 0
            assert torch.all(bufs[r][slot] == -32768)
            assert torch.equal(out_c, exp_parts[0])
            assert torch.equal(out_s, exp_parts[1])
            assert torch.equal(out_i, exp_parts[2])
            assert torch.equal(out_w, exp_parts[3])

        for dst_r in range(world):
            for src_r in range(world):
                if src_r == dst_r:
                    continue
                n_s = counts[src_r]
                c, s, idx, wt = per_rank[src_r]
                exp = torch.zeros((n_s, row_w), dtype=torch.int16)
                exp[:, :codes_w] = c
                exp[:, codes_w : codes_w + scales_w] = s
                exp[:, codes_w + scales_w : codes_w + scales_w + ids_w] = idx
                exp[:, codes_w + scales_w + ids_w : codes_w + scales_w + ids_w + weights_w] = wt
                exp[exp == -32768] = 0
                got = send_targets[dst_r][slot, src_r, : n_s * row_w].view(n_s, row_w)
                assert torch.equal(got, exp)


@pytest.mark.parametrize(
    ("world", "counts"),
    [
        (4, [3, 1, 4, 2]),
        (8, [2, 1, 3, 1, 4, 2, 1, 3]),
    ],
    ids=("dep4_ragged", "dep8_ragged"),
)
def test_dispatch_kernel_interpret_multi_round_and_sentinels(world: int, counts: list[int]) -> None:
    """Verify rank-rotated dispatch send/recv, -0x8000 sanitization, 3-slot rotation, and
    sentinel restoration via Triton's CPU interpreter."""
    _run_interpret_helper("_impl_dispatch_interpret", {"world": world, "counts": counts})


def _f32_to_bf16_rtz(x: torch.Tensor) -> torch.Tensor:
    """Match Triton's CPU interpreter (``TRITON_INTERPRET=1``) FP32->BF16 truncation so tests
    verify 0-bit exact rank-order accumulation; GPU PTX emits ``cvt.rn.bf16.f32`` (checked below)."""
    return (x.contiguous().view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)


def _impl_combine_interpret(world: int, counts: list[int], norm: bool) -> None:
    k = _load_kernel_module()
    hidden = 96
    block = triton.next_power_of_2(hidden)
    max_rows = max(counts)
    region = max_rows * hidden
    padded = list(counts) + [0] * (8 - len(counts))
    total = sum(counts)

    bufs = [torch.full((3, world, region), -32768, dtype=torch.int16) for _ in range(world)]
    states = [torch.zeros(2, dtype=torch.int32) for _ in range(world)]
    gen = torch.Generator().manual_seed(73 + world + int(norm))

    for rnd in range(4):
        slot = rnd % 3
        partials = [
            torch.randn((total, hidden), generator=gen, dtype=torch.bfloat16) for _ in range(world)
        ]
        partials[0][0, 0] = -0.0
        partials[1][0, 1] = -0.0
        shareds = [
            torch.randn((counts[r], hidden), generator=gen, dtype=torch.bfloat16)
            for r in range(world)
        ]
        residuals = [
            torch.randn((counts[r], hidden), generator=gen, dtype=torch.bfloat16)
            for r in range(world)
        ]
        res_copies = [r.clone() for r in residuals]
        weight = torch.randn((hidden,), generator=gen, dtype=torch.bfloat16)
        outs = [torch.empty_like(shareds[r]) for r in range(world)]

        for r in range(world):
            lo = sum(counts[:r])
            hi = lo + counts[r]
            for p in range(world):
                if p == r:
                    continue
                words = partials[p][lo:hi].view(torch.int16).clone()
                words[words == -32768] = 0
                bufs[r][slot, p, : counts[r] * hidden] = words.reshape(-1)

        send_targets = [
            torch.full((3, world, region), -32768, dtype=torch.int16) for _ in range(world)
        ]
        send_peers = torch.tensor([b.data_ptr() for b in send_targets], dtype=torch.int64)

        for r in range(world):
            k.combine_kernel[(min(world * max_rows, 8),)](
                partials[r],
                shareds[r],
                res_copies[r] if norm else shareds[r],
                res_copies[r] if norm else outs[r],
                weight if norm else shareds[r],
                outs[r],
                bufs[r],
                send_peers,
                states[r],
                r,
                *padded,
                region,
                1e-5,
                H=hidden,
                WORLD=world,
                BLOCK=block,
                NORM=norm,
                num_warps=4,
            )
            assert int(states[r][0].item()) == (slot + 1) % 3
            assert int(states[r][1].item()) == 0
            assert torch.all(bufs[r][slot] == -32768)

            # Verify 0-bit exact strict rank-order FP32 summation matching TRITON_INTERPRET's RTZ.
            lo = sum(counts[:r])
            hi = lo + counts[r]
            acc = torch.zeros((counts[r], hidden), dtype=torch.float32)
            for p in range(world):
                blk = partials[p][lo:hi].float()
                if p != r:
                    blk = torch.where(blk == 0, torch.zeros_like(blk), blk)
                acc += blk
            mlp = _f32_to_bf16_rtz(_f32_to_bf16_rtz(acc).float() + shareds[r].float())
            if not norm:
                assert torch.equal(outs[r].view(torch.int16), mlp.view(torch.int16))
            else:
                new_res = _f32_to_bf16_rtz(residuals[r].float() + mlp.float())
                assert torch.equal(res_copies[r].view(torch.int16), new_res.view(torch.int16))
                hf = new_res.float()
                inv = torch.rsqrt((hf * hf).sum(dim=-1, keepdim=True) / hidden + 1e-5)
                ref_out = _f32_to_bf16_rtz(hf * inv * weight.float())
                torch.testing.assert_close(outs[r].float(), ref_out.float(), atol=2e-2, rtol=2e-2)

        for dst_r in range(world):
            lo = sum(counts[:dst_r])
            hi = lo + counts[dst_r]
            for src_r in range(world):
                if src_r == dst_r:
                    continue
                got = send_targets[dst_r][slot, src_r, : counts[dst_r] * hidden].view(
                    counts[dst_r], hidden
                )
                exp = partials[src_r][lo:hi].view(torch.int16).clone()
                exp[exp == -32768] = 0
                assert torch.equal(got, exp)


@pytest.mark.parametrize(
    ("world", "counts", "norm"),
    [
        (4, [3, 1, 4, 2], False),
        (4, [3, 1, 4, 2], True),
        (8, [2, 1, 3, 1, 4, 2, 1, 3], False),
        (8, [2, 1, 3, 1, 4, 2, 1, 3], True),
    ],
    ids=("dep4_mlp", "dep4_norm", "dep8_mlp", "dep8_norm"),
)
def test_combine_kernel_interpret_multi_round_and_sentinels(
    world: int, counts: list[int], norm: bool
) -> None:
    """Verify parallel wave polling, strict rank-order summation, deferred streaming sentinel
    clearing, and 3-slot rotation across DEP4 and DEP8."""
    _run_interpret_helper(
        "_impl_combine_interpret", {"world": world, "counts": counts, "norm": norm}
    )


def _impl_combine_delayed_out_of_order() -> None:
    k = _load_kernel_module()
    world = 8
    rank = 3
    hidden = 64
    block = 64
    counts = (1,) * world
    padded = list(counts)
    region = hidden

    gen = torch.Generator().manual_seed(999)
    partials = [
        torch.randn((world, hidden), generator=gen, dtype=torch.bfloat16) for _ in range(world)
    ]
    shared = torch.randn((1, hidden), generator=gen, dtype=torch.bfloat16)
    out = torch.empty_like(shared)
    buf = torch.full((3, world, region), -32768, dtype=torch.int16)
    send_targets = [torch.full((3, world, region), -32768, dtype=torch.int16) for _ in range(world)]
    send_peers = torch.tensor([b.data_ptr() for b in send_targets], dtype=torch.int64)
    state = torch.zeros(2, dtype=torch.int32)

    def _write_peer(p: int, lo_col: int, hi_col: int) -> None:
        w = partials[p][rank, lo_col:hi_col].view(torch.int16).clone()
        w[w == -32768] = 0
        buf[0, p, lo_col:hi_col] = w

    # Pre-populate only peers 2 and 6; deliver the remaining peers out of order and split in half.
    _write_peer(2, 0, hidden)
    _write_peer(6, 0, hidden)

    def _delayed_producer() -> None:
        time.sleep(0.005)
        _write_peer(1, 0, hidden // 2)
        _write_peer(7, 0, hidden)
        time.sleep(0.005)
        _write_peer(0, 0, hidden)
        _write_peer(5, 0, hidden // 2)
        time.sleep(0.005)
        _write_peer(1, hidden // 2, hidden)
        _write_peer(4, 0, hidden)
        _write_peer(5, hidden // 2, hidden)

    producer = threading.Thread(target=_delayed_producer, daemon=True)
    producer.start()
    try:
        k.combine_kernel[(1,)](
            partials[rank],
            shared,
            shared,
            out,
            shared,
            out,
            buf,
            send_peers,
            state,
            rank,
            *padded,
            region,
            1e-5,
            H=hidden,
            WORLD=world,
            BLOCK=block,
            NORM=False,
            num_warps=4,
        )
    finally:
        producer.join(timeout=5.0)
    assert not producer.is_alive()
    assert int(state[0].item()) == 1
    assert torch.all(buf[0] == -32768)

    acc = torch.zeros((hidden,), dtype=torch.float32)
    for p in range(world):
        blk = partials[p][rank].float()
        if p != rank:
            blk = torch.where(blk == 0, torch.zeros_like(blk), blk)
        acc += blk
    ref = _f32_to_bf16_rtz(_f32_to_bf16_rtz(acc).float() + shared[0].float())
    assert torch.equal(out[0].view(torch.int16), ref.view(torch.int16))


def test_combine_parallel_wave_delayed_out_of_order_arrival() -> None:
    """Exercise ``_poll_wave4``'s retry loop under delayed, split-half, out-of-order peer
    writes on an 8-rank workspace."""
    _run_interpret_helper("_impl_combine_delayed_out_of_order", {})


def test_ptx_vectorization_streaming_clear_and_barrier_budget() -> None:
    """Offline SM100 PTX compile verifies vectorized volatile loads, streaming L2::cs sentinel
    stores, and reduced CTA barrier counts for DEP4 and DEP8."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    k = _load_kernel_module()
    target = GPUTarget("cuda", 100, 32)
    combine_sig = {
        "partial": "*bf16",
        "shared": "*bf16",
        "residual": "*bf16",
        "residual_out": "*bf16",
        "weight": "*bf16",
        "out": "*bf16",
        "buf": "*i16",
        "peers": "*i64",
        "state": "*i32",
        "rank": "i32",
        "c0": "i32",
        "c1": "i32",
        "c2": "i32",
        "c3": "i32",
        "c4": "i32",
        "c5": "i32",
        "c6": "i32",
        "c7": "i32",
        "region": "i32",
        "eps": "fp32",
    }
    combine_attrs = {(i,): [["tt.divisibility", 16]] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 18)}
    for world, max_barriers in ((4, 14), (8, 18)):
        compiled = triton.compile(
            ASTSource(
                fn=k.combine_kernel,
                signature=combine_sig,
                constexprs={"H": 6144, "WORLD": world, "BLOCK": 8192, "NORM": True},
                attrs=combine_attrs,
            ),
            target=target,
            options={"num_warps": 8},
        )
        ptx = compiled.asm["ptx"]
        assert ptx.count("ld.volatile.global.v4") >= 8 * world
        assert ptx.count("st.global.cs.v4") >= 4 * world
        assert "cvt.rn.bf16" in ptx
        assert ptx.count("bar.sync") <= max_barriers

    dispatch_sig = {
        "codes": "*i16",
        "scales": "*i16",
        "ids": "*i16",
        "weights": "*i16",
        "out_codes": "*i16",
        "out_scales": "*i16",
        "out_ids": "*i16",
        "out_weights": "*i16",
        "buf": "*i16",
        "peers": "*i64",
        "state": "*i32",
        "rank": "i32",
        "c0": "i32",
        "c1": "i32",
        "c2": "i32",
        "c3": "i32",
        "c4": "i32",
        "c5": "i32",
        "c6": "i32",
        "c7": "i32",
        "region": "i32",
    }
    dispatch_attrs = {
        (i,): [["tt.divisibility", 16]] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20)
    }
    for world in (4, 8):
        compiled = triton.compile(
            ASTSource(
                fn=k.dispatch_kernel,
                signature=dispatch_sig,
                constexprs={
                    "CODES": 1536,
                    "SCALES": 192,
                    "IDS": 8,
                    "WEIGHTS": 4,
                    "ROW": 1744,
                    "WORLD": world,
                    "BLOCK": 2048,
                },
                attrs=dispatch_attrs,
            ),
            target=target,
            options={"num_warps": 4},
        )
        ptx = compiled.asm["ptx"]
        assert "st.global.cs.v4" in ptx
        assert ".local" not in ptx
