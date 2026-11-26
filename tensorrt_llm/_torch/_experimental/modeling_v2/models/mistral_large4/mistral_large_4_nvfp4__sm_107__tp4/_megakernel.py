# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent TP16 whole-layer megakernel for Mistral Large 4 on VR200 (SM107, B1 M=4).

Opt-in via ``TRTLLM_ML4_MEGAKERNEL=1``. Repacks weights directly from the loaded
``DecodeTarget`` layer views without modifying ``weights.py`` or non-megakernel paths.
"""

from __future__ import annotations

import os
import struct
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .modeling import DecodeTarget

# Parameter block fields, matching ml4::Param in ml4Megakernel.cu 1-to-1.
_FIELDS: tuple[str, ...] = (
    "Magic",
    "KvBase",
    "StrideRows",
    "LayerRows",
    "MaxBlocks",
    "TokensPerBlock",
    "Rope",
    "Wq",
    "SsmaxS",
    "SsmaxB",
    "WkvA",
    "Kb",
    "Vb",
    "WO",
    "WOScale",
    "WR",
    "WGu",
    "WD",
    "GammaKv",
    "GammaPost",
    "GammaNext",
    "W13",
    "W2",
    "Alpha1",
    "Alpha2",
    "Arena",
    "PartO",
    "PartStats",
    "Fc2out",
    "H1Mc",
    "H1Uc",
    "H2Mc",
    "H2Uc",
    "TkMc",
    "TkUc",
    "IdxHistMc",
    "IdxHistUc",
    "IdxBndMc",
    "IdxBndUc",
    "HopBytes",
    "Rank",
    "Counters",
    "Err",
    "MaxRoles",
    "TimeoutNs",
    "Flags",
    "Copy",
    "NextCopy",
    "PrevCopy",
    "NumCopies",
    "IdxCopy",
    "NextIdxCopy",
    "IdxArena",
    "IdxLogits",
    "IdxCounters",
    "IdxClearMeta",
    "SmidMap",
    "NDie0",
    "RmsEps",
    "IdxEps",
    "CaptureOut",
    "CaptureStride",
    "IdxQNormW",
    "IdxQNormB",
    "IdxWq",
    "IdxKnW",
    "IdxKnB",
    "IdxSsmaxS",
    "IdxSsmaxB",
    "IdxPool",
    "IdxBlockStride",
    "IdxTmap",
    "IdxBlocks",
    "TopkOut",
    "UgpuState",
)
_P: dict[str, int] = {name: i for i, name in enumerate(_FIELDS)}

# Stage codes, matching ml4::Stage in ml4Megakernel.cu 1-to-1.
_ST: dict[str, int] = dict(
    Wq=1,
    WkvA=2,
    Kb=3,
    KvAppend=4,
    Attn=8,
    Merge=9,
    Vup=10,
    Oproj=11,
    Clear=12,
    H1=13,
    RouterGemv=14,
    Shared=15,
    Moe=16,
    Owner=18,
    H2=19,
)

# Flags, matching ml4::Flag in ml4Megakernel.cu 1-to-1.
_FLAG_HOLD_EXIT = 1 << 0
_FLAG_NEXT_CHAINED = 1 << 1
_FLAG_EVICT_FIRST = 1 << 2
_FLAG_MOE_L2 = 1 << 4
_FLAG_EARLY_X = 1 << 5
_FLAG_LEADER = 1 << 6
_FLAG_NOPE = 1 << 7
_FLAG_KV_OWNER = 1 << 8
_FLAG_DIE_AWARE = 1 << 9
_FLAG_HOP_SELF_CLEAR = 1 << 10
_FLAG_X_PRIME = 1 << 11
_FLAG_H_PRIME = 1 << 12
_FLAG_HOP1_LINES = 1 << 13
_FLAG_SKIP_FRONT = 1 << 16
_FLAG_TEST_RANK0_CLEAR = 1 << 17

_BASE_FLAGS = (
    _FLAG_HOLD_EXIT
    | _FLAG_EVICT_FIRST
    | _FLAG_MOE_L2
    | _FLAG_EARLY_X
    | _FLAG_HOP_SELF_CLEAR
    | _FLAG_X_PRIME
    | _FLAG_H_PRIME
    | _FLAG_HOP1_LINES
)

_MAGIC = 0x4D4C345F4D4B3035  # "ML4_MK05"
_CTAS = 212
_DEFAULT_NDIE0 = 106
_SHARED_CTAS = 24
_ROUTED_CTAS = _CTAS - _SHARED_CTAS  # 188
_TOK = 4
_HIDDEN = 6144
_RANKS = 16
_HEADS = 4
_TOPK = 2048
_EXPERTS = 128
_MOE_INTER = 384
_SHARED_INTER = 384
_MAX_SLOTS = 16
_MAX_ROLES = 16
_NUM_COPIES = 3
_SMEM_MAIN = 200704
_SMEM_FIXED = 16384 + 3072 + 3072
_SMEM_BYTES = _SMEM_MAIN + _SMEM_FIXED  # 223,232 B
_IDX_SMEM_BYTES = 135168
_ATTN_OFF = 131072
_HOP_BYTES = _TOK * _RANKS * _HIDDEN * 2  # 786,432 B
_TK_BYTES = _TOK * _TOPK * 4  # 32,768 B
_IDX_HIST_WORDS = 1088  # 1024 fine u16x2 words + 64 coarse u32 group totals
_IDX_HIST_BYTES = 3 * _TOK * _RANKS * _IDX_HIST_WORDS * 4  # 835,584 B
_IDX_BND_BYTES = _TOK * 4096 * 2 * 4  # 131,072 B
_IDX_ARENA_HIST0_OFF = 18816  # 1280 + 512 + 128 + 16384 + 512
_IDX_ARENA_BYTES = 51584
_IDX_LOGITS_BYTES = _TOK * ((2 << 20) // _RANKS) * 4  # 2,097,152 B
_ARENA_CLEAR_BYTES = 597504
_ARENA_BYTES = 630272
_CLEAR_UNIT_BYTES = 16384
_CLEAR_UNITS = (_ARENA_CLEAR_BYTES + _CLEAR_UNIT_BYTES - 1) // _CLEAR_UNIT_BYTES  # 37
_NUM_COUNTERS = 64
_MAX_KEYS = 2 << 20
_MXF8_PIECE_BYTES = 128 * 128 + 512  # 16,896 B

# Static & dynamic SMEM unit byte sizes (must match ml4Megakernel.cu).
_UNIT_BYTES_BASE: dict[int, int] = {
    _ST["Wq"]: 64 * 1024,  # 65,536 B (dynamic TCGen05 MXFP8 ring; leaves >=64 KiB above for Kb!)
    _ST["WkvA"]: 96 * 2048,  # 196,608 B
    _ST["Attn"]: 64 * 672 + 8 * 1168 + 4 * 8 * 144 + 2048,  # 59,008 B @ _ATTN_OFF
    _ST["Vup"]: 8 * 2048,  # 16,384 B
    _ST["Oproj"]: 2 * 4 * 2048,  # 16,384 B
    _ST["RouterGemv"]: 24 * 2048,  # 49,152 B
    _ST["Shared"]: 96 * 1024,  # 98,304 B (dynamic TCGen05 MXFP8 SharedGu + SharedDn)
    _ST["Moe"]: 3 * 55296 + 28160,  # 194,048 B (dynamic)
}
_DYNAMIC = frozenset({_ST["Wq"], _ST["Attn"], _ST["Shared"], _ST["Moe"]})

_DIE_CHUNK_BYTES = 2 << 20
_NATIVE_OPS: Any = None


def _float_to_u32(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(val)))[0]


def _load_native_ops() -> Any:
    """Locate and load libth_ml4_megakernel.so (or family modeling_v2 library) once."""
    global _NATIVE_OPS
    if _NATIVE_OPS is not None:
        return _NATIVE_OPS

    from . import fake_ops

    if hasattr(torch.ops, "ml4_megakernel") and hasattr(
        torch.ops.ml4_megakernel, "ml4_megakernel_info"
    ):
        fake_ops.ensure_fakes_registered("ml4_megakernel")
        _NATIVE_OPS = torch.ops.ml4_megakernel
        return _NATIVE_OPS

    here = Path(__file__).resolve()
    libs_dir = here.parents[6] / "libs"
    repo_root = here.parents[7]
    candidates: list[Path] = []
    env_lib = os.environ.get("TRTLLM_ML4_MEGAKERNEL_LIB")
    if env_lib:
        candidates.append(Path(env_lib))
    candidates.extend(
        [
            libs_dir / "libth_ml4_megakernel.so",
            libs_dir / "libth_modeling_v2_mistral_large4.so",
            here.parent / "csrc" / "libth_ml4_megakernel.so",
        ]
    )
    for build_dir in sorted(repo_root.glob("cpp/build*")):
        candidates.append(build_dir / "ml4_megakernel" / "libth_ml4_megakernel.so")

    for path in candidates:
        if path.is_file():
            torch.ops.load_library(str(path))
            if hasattr(torch.ops, "ml4_megakernel") and hasattr(
                torch.ops.ml4_megakernel, "ml4_megakernel_info"
            ):
                fake_ops.ensure_fakes_registered("ml4_megakernel")
                _NATIVE_OPS = torch.ops.ml4_megakernel
                return _NATIVE_OPS

    raise RuntimeError(
        "TRTLLM_ML4_MEGAKERNEL=1 requested, but libth_ml4_megakernel.so was not found. "
        "Build target `th_ml4_megakernel` (cmake --build <build_dir> --target th_ml4_megakernel) "
        "or set TRTLLM_ML4_MEGAKERNEL_LIB=/path/to/libth_ml4_megakernel.so."
    )


# ----------------------------------------------------------------------------------------------------------------
# Weight repacking helpers (pure torch, zero changes to weights.py)
# ----------------------------------------------------------------------------------------------------------------
def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """[M, K] BF16 -> [M/16, K/64, 2048] uint8 for mma.sync m16n8k16."""
    m, k = w.shape
    assert w.dtype == torch.bfloat16 and m % 16 == 0 and k % 64 == 0, (w.dtype, w.shape)
    v = w.contiguous().view(m // 16, 2, 8, k // 64, 4, 2, 8)
    v = v.permute(0, 3, 1, 5, 2, 4, 6)
    return v.contiguous().view(torch.uint8).reshape(m // 16, k // 64, 2048)


def pack_fp8_pb_wo(
    w: torch.Tensor, w_scale_r128c4: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] E4M3 + R128c4 K32 UE8M0 scales -> ([N/16, K/128, 2048] uint8, [N/128, K/128] fp32)."""
    n, k = w.shape
    assert w.dtype == torch.float8_e4m3fn and n % 128 == 0 and k % 128 == 0, (w.dtype, w.shape)
    w_u8 = w.contiguous().view(torch.uint8)
    v = w_u8.view(n // 16, 2, 8, k // 128, 2, 4, 16)
    v = v.permute(0, 3, 1, 4, 2, 5, 6)
    w_packed = v.contiguous().reshape(n // 16, k // 128, 2048)

    cols_k32 = k // 32
    exp_full = (
        w_scale_r128c4.contiguous()
        .view(torch.uint8)
        .reshape(n // 128, cols_k32 // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(n, cols_k32)
    )
    exp_128 = exp_full[0::128, 0::4].contiguous()
    s_fp32 = (exp_128.to(torch.int32) << 23).view(torch.float32).contiguous()
    return w_packed, s_fp32


def swizzle_64b(t: torch.Tensor) -> torch.Tensor:
    """[..., R, 64] uint8 -> TCGen05 SWIZZLE_64B 8-row x 64 B atoms."""
    *lead, rows, width = t.shape
    assert width == 64
    v = t.reshape(-1, rows, 4, 16)
    r = torch.arange(rows, device=t.device).view(rows, 1)
    dst = torch.arange(4, device=t.device).view(1, 4) ^ ((r >> 1) & 3)
    out = torch.empty_like(v)
    out.scatter_(2, dst.view(1, rows, 4, 1).expand(v.shape[0], rows, 4, 16), v)
    return out.view(*lead, rows, 64)


def swizzle_128b(t: torch.Tensor) -> torch.Tensor:
    """[..., R, 128] uint8 -> TCGen05 SWIZZLE_128B 8-row x 128 B atoms."""
    *lead, rows, width = t.shape
    assert width == 128
    v = t.reshape(-1, rows, 8, 16)
    r = torch.arange(rows, device=t.device).view(rows, 1)
    dst = torch.arange(8, device=t.device).view(1, 8) ^ (r & 7)
    out = torch.empty_like(v)
    out.scatter_(2, dst.view(1, rows, 8, 1).expand(v.shape[0], rows, 8, 16), v)
    return out.view(*lead, rows, 128)


def pack_mxf8_tiles(w: torch.Tensor, w_scale_r128c4: torch.Tensor) -> torch.Tensor:
    """[N, K] E4M3 + R128c4 K32 UE8M0 scales -> [N/128, K/128, 16896] uint8 for TCGen05 MXFP8."""
    n, k = w.shape
    assert w.dtype == torch.float8_e4m3fn and n % 128 == 0 and k % 128 == 0, (w.dtype, w.shape)
    w_u8 = (
        w.contiguous()
        .view(torch.uint8)
        .reshape(n // 128, 128, k // 128, 128)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    w_sw = swizzle_128b(w_u8).reshape(n // 128, k // 128, 16384)
    s_atoms = w_scale_r128c4.contiguous().view(torch.uint8).reshape(n // 128, k // 128, 512)
    return torch.cat([w_sw, s_atoms], dim=2).contiguous()


def repack_nvfp4_moe(
    fc1_w: torch.Tensor,
    fc1_s: torch.Tensor,
    fc2_w: torch.Tensor,
    fc2_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repack weights.py TRTLLM-Gen NVFP4 MoE tensors into 55,296-byte TCGen05 SWIZZLE_64B pieces."""
    assert tuple(fc1_w.shape) == (_EXPERTS, 2 * _MOE_INTER, _HIDDEN // 2), fc1_w.shape
    assert tuple(fc1_s.shape) == (_EXPERTS, 2 * _MOE_INTER, _HIDDEN // 16), fc1_s.shape
    assert tuple(fc2_w.shape) == (_EXPERTS, _HIDDEN, _MOE_INTER // 2), fc2_w.shape
    assert tuple(fc2_s.shape) == (_EXPERTS, _HIDDEN, _MOE_INTER // 16), fc2_s.shape

    w13_codes = (
        fc1_w.contiguous()
        .view(torch.uint8)
        .reshape(_EXPERTS, 6, 128, 48, 64)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    w13_codes = swizzle_64b(w13_codes).reshape(_EXPERTS, 6, 8, 6 * 8192)
    w13_scales = fc1_s.contiguous().view(torch.uint8).reshape(_EXPERTS, 6, 8, 12 * 512)
    w13 = torch.cat([w13_codes, w13_scales], dim=3).contiguous()

    w2_codes = (
        fc2_w.contiguous()
        .view(torch.uint8)
        .reshape(_EXPERTS, 24, 2, 128, 3, 64)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )
    w2_codes = swizzle_64b(w2_codes).reshape(_EXPERTS, 24, 6 * 8192)
    w2_scales = fc2_s.contiguous().view(torch.uint8).reshape(_EXPERTS, 24, 12 * 512)
    w2 = torch.cat([w2_codes, w2_scales], dim=2).contiguous()
    return w13, w2


# ----------------------------------------------------------------------------------------------------------------
# Die-Aware 212-CTA Role Table Builder (pure Python / torch, testable on CPU)
# ----------------------------------------------------------------------------------------------------------------
def build_roles(
    leader: bool = False,
    nope: bool = False,
    kv_owner: bool = True,
    n_die0: int = _DEFAULT_NDIE0,
    smem_main: int = _SMEM_MAIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build symmetric Die-Aware 212-CTA (roles, role_smem, role_issue) tables for ml4LayerKernel.

    When ``leader=True``, ``ml4LeaderFrontKernel`` already ran Wq/Kb/WkvA/KvAppend/IdxQ/IdxK/IdxScore/IdxTopK,
    so ``ml4LayerKernel`` starts directly at ``Attn`` (``kFlagSkipFront``).
    For MoE, 24 CTAs (64..75 Die-0, 170..181 Die-1) run dedicated parallel TCGen05 MXFP8 Shared+Owner,
    while the remaining 188 CTAs (94 Die-0 + 94 Die-1, with H1 CTAs at highest local indices 86..93)
    run zero-stall Routed MoE!
    """
    if n_die0 != _DEFAULT_NDIE0:
        raise ValueError(
            f"ML4 megakernel symmetric die role table requires n_die0={_DEFAULT_NDIE0}, got {n_die0}"
        )
    d1 = n_die0
    lists: list[list[tuple[int, int]]] = [[] for _ in range(_CTAS)]

    def add(cta: int, stage: str, unit: int) -> None:
        lists[cta].append((_ST[stage], unit))

    # --- Stage 1 (Plain layers only; Leader layers run front-end in ml4LeaderFrontKernel!) ---
    if not leader:
        # 32-CTA TCGen05 MXFP8 Wq (4 heads x 8 K-splits = 6 K-tiles / 101 KB per CTA, 1-wave pollStrided<7> reduce!):
        # 16 on Die 0 (heads 0..1), 16 on Die 1 (heads 2..3).
        for u in range(16):
            add(u, "Wq", u)
            add(d1 + u, "Wq", 16 + u)
        # WkvA (36 RoPE / 32 NoPE tiles) on CTAs 32..49 and d1+32..d1+49.
        kv_tiles = (32 if nope else 36) if kv_owner else 0
        half_kv = kv_tiles // 2
        for u in range(half_kv):
            add(32 + u, "WkvA", u)
            add(d1 + 32 + u, "WkvA", half_kv + u)
        # Kb (32 units) on FREE CTAs 51..66 and d1+51..d1+66 (prefetched at T=0, starts polling kArQNope immediately!).
        for u in range(16):
            add(51 + u, "Kb", u)
            add(d1 + 51 + u, "Kb", 16 + u)
        if kv_owner:
            add(50, "KvAppend", 0)

    # --- Stage 3: Sparse MLA (128 CTAs) -> Merge (16) -> Vup (32) -> Oproj (192) ---
    for u in range(64):
        add(u, "Attn", u)
        add(d1 + u, "Attn", 64 + u)
    for u in range(8):
        add(64 + u, "Merge", u)
        add(d1 + 64 + u, "Merge", 8 + u)
    for u in range(16):
        add(72 + u, "Vup", u)
        add(d1 + 72 + u, "Vup", 16 + u)
    for u in range(96):
        add(u, "Oproj", u)
        add(d1 + u, "Oproj", 96 + u)

    # Lean 37-unit Clear (592 KB main Arena) on CTAs 76..94 (Die 0) and d1+76..d1+93 (Die 1):
    # hidden completely behind H1's 16-rank MNNVL poll!
    half_clear = (_CLEAR_UNITS + 1) // 2  # 19 on Die 0, 18 on Die 1
    for u in range(half_clear):
        add(76 + u, "Clear", u)
    for u in range(_CLEAR_UNITS - half_clear):
        add(d1 + 76 + u, "Clear", half_clear + u)

    # --- Stage 4: H1 (16) -> RouterGemv (32) -> [24 Shared+Owner || 188 Routed Moe] -> H2 (16) ---
    for u in range(8):
        add(98 + u, "H1", u)
        add(d1 + 98 + u, "H1", 8 + u)
    for u in range(16):
        add(u, "RouterGemv", u)
        add(d1 + u, "RouterGemv", 16 + u)

    # 24 Dedicated Shared+Owner CTAs (12 Die 0: 64..75, 12 Die 1: d1+64..d1+75; 11.8 us hidden behind Routed MoE!).
    for u in range(12):
        add(64 + u, "Shared", u)
        add(64 + u, "Owner", u)
        add(d1 + 64 + u, "Shared", 12 + u)
        add(d1 + 64 + u, "Owner", 12 + u)

    # 188 Dedicated Routed-MoE CTAs (94 Die 0, 94 Die 1), placing H1 CTAs at highest local index 86..93!
    die0_routed = list(range(0, 64)) + list(range(76, 98)) + list(range(98, 106))
    die1_routed = [d1 + c for c in die0_routed]
    assert len(die0_routed) == 94 and len(die1_routed) == 94
    for rc, cta in enumerate(die0_routed):
        add(cta, "Moe", rc)
    for rc, cta in enumerate(die1_routed):
        add(cta, "Moe", 94 + rc)

    for u in range(8):
        add(98 + u, "H2", u)
        add(d1 + 98 + u, "H2", 8 + u)

    unit_bytes = dict(_UNIT_BYTES_BASE)
    unit_bytes[_ST["Kb"]] = 16384 if nope else 8192

    roles = torch.zeros(_CTAS, _MAX_ROLES, dtype=torch.int32)
    smem = torch.zeros(_CTAS, _MAX_ROLES, dtype=torch.int32)
    issue = torch.full((_CTAS, _MAX_ROLES), -2, dtype=torch.int32)

    for c, units in enumerate(lists):
        assert len(units) <= _MAX_ROLES, f"CTA {c} exceeded {_MAX_ROLES} roles: {len(units)}"
        placed: list[tuple[int, int, int]] = []
        for i, (stage, unit) in enumerate(units):
            roles[c, i] = (stage << 16) | unit
            size = unit_bytes.get(stage, 0)
            if size == 0:
                continue
            if stage in _DYNAMIC:
                off = _ATTN_OFF if stage == _ST["Attn"] else 0
                assert off + size <= smem_main, (c, stage, off, size, smem_main)
            else:
                cands = sorted({0, *[((o + s + 1023) // 1024) * 1024 for (o, s, _) in placed]})
                best: tuple[int, int] | None = None
                for off_c in cands:
                    if off_c + size > smem_main:
                        continue
                    blk = max(
                        [j for (o, s, j) in placed if o < off_c + size and off_c < o + s],
                        default=-1,
                    )
                    if best is None or blk < best[0]:
                        best = (blk, off_c)
                assert best is not None, f"SMEM placement failed on CTA {c} stage {stage}"
                blocking, off = best
                issue[c, i] = blocking
            smem[c, i] = off
            placed.append((off, size, i))
    return roles, smem, issue


def compute_copy_schedule(num_layers: int) -> tuple[int, list[tuple[int, int, int]]]:
    """Return (num_copies, [(copy, next_copy, prev_copy), ...]) for cyclic CUDA-graph replay."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if num_layers == 1:
        return 3, [(0, 1, 2)]
    if num_layers == 2:
        return 3, [(0, 1, 2), (1, 2, 0)]
    if num_layers % 3 == 0:
        copies = [i % 3 for i in range(num_layers)]
        num_copies = 3
    elif num_layers == 5:
        copies = [0, 1, 2, 3, 4]
        num_copies = 5
    else:
        num_copies = 4
        if num_layers % 4 == 1:
            n_three = 9
            n_four = num_layers - n_three
            copies = [i % 4 for i in range(n_four)] + [i % 3 for i in range(n_three)]
        else:
            copies = [i % 4 for i in range(num_layers)]

    schedule: list[tuple[int, int, int]] = []
    for i in range(num_layers):
        cur = copies[i]
        nxt = copies[(i + 1) % num_layers]
        prv = copies[(i - 1) % num_layers]
        assert len({prv, cur, nxt}) == 3, (num_layers, i, prv, cur, nxt)
        schedule.append((cur, nxt, prv))
    return num_copies, schedule


def build_smid_bijection(
    probe_rows: list[tuple[int, int, int]], min_gap: int = 40
) -> tuple[list[int], int] | None:
    """Build hardware %smid -> logical_cta bijection from probe (smid, die0_cycles, die1_cycles) rows."""
    if len(probe_rows) != _CTAS:
        return None
    by_smid: dict[int, tuple[int, int]] = {}
    for sm, c0, c1 in probe_rows:
        if sm < 0 or sm in by_smid:
            return None
        if abs(c0 - c1) < min_gap:
            return None
        by_smid[sm] = (c0, c1)
    die0_sms = sorted(sm for sm, (c0, c1) in by_smid.items() if c0 < c1)
    die1_sms = sorted(sm for sm, (c0, c1) in by_smid.items() if c1 < c0)
    if len(die0_sms) != _DEFAULT_NDIE0 or len(die1_sms) != _CTAS - _DEFAULT_NDIE0:
        return None
    max_sm = max(by_smid)
    table = [-1] * (max_sm + 1)
    for idx, sm in enumerate(die0_sms):
        table[sm] = idx
    for idx, sm in enumerate(die1_sms):
        table[sm] = _DEFAULT_NDIE0 + idx
    return table, _DEFAULT_NDIE0


def build_smid_bijection_from_domains(
    domain_sms: list[list[int]], num_sms: int = _CTAS
) -> tuple[list[int], int] | None:
    """Build hardware %smid -> logical_cta bijection from driver-attested locality domain SM lists."""
    if len(domain_sms) != 2:
        return None
    d0 = sorted(set(domain_sms[0]))
    d1 = sorted(set(domain_sms[1]))
    if not d0 or not d1 or (set(d0) & set(d1)):
        return None
    if len(d0) > _DEFAULT_NDIE0 or len(d1) > num_sms - _DEFAULT_NDIE0:
        return None
    assigned = set(d0) | set(d1)
    rem = [sm for sm in range(num_sms) if sm not in assigned]
    if len(d0) + len(d1) + len(rem) != num_sms:
        return None
    need0 = _DEFAULT_NDIE0 - len(d0)
    die0_all = d0 + rem[:need0]
    die1_all = d1 + rem[need0:]
    if len(die0_all) != _DEFAULT_NDIE0 or len(die1_all) != num_sms - _DEFAULT_NDIE0:
        return None
    max_sm = max(max(die0_all), max(die1_all))
    table = [-1] * (max_sm + 1)
    for idx, sm in enumerate(die0_all):
        table[sm] = idx
    for idx, sm in enumerate(die1_all):
        table[sm] = _DEFAULT_NDIE0 + idx
    return table, _DEFAULT_NDIE0


_SMID_PROBE_SRC = r"""
extern "C" __global__ void ml4_smid_probe(int* out) {
    if (threadIdx.x == 0) {
        unsigned s;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
        out[blockIdx.x] = (int)s;
    }
}
"""
_LIBCUDA = None
_DIE_MAP_CACHE: dict[int, tuple[torch.Tensor | None, int]] = {}


def _pointer_locality_domain(ptr: int) -> int:
    """Query CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL for a device pointer."""
    global _LIBCUDA
    import ctypes

    import cuda.bindings.driver as cuda

    if _LIBCUDA is None:
        _LIBCUDA = ctypes.CDLL("libcuda.so.1")
    out = ctypes.c_int(-9)
    attr = int(cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL)
    err = _LIBCUDA.cuPointerGetAttribute(
        ctypes.byref(out), ctypes.c_int(attr), ctypes.c_uint64(ptr)
    )
    if err != 0:
        raise RuntimeError(f"cuPointerGetAttribute(LOCALITY_DOMAIN_ORDINAL) failed: {err}")
    return int(out.value)


class _SplitDieMemory:
    """Contiguous virtual address range whose first half lives on Die 0 and second half on Die 1."""

    def __init__(
        self, nbytes: int, split_byte: int, device_index: int, verify_domains: bool = True
    ) -> None:
        import cuda.bindings.driver as cuda

        def checked(res, name: str):
            status, *vals = res
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"{name} failed: {status}")
            return vals[0] if len(vals) == 1 else (tuple(vals) if vals else None)

        cu_dev = int(checked(cuda.cuDeviceGet(device_index), "cuDeviceGet"))

        def make_prop(die: int):
            prop = cuda.CUmemAllocationProp()
            prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            prop.requestedHandleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
            prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
            prop.location.localized.deviceId = cu_dev
            prop.location.localized.localityDomainId = die
            return prop

        prop0 = make_prop(0)
        gran = int(
            checked(
                cuda.cuMemGetAllocationGranularity(
                    prop0, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
                ),
                "cuMemGetAllocationGranularity",
            )
        )
        gran = max(gran, _DIE_CHUNK_BYTES)
        if split_byte <= 0:
            half0 = 0
        elif split_byte >= nbytes:
            half0 = ((nbytes + gran - 1) // gran) * gran
        else:
            half0 = max(gran, int(round(split_byte / gran)) * gran)
            if half0 >= nbytes and nbytes > gran:
                half0 = (nbytes // gran) * gran
        rem = max(0, nbytes - half0)
        half1 = ((rem + gran - 1) // gran) * gran if rem > 0 else 0
        self.half0 = half0
        self.half1 = half1
        self.total_bytes = half0 + half1
        self.base = int(checked(cuda.cuMemAddressReserve(self.total_bytes, 0, 0, 0), "reserve"))
        self._handles: list[tuple[Any, int]] = []
        self._mapped = False
        try:
            if half0 > 0:
                h0 = checked(cuda.cuMemCreate(half0, prop0, 0), "cuMemCreate(die0)")
                self._handles.append((h0, half0))
                checked(cuda.cuMemMap(self.base, half0, 0, h0, 0), "cuMemMap(die0)")
            if half1 > 0:
                h1 = checked(cuda.cuMemCreate(half1, make_prop(1), 0), "cuMemCreate(die1)")
                self._handles.append((h1, half1))
                checked(cuda.cuMemMap(self.base + half0, half1, 0, h1, 0), "cuMemMap(die1)")
            self._mapped = True
            access = cuda.CUmemAccessDesc()
            access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = cu_dev
            access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            checked(cuda.cuMemSetAccess(self.base, self.total_bytes, [access], 1), "cuMemSetAccess")
            if verify_domains:
                if half0 > 0:
                    for addr in (self.base, self.base + half0 - 1):
                        dom = _pointer_locality_domain(addr)
                        if dom != 0:
                            raise RuntimeError(f"Die-0 VMM byte {addr:#x} landed in domain {dom}")
                if half1 > 0:
                    for addr in (self.base + half0, self.base + self.total_bytes - 1):
                        dom = _pointer_locality_domain(addr)
                        if dom != 1:
                            raise RuntimeError(f"Die-1 VMM byte {addr:#x} landed in domain {dom}")
        except Exception:
            self._release(self.base, self.total_bytes, self._handles, [self._mapped])
            raise
        self._finalizer = weakref.finalize(
            self,
            _SplitDieMemory._release,
            self.base,
            self.total_bytes,
            self._handles,
            [self._mapped],
        )

    @staticmethod
    def _release(
        base: int, nbytes: int, handles: list[tuple[Any, int]], mapped_box: list[bool]
    ) -> None:
        import cuda.bindings.driver as cuda

        off = 0
        for h, sz in handles:
            if mapped_box[0]:
                cuda.cuMemUnmap(base + off, sz)
            cuda.cuMemRelease(h)
            off += sz
        mapped_box[0] = False
        handles.clear()
        cuda.cuMemAddressFree(base, nbytes)

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        return {
            "shape": (self.total_bytes,),
            "typestr": "|u1",
            "data": (self.base, False),
            "version": 3,
            "strides": None,
            "stream": None,
        }


def _try_split_die_copy(tensor: torch.Tensor, enable_die_aware: bool) -> torch.Tensor:
    """Copy a weight tensor into a verified Die-0 / Die-1 split VMM allocation when available."""
    if not enable_die_aware or not tensor.is_cuda or tensor.nbytes < _DIE_CHUNK_BYTES:
        return tensor.contiguous()
    try:
        src = tensor.contiguous()
        nbytes = src.nbytes
        mem = _SplitDieMemory(nbytes, nbytes // 2, src.device.index, verify_domains=True)
        raw = torch.as_tensor(mem, device=src.device)
        dst = raw[:nbytes].view(src.dtype).view(src.shape)
        dst.copy_(src)
        return dst
    except Exception:
        return tensor.contiguous()


def _probe_hardware_sm_die_map(
    native_ops: Any, device: torch.device
) -> tuple[torch.Tensor | None, int]:
    """Probe physical SM -> uGPU die bijection via CUDA 13.4+ locality-domain green contexts."""
    if os.environ.get("TRTLLM_ML4_MEGAKERNEL_DIE_AWARE", "1") == "0":
        return None, _DEFAULT_NDIE0
    dev = device.index or 0
    if dev in _DIE_MAP_CACHE:
        return _DIE_MAP_CACHE[dev]
    try:
        import ctypes

        from cuda.bindings import driver as cuda
        from cuda.bindings import nvrtc

        def checked(res, name: str):
            status, *vals = res
            if int(status) != 0:
                raise RuntimeError(f"{name} failed: {status}")
            return vals[0] if len(vals) == 1 else (tuple(vals) if vals else None)

        cu_dev = checked(cuda.cuDeviceGet(dev), "cuDeviceGet")
        n_domains = int(
            checked(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, cu_dev
                ),
                "LOCALITY_DOMAIN_COUNT",
            )
        )
        if n_domains != 2:
            _DIE_MAP_CACHE[dev] = (None, _DEFAULT_NDIE0)
            return None, _DEFAULT_NDIE0

        sms_per_dom = int(
            checked(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT,
                    cu_dev,
                ),
                "LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT",
            )
        )
        major = int(
            checked(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_dev
                ),
                "CC_MAJOR",
            )
        )
        minor = int(
            checked(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_dev
                ),
                "CC_MINOR",
            )
        )
        full_sm = checked(
            cuda.cuDeviceGetDevResource(cu_dev, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
            "cuDeviceGetDevResource",
        )
        params = []
        for dom in range(n_domains):
            p = cuda.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
            p.smCount = sms_per_dom
            p.flags = cuda.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
            p.localityDomainId = dom
            params.append(p)
        dom_res, _rem = checked(
            cuda.cuDevSmResourceSplit(n_domains, full_sm, 0, params), "cuDevSmResourceSplit"
        )

        prog = checked(
            nvrtc.nvrtcCreateProgram(_SMID_PROBE_SRC.encode(), b"ml4_smid_probe.cu", 0, [], []),
            "nvrtcCreateProgram",
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
        if int(err) != 0:
            raise RuntimeError(f"nvrtcCompileProgram failed: {err}")
        cubin_sz = int(checked(nvrtc.nvrtcGetCUBINSize(prog), "nvrtcGetCUBINSize"))
        cubin = b" " * cubin_sz
        checked(nvrtc.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")
        nvrtc.nvrtcDestroyProgram(prog)

        nb = 4 * sms_per_dom
        buf = torch.full((nb,), -1, dtype=torch.int32, device=device)
        domain_sms: list[list[int]] = []
        for dom in range(n_domains):
            desc = checked(
                cuda.cuDevResourceGenerateDesc([dom_res[dom]], 1), "cuDevResourceGenerateDesc"
            )
            gctx = checked(
                cuda.cuGreenCtxCreate(
                    desc, cu_dev, cuda.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
                ),
                "cuGreenCtxCreate",
            )
            gst = checked(
                cuda.cuGreenCtxStreamCreate(gctx, cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, 0),
                "cuGreenCtxStreamCreate",
            )
            ctx = checked(cuda.cuCtxFromGreenCtx(gctx), "cuCtxFromGreenCtx")
            checked(cuda.cuCtxPushCurrent(ctx), "cuCtxPushCurrent")
            try:
                mod = checked(cuda.cuModuleLoadData(cubin), "cuModuleLoadData")
                fn = checked(
                    cuda.cuModuleGetFunction(mod, b"ml4_smid_probe"), "cuModuleGetFunction"
                )
                buf.fill_(-1)
                torch.cuda.synchronize(device)
                arg = ctypes.c_void_p(buf.data_ptr())
                checked(
                    cuda.cuLaunchKernel(
                        fn, nb, 1, 1, 32, 1, 1, 0, gst, ((arg,), (ctypes.c_void_p,)), 0
                    ),
                    "cuLaunchKernel(ml4_smid_probe)",
                )
                checked(cuda.cuStreamSynchronize(gst), "cuStreamSynchronize")
                sms = sorted({int(x) for x in buf.tolist() if x >= 0})
                cuda.cuModuleUnload(mod)
            finally:
                cuda.cuCtxPopCurrent()
                cuda.cuStreamDestroy(gst)
                cuda.cuGreenCtxDestroy(gctx)
            if len(sms) != sms_per_dom:
                raise RuntimeError(f"domain {dom} probed {len(sms)} SMs, expected {sms_per_dom}")
            domain_sms.append(sms)

        _ = _SplitDieMemory(2 * _DIE_CHUNK_BYTES, _DIE_CHUNK_BYTES, dev, verify_domains=True)
        res = build_smid_bijection_from_domains(domain_sms, num_sms=_CTAS)
        if res is None:
            _DIE_MAP_CACHE[dev] = (None, _DEFAULT_NDIE0)
            return None, _DEFAULT_NDIE0
        table, n_die0 = res
        smid_map = torch.tensor(table, dtype=torch.int32, device=device)
        _DIE_MAP_CACHE[dev] = (smid_map, n_die0)
        return smid_map, n_die0
    except Exception:
        _DIE_MAP_CACHE[dev] = (None, _DEFAULT_NDIE0)
        return None, _DEFAULT_NDIE0


# ----------------------------------------------------------------------------------------------------------------
# Shared Workspace & Per-Layer Repacked Parameters
# ----------------------------------------------------------------------------------------------------------------
class _Shared:
    """Reusable multi-copy main & indexer workspaces, MNNVL multicast buffers, and Die-Aware role tables."""

    def __init__(
        self, mapping: Any, num_copies: int = _NUM_COPIES, num_idx_copies: int = _NUM_COPIES
    ) -> None:
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )
        from tensorrt_llm._utils import prefer_pinned

        self.ops = _load_native_ops()
        (
            self.magic,
            num_params,
            arena_bytes,
            clear_units,
            smem_bytes,
            num_counters,
            hop_bytes,
            tk_bytes,
            idx_hist_bytes,
            self.max_keys,
            idx_bnd_bytes,
            idx_arena_bytes,
            idx_logits_bytes,
            idx_smem_bytes,
        ) = self.ops.ml4_megakernel_info()
        assert self.magic == _MAGIC and num_params == len(_FIELDS), (self.magic, num_params)
        assert arena_bytes == _ARENA_BYTES and clear_units == _CLEAR_UNITS
        assert smem_bytes == _SMEM_BYTES and num_counters == _NUM_COUNTERS
        assert (
            hop_bytes == _HOP_BYTES
            and tk_bytes == _TK_BYTES
            and idx_hist_bytes == _IDX_HIST_BYTES
            and idx_bnd_bytes == _IDX_BND_BYTES
            and idx_arena_bytes == _IDX_ARENA_BYTES
            and idx_logits_bytes == _IDX_LOGITS_BYTES
            and idx_smem_bytes == _IDX_SMEM_BYTES
        )

        self.copies = num_copies
        self.idx_copies = num_idx_copies
        device = torch.device("cuda", torch.cuda.current_device())
        self.smid_map, self.n_die0 = _probe_hardware_sm_die_map(self.ops, device)
        self.die_aware = self.smid_map is not None

        self.tables = {
            key: build_roles(*key, n_die0=self.n_die0)
            for key in (
                (False, False, True),  # 28 Plain layers (RoPE owner, shared top-k)
                (True, False, True),  # 25 RoPE Owner Leader layers
                (True, True, True),  # 9 NoPE Owner Leader layers
                (True, True, False),  # 9 NoPE Layer-35 KV-Sharer Leader layers
            )
        }

        def empty_lamport(num_bytes: int) -> torch.Tensor:
            return torch.full((num_bytes // 4,), -0.0, dtype=torch.float32, device=device)

        self.arena = empty_lamport(self.copies * _ARENA_BYTES)
        self.idx_arena = empty_lamport(self.idx_copies * _IDX_ARENA_BYTES)
        idx_ar_u8 = self.idx_arena.view(torch.uint8).view(self.idx_copies, _IDX_ARENA_BYTES)
        idx_ar_u8[:, _IDX_ARENA_HIST0_OFF:].zero_()
        self.idx_logits = torch.zeros(
            (self.idx_copies * (_IDX_LOGITS_BYTES // 4),), dtype=torch.float32, device=device
        )
        self.idx_counters = torch.zeros(
            self.idx_copies * num_counters, dtype=torch.int32, device=device
        )
        self.idx_clear_meta = torch.zeros(
            self.idx_copies * _TOK * 2, dtype=torch.int32, device=device
        )
        self.ugpu_state = torch.zeros(20, dtype=torch.int32, device=device)

        self.part_o = empty_lamport(_TOK * 32 * _HEADS * 512 * 4)
        self.part_stats = empty_lamport(_TOK * 32 * _HEADS * 2 * 4)
        self.fc2out = empty_lamport(_MAX_SLOTS * _TOK * _HIDDEN * 2)
        self.counters = torch.zeros(self.copies * num_counters, dtype=torch.int32, device=device)
        self.err = torch.zeros(16 + 5 * _CTAS, dtype=torch.int32, pin_memory=prefer_pinned())
        self.leader_topk: dict[int, torch.Tensor] = {}

        self.comm = _get_mnnvl_workspace_comm(mapping)
        error = None
        try:
            hop_total = self.copies * 2 * _HOP_BYTES
            tk_total = self.idx_copies * _TK_BYTES
            ih_total = self.idx_copies * _IDX_HIST_BYTES
            ib_total = self.idx_copies * _IDX_BND_BYTES
            total = hop_total + tk_total + ih_total + ib_total
            self.handle = _make_mnnvl_mcast_buffer(
                self.comm, total, mapping, use_fabric_handle=mapping.is_multi_node()
            )
            self.uc = self.handle.get_uc_buffer(mapping.tp_rank, (total // 4,), torch.float32, 0)
            self.mc = self.handle.get_mc_buffer((total // 4,), torch.float32, 0)
            self.uc.fill_(-0.0)
            # IdxHist and IdxBnd use 0xFFFFFFFF (kEmptyHist) empty sentinel!
            hist_off = hop_total + tk_total
            self.uc.view(torch.uint8)[hist_off : hist_off + ih_total + ib_total].fill_(0xFF)
            torch.cuda.synchronize()
        except Exception as exc:
            error = exc
        if not _mnnvl_workspace_all_succeeded(self.comm, error is None):
            raise RuntimeError(
                "ML4 megakernel MNNVL multicast workspace creation failed"
            ) from error

        self.mutable = [
            self.arena,
            self.idx_arena,
            self.idx_logits,
            self.idx_counters,
            self.idx_clear_meta,
            self.ugpu_state,
            self.part_o,
            self.part_stats,
            self.fc2out,
            self.counters,
            self.uc,
            self.mc,
            self.err,
        ]
        if self.smid_map is not None:
            self.mutable.append(self.smid_map)


class _Layer:
    """Repacked weights and capture-stable native parameters for one ML4 MoE layer."""

    def __init__(
        self,
        decode: DecodeTarget,
        layer_idx: int,
        shared: _Shared,
        copy: int,
        next_copy: int,
        prev_copy: int,
        idx_copy: int,
        next_idx_copy: int,
        chained: bool,
        next_chained: bool,
    ) -> None:
        self.decode = decode
        self.core = core = decode.core
        self.layer = L = decode._layers[layer_idx]
        self.layer_idx = layer_idx
        self.shared = shared
        self.copy = copy
        self.next_copy = next_copy
        self.prev_copy = prev_copy
        self.idx_copy = idx_copy
        self.next_idx_copy = next_idx_copy
        self.leader = L.indexer is not None
        self.nope = not L.rope
        self.kv_owner = L.owner
        self.kv_source = L.kv_source
        self.leader_idx = max(
            (j for j in range(layer_idx) if decode._layers[j].indexer is not None),
            default=None,
        )
        self.pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"

        flags = _BASE_FLAGS
        if next_chained:
            flags |= _FLAG_NEXT_CHAINED
        if not chained:
            flags &= ~_FLAG_EARLY_X
        if self.leader:
            flags |= _FLAG_LEADER | _FLAG_SKIP_FRONT
        if self.nope:
            flags |= _FLAG_NOPE
        if self.kv_owner:
            flags |= _FLAG_KV_OWNER
        if shared.die_aware:
            flags |= _FLAG_DIE_AWARE
        self.flags = flags

        da = shared.die_aware
        device = L.wq.device

        # 1. MLA projections: Wq packed for TCGen05 MXFP8, WkvA/Kb/Vb BF16, WO warp FP8_PB_WO.
        self.w_q = _try_split_die_copy(pack_mxf8_tiles(L.wq, L.wq_s), da)
        self.ssmax_s = L.ssmax_s.contiguous()
        self.ssmax_b = L.ssmax_b.contiguous()

        kv_parts: list[torch.Tensor] = []
        if self.kv_owner:
            kv_parts.append(L.wkv_a.t().contiguous())
        if self.leader:
            kv_parts.append(L.indexer.wk_wp.t().contiguous())
        if kv_parts:
            self.w_kva = _try_split_die_copy(pack_bf16(torch.cat(kv_parts, dim=0)), da)
        else:
            self.w_kva = torch.empty(1, dtype=torch.uint8, device=device)

        kb_t = L.k_b.transpose(1, 2).contiguous().reshape(_HEADS * 512, L.nope)
        self.w_kb = pack_bf16(kb_t)
        vb_t = L.v_b_t.transpose(1, 2).contiguous().reshape(_HEADS * 128, 512)
        self.w_vb = pack_bf16(vb_t)

        wo_p, wo_s = pack_fp8_pb_wo(L.wo, L.wo_s)
        self.w_o = _try_split_die_copy(wo_p, da)
        self.w_o_s = wo_s.to(device)

        self.gamma_kv = (
            L.kv_norm.contiguous()
            if self.kv_owner
            else torch.empty(1, dtype=torch.bfloat16, device=device)
        )
        self.gamma_post = L.ffn_norm.contiguous()
        self.gamma_next = L.next_norm.contiguous()

        # 2. MoE: router BF16, Shared expert TCGen05 MXFP8, Routed experts NVFP4 (Die-0/Die-1).
        m = L.moe
        assert m is not None, f"Layer {layer_idx} is not a MoE layer"
        self.w_r = pack_bf16(m.router.t().contiguous())
        self.w_gu = _try_split_die_copy(pack_mxf8_tiles(L.mlp_gu, L.mlp_gu_s), da)
        self.w_d = _try_split_die_copy(pack_mxf8_tiles(L.mlp_dn, L.mlp_dn_s), da)

        w13, w2 = repack_nvfp4_moe(m.fc1_w, m.fc1_s, m.fc2_w, m.fc2_s)
        self.w13 = _try_split_die_copy(w13, da)
        self.w2 = _try_split_die_copy(w2, da)
        isc2 = core.w[f"l{layer_idx}_isc2"].detach().float().reshape(1).to(device)
        self.alpha1 = torch.cat(
            [m.out1_gate.float().reshape(_EXPERTS), m.gscale.float().reshape(1)]
        ).contiguous()
        self.alpha2 = torch.cat(
            [m.out2.float().reshape(_EXPERTS), (1.0 / isc2).reshape(1)]
        ).contiguous()

        # 3. Leader Indexer weights (TCGen05 MXFP8 ix_wq_b).
        if self.leader:
            ix = L.indexer
            self.idx_q_norm_w = ix.q_norm_w.contiguous()
            self.idx_q_norm_b = ix.q_norm_b.contiguous()
            self.idx_wq = _try_split_die_copy(pack_mxf8_tiles(ix.wq_b, ix.wq_b_s), da)
            self.idx_kn_w = ix.k_norm_w.float().contiguous()
            self.idx_kn_b = ix.k_norm_b.float().contiguous()
            self.idx_ssmax_s = ix.ssmax_s.contiguous()
            self.idx_ssmax_b = ix.ssmax_b.contiguous()
            self.topk_out = torch.full((_TOK, _TOPK), -1, dtype=torch.int32, device=device)
            shared.leader_topk[layer_idx] = self.topk_out
        else:
            self.topk_out = None

        self.params_host: torch.Tensor | None = None
        self.params: torch.Tensor | None = None
        self.step_params: dict[tuple, tuple] = {}
        self.idx_tmap: torch.Tensor | None = None
        self.idx_block_stride = 0

    def bind(self, manager: Any) -> None:
        """Bind KV pool raw pointers & build precomputed weight table once per KV manager."""
        from tensorrt_llm._utils import prefer_pinned

        shared = self.shared
        core = self.core
        device = self.w_q.device
        kv_pool = manager.get_latent_pool()
        tpb = int(kv_pool.shape[2])
        assert tpb in (32, 64), f"ML4 megakernel requires tokens_per_block in (32, 64), got {tpb}"
        owners = int(kv_pool.shape[1])
        pool_row = int(manager.latent_pool_row(self.layer_idx))

        kbuf = None
        if self.leader:
            kbuf = manager.get_indexer_k_cache_buffers(self.layer_idx)
            assert kbuf.dtype == torch.uint8 and tuple(kbuf.shape[1:]) == (tpb, 1, 132), kbuf.shape
            self.idx_block_stride = int(kbuf.stride(0))
            assert self.idx_block_stride % 128 == 0
            rows = ((kbuf.shape[0] - 1) * self.idx_block_stride + tpb * 132) // 128
            self.idx_tmap = shared.ops.ml4_megakernel_indexer_tmap(kbuf, rows, tpb).to(device)

        rope_tbl = core._rope.indexer.contiguous()
        roles, role_smem, role_issue = shared.tables[(self.leader, self.nope, self.kv_owner)]
        tables = torch.cat([roles.reshape(-1), role_smem.reshape(-1), role_issue.reshape(-1)])
        wt_offset = len(_FIELDS) + tables.numel() // 2
        p = torch.zeros(
            wt_offset + _CTAS * _MAX_ROLES * 3, dtype=torch.int64, pin_memory=prefer_pinned()
        )
        p[len(_FIELDS) : wt_offset] = tables.view(torch.int64)

        def put(name: str, value: int | torch.Tensor) -> None:
            p[_P[name]] = value.data_ptr() if isinstance(value, torch.Tensor) else int(value)

        c = shared.copies
        ic = shared.idx_copies
        h1_uc = shared.uc.data_ptr()
        h1_mc = shared.mc.data_ptr()
        h2_uc = h1_uc + c * _HOP_BYTES
        h2_mc = h1_mc + c * _HOP_BYTES
        tk_uc = h2_uc + c * _HOP_BYTES
        tk_mc = h2_mc + c * _HOP_BYTES
        ih_uc = tk_uc + ic * _TK_BYTES
        ih_mc = tk_mc + ic * _TK_BYTES
        ib_uc = ih_uc + ic * _IDX_HIST_BYTES
        ib_mc = ih_mc + ic * _IDX_HIST_BYTES

        for name, value in (
            ("Magic", shared.magic),
            ("KvBase", kv_pool),
            ("StrideRows", owners * tpb),
            ("LayerRows", pool_row * tpb),
            ("TokensPerBlock", tpb),
            ("Rope", rope_tbl),
            ("Wq", self.w_q),
            ("SsmaxS", self.ssmax_s),
            ("SsmaxB", self.ssmax_b),
            ("WkvA", self.w_kva),
            ("Kb", self.w_kb),
            ("Vb", self.w_vb),
            ("WO", self.w_o),
            ("WOScale", self.w_o_s),
            ("WR", self.w_r),
            ("WGu", self.w_gu),
            ("WD", self.w_d),
            ("GammaKv", self.gamma_kv),
            ("GammaPost", self.gamma_post),
            ("GammaNext", self.gamma_next),
            ("W13", self.w13),
            ("W2", self.w2),
            ("Alpha1", self.alpha1),
            ("Alpha2", self.alpha2),
            ("Arena", shared.arena),
            ("PartO", shared.part_o),
            ("PartStats", shared.part_stats),
            ("Fc2out", shared.fc2out),
            ("H1Mc", h1_mc),
            ("H1Uc", h1_uc),
            ("H2Mc", h2_mc),
            ("H2Uc", h2_uc),
            ("TkMc", tk_mc),
            ("TkUc", tk_uc),
            ("IdxHistMc", ih_mc),
            ("IdxHistUc", ih_uc),
            ("IdxBndMc", ib_mc),
            ("IdxBndUc", ib_uc),
            ("HopBytes", _HOP_BYTES),
            ("Rank", core.tp_rank),
            ("Counters", shared.counters),
            ("Err", shared.err),
            ("MaxRoles", _MAX_ROLES),
            ("TimeoutNs", 20_000_000_000),
            ("Flags", self.flags),
            ("Copy", self.copy),
            ("NextCopy", self.next_copy),
            ("PrevCopy", self.prev_copy),
            ("NumCopies", c),
            ("IdxCopy", self.idx_copy),
            ("NextIdxCopy", self.next_idx_copy),
            ("IdxArena", shared.idx_arena),
            ("IdxLogits", shared.idx_logits),
            ("IdxCounters", shared.idx_counters),
            ("IdxClearMeta", shared.idx_clear_meta),
            ("SmidMap", shared.smid_map if shared.smid_map is not None else 0),
            ("NDie0", shared.n_die0),
            ("RmsEps", _float_to_u32(core.eps)),
            ("IdxEps", _float_to_u32(core.idx_eps)),
            ("UgpuState", shared.ugpu_state),
        ):
            put(name, value)

        if self.leader and kbuf is not None and self.idx_tmap is not None:
            for name, value in (
                ("IdxQNormW", self.idx_q_norm_w),
                ("IdxQNormB", self.idx_q_norm_b),
                ("IdxWq", self.idx_wq),
                ("IdxKnW", self.idx_kn_w),
                ("IdxKnB", self.idx_kn_b),
                ("IdxSsmaxS", self.idx_ssmax_s),
                ("IdxSsmaxB", self.idx_ssmax_b),
                ("IdxPool", kbuf),
                ("IdxBlockStride", self.idx_block_stride),
                ("IdxTmap", self.idx_tmap),
                ("TopkOut", self.topk_out),
            ):
                put(name, value)

        self.params = p.to(device)
        shared.ops.ml4_megakernel_weight_table(self.params, wt_offset)
        p[wt_offset:] = self.params[wt_offset:].cpu()
        self.params_host = p
        self.step_params = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        step: dict[str, Any],
        metadata: Any,
        fallback_topk: torch.Tensor,
        spec_metadata: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.leader:
            topk = self.topk_out
            idx_blocks = step["indexer_k_cache_block_offsets"][0]
        else:
            topk = self.shared.leader_topk.get(self.leader_idx, fallback_topk)
            idx_blocks = None

        block_table = step["block_table"][0]
        kv_lens = step["kv_lens_cuda"][:1]
        max_blocks = int(step["block_table"].shape[1])

        cap_ptr = 0
        cap_stride = 0
        cap_buf = None
        if spec_metadata is not None:
            cap_buf = getattr(spec_metadata, "captured_hidden_states", None)
            layer_to_idx = getattr(spec_metadata, "_layer_to_idx", None)
            if cap_buf is not None and layer_to_idx is not None and self.layer_idx in layer_to_idx:
                slot = int(layer_to_idx[self.layer_idx])
                cap_ptr = cap_buf.data_ptr() + slot * _HIDDEN * cap_buf.element_size()
                cap_stride = int(cap_buf.stride(0))

        key = (
            None if idx_blocks is None else idx_blocks.data_ptr(),
            max_blocks,
            cap_ptr,
            cap_stride,
        )
        entry = self.step_params.get(key)
        if entry is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up the ML4 megakernel metadata before CUDA graph capture")
            assert self.params_host is not None and self.params is not None
            if idx_blocks is not None:
                self.params_host[_P["IdxBlocks"]] = idx_blocks.data_ptr()
            self.params_host[_P["MaxBlocks"]] = max_blocks
            self.params_host[_P["CaptureOut"]] = cap_ptr
            self.params_host[_P["CaptureStride"]] = cap_stride
            params = self.params_host.to(self.params.device)
            entry = (params, idx_blocks, cap_buf)
            self.step_params[key] = entry

        manager = metadata.kv_cache_manager
        kv_cache = [manager.get_latent_pool()]
        if self.leader:
            kv_cache.append(manager.get_indexer_k_cache_buffers(self.layer_idx))
        mutable = self.shared.mutable if cap_buf is None else [*self.shared.mutable, cap_buf]
        x_out, res_out = self.shared.ops.ml4_megakernel(
            entry[0],
            hidden_states,
            residual,
            topk,
            block_table,
            kv_lens,
            positions.to(torch.int32).contiguous(),
            mutable,
            kv_cache,
            self.pdl,
            self.leader,
        )
        if spec_metadata is not None and cap_ptr == 0:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, res_out)
        return x_out, res_out


def select_megakernel_layers(
    full_indexer: list[bool], dense_layers: int, mode: str = "all"
) -> list[int]:
    """Return ordered list of MoE layer indices enabled for megakernel execution."""
    moe_indices = list(range(dense_layers, len(full_indexer)))
    if mode == "plain":
        return [i for i in moe_indices if not full_indexer[i]]
    if mode == "all":
        return moe_indices
    raise ValueError(f"Unsupported TRTLLM_ML4_MEGAKERNEL_MODE={mode!r}; expected 'plain' or 'all'")


class Ml4Megakernel:
    """Opt-in persistent whole-layer replacement for ML4 TP16 B1 M=4 MoE layers."""

    def __init__(self, decode: DecodeTarget, layer_indices: list[int]) -> None:
        from tensorrt_llm.logger import logger

        num_copies, copy_schedule = compute_copy_schedule(len(layer_indices))
        leader_indices = [idx for idx in layer_indices if decode._layers[idx].indexer is not None]
        if leader_indices:
            num_idx_copies, idx_schedule = compute_copy_schedule(len(leader_indices))
            idx_map = {idx: (cur, nxt) for idx, (cur, nxt, _) in zip(leader_indices, idx_schedule)}
        else:
            num_idx_copies = _NUM_COPIES
            idx_map = {}

        self.shared = _Shared(
            decode.core.model_config.mapping, num_copies=num_copies, num_idx_copies=num_idx_copies
        )
        self.layers: dict[int, _Layer] = {}
        self.manager = None
        indices_set = set(layer_indices)
        for ordinal, idx in enumerate(layer_indices):
            cur_copy, next_copy, prev_copy = copy_schedule[ordinal]
            cur_ic, next_ic = idx_map.get(idx, (0, 1))
            self.layers[idx] = _Layer(
                decode,
                idx,
                self.shared,
                cur_copy,
                next_copy,
                prev_copy,
                cur_ic,
                next_ic,
                (idx - 1) in indices_set,
                (idx + 1) in indices_set,
            )
        logger.info(
            f"[ml4_megakernel] prepared {len(self.layers)} MoE layers for TP16/B1/M=4 ("
            f"copies={num_copies}, idx_copies={num_idx_copies}, die_aware={self.shared.die_aware}, "
            f"n_die0={self.shared.n_die0})"
        )

    @classmethod
    def create(cls, decode: DecodeTarget) -> Ml4Megakernel | None:
        if os.environ.get("TRTLLM_ML4_MEGAKERNEL", "0") != "1":
            return None
        if decode.core.tp_size != _RANKS:
            return None
        prev_decode = getattr(decode.core, "_decode", None)
        if prev_decode is not None and getattr(prev_decode, "_megakernel", None) is not None:
            return prev_decode._megakernel
        from tensorrt_llm._utils import prefer_pinned

        if not prefer_pinned():
            raise RuntimeError(
                "ML4 megakernel requires pinned host memory for its CUDA error record"
            )
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if (props.major, props.minor) != (10, 7) or props.multi_processor_count != _CTAS:
            raise ValueError("ML4 megakernel requires SM107 (VR200) with 212 SMs")
        mode = os.environ.get("TRTLLM_ML4_MEGAKERNEL_MODE", "all")
        indices = select_megakernel_layers(
            decode.core.full_indexer, decode.core.dense_layers, mode=mode
        )
        if not indices:
            return None
        return cls(decode, indices)

    def begin(self, hidden_states: torch.Tensor, metadata: Any, step: dict[str, Any]) -> bool:
        from tensorrt_llm.logger import logger

        if tuple(hidden_states.shape) != (_TOK, _HIDDEN) or hidden_states.dtype != torch.bfloat16:
            return False
        if step["num_contexts"] != 0 or step["num_seqs"] != 1 or step["num_tokens"] != _TOK:
            return False
        manager = metadata.kv_cache_manager
        tpb = int(step.get("tokens_per_block", 0))
        if manager is None or tpb not in (32, 64) or metadata.max_seq_len > self.shared.max_keys:
            if not getattr(self, "_logged_reject", False):
                self._logged_reject = True
                logger.warning(
                    f"[ml4_megakernel] rejected B1/M=4 step: manager={type(manager).__name__}, "
                    f"tpb={tpb}, max_seq_len={getattr(metadata, 'max_seq_len', None)}, "
                    f"max_keys={self.shared.max_keys}"
                )
            return False
        if self.manager is None or self.manager() is not manager:
            if torch.cuda.is_current_stream_capturing():
                if not getattr(self, "_logged_cap_reject", False):
                    self._logged_cap_reject = True
                    logger.warning(
                        "[ml4_megakernel] rejected B1/M=4 capture: manager not bound before capture"
                    )
                return False
            for impl in self.layers.values():
                impl.bind(manager)
            self.manager = weakref.ref(manager)
            logger.info(
                f"[ml4_megakernel] bound KV cache for TP16/B1/M=4 decode "
                f"(tpb={tpb}, max_seq_len={metadata.max_seq_len}, layers={len(self.layers)})"
            )
        if torch.cuda.is_current_stream_capturing() and not getattr(
            self, "_logged_captured", False
        ):
            self._logged_captured = True
            logger.info("[ml4_megakernel] captured B1/M=4 megakernel into CUDA graph")
        return True
