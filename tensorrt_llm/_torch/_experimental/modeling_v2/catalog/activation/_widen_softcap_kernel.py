# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Triton kernel of `triton_widen_softcap`: bf16 logits widened to fp32 and soft-capped in
one pass, bit-identical to the fp32 upcast followed by tanh_softcap's three torch ops.

It reads a `[blocks, rows_cap, shard]` bf16 buffer and writes block `p` of row `r` to columns
`p * shard` onward: one block holding whole rows for `triton_widen_softcap`, and the slot of a
tensor-parallel target's symmetric logits gather, one block per rank's vocabulary shard.

Launched with `enable_fp_fusion=False`: each multiply rounds to fp32 as the torch ops do.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _mul_rn(a, b):
    """fp32 product rounded to nearest with subnormals kept, as the torch kernels compute it."""
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def softcap_gather_kernel(
    buf_ptr, rows_cap, shard, out_ptr, out_s_row, inv_cap, cap, BLOCK_V: tl.constexpr
):
    """`out[r, p * shard + v] = cap * tanh(float(buf[p, r, v]) * inv_cap)`, grid `(rows, blocks,
    cdiv(shard, BLOCK_V))`. `x * inv_cap` is the torch division by a Python float; tanh is
    libdevice's `tanhf`, which Triton links flushing subnormals, so a subnormal argument (where
    tanhf is the identity) bypasses it."""
    r = tl.program_id(0).to(tl.int64)
    peer = tl.program_id(1)
    chunk = tl.program_id(2)
    v = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = v < shard
    x = tl.load(buf_ptr + (peer * rows_cap + r) * shard + v, mask=mask, other=0.0)
    y = _mul_rn(x.to(tl.float32), inv_cap)
    t = tl.where(tl.abs(y) < 1.1754943508222875e-38, y, libdevice.tanh(y))
    tl.store(out_ptr + r * out_s_row + peer * shard + v, _mul_rn(t, cap), mask=mask)
