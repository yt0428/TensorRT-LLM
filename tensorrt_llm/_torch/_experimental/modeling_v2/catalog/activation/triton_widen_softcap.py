# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The final-logit soft-cap with the fp32 upcast before it, in one pass: `cap * tanh(x / cap)`
over bf16 lm_head logits that need no vocabulary gather (a full-vocabulary lm_head under
attention DP), written as fp32.

It stands in for the logits processor's `.float()` and tanh_softcap's three ops, and is
bit-identical to them: the upcast is exact, and the soft-cap rounds `x * fl(1 / cap)`, libdevice's
`tanhf` (the function the torch op calls) and the product with `cap` to fp32 as they do
(`_widen_softcap_kernel`, whose read of a tensor-parallel target's symmetric gather slot is the
same kernel over one block per rank).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import triton

from .._op import Arch, Cell, OpWrapper
from . import _widen_softcap_kernel as _kernel

#: The checkpoint's vocabulary and `softmax_tanh_gating`.
VOCAB_SIZE = 204800
SOFTCAP = 30.0
#: Columns per CTA.
_BLOCK_V = 4096


class TritonWidenSoftcap(OpWrapper):
    """`torch.tanh(x.float() / cap) * cap` for bf16 `[rows, cols]` `x`, contiguous; returns a
    new `[rows, cols]` fp32 tensor. Rows are independent.

    Bit-identical to the reference, the torch chain: the upcast is a copy, `x / cap` with a
    Python-float `cap` is the multiply by `fl(1 / cap)` torch's CUDA kernel applies, and tanh
    is the same libdevice `tanhf`, bypassed for a subnormal argument where Triton links it
    flushing subnormals and tanhf is the identity.
    """

    ARCHS = frozenset({Arch.SM_107})
    note = """
    Offsets are int64: rows x cols may pass 2^31. `cap` is a Python float; `is_valid` refuses a
    zero, infinite or nan one, which the op takes and answers with zeros or nan.
    """

    # spec: rows, cols, cap; `graph` True also captures the call in a CUDA graph and replays it
    # on fresh logits written into the captured input. Rows per call at dep4: the verified
    # tokens of a decode step's sequences (4 a sequence, up to 32 sequences a rank) for the
    # target, the draft tokens (3 a sequence) for the draft, both captured; a mixed step's
    # context requests add one row each, eager.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"dep4 decode {what} logits, {batch} sequence(s): {rows} rows, captured",
            spec=dict(rows=rows, cols=VOCAB_SIZE, cap=SOFTCAP, graph=True),
        )
        for batch in (1, 2, 4, 8, 16, 32)
        for what, rows in (("target", 4 * batch), ("draft", 3 * batch))
    ) + (
        Cell(
            why="dep4 mixed step, 8 context and 24 generation requests: 104 target rows, eager",
            spec=dict(rows=8 + 4 * 24, cols=VOCAB_SIZE, cap=SOFTCAP, graph=False),
        ),
    )

    def __call__(self, x: torch.Tensor, cap: float) -> torch.Tensor:
        rows, cols = x.shape
        out = x.new_empty((rows, cols), dtype=torch.float32)
        inv_cap = float(np.float32(1.0) / np.float32(cap))
        _kernel.softcap_gather_kernel[(rows, 1, triton.cdiv(cols, _BLOCK_V))](
            x, rows, cols, out, out.stride(0), inv_cap, float(cap), BLOCK_V=_BLOCK_V,
            num_warps=4, enable_fp_fusion=False,
        )  # fmt: skip
        return out

    def reference(self, x: torch.Tensor, cap: float) -> torch.Tensor:
        return torch.tanh(x.float() / cap) * cap

    def is_valid(self, x: torch.Tensor, cap: float) -> None:
        if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous():
            raise ValueError(
                f"triton_widen_softcap: x must be contiguous 2-D bf16, got {x.dtype} "
                f"{tuple(x.shape)} strides {x.stride()}; the kernel reads row r at r * cols"
            )
        if not (math.isfinite(cap) and cap != 0.0):
            raise ValueError(
                f"triton_widen_softcap: cap must be finite and non-zero, got {cap!r}; "
                "a zero cap zeroes every logit and an infinite or nan one makes them nan"
            )


triton_widen_softcap = TritonWidenSoftcap()
