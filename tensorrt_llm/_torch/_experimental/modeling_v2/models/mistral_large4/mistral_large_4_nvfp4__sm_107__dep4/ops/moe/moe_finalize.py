# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 MoE finalize: each token's top-k expert outputs, read out of the expert runner's
permuted FC2 output and weighted, summed into its bf16 row. The same sum as the runner's own
finalize kernel, launched as a Triton kernel that reads each row's indices and weights once."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Mistral Large 4: hidden 6144, top-4.
HIDDEN = 6144
TOPK = 4


@triton.jit
def _moe_finalize_kernel(
    perm_ptr,  # bf16 [permuted rows, >= H], unit stride over H
    perm_st,
    idx_ptr,  # int32 [rows, TOPK]: permuted row of each (token, k), -1 when not this rank's
    wts_ptr,  # bf16 [rows, TOPK]
    out_ptr,  # bf16 [rows, H], unit stride over H
    out_st,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    if USE_PDL:
        gdc_wait()
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        p = tl.load(idx_ptr + row * TOPK + k)
        w = tl.load(wts_ptr + row * TOPK + k).to(tl.float32)
        x = tl.load(perm_ptr + tl.maximum(p, 0).to(tl.int64) * perm_st + h, mask=p >= 0, other=0.0)
        # The runner's finalize adds `scale * expert` into its fp32 sum and skips an expert
        # outside this rank's window: an fma, and the sum left as it was.
        acc = tl.where(p >= 0, acc + w * x.to(tl.float32), acc)
    tl.store(out_ptr + row * out_st + h, acc.to(tl.bfloat16))


def _launch_config(rows: int) -> tuple[int, int]:
    """(BLOCK_H, num_warps) for `rows` tokens: one program per token and BLOCK_H columns.
    Measured on VR200 at 256 and 512 rows over 256..2048 columns and 1..8 warps."""
    return (2048, 4) if rows >= 512 else (1024, 4)


class MoeFinalize(OpWrapper):
    """`out[t] = sum_k weights[t, k] * permuted[idx[t, k]]` over the k with `idx[t, k] >= 0`,
    accumulated in fp32 in k order and rounded once to bf16: the expert runner's finalize of its
    `do_finalize=False` outputs (the permuted FC2 output and the expanded-to-permuted index map),
    with the routing weights the runner was called with. A token with no expert in this rank's
    window comes back +0.0.

    `permuted` is bf16 `[permuted rows, hidden]` with unit stride over hidden, `idx` int32 and
    `weights` bf16 `[rows, topk]`. Writes `out` (bf16 `[rows, hidden]`, unit stride over hidden)
    and returns it.

    Fusion boundary: replaces the runner's finalize kernel; the runner's GEMMs are untouched.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The Mistral Large 4 dep4 decode target's expert window over every rank's gathered rows on
    # each of the 71 MoE layers: 4 or 8 ranks of up to 128 rows each.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep4/dep8 decode, gathered rows 1/64/256/512/1024: hidden 6144, top-4 over a "
            "16-expert window, bitwise equal to the runner finalize's sum, graph replay included",
            spec=dict(rows=(1, 64, 256, 512, 1024), local_experts=16),
        ),
    )

    note = """
    Bits. The same fp32 sum, in the same order, of the same products as the runner's finalize
    kernel, rounded once: bitwise equal to it. A product of two bf16 values is exact in fp32, so
    the sum is the same whether or not an add is contracted into an fma.

    Cost. Each program reads its token's indices and weights once and issues its experts' row
    loads together. Measured on VR200 inside the ML4 dep8 runner call (16 experts, layer 1
    scales): at 512 gathered rows 4.6 us against the runner's 7.5, the FC2 end to the finalize
    end 2.8 us against 5.5, the whole call 105.5 us against 109.4; at 256 rows that span 2.3 us
    against 2.5.

    Not certified: other than hidden 6144 and top-4; NaN or infinite expert outputs.
    """

    def __call__(
        self,
        permuted: torch.Tensor,
        idx: torch.Tensor,
        weights: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        rows, topk = idx.shape
        hidden = out.shape[1]
        if rows:
            block_h, warps = _launch_config(rows)
            _moe_finalize_kernel[(rows, hidden // block_h)](
                permuted, permuted.stride(0), idx, weights, out, out.stride(0),
                TOPK=topk, BLOCK_H=block_h, USE_PDL=True, num_warps=warps, launch_pdl=True,
            )  # fmt: skip
        return out

    def reference(
        self,
        permuted: torch.Tensor,
        idx: torch.Tensor,
        weights: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """The k-ordered fp32 sum of the weighted rows, rounded to bf16."""
        hidden = out.shape[1]
        acc = torch.zeros(idx.shape[0], hidden, dtype=torch.float32, device=out.device)
        for k in range(idx.shape[1]):
            p = idx[:, k].long()
            rows = permuted[p.clamp_min(0), :hidden].float()
            acc = torch.where(
                (p >= 0)[:, None], torch.addcmul(acc, weights[:, k : k + 1].float(), rows), acc
            )
        return acc.to(torch.bfloat16)

    def is_valid(
        self,
        permuted: torch.Tensor,
        idx: torch.Tensor,
        weights: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        assert permuted.dtype == torch.bfloat16 and permuted.stride(1) == 1, (
            f"permuted must be bf16 with unit stride over hidden; got {permuted.dtype}, "
            f"strides {tuple(permuted.stride())}"
        )
        assert idx.dtype == torch.int32 and weights.shape == idx.shape, (
            f"idx must be int32 and weights its shape; got {idx.dtype} {tuple(idx.shape)}, "
            f"{tuple(weights.shape)}"
        )
        assert out.dtype == torch.bfloat16 and out.shape[0] == idx.shape[0], (
            f"out must be bf16 [rows, hidden]; got {out.dtype} {tuple(out.shape)}"
        )
        hidden = out.shape[1]
        block_h = _launch_config(idx.shape[0])[0]
        assert hidden % block_h == 0 and permuted.shape[1] >= hidden, (
            f"hidden {hidden} must split into blocks of {block_h}"
        )


moe_finalize = MoeFinalize()
