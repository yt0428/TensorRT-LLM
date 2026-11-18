# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the attention-DP expert path's collectives on sm_107:
`lamport_moe_dispatch`, `lamport_moe_combine`, `allgather_list`, and the ragged cells of
`reducescatter`.

Every rank runs under MPI, as a dep4 or dep8 engine's ranks do: the NCCL entries on the
MPI-session communicator, the one-shot entries on a `MoeCommWorkspace` whose process group
comes up from rank 0's address broadcast over the MPI communicator (`MPI.COMM_WORLD` here).
Every rank regenerates every rank's inputs from a shared seed, so each reference is arithmetic
rather than a second collective.

- the one-shot entries, bitwise against their references (the combine's norm output within
  one bf16 ulp: the reference's norm is fp64), eager at every cell's counts, ragged counts
  swept, every 16-bit word pattern in the payload; captured decode graphs of one dispatch and
  one combine per MoE layer at every graph batch, replayed on new rows between eager calls;
- a long call sequence whose counts change call to call (eager and replayed), so every slot of
  the rotation is reused many times over;
- `allgather_list` bitwise against the concatenations, uniform and ragged;
- `reducescatter`'s ragged cells on payloads whose every partial sum is exact in bf16, so the
  NCCL reduction's order does not matter, bitwise.

Two jobs, both this file:

- the world-4 cells, on one node. Run plainly, this script starts `mpirun -n 4` over the
  devices in CUDA_VISIBLE_DEVICES under a deadline it enforces, since a collective whose ranks
  disagree hangs rather than raising.

      CUDA_VISIBLE_DEVICES=0,1,2,3 python _moe_comm_op_matrix.py

- the world-8 cells, over two nodes of four devices: started once per rank by an MPI launcher
  with `--mpi-rank-worker`, as the dep8 engine's ranks are.

      <launcher> python _moe_comm_op_matrix.py --mpi-rank-worker

Not a pytest module; the collected entry point is `test_modeling_v2_moe_comm.py` beside this
file. Only torch and the standard library are imported at module scope.
"""

import faulthandler
import itertools
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

_MPI_WORKER_FLAG = "--mpi-rank-worker"
MPI_DEADLINE_S = 1200
WORLD_SIZE = 4
#: Columns of a combine row, and the payload widths of a dispatch row in bytes.
HIDDEN = 6144
TOPK = 4
#: Replays of each captured graph, each on new rows.
REPLAYS = 2
#: Calls of the rotation check, each at its own counts.
SEQUENCE_CALLS = 600

# Bound in the rank body.
RANK = 0
WORLD = 1
NODES = 1
GROUP: List[int] = []
MPI_COMM: Any = None
WS: Any = None
dispatch: Any = None
combine: Any = None
gather_list: Any = None
rs: Any = None
DRIVEN: Dict[str, int] = {}


def _gen(seed: int, rank: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed * 7919 + rank)


def _dispatch_inputs(n: int, seed: int, rank: int, bits: bool = False):
    """Rank `rank`'s rows of a dispatch: random codes and positive e4m3 scales, ids in -1..127,
    softmax-like weights. With `bits`, the codes cycle through every 16-bit word pattern."""
    g = _gen(seed, rank)
    codes = torch.randint(0, 256, (n, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=g)
    if bits:
        words = torch.arange(n * HIDDEN // 4, device="cuda", dtype=torch.int64) + rank * 4099
        codes.view(torch.int16).view(-1)[:] = (words % 65536 - 32768).to(torch.int16)
    scales = torch.randint(
        0, 0x7F, (n * HIDDEN // 16,), dtype=torch.uint8, device="cuda", generator=g
    )
    ids = torch.randint(-1, 128, (n, TOPK), dtype=torch.int32, device="cuda", generator=g)
    weights = torch.rand((n, TOPK), device="cuda", generator=g).to(torch.bfloat16)
    return codes, scales, ids, weights


def _combine_inputs(counts, seed: int, rank: int, bits: bool = False, exact: bool = False):
    """Rank `rank`'s partial over every rank's rows, and its own rows' shared output and
    residual. `bits`: partials of every bf16 pattern. `exact`: small integers, whose sums in
    any order are exact in bf16."""
    g = _gen(seed, rank)
    total, n = sum(counts), counts[rank]
    if exact:
        partial = torch.randint(-8, 9, (total, HIDDEN), device="cuda", generator=g).to(
            torch.bfloat16
        )
    else:
        partial = (torch.randn((total, HIDDEN), device="cuda", generator=g) * 0.3).to(
            torch.bfloat16
        )
    if bits:
        words = torch.arange(total * HIDDEN, device="cuda", dtype=torch.int64) + rank * 7
        partial.view(torch.int16).view(-1)[:] = (words % 65536 - 32768).to(torch.int16)
    shared = (torch.randn((n, HIDDEN), device="cuda", generator=g) * 0.3).to(torch.bfloat16)
    residual = (torch.randn((n, HIDDEN), device="cuda", generator=g) * 4).to(torch.bfloat16)
    return partial, shared, residual


def _norm_weight(seed: int) -> torch.Tensor:
    return (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=_gen(seed, 99))).to(
        torch.bfloat16
    )


def _bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    assert out.shape == ref.shape and out.dtype == ref.dtype, (where, out.shape, ref.shape)
    a, b = out.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
    if not torch.equal(a, b):
        bad = (a != b).nonzero()
        raise AssertionError(f"{where}: {bad.shape[0]} bytes differ, first at {bad[0].tolist()}")


def _same_values(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    """Bitwise, except that any NaN matches any NaN (a NaN's payload is not part of a sum)."""
    nan = out.isnan()
    assert torch.equal(nan, ref.isnan()), f"{where}: NaN positions differ"
    _bitwise(torch.where(nan, 0, out), torch.where(nan, 0, ref), where)


def _within_one_ulp(out: torch.Tensor, ref: torch.Tensor, where: str) -> int:
    """`out` within one bf16 ulp of `ref` everywhere; returns how many elements differ."""
    o, r = out.float(), ref.float()
    ulp = torch.exp2(torch.floor(torch.log2(r.abs().clamp_min(2.0**-126))) - 7)
    worst = ((o - r).abs() / ulp).max().item()
    assert worst <= 1.0, f"{where}: {worst} ulp from the fp64 norm"
    return int((o != r).sum())


# ---------------------------------------------------------------------------- one-shot entries


def _check_dispatch(counts, seed: int, bits: bool = False) -> None:
    mine = _dispatch_inputs(counts[RANK], seed, RANK, bits)
    out = dispatch(*mine, counts, WS)
    every = [_dispatch_inputs(counts[r], seed, r, bits) for r in range(WORLD)]
    for o, r, name in zip(
        out, dispatch.reference(every, counts, WS), "codes scales ids weights".split()
    ):
        _bitwise(o, r, f"dispatch {counts} {name}")


def _check_combine(counts, seed: int, norm: bool, bits: bool = False) -> int:
    partial, shared, residual = _combine_inputs(counts, seed, RANK, bits)
    w = _norm_weight(seed)
    res_in = residual.clone()
    out = combine(
        partial, shared, residual if norm else None, w if norm else None, 1e-5, counts, WS
    )
    partials = [_combine_inputs(counts, seed, r, bits)[0] for r in range(WORLD)]
    ref, ref_res = combine.reference(
        partials, shared, res_in if norm else None, w if norm else None, 1e-5, counts, WS
    )
    if not norm:
        _same_values(out, ref, f"combine {counts}")
        return 0
    _bitwise(residual, ref_res, f"combine {counts} residual")
    return _within_one_ulp(out, ref, f"combine {counts} norm")


def _cells(entry) -> List[Any]:
    return [c for c in entry.CELLS if c.spec["world_size"] == WORLD]


def check_one_shot_eager_cells() -> None:
    """Every eager cell of both one-shot entries, at its counts or over its sweep."""
    for cell in _cells(dispatch):
        spec = cell.spec
        if spec.get("graph"):
            continue
        if "sweep_counts" in spec:
            lo, hi = spec["sweep_counts"]
            for i, counts in enumerate(itertools.product(range(lo, hi + 1), repeat=WORLD)):
                _check_dispatch(counts, cell.seed_base() + i)
        else:
            _check_dispatch(spec["counts"], cell.seed_base(), spec.get("payload") == "bits")
        DRIVEN[cell.why] = 1
    off_by_ulp = 0
    for cell in _cells(combine):
        spec = cell.spec
        if spec.get("graph"):
            continue
        if "sweep_counts" in spec:
            lo, hi = spec["sweep_counts"]
            for i, counts in enumerate(itertools.product(range(lo, hi + 1), repeat=WORLD)):
                off_by_ulp += _check_combine(counts, cell.seed_base() + i, spec["norm"])
        else:
            bits = spec.get("payload") == "bits"
            off_by_ulp += _check_combine(spec["counts"], cell.seed_base(), spec["norm"], bits)
        DRIVEN[cell.why] = 1
    if RANK == 0:
        print(f"[mpi] eager combine norms one ulp off the fp64 norm: {off_by_ulp} elements")


def check_one_shot_graph_cells() -> None:
    """Per graph batch: one captured decode step's 71 dispatches and 71 combines (each layer
    its own inputs, as the step's layers are), replayed on new rows, between eager calls at
    other counts that move the slot counter between replays."""
    d_cells = {c.spec["counts"]: c for c in _cells(dispatch) if c.spec.get("graph")}
    c_cells = {c.spec["counts"]: c for c in _cells(combine) if c.spec.get("graph")}
    assert d_cells.keys() == c_cells.keys()
    w = _norm_weight(1)
    stream = torch.cuda.Stream()
    for counts, cell in d_cells.items():
        sites = cell.spec["sites"]
        n = counts[RANK]
        d_in = [_dispatch_inputs(n, 0, RANK) for _ in range(sites)]
        c_in = [_combine_inputs(counts, 0, RANK) for _ in range(sites)]
        # Warm the kernels at this shape before the capture, as the engine's warmup does.
        dispatch(*d_in[0], counts, WS)
        combine(c_in[0][0], c_in[0][1], c_in[0][2].clone(), w, 1e-5, counts, WS)
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            d_out = [dispatch(*d_in[s], counts, WS) for s in range(sites)]
            c_out = [combine(p, sh, res, w, 1e-5, counts, WS) for p, sh, res in c_in]
        torch.cuda.current_stream().wait_stream(stream)
        for replay in range(REPLAYS):
            seed = cell.seed_base() + 1000 * replay
            for s in range(sites):
                for dst, src in zip(d_in[s], _dispatch_inputs(n, seed + s, RANK)):
                    dst.copy_(src)
                for dst, src in zip(c_in[s], _combine_inputs(counts, seed + s, RANK)):
                    dst.copy_(src)
            res_in = [c[2].clone() for c in c_in]
            graph.replay()
            for s in range(sites):
                every = [_dispatch_inputs(counts[r], seed + s, r) for r in range(WORLD)]
                for o, r in zip(d_out[s], dispatch.reference(every, counts, WS)):
                    _bitwise(o, r, f"graph {counts} site {s} dispatch")
                partials = [_combine_inputs(counts, seed + s, r)[0] for r in range(WORLD)]
                ref, ref_res = combine.reference(
                    partials, c_in[s][1], res_in[s], w, 1e-5, counts, WS
                )
                _bitwise(c_in[s][2], ref_res, f"graph {counts} site {s} residual")
                _within_one_ulp(c_out[s], ref, f"graph {counts} site {s} norm")
            # Eager calls at other counts between replays: the replay must land on whichever
            # slot the device counter then holds.
            _check_dispatch(tuple(range(1, WORLD + 1)), seed + 77)
            _check_combine(tuple(range(WORLD, 0, -1)), seed + 78, norm=True)
        DRIVEN[cell.why] = 1
        DRIVEN[c_cells[counts].why] = 1
        del graph, d_out, c_out, d_in, c_in
        torch.cuda.empty_cache()


def check_slot_rotation_sequence() -> None:
    """`SEQUENCE_CALLS` calls alternating dispatch and combine, each at counts drawn from one
    shared generator (1..512 rows), so every slot is reused many times at changing sizes."""
    rng = torch.Generator().manual_seed(4242)
    for i in range(SEQUENCE_CALLS):
        if i % 50 == 0:
            counts = tuple(WS.max_rows for _ in range(WORLD))
        else:
            hi = 512 if i % 7 == 0 else 24
            counts = tuple(int(c) for c in torch.randint(1, hi + 1, (WORLD,), generator=rng))
        if i % 2 == 0:
            _check_dispatch(counts, 50000 + i)
        else:
            _check_combine(counts, 50000 + i, norm=bool(i % 4 == 1))


# ---------------------------------------------------------------------------- NCCL entries


def _payload(n: int, seed: int, rank: int, spec) -> List[torch.Tensor]:
    g = _gen(seed, rank)
    out = []
    for trailing, dtype in spec["payload"]:
        if dtype.is_floating_point:
            out.append(torch.randn((n, *trailing), device="cuda", generator=g).to(dtype))
        else:
            info = torch.iinfo(dtype)
            lo, hi = max(info.min, -(2**31)), min(info.max, 2**31 - 1)
            out.append(torch.randint(lo, hi, (n, *trailing), device="cuda", generator=g).to(dtype))
    return out


def _check_gather_list(counts, spec, seed: int) -> None:
    sizes = None if len(set(counts)) == 1 else list(counts)
    if "sizes" in spec and spec["sizes"] is None:
        assert sizes is None, spec
    out = gather_list(_payload(counts[RANK], seed, RANK, spec), sizes, GROUP)
    every = [_payload(counts[r], seed, r, spec) for r in range(WORLD)]
    for i, (o, r) in enumerate(zip(out, gather_list.reference(every, sizes, GROUP))):
        _bitwise(o, r, f"allgather_list {counts} tensor {i}")


def _sweep(spec):
    lo, hi, step = spec["sweep"]
    for mine in range(lo, hi + 1, step):
        for r in range(WORLD):
            yield tuple(mine if q == r else spec["others"][(q - r - 1) % len(spec["others"])]
                        for q in range(WORLD))  # fmt: skip


def check_allgather_list_cells() -> None:
    for cell in _cells(gather_list):
        spec = cell.spec
        patterns = list(_sweep(spec)) if "sweep" in spec else [spec["counts"]]
        for i, counts in enumerate(patterns):
            _check_gather_list(counts, spec, cell.seed_base() + i)
        DRIVEN[cell.why] = 1


def check_reducescatter_ragged_cells() -> None:
    cells = [c for c in rs.CELLS if c.spec.get("ragged") and c.spec["world_size"] == WORLD]
    assert cells, "no ragged reducescatter cell at this world"
    for cell in cells:
        spec = cell.spec
        patterns = list(_sweep(spec)) if "sweep" in spec else [spec["counts"]]
        for i, counts in enumerate(patterns):
            seed = cell.seed_base() + i
            sizes = None if len(set(counts)) == 1 else list(counts)
            partial = _combine_inputs(counts, seed, RANK, exact=True)[0]
            out = rs(partial, sizes, GROUP)
            lo = sum(counts[:RANK])
            blocks = [
                _combine_inputs(counts, seed, r, exact=True)[0][lo : lo + counts[RANK]]
                for r in range(WORLD)
            ]
            _bitwise(out, rs.reference(blocks), f"reducescatter {counts}")
        DRIVEN[cell.why] = 1


# ---------------------------------------------------------------------------- the job

MPI_CHECKS: List[Callable[[], None]] = [
    check_allgather_list_cells,
    check_reducescatter_ragged_cells,
    check_one_shot_eager_cells,
    check_one_shot_graph_cells,
    check_slot_rotation_sequence,
]


def _validating(entry_obj) -> None:
    """Interpose the entry's `is_valid` on its `__call__`, as `_validating.validating` does."""
    cls = type(entry_obj)
    call = cls.__dict__["__call__"]

    def checked(self, *args, **kwargs):
        self.is_valid(*args, **kwargs)
        return call(self, *args, **kwargs)

    cls.__call__ = checked


def _run_one_mpi_rank() -> int:
    global RANK, WORLD, NODES, GROUP, MPI_COMM, WS, dispatch, combine, gather_list, rs
    faulthandler.dump_traceback_later(MPI_DEADLINE_S, exit=True)
    RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])
    WORLD = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]))

    from mpi4py import MPI

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather_list import (
        allgather_list,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.lamport_moe_combine import (
        lamport_moe_combine,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.lamport_moe_dispatch import (
        build_moe_comm_workspace,
        lamport_moe_dispatch,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import (
        reducescatter,
    )
    from tensorrt_llm._utils import mpi_comm

    MPI_COMM = mpi_comm()
    NODES = len(set(MPI_COMM.allgather(socket.gethostname())))
    GROUP = list(range(WORLD))
    dispatch, combine, gather_list, rs = (
        lamport_moe_dispatch,
        lamport_moe_combine,
        allgather_list,
        reducescatter,
    )
    for e in (dispatch, combine, gather_list, rs):
        _validating(e)
    WS = build_moe_comm_workspace(HIDDEN, 512, RANK, WORLD)

    for check in MPI_CHECKS:
        started = time.monotonic()
        try:
            check()
            torch.cuda.synchronize()
        except BaseException:
            import traceback

            print(f"[mpi rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(13)
        MPI_COMM.Barrier()
        if RANK == 0:
            print(f"[mpi] {check.__name__}: {time.monotonic() - started:.1f}s", flush=True)
    every = [c.why for e in (dispatch, combine, gather_list) for c in _cells(e)]
    every += [c.why for c in rs.CELLS if c.spec.get("ragged") and c.spec["world_size"] == WORLD]
    missing = [w for w in every if w not in DRIVEN]
    assert not missing, f"cells not driven: {missing}"
    print(f"[mpi rank {RANK}] {len(DRIVEN)} cells passed at world {WORLD} on {NODES} node(s)")
    sys.stdout.flush()
    faulthandler.cancel_dump_traceback_later()
    MPI.Finalize()
    os._exit(0)


def _end_job(job: Any) -> None:
    """End mpirun and every rank of its session."""
    if job.poll() is None:
        job.terminate()
        try:
            job.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
        except OSError:
            continue
        if int(stat[stat.rindex(")") + 2 :].split()[3]) == job.pid:
            try:
                os.kill(int(d.name), signal.SIGKILL)
            except ProcessLookupError:
                pass
    if job.poll() is None:
        job.kill()
    job.wait()


def main() -> None:
    """Start the world-4 job under mpirun on this node and wait for it."""
    mpirun = shutil.which("mpirun")
    assert mpirun is not None, "the ranks are started with mpirun; none on PATH"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_", "PRTE_"))
        and k not in ("TLLM_DISABLE_MPI", "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    command = [mpirun, "-n", str(WORLD_SIZE), "--oversubscribe", "--allow-run-as-root"]
    command += [sys.executable, str(Path(__file__).resolve()), _MPI_WORKER_FLAG]
    job = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = job.wait(timeout=MPI_DEADLINE_S + 60)
    except subprocess.TimeoutExpired:
        _end_job(job)
        raise SystemExit(f"the {WORLD_SIZE}-rank job did not finish (wedged)")
    if code != 0:
        _end_job(job)
        raise SystemExit(f"the {WORLD_SIZE}-rank job failed: mpirun exited {code}")


if __name__ == "__main__":
    if _MPI_WORKER_FLAG in sys.argv:
        _run_one_mpi_rank()
    else:
        main()
