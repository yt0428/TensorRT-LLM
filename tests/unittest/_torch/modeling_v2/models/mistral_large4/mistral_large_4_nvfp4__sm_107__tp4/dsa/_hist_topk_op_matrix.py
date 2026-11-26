# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for ML4 ISS-8/16 hist_topk (exact global top-2048 over MNNVL).

Single-node 4-rank check:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python _hist_topk_op_matrix.py

Multi-node 8-rank (ISS-8) or 16-rank (ISS-16) check:
    srun -N2 --ntasks-per-node=4 env MASTER_ADDR=<node0> MASTER_PORT=<port> \
        python _hist_topk_op_matrix.py --rank-worker
    srun -N4 --ntasks-per-node=4 env MASTER_ADDR=<node0> MASTER_PORT=<port> \
        python _hist_topk_op_matrix.py --rank-worker
"""

import os
import signal
import socket
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, List

import torch
import torch.distributed as dist

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 1200

op: Any = None
canonical_rows: Any = None
shard_size: Any = None
shard_capacity: Any = None
validating: Any = None
WORKSPACE: Any = None
RANK = 0
WORLD = 1

_SPECIAL_BITS = (
    0x7F800000,
    -0x00800000,
    0x00000000,
    -0x80000000,
    0x7FC00000,
    -0x00400000,
    0x7FFFFFFF,
)


def _full_logits(spec: dict, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    rows, kv_len, pattern = spec["rows"], spec["kv_len"], spec["pattern"]
    if pattern == "ties":
        logits = torch.randint(0, 64, (rows, kv_len), generator=gen).float()
    elif pattern == "equal":
        logits = torch.ones(rows, kv_len)
    elif pattern == "saturated":
        logits = 1.0 + torch.rand(rows, kv_len, generator=gen) * 2.0**-10
    else:
        logits = torch.randn(rows, kv_len, generator=gen)
    if pattern == "special":
        values = torch.tensor(_SPECIAL_BITS, dtype=torch.int64).to(torch.int32).view(torch.float32)
        for value in values:
            index = torch.randint(0, kv_len, (rows, 64), generator=gen)
            logits.scatter_(1, index, value.expand(rows, 64).contiguous())
    return logits.cuda()


def _shard(full: torch.Tensor, kv_len: int, rank: int, width: int) -> torch.Tensor:
    size = shard_size(kv_len, WORLD)
    n = min(max(kv_len - rank * size, 0), size)
    shard = torch.full((full.shape[0], width), float("nan"), device="cuda")
    shard[:, :n] = full[:, rank * size : rank * size + n]
    return shard


def _assert_same_on_every_rank(t: torch.Tensor, where: str) -> None:
    lo, hi = t.clone(), t.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    assert torch.equal(lo, hi), f"{where}: ranks produced different outputs"


def _mine() -> List[int]:
    return [i for i, c in enumerate(op.CELLS) if c.spec["group_size"] == WORLD]


def check_certified_cells() -> None:
    mine = _mine()
    assert mine, f"no cells for group_size={WORLD}"
    for idx in mine:
        cell = op.CELLS[idx]
        spec = cell.spec
        seed = zlib.crc32(cell.why.encode()) % 2**31
        kv_len, rows, next_n = spec["kv_len"], spec["rows"], spec["next_n"]
        width = shard_capacity(kv_len, WORLD)
        full = _full_logits(spec, seed)
        shards = [_shard(full, kv_len, r, width) for r in range(WORLD)]
        kv_lens = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
        out = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
        with validating(op):
            got = op(
                shards[RANK],
                kv_lens,
                out,
                WORKSPACE.comm_buffer,
                WORKSPACE.state,
                next_n,
                WORLD,
            )
        torch.cuda.synchronize()
        assert got is out
        assert int(WORKSPACE.state[2].item()) == 0, f"{cell.why}: peer wait timed out"
        _assert_same_on_every_rank(out, cell.why)
        expected = op.reference(
            shards, kv_lens, out, WORKSPACE.comm_buffer, WORKSPACE.state, next_n, WORLD
        )
        assert torch.equal(canonical_rows(out), expected), f"{cell.why}: exact top-2048 mismatch"


def check_cuda_graph_replay_and_buffer_rotation() -> None:
    rows, next_n = 4, 4
    max_len = min(WORLD * 98304, 1_048_584)
    width = shard_capacity(max_len, WORLD)
    shard_buf = torch.empty((rows, width), dtype=torch.float32, device="cuda")
    kv_lens = torch.tensor([262_144], dtype=torch.int32, device="cuda")
    out = torch.empty((rows, 2048), dtype=torch.int32, device="cuda")

    # Warmup outside capture.
    full = _full_logits(dict(rows=rows, kv_len=262_144, pattern="randn"), seed=901)
    shard_buf.copy_(_shard(full, 262_144, RANK, width))
    op(shard_buf, kv_lens, out, WORKSPACE.comm_buffer, WORKSPACE.state, next_n, WORLD)
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        op(shard_buf, kv_lens, out, WORKSPACE.comm_buffer, WORKSPACE.state, next_n, WORLD)

    # Replay across 6 rounds (> 3 Lamport buffers) with changing lengths including <= 2048.
    replay_lens = [1500, 2051, 131_073, 262_145, min(530_000, max_len), max_len]
    for rnd, kv_len in enumerate(replay_lens):
        full = _full_logits(dict(rows=rows, kv_len=kv_len, pattern="randn"), seed=1000 + rnd)
        shards = [_shard(full, kv_len, r, width) for r in range(WORLD)]
        shard_buf.copy_(shards[RANK])
        kv_lens.fill_(kv_len)
        out.fill_(-9)
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        assert int(WORKSPACE.state[2].item()) == 0
        _assert_same_on_every_rank(out, f"graph replay len={kv_len}")
        expected = op.reference(
            shards, kv_lens, out, WORKSPACE.comm_buffer, WORKSPACE.state, next_n, WORLD
        )
        assert torch.equal(canonical_rows(out), expected), f"graph replay mismatch len={kv_len}"


CHECKS = (
    check_certified_cells,
    check_cuda_graph_replay_and_buffer_rotation,
)


def _run_one_rank() -> int:
    global op, canonical_rows, shard_size, shard_capacity, validating, WORKSPACE, RANK, WORLD
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", str(rank))))
    local_world = int(
        os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("SLURM_NTASKS_PER_NODE", str(world)))
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method="env://",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", local_rank),
    )

    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.dsa import (  # noqa: E501
        _hist_topk,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.dsa import (  # noqa: E501
        hist_topk as entry,
    )
    from tensorrt_llm.mapping import Mapping

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating as _validating_cm

    op = entry.hist_topk
    canonical_rows = entry.canonical_rows
    shard_size = _hist_topk.shard_size
    shard_capacity = _hist_topk.shard_capacity
    validating = _validating_cm
    RANK, WORLD = rank, world
    assert 2 <= WORLD <= _hist_topk.MAX_RANKS, f"unsupported world size {WORLD}"
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=local_world, tp_size=world)
    WORKSPACE = _hist_topk.create_hist_topk_workspace(mapping, comm=dist.group.WORLD)

    for check in CHECKS:
        try:
            check()
            dist.barrier()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(13)
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed (world={WORLD})", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, f"need at least 2 GPUs, found {world_size}"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        LOCAL_WORLD_SIZE=str(world_size),
        TLLM_DISABLE_MPI="1",
    )
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
            start_new_session=True,
        )
        for r in range(world_size)
    ]

    def _kill_all() -> None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world_size))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for r in sorted(live):
            code = children[r].poll()
            if code is None:
                continue
            live.discard(r)
            if code != 0:
                failure = f"rank {r} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        _kill_all()
        raise AssertionError(f"the {world_size}-rank run failed: {failure}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
