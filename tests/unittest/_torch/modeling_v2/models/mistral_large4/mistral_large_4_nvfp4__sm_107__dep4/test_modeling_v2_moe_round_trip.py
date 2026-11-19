# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One real MoE layer's attention-DP round trip through the sm_107 dep4 target's `_moe`, on four
GPUs of one node and on eight ranks over two nodes of four (dep8).

Each rank routes and NVFP4-quantizes its own rows, gathers every rank's codes, scales, expert ids
and weights unpadded, runs its expert window (32 experts at four ranks, 16 at eight) over the
gathered rows in calls of at most `_EXPERT_CALL_TOKENS[world]` rows (or
`_EXPERT_CALL_TOKENS_CHUNKED[world]` under chunked prefill), and reduce-scatters the partials
back, adding the shared expert -- through the one-shot kernels while no rank holds more than
`_ONE_SHOT_ROWS` rows, through NCCL past it; the decode target's one-shot combine also adds the
residual and applies the next layer's norm. A gather or combine defect raises nothing: the
collectives move rows, and every rank simply gets other rows.

On layer 1 of Mistral-Large-4-1T-A52B-NVFP4 itself (its router, its shared expert, each rank's
own window of its routed experts, laid out by the dep4 weight table's own manifest rows), every
rank runs `PrefillTarget._moe` or `DecodeTarget._moe` on its own rows, and every rank's result is
compared with the layer computed on one rank over all ranks' tokens concatenated: the shared
expert plus every window's partial sum, added in fp32 (`_reference`). The decode target is run
with a zero residual, so its new residual is the layer's output itself, and its norm output is
compared with the fp64 norm of the reference.

Cases, per world: unequal counts on the one-shot path (ranks prefilling beside ranks decoding
and the one-row dummy), the one-shot path at its workspace, unequal counts on the NCCL path,
equal counts on the NCCL path, and the gathered maximum (one rank at max_num_tokens); and
captured decode graphs (every rank decoding the same batch, 1, 16, 64 and 128 rows: one row per
request or DSpark's four, at batches up to 32), captured once and replayed on new tokens.

What the gather hands the runner is also checked against the computation the target made
before routing moved ahead of the gather (the router, the routing and the NVFP4 quantization over
the gathered bf16 rows): the codes and scales bit for bit (the one-shot gather stores the 16-bit
word 0x8000, an e2m1 -0 code beside three +0 codes, as +0), the expert ids equal except in rows
whose 4th and 5th router logits are within `TIE` of each other, and the count of rows whose bf16
weights differ in any bit is reported. The router GEMM's tactic depends on its row count.

The gate is the runner entry's, against the row's scale: 16 bf16 ulp per element, 2 ulp relative
RMS; near-tie rows are not compared. One defect is shown to land far outside it: the combine
handed the partial rolled by one rank's block.

The ranks are MPI ranks, as the engine's are: the four-rank test starts `mpirun -n 4` over the
first four visible devices; the eight-rank test runs in each of eight ranks a launcher started
over two nodes of four (`get_mpi_world_size() == 8`). Skipped where the checkpoint is absent
(`Mistral-Large-4-1T-A52B-NVFP4` or `mistral-large-4-nvfp4` under `LLM_MODELS_ROOT`) or off
sm_107.
"""

from __future__ import annotations

import collections
import faulthandler
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SPAWNED_WORLD = 4
MPI_WORLD = 8
DEADLINE_S = 2400
_WORKER_FLAG = "--rank-worker"
CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")
SHARD_INDEX = "consolidated.safetensors.index.json"
LAYER = 1
#: Per world: (name, rows per rank, target per rank).
CASES = {
    4: (
        ("one-shot unequal", (300, 1, 16, 257), ("prefill", "decode", "decode", "prefill")),
        ("one-shot workspace", (512, 1, 64, 3), ("prefill", "decode", "decode", "decode")),
        ("nccl unequal", (1003, 1, 16, 257), ("prefill", "decode", "decode", "prefill")),
        ("nccl equal", (600, 600, 600, 600), ("prefill", "prefill", "decode", "prefill")),
        ("gathered maximum", (32768, 1, 16, 9001), ("prefill", "decode", "decode", "prefill")),
    ),
    8: (
        (
            "one-shot unequal",
            (300, 1, 16, 257, 1, 40, 500, 3),
            ("prefill", "decode", "decode", "prefill", "decode", "prefill", "prefill", "decode"),
        ),
        (
            "nccl unequal",
            (1003, 1, 16, 257, 1, 40, 700, 3),
            ("prefill", "decode", "decode", "prefill", "decode", "prefill", "prefill", "decode"),
        ),
        (
            "gathered maximum",
            (32768, 1, 16, 9001, 1, 2, 300, 32768),
            ("prefill", "decode", "decode", "prefill", "decode", "prefill", "prefill", "prefill"),
        ),
    ),
}
#: The expert call tables, by engine: without chunked prefill (the 32K config) and with it.
EXPERT_CALLS = {"32k": "_EXPERT_CALL_TOKENS", "1m": "_EXPERT_CALL_TOKENS_CHUNKED"}
GRAPH_BATCHES = (1, 16, 64, 128)
ULP = 2.0**-8
GATE_ELT, GATE_RMS = 16.0, 2.0
#: A routing the router GEMM's row count may flip: the 4th and 5th logits closer than this
#: fraction of the row's largest logit.
TIE = 1e-4
E2M1_MAX, E4M3_MAX = 6.0, 448.0
EPS = 1e-5
PARAMS = (
    "router", "mlp_gu", "mlp_gu_s", "mlp_dn", "mlp_dn_s", "fc1_w", "fc1_s", "fc2_w", "fc2_s",
    "isc1", "isc2", "ws2_13", "ws2_2",
)  # fmt: skip
WINDOW_PARAMS = ("fc1_w", "fc1_s", "fc2_w", "fc2_s", "ws2_13", "ws2_2")


def _checkpoint_dir() -> Path | None:
    from test_common.llm_data import llm_models_root

    root = llm_models_root()
    if root is None:
        return None
    for name in CHECKPOINTS:
        path = Path(root) / name
        if (path / "params.json").is_file() and (path / SHARD_INDEX).is_file():
            return path
    return None


def _modules():
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__dep4 import (  # noqa: E501
        modeling as M,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__dep4 import (  # noqa: E501
        weights as W,
    )

    return M, W


def _config(ck: Path):
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader

    return MistralConfigLoader().load_pretrained_config(str(ck))


def _core(config, rank: int, world: int) -> SimpleNamespace:
    """The surface the weight table reads off the core for rank `rank` of `world`, stated, so
    one rank can load the other ranks' windows for the reference."""
    mapping = SimpleNamespace(
        world_size=world, rank=rank, tp_size=world, tp_rank=rank, moe_tp_size=1,
        moe_ep_size=world, moe_ep_rank=rank, enable_attention_dp=True,
    )  # fmt: skip
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=config, mapping=mapping, torch_dtype=torch.bfloat16
        )
    )


def _load(W, ck: Path, config, weight_map: dict, rank: int, world: int, names) -> tuple:
    """Layer `LAYER`'s parameters `names` as rank `rank` of `world`'s `load` leaves them: every
    manifest source read off the shard files and copied to its index through its transform,
    then each row's `layout` on every expert slot. Also the rank's dimension bundle."""
    from safetensors import safe_open

    core = _core(config, rank, world)
    d = W.MODEL_WEIGHTS.dims(core)
    manifest = W.MODEL_WEIGHTS.manifest(core)
    rows = {
        r.name: r
        for r in W.MODEL_WEIGHTS.WEIGHTS
        if r.name in names and LAYER in (r.layer_indices(d) or [])
    }
    keys = {k for n in names for ckpt, *_ in manifest[f"l{LAYER}_{n}"] for k in W._keys(ckpt)}
    by_file = collections.defaultdict(list)
    for key in keys:
        by_file[weight_map[key]].append(key)
    stored = {}
    for name, file_keys in by_file.items():
        with safe_open(str(ck / name), framework="pt", device="cpu") as handle:
            for key in file_keys:
                stored[key] = handle.get_tensor(key)
    out = {}
    for n in names:
        row = rows[n]
        param = torch.empty(row.shape(d), dtype=row.dtype or d.dtype, device="cuda")
        for ckpt, index, transform in manifest[f"l{LAYER}_{n}"]:
            dst = param if index is None else param[index]
            if isinstance(ckpt, tuple):
                src = transform(tuple(stored[k].cuda() for k in ckpt), core)
            else:
                src = stored[ckpt].cuda()
                if transform is not None:
                    src = transform(src.contiguous(), core)
            dst.copy_(src)
        if row.layout is not None:
            for slot in range(param.shape[0]):
                param[slot].copy_(row.layout(param[slot]))
        out[n] = param
    return out, d


def _layer(M, p: dict, window: dict, next_norm: torch.Tensor) -> SimpleNamespace:
    """The `L` `_moe` reads, as `_layer_views` takes it."""
    return SimpleNamespace(
        mlp_gu=p["mlp_gu"],
        mlp_gu_s=M.prepare_weight_scale(p["mlp_gu_s"].t(), p["mlp_gu"].shape[1]),
        mlp_dn=p["mlp_dn"],
        mlp_dn_s=M.prepare_weight_scale(p["mlp_dn_s"].t(), p["mlp_dn"].shape[1]),
        next_norm=next_norm,
        moe=SimpleNamespace(
            router=p["router"].t(),
            fc1_w=window["fc1_w"],
            fc1_s=window["fc1_s"],
            fc2_w=window["fc2_w"],
            fc2_s=window["fc2_s"],
            **vars(M._moe_scalars(p["isc1"], p["isc2"], window["ws2_13"], window["ws2_2"])),
        ),
    )


def _target(M, kind: str, d, rank: int, world: int, side, call: int, ws) -> object:
    """The target class `kind` over the surface `_moe` reads off the core, on the MPI-session
    communicator (`dp_pg` None, as the core holds it under MPI), in expert calls of `call`."""
    target = object.__new__(M.PrefillTarget if kind == "prefill" else M.DecodeTarget)
    target.core = SimpleNamespace(
        hidden=6144, eps=EPS, topk=4, num_experts=d.num_experts, moe_inter=d.moe_inter,
        expert_offset=d.expert_offset, local_experts=d.local_experts, expert_call_tokens=call,
        expert_tune_tokens=call, dp_group=list(range(world)), dp_rank=rank, dp_pg=None,
        _side_stream=side, moe_comm=ws,
    )  # fmt: skip
    return target


def _tokens(total: int, seed: int, scale: float) -> torch.Tensor:
    """Every rank's tokens concatenated in rank order, the same on every rank."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(total, 6144, generator=g, device="cuda") * scale).to(torch.bfloat16)


def _reference(M, x, L, windows: dict, offsets: dict, shared_of, call: int) -> tuple:
    """The layer on one rank over `x`, the concatenated tokens, as the target computed it
    before routing moved ahead of the gather: the router, the routing and the quantization over
    all rows, every window's partials summed in fp32 plus the shared expert. Also each token's
    router near-tie flag and the gathered inputs `(codes, scales, ids, weights)`."""
    acc = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    tie = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    inputs = []
    for lo in range(0, x.shape[0], call):
        hi = min(lo + call, x.shape[0])
        chunk = x[lo:hi]
        logits = M.cublas_mm(chunk, L.moe.router, None, torch.float32)
        top = logits.topk(5, dim=1).values
        tie[lo:hi] = (top[:, 3] - top[:, 4]) < TIE * top.abs().amax(dim=1)
        ids, weights = M.renorm_moe_routing_op(logits, 4, torch.bfloat16)
        xq, xs = M.fp4_quantize(chunk, L.moe.gscale, 16, False, False)
        inputs.append((xq, xs, ids, weights))
        for w, m in windows.items():
            part = M.fp4_block_scale_moe_runner(
                None, None, xq, xs.view(torch.float8_e4m3fn), m.fc1_w, m.fc1_s, None, None,
                None, None, m.fc2_w, m.fc2_s, None, m.out1, m.out1_gate, m.out2, 128, 4, None,
                None, 6144, offsets[w], m.fc1_w.shape[0], None, 1, True, 0,
                topk_weights=weights, topk_ids=ids, tune_max_num_tokens=call,
            )[0]  # fmt: skip
            acc[lo:hi] += part.float()
    shared = shared_of(x)
    gathered = tuple(torch.cat(parts) for parts in zip(*inputs))
    return (acc.to(torch.bfloat16).float() + shared.float()).to(torch.bfloat16), tie, gathered


def _rmsnorm64(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    h = x.double()
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * w.double()).to(torch.bfloat16)


def _deviation(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(worst element deviation against its row's scale, relative RMS deviation), in ulp."""
    o, r = out.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-30)).item() / ULP
    return elt, rms


def _run_rank(ck: Path, world: int, engine: str) -> list[str]:
    """One rank's body under MPI. Returns the rank's gate failures; raises on anything else."""
    from tensorrt_llm._utils import mpi_comm, mpi_rank

    M, W = _modules()
    call = getattr(M, EXPERT_CALLS[engine])[world]
    rank = mpi_rank()
    print(f"[rank {rank}] world {world}, {engine}: expert calls of {call} rows", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = _config(ck)
    weight_map = json.loads((ck / SHARD_INDEX).read_text())["weight_map"]
    p, d = _load(W, ck, config, weight_map, rank, world, PARAMS)
    windows, offsets = {rank: p}, {rank: d.expert_offset}
    for w in range(world):
        if w != rank:
            windows[w], dw = _load(W, ck, config, weight_map, w, world, WINDOW_PARAMS)
            offsets[w] = dw.expert_offset
    assert offsets == {w: w * d.local_experts for w in range(world)}, offsets
    g = torch.Generator(device="cuda").manual_seed(77)
    next_norm = (1 + 0.1 * torch.randn(6144, device="cuda", generator=g)).to(torch.bfloat16)
    L = _layer(M, p, p, next_norm)
    ref_windows = {w: _layer(M, p, windows[w], next_norm).moe for w in sorted(windows)}
    side = torch.cuda.Stream()
    ws = M.build_moe_comm_workspace(6144, M._ONE_SHOT_ROWS, rank, world)
    shared_of = _target(M, "prefill", d, rank, world, side, call, ws)._dense
    # Activations at the amax the layer's static input scale was calibrated for.
    scale = E4M3_MAX * E2M1_MAX * float(p["isc1"]) / 5
    failures = []
    report = {}

    def gate(name, out, ref, keep):
        elt, rms = _deviation(out[keep], ref[keep])
        n = int(keep.sum())
        print(f"[rank {rank}] {name}: {n}/{keep.numel()} rows compared; "
              f"{elt:.2f} elt / {rms:.3f} rms ulp", flush=True)  # fmt: skip
        if not (n > 0 and elt <= GATE_ELT and rms <= GATE_RMS):
            failures.append(f"{name}: {elt:.2f} elt / {rms:.3f} rms ulp over {n} rows")
        return rms

    def check_gathered(name, counts, got, want, tie):
        """The gathered inputs against the computation over the gathered bf16 rows."""
        codes, scales, ids, weights = got
        r_codes, r_scales, r_ids, r_weights = want
        if M._one_shot(SimpleNamespace(moe_comm=ws), counts):
            r_codes = r_codes.clone()
            words = r_codes.view(torch.int16)
            words.masked_fill_(words == -32768, 0)
        if not torch.equal(codes, r_codes) or not torch.equal(scales.view(-1), r_scales.view(-1)):
            failures.append(f"{name}: the gathered NVFP4 codes or scales differ")
        id_rows = (ids != r_ids).any(dim=1)
        if (id_rows & ~tie).any():
            failures.append(f"{name}: {int((id_rows & ~tie).sum())} non-tie rows route elsewhere")
        w_rows = int((weights.view(torch.int16) != r_weights.view(torch.int16)).any(dim=1).sum())
        report[name] = dict(rows=int(ids.shape[0]), ids_differ=int(id_rows.sum()),
                            near_tie=int(tie.sum()), weights_differ=w_rows)  # fmt: skip
        print(f"[rank {rank}] {name}: routing against the gathered-row routing {report[name]}",
              flush=True)  # fmt: skip

    def round_trip(name, counts, kinds, x, roll=False):
        lo = sum(counts[:rank])
        own = x[lo : lo + counts[rank]].contiguous()
        target = _target(M, kinds[rank], d, rank, world, side, call, ws)
        runner, dispatch = M.fp4_block_scale_moe_runner, M._dispatch
        combine, rs = M.lamport_moe_combine, M.reducescatter
        calls, gathered = [], []

        def counted(*args, **kwargs):
            calls.append(args[2].shape[0])
            return runner(*args, **kwargs)

        def seen(*args, **kwargs):
            gathered.append(dispatch(*args, **kwargs))
            return gathered[-1]

        def rolled_combine(partial, *args, **kwargs):
            return combine(torch.roll(partial, -counts[0], 0), *args, **kwargs)

        def rolled_rs(partial, *args, **kwargs):
            return rs(torch.roll(partial, -counts[0], 0), *args, **kwargs)

        M.fp4_block_scale_moe_runner, M._dispatch = counted, seen
        if roll:
            M.lamport_moe_combine, M.reducescatter = rolled_combine, rolled_rs
        try:
            if kinds[rank] == "prefill":
                out = target._moe(own, L, counts)
                norm = None
            else:
                residual = torch.zeros_like(own)
                norm, out = target._moe(own, L, residual, counts)
        finally:
            M.fp4_block_scale_moe_runner, M._dispatch = runner, dispatch
            M.lamport_moe_combine, M.reducescatter = combine, rs
        torch.cuda.synchronize()
        total = sum(counts)
        expected = [min(call, total - lo) for lo in range(0, total, call)]
        if calls != expected:
            failures.append(f"{name}: expert calls {calls}, expected {expected}")
        return out, norm, gathered[0]

    for seed, (name, counts, kinds) in enumerate(CASES[world]):
        x = _tokens(sum(counts), seed, scale)
        out, norm, got = round_trip(name, counts, kinds, x)
        ref, tie, want = _reference(M, x, L, ref_windows, offsets, lambda t: shared_of(t, L), call)
        check_gathered(name, counts, got, want, tie)
        lo = sum(counts[:rank])
        mine = slice(lo, lo + counts[rank])
        gate(name, out, ref[mine], ~tie[mine])
        if norm is not None:
            gate(f"{name} norm", norm, _rmsnorm64(ref[mine], next_norm), ~tie[mine])
        del x, out, ref

    # The defect the comparison must see: every rank's combine handed the partial rolled by one
    # rank's block, on the one-shot path and on the NCCL path.
    for name, counts, kinds in (CASES[world][0], CASES[world][-2]):
        x = _tokens(sum(counts), 5, scale)
        out, _, _ = round_trip(name, counts, kinds, x, roll=True)
        ref, tie, _ = _reference(M, x, L, ref_windows, offsets, lambda t: shared_of(t, L), call)
        lo = sum(counts[:rank])
        keep = ~tie[lo : lo + counts[rank]]
        _, rms = _deviation(out[keep], ref[lo : lo + counts[rank]][keep])
        print(f"[rank {rank}] defect, rolled partial ({name}): {rms:.3f} rms ulp", flush=True)
        if rms <= 5 * GATE_RMS:
            failures.append(f"a rolled partial passes ({name}): {rms:.3f} rms ulp")

    # CUDA graphs: every rank decoding the same batch, captured once, replayed on new tokens.
    for b in GRAPH_BATCHES:
        counts = (b,) * world
        target = _target(M, "decode", d, rank, world, side, call, ws)
        static = torch.zeros(b, 6144, dtype=torch.bfloat16, device="cuda")
        residual = torch.zeros_like(static)
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(2):
                target._moe(static, L, residual, counts)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            norm, _ = target._moe(static, L, residual, counts)
        for replay in range(2):
            x = _tokens(world * b, 100 + 10 * b + replay, scale)
            static.copy_(x[rank * b : (rank + 1) * b])
            residual.zero_()
            graph.replay()
            torch.cuda.synchronize()
            ref, tie, _ = _reference(M, x, L, ref_windows, offsets, lambda t: shared_of(t, L), call)
            mine = slice(rank * b, (rank + 1) * b)
            gate(f"graph {b} rows, replay {replay}", residual.clone(), ref[mine], ~tie[mine])
            gate(f"graph {b} rows, replay {replay} norm", norm.clone(),
                 _rmsnorm64(ref[mine], next_norm), ~tie[mine])  # fmt: skip
        del graph
    every = mpi_comm().allgather(report)
    if rank == 0:
        print(f"[routing] {json.dumps(every)}", flush=True)
    return failures


def _rank_main(ck: Path, world: int, engine: str) -> None:
    """Run the body; every rank sees every rank's failures, so all pass or fail together."""
    from tensorrt_llm._utils import mpi_comm, mpi_rank

    faulthandler.dump_traceback_later(DEADLINE_S, exit=True)
    rank = mpi_rank()
    try:
        failures = _run_rank(ck, world, engine)
    except BaseException:
        import traceback

        print(f"[rank {rank}] raised", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(13)
    every = mpi_comm().allgather(failures)
    faulthandler.cancel_dump_traceback_later()
    bad = [f"rank {r}: {f}" for r, fs in enumerate(every) for f in fs]
    assert not bad, "; ".join(bad)
    print(f"[rank {rank}] all checks passed on every rank", flush=True)


def _worker() -> None:
    torch.cuda.set_device(int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]))
    try:
        _rank_main(Path(os.environ["ML4_CHECKPOINT"]), SPAWNED_WORLD, os.environ["ML4_ENGINE"])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(13)
    sys.stdout.flush()
    from mpi4py import MPI

    MPI.Finalize()
    os._exit(0)


def _mpirun(checkpoint: Path, engine: str) -> None:
    """`mpirun -n 4` over the first four visible devices, outside any enclosing MPI job or Slurm
    step, under the deadline; a wedged collective is ended with its session."""
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("the four ranks are started with mpirun; none on PATH")
    repo_root = Path(__file__).resolve().parents[7]
    assert (repo_root / "tensorrt_llm").is_dir(), repo_root
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [v for v in visible.split(",") if v.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_", "PRTE_"))
        and k not in ("TLLM_DISABLE_MPI", "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    tests_root = repo_root / "tests"
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(devices[:SPAWNED_WORLD]),
        ML4_CHECKPOINT=str(checkpoint),
        ML4_ENGINE=engine,
        PYTHONPATH=os.pathsep.join(
            q for q in (str(repo_root), str(tests_root), env.get("PYTHONPATH", "")) if q
        ),
    )
    command = [mpirun, "-n", str(SPAWNED_WORLD), "--oversubscribe", "--allow-run-as-root"]
    command += [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG]
    job = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = job.wait(timeout=DEADLINE_S + 120)
    except subprocess.TimeoutExpired:
        os.killpg(job.pid, signal.SIGKILL)
        job.wait()
        raise AssertionError(f"the {SPAWNED_WORLD}-rank round trip ({engine}) wedged") from None
    assert code == 0, f"the {SPAWNED_WORLD}-rank round trip ({engine}) failed: mpirun exited {code}"


def _launched_world() -> int:
    from tensorrt_llm._utils import mpi_world_size
    from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size

    world = get_mpi_world_size()
    return world if world == mpi_world_size() else 1


@pytest.mark.no_xdist
@pytest.mark.parametrize("engine", list(EXPERT_CALLS))
def test_moe_round_trip_on_four_ranks(engine):
    checkpoint = _checkpoint_dir()
    if checkpoint is None:
        pytest.skip(f"checkpoint {CHECKPOINTS} is not under LLM_MODELS_ROOT")
    if torch.cuda.device_count() < SPAWNED_WORLD:
        pytest.skip(f"needs {SPAWNED_WORLD} GPUs")
    if _launched_world() not in (1, SPAWNED_WORLD):
        pytest.skip("the four-rank round trip starts its own ranks")
    if torch.cuda.get_device_capability(0) != (10, 7):
        pytest.skip("this target is certified on sm_107")
    _mpirun(checkpoint, engine)


@pytest.mark.no_xdist
@pytest.mark.threadleak(enabled=False)
def test_moe_round_trip_on_eight_ranks():
    from tensorrt_llm._utils import get_sm_version, local_mpi_rank

    checkpoint = _checkpoint_dir()
    if checkpoint is None:
        pytest.skip(f"checkpoint {CHECKPOINTS} is not under LLM_MODELS_ROOT")
    if _launched_world() != MPI_WORLD:
        pytest.skip(f"needs {MPI_WORLD} launched MPI ranks")
    if get_sm_version() != 107:
        pytest.skip("this target is certified on sm_107")
    torch.cuda.set_device(local_mpi_rank())
    _rank_main(checkpoint, MPI_WORLD, "32k")


if __name__ == "__main__" and _WORKER_FLAG in sys.argv:
    _worker()
