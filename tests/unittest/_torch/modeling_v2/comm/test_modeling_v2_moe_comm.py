# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the attention-DP expert path's collectives on sm_107:
`lamport_moe_dispatch`, `lamport_moe_combine`, `allgather_list` and `reducescatter`'s ragged
cells.

The matrix is `_moe_comm_op_matrix.py` beside this file, which drives those entries' `CELLS`
in two jobs, one test each:

- the world-4 cells: the matrix is its own launcher, `mpirun -n 4` over 4 devices of one node,
  started here in a fresh interpreter that carries no CUDA or torch.distributed state;
- the world-8 cells: 8 MPI ranks over two nodes of 4 devices, started by the launcher named in
  `MODELING_V2_MPI_LAUNCH` (a command prefix that starts the command after it once per rank,
  four per node); skipped without it.
"""

import importlib.util
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires VR200/SM107",
)

_HERE = Path(__file__).resolve().parent
_MATRIX = _HERE / "_moe_comm_op_matrix.py"
_LAUNCH_ENV = "MODELING_V2_MPI_LAUNCH"
_LAUNCHER_GRACE_S = 300
_DROPPED = ("TLLM_DISABLE_MPI",)
_DROPPED_PREFIXES = ("PMIX_", "PMI_", "OMPI_", "SLURM_", "PRTE_")


def _env() -> dict:
    # tests/unittest/_torch/modeling_v2/comm -> repo root.
    repo_root = _HERE.parents[4]
    assert (repo_root / "tensorrt_llm").is_dir(), f"expected the repo root at {repo_root}"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _DROPPED and not k.startswith(_DROPPED_PREFIXES)
    }
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    return env


def _deadline() -> int:
    spec = importlib.util.spec_from_file_location(_MATRIX.stem, _MATRIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MPI_DEADLINE_S + _LAUNCHER_GRACE_S


# Starts its own ranks over the visible devices; under xdist workers would fight for the GPUs.
@pytest.mark.no_xdist
def test_moe_comm_world4() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    if len(devices) < 4:
        pytest.skip(f"the world-4 cells need 4 devices, {len(devices)} visible")
    env = dict(_env(), CUDA_VISIBLE_DEVICES=",".join(devices[:4]))
    completed = subprocess.run(
        [sys.executable, str(_MATRIX)], env=env, capture_output=True, text=True, timeout=_deadline()
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{_MATRIX.name} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )


@pytest.mark.no_xdist
def test_moe_comm_world8_mpi_ranks() -> None:
    launch = os.environ.get(_LAUNCH_ENV, "").strip()
    if not launch:
        pytest.skip(
            f"the world-8 cells need 8 ranks over two nodes, which this process cannot start; "
            f"set {_LAUNCH_ENV} to a launcher that starts them"
        )
    command = [*shlex.split(launch), sys.executable, str(_MATRIX), "--mpi-rank-worker"]
    job = subprocess.Popen(command, env=_env(), start_new_session=True)
    try:
        code = job.wait(timeout=_deadline())
    except subprocess.TimeoutExpired:
        os.killpg(job.pid, signal.SIGTERM)
        try:
            job.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(job.pid, signal.SIGKILL)
            job.wait()
        raise AssertionError("the world-8 job did not finish (wedged)") from None
    assert code == 0, f"the world-8 job exited {code}: {shlex.join(command)}"
