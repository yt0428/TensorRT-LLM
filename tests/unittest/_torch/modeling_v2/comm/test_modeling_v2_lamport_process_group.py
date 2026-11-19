# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the Lamport process group's cross-node rendezvous."""

import os
import socket
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.distributed as dist

from tensorrt_llm import _utils as trtllm_utils
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import _lamport
from tensorrt_llm._torch.disaggregation.native import utils as native_utils

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def process_group(monkeypatch):
    for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    initialize = Mock()
    monkeypatch.setattr(dist, "init_process_group", initialize)
    return initialize


def _mpi_comm(monkeypatch, rank, address=None, size=8):
    """The engine's communicator (`tensorrt_llm._utils.mpi_comm()`) as rank `rank` of `size`.
    `MPI.COMM_WORLD` is a larger world that nothing may use."""
    comm = Mock()
    comm.Get_size.return_value = size
    comm.Get_rank.return_value = rank
    comm.bcast.side_effect = lambda value, root: value if rank == root else address
    monkeypatch.setattr(trtllm_utils, "mpi_comm", lambda: comm)
    world = Mock(side_effect=AssertionError("COMM_WORLD is not the engine's communicator"))
    world.Get_size.side_effect = world.Get_rank.side_effect = world.bcast.side_effect = (
        AssertionError("COMM_WORLD is not the engine's communicator")
    )
    monkeypatch.setitem(
        sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(COMM_WORLD=world))
    )
    return comm


def test_mpi_root_broadcasts_routable_address_for_loopback_hostname(monkeypatch, process_group):
    comm = _mpi_comm(monkeypatch, rank=0)
    monkeypatch.setattr(socket, "gethostname", lambda: "rank-zero")
    monkeypatch.setattr(socket, "gethostbyname", lambda name: "127.0.1.1")
    listener = MagicMock()
    listener.__enter__.return_value = listener
    listener.getsockname.return_value = ("0.0.0.0", 23456)
    route = MagicMock()
    route.__enter__.return_value = route
    route.getsockname.return_value = ("10.20.30.40", 54321)
    monkeypatch.setattr(
        socket,
        "socket",
        lambda family=socket.AF_INET, kind=socket.SOCK_STREAM: (
            route if kind == socket.SOCK_DGRAM else listener
        ),
    )

    assert _lamport.ensure_process_group(rank=0, world=8) is True

    listener.bind.assert_called_once_with(("", 0))
    route.connect.assert_called_once_with(("10.255.255.255", 1))
    comm.bcast.assert_called_once_with(("10.20.30.40", 23456), root=0)
    assert os.environ["MASTER_ADDR"] == "10.20.30.40"
    assert os.environ["MASTER_PORT"] == "23456"
    process_group.assert_called_once_with(
        backend="nccl", rank=0, world_size=8, device_id=torch.device("cuda", 2)
    )


def test_explicit_rendezvous_environment_skips_discovery(monkeypatch, process_group):
    for name, value in {
        "MASTER_ADDR": "explicit-master.example",
        "MASTER_PORT": "34567",
        "RANK": "3",
        "WORLD_SIZE": "8",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    discover = Mock(side_effect=AssertionError("Explicit rendezvous must not discover an address"))
    monkeypatch.setattr(native_utils, "get_local_ip", discover)

    assert _lamport.ensure_process_group(rank=3, world=8) is True

    discover.assert_not_called()
    assert os.environ["MASTER_ADDR"] == "explicit-master.example"
    assert os.environ["MASTER_PORT"] == "34567"
    process_group.assert_called_once_with(
        backend="nccl", rank=3, world_size=8, device_id=torch.device("cuda", 2)
    )


def test_mpi_nonroot_uses_broadcast_address(monkeypatch, process_group):
    comm = _mpi_comm(monkeypatch, rank=5, address=("10.20.30.40", 23456))
    discover = Mock(side_effect=AssertionError("Only rank zero chooses the rendezvous address"))
    monkeypatch.setattr(native_utils, "get_local_ip", discover)
    open_socket = Mock(side_effect=AssertionError("Only rank zero chooses the rendezvous port"))
    monkeypatch.setattr(socket, "socket", open_socket)

    assert _lamport.ensure_process_group(rank=5, world=8) is True

    discover.assert_not_called()
    open_socket.assert_not_called()
    comm.bcast.assert_called_once_with(None, root=0)
    assert os.environ["MASTER_ADDR"] == "10.20.30.40"
    assert os.environ["MASTER_PORT"] == "23456"
    process_group.assert_called_once_with(
        backend="nccl", rank=5, world_size=8, device_id=torch.device("cuda", 2)
    )


def test_existing_group_is_used_as_is(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(dist, "get_rank", lambda: 6)
    initialize = Mock(side_effect=AssertionError("An existing group must not be replaced"))
    monkeypatch.setattr(dist, "init_process_group", initialize)
    monkeypatch.setitem(sys.modules, "mpi4py", None)

    assert _lamport.ensure_process_group(rank=6, world=8) is False

    initialize.assert_not_called()


def test_disaggregated_server_rendezvous_on_its_own_communicator(monkeypatch, process_group):
    # Rank 5 of an eight-rank generation server inside a twelve-rank MPI world (four context
    # ranks first): the launcher's RANK / WORLD_SIZE describe the world, not the server.
    for name, value in {
        "MASTER_ADDR": "world-master.example",
        "MASTER_PORT": "29500",
        "RANK": "9",
        "WORLD_SIZE": "12",
    }.items():
        monkeypatch.setenv(name, value)
    comm = _mpi_comm(monkeypatch, rank=5, address=("10.20.30.40", 23456))

    assert _lamport.ensure_process_group(rank=5, world=8) is True

    comm.bcast.assert_called_once_with(None, root=0)
    assert os.environ["MASTER_ADDR"] == "10.20.30.40"
    assert os.environ["MASTER_PORT"] == "23456"
    process_group.assert_called_once_with(
        backend="nccl", rank=5, world_size=8, device_id=torch.device("cuda", 2)
    )


def test_communicator_that_is_not_the_group_is_refused(monkeypatch, process_group):
    _mpi_comm(monkeypatch, rank=9, size=12)

    with pytest.raises(AssertionError, match="rank 9 of 12"):
        _lamport.ensure_process_group(rank=5, world=8)

    process_group.assert_not_called()
