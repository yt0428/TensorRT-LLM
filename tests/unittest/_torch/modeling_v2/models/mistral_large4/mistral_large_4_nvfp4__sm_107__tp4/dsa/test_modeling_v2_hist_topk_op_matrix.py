# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for ML4 SM107 ISS hist_topk certification matrix."""

import _iss_rank_job
import pytest
import torch

from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_sm_version() != 107
    or torch.cuda.device_count() < _iss_rank_job.WORLD_SIZE,
    reason="requires SM107 and at least four visible GPUs",
)


@pytest.mark.no_xdist
def test_hist_topk_op_matrix() -> None:
    _iss_rank_job.run("hist_topk")
