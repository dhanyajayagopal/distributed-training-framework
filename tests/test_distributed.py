"""Smoke tests for DistributedTrainer spawn/exit behaviour."""
from __future__ import annotations

import os
import sys

import pytest

from trainer import DistributedTrainer

from ._helpers import hello_allreduce


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") is None,
    reason="mp.spawn under pytest is flaky on local macOS; run in CI or directly.",
)
def test_two_worker_allreduce():
    DistributedTrainer(world_size=2).run(hello_allreduce)


@pytest.mark.timeout(30)
def test_single_process_run():
    """world_size=1 should still work and not deadlock."""
    DistributedTrainer(world_size=1).run(hello_allreduce)
