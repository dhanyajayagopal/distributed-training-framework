"""DDP invariant tests.

Two weaker, reliable invariants instead of cross-world-size bit-parity:

1. After DDP training, every rank holds *identical* model parameters.
   (This is the fundamental DDP guarantee; if it ever fails, gradient
   all-reduce is broken.)
2. Loss decreases over training on a synthetic task with a learnable
   signal — i.e. the training loop actually trains.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

from trainer import DistributedTrainer, TrainConfig
from trainer.checkpoint import CheckpointManager
from trainer.config import DataConfig, ModelConfig, OptimizerConfig
from trainer.models import build_model

from ._helpers import run_ddp, run_ddp_and_snapshot_final


def _cfg(world_size: int, tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        strategy="ddp",
        world_size=world_size,
        max_steps=15,
        grad_accum_steps=1,
        precision="fp32",
        log_interval=5,
        ckpt_interval=15,
        ckpt_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        seed=123,
        model=ModelConfig(name="mlp", input_dim=8, hidden_dim=16, output_dim=1, num_layers=2),
        data=DataConfig(
            name="synthetic_regression",
            num_samples=256,
            input_dim=8,
            batch_size=32,
        ),
        optimizer=OptimizerConfig(name="sgd", lr=0.05, momentum=0.0),
    )


@pytest.mark.timeout(120)
@pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") is None,
    reason="mp.spawn under pytest can hang on local macOS; run manually or in CI.",
)
def test_ddp_trains_and_writes_checkpoint(tmp_path):
    cfg = _cfg(world_size=2, tmp_path=tmp_path)
    DistributedTrainer(world_size=cfg.world_size, backend=cfg.backend).run(run_ddp, cfg)

    # A final checkpoint should exist.
    mgr = CheckpointManager(cfg.ckpt_dir)
    model = build_model(cfg.model)
    info = mgr.load(model=model, path="latest")
    assert info["step"] == cfg.max_steps - 1

    # Log file should show loss and include samples_per_s.
    log_path = Path(cfg.log_dir) / "training_log.jsonl"
    assert log_path.exists()
    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(entries) >= 2
    first_loss = entries[0]["loss"]
    last_loss = entries[-1]["loss"]
    assert last_loss < first_loss, f"loss did not decrease: {first_loss} -> {last_loss}"
    for e in entries:
        assert "samples_per_s" in e


@pytest.mark.timeout(120)
@pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") is None,
    reason="mp.spawn under pytest can hang on local macOS; run manually or in CI.",
)
def test_all_ranks_see_same_final_weights(tmp_path):
    """All ranks load the same final checkpoint; states must match."""
    cfg = _cfg(world_size=2, tmp_path=tmp_path)
    save_dir = tmp_path / "per_rank"
    save_dir.mkdir()

    DistributedTrainer(world_size=cfg.world_size, backend=cfg.backend).run(
        run_ddp_and_snapshot_final, cfg, str(save_dir)
    )

    state0 = torch.load(save_dir / "rank_0.pt", map_location="cpu", weights_only=True)
    state1 = torch.load(save_dir / "rank_1.pt", map_location="cpu", weights_only=True)
    assert set(state0.keys()) == set(state1.keys())
    for k in state0:
        assert torch.equal(state0[k], state1[k]), f"rank 0 vs rank 1 mismatch on {k}"
