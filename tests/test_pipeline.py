"""Pipeline-parallel tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from trainer import DistributedTrainer, TrainConfig
from trainer.config import DataConfig, ModelConfig, OptimizerConfig
from trainer.models import build_pipeline_stages

from ._helpers import run_pipeline


def test_pipeline_stages_cover_full_model():
    cfg = ModelConfig(name="mlp", input_dim=8, hidden_dim=16, output_dim=4, num_layers=4)
    stages_2 = build_pipeline_stages(cfg, world_size=2)
    assert len(stages_2) == 2

    # Concatenated stages must match the sequential MLP's forward on a sample input.
    full = torch.nn.Sequential(*[m for s in stages_2 for m in s.children()])
    torch.manual_seed(0)
    x = torch.randn(3, 8)
    out_via_stages = x
    for s in stages_2:
        out_via_stages = s(out_via_stages)
    assert torch.allclose(out_via_stages, full(x))


def test_split_handles_remainder():
    cfg = ModelConfig(name="mlp", input_dim=4, hidden_dim=8, output_dim=2, num_layers=3)
    # MLP with num_layers=3 produces a Sequential of 5 children (Linear, ReLU, Linear, ReLU, Linear).
    stages = build_pipeline_stages(cfg, world_size=2)
    child_counts = [len(list(s.children())) for s in stages]
    assert sum(child_counts) == 5
    assert min(child_counts) >= 1


def test_invalid_world_size_raises():
    cfg = ModelConfig(name="mlp", input_dim=4, hidden_dim=8, output_dim=2, num_layers=1)
    # num_layers=1 -> only 1 child layer, can't split across 2.
    with pytest.raises(ValueError):
        build_pipeline_stages(cfg, world_size=2)


@pytest.mark.timeout(120)
@pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI") is None,
    reason="mp.spawn under pytest can hang on local macOS; run manually or in CI.",
)
def test_pipeline_runs_end_to_end(tmp_path):
    cfg = TrainConfig(
        strategy="pipeline",
        world_size=2,
        max_steps=6,
        num_micro_batches=2,
        log_interval=1,
        ckpt_interval=0,
        log_dir=str(tmp_path / "logs"),
        seed=7,
        model=ModelConfig(name="mlp", input_dim=8, hidden_dim=12, output_dim=3, num_layers=4),
        data=DataConfig(
            name="synthetic_classification",
            num_samples=128,
            input_dim=8,
            num_classes=3,
            batch_size=16,
        ),
        optimizer=OptimizerConfig(name="sgd", lr=0.05, momentum=0.0),
    )

    DistributedTrainer(world_size=cfg.world_size, backend=cfg.backend).run(run_pipeline, cfg)

    log_file = Path(cfg.log_dir) / "training_log.jsonl"
    # Only rank world_size-1 logs losses in the pipeline path.
    assert log_file.exists()
    assert log_file.stat().st_size > 0
