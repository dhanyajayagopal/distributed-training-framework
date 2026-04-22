"""Tests for the YAML config loader."""
from __future__ import annotations

from pathlib import Path

from trainer.config import TrainConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_ddp_synthetic():
    cfg = load_config(REPO_ROOT / "configs" / "ddp_synthetic.yaml")
    assert cfg.strategy == "ddp"
    assert cfg.world_size == 2
    assert cfg.data.name == "synthetic_regression"
    assert cfg.optimizer.name == "sgd"
    assert cfg.model.num_layers == 3


def test_load_pipeline_config():
    cfg = load_config(REPO_ROOT / "configs" / "pipeline_synthetic.yaml")
    assert cfg.strategy == "pipeline"
    assert cfg.num_micro_batches == 4


def test_defaults_are_reasonable():
    cfg = TrainConfig()
    assert cfg.strategy == "ddp"
    assert cfg.precision == "fp32"
    assert cfg.grad_accum_steps == 1
