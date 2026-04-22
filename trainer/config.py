"""Training configuration.

A single dataclass describes a training run. Configs can be loaded from YAML
so one CLI entry-point can run many different experiments.
"""
from __future__ import annotations

import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OptimizerConfig:
    name: str = "sgd"  # "sgd" or "adam"
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class DataConfig:
    name: str = "synthetic_regression"  # "synthetic_regression", "synthetic_classification", "mnist"
    num_samples: int = 2048
    input_dim: int = 10
    num_classes: int = 10
    batch_size: int = 64
    num_workers: int = 0
    data_dir: str = "./data"


@dataclass
class ModelConfig:
    name: str = "mlp"  # "mlp", "pipeline_mlp"
    input_dim: int = 10
    hidden_dim: int = 128
    output_dim: int = 1
    num_layers: int = 2


@dataclass
class TrainConfig:
    strategy: str = "ddp"  # "ddp" or "pipeline"
    world_size: int = 2
    max_steps: int = 50
    grad_accum_steps: int = 1
    precision: str = "fp32"  # "fp32", "fp16", "bf16"
    grad_clip: float | None = None
    log_interval: int = 10
    ckpt_interval: int = 0  # 0 = disabled
    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    resume_from: str | None = None  # path or "latest"
    seed: int = 42
    backend: str | None = None  # None = auto (nccl if cuda else gloo)
    # Pipeline-only
    num_micro_batches: int = 4
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


def _resolve_hints(cls):
    """Resolve string annotations (from ``from __future__ import annotations``)."""
    return typing.get_type_hints(cls)


def _from_dict(cls, data: dict[str, Any]):
    """Construct a dataclass from a dict, recursively."""
    if not is_dataclass(cls):
        return data
    hints = _resolve_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints.get(f.name, f.type)
        if isinstance(value, dict) and isinstance(hint, type) and is_dataclass(hint):
            kwargs[f.name] = _from_dict(hint, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> TrainConfig:
    """Load a YAML config file into a TrainConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(TrainConfig, raw)


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return asdict(cfg)
