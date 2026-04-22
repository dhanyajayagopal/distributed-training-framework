"""Example models.

``MLP`` is the general-purpose network. ``build_pipeline_stages`` returns a
list of ``nn.Sequential`` stages, one per pipeline rank, that compose into
the same function as the full MLP — useful for pipeline-parallel tests.
"""
from __future__ import annotations

import torch.nn as nn

from .config import ModelConfig


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.name == "mlp":
        return MLP(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            num_layers=cfg.num_layers,
        )
    raise ValueError(f"Unknown model: {cfg.name}")


def build_pipeline_stages(cfg: ModelConfig, world_size: int) -> list[nn.Sequential]:
    """Return a list of ``nn.Sequential`` pipeline stages (one per rank).

    The stages concatenated are functionally equivalent to ``build_model(cfg)``.
    """
    model = build_model(cfg)
    # Unwrap the top-level Sequential so we can split the raw layer list.
    assert isinstance(model, MLP)
    layers = list(model.net.children())
    if world_size < 1 or world_size > len(layers):
        raise ValueError(
            f"world_size={world_size} incompatible with {len(layers)} layers"
        )

    # Even split, with remainder distributed to early stages.
    per = len(layers) // world_size
    rem = len(layers) % world_size
    stages: list[nn.Sequential] = []
    idx = 0
    for rank in range(world_size):
        take = per + (1 if rank < rem else 0)
        stages.append(nn.Sequential(*layers[idx : idx + take]))
        idx += take
    assert idx == len(layers)
    return stages
