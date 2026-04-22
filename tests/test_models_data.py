"""Sanity checks for models and data builders."""
from __future__ import annotations

import torch

from trainer.config import DataConfig, ModelConfig
from trainer.data import build_dataset
from trainer.models import build_model


def test_mlp_forward_shapes():
    m = build_model(ModelConfig(input_dim=10, hidden_dim=32, output_dim=4, num_layers=3))
    out = m(torch.randn(5, 10))
    assert out.shape == (5, 4)


def test_synthetic_regression_has_learnable_signal():
    """Fitting a linear model should reach low loss on the synthetic set."""
    dataset, loss_fn = build_dataset(DataConfig(name="synthetic_regression", num_samples=256, input_dim=6))
    xs, ys = zip(*(dataset[i] for i in range(len(dataset))))
    x = torch.stack(xs)
    y = torch.stack(ys)
    # Closed-form linear regression solution.
    result = torch.linalg.lstsq(x, y)
    pred = x @ result.solution
    assert loss_fn(pred, y).item() < 0.05


def test_synthetic_classification_produces_labels_in_range():
    cfg = DataConfig(name="synthetic_classification", num_samples=100, input_dim=8, num_classes=5)
    dataset, _ = build_dataset(cfg)
    for i in range(len(dataset)):
        _, y = dataset[i]
        assert 0 <= int(y) < cfg.num_classes
