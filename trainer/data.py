"""Dataset builders.

Exposes:
- ``synthetic_regression``: random x/y, great for fast CPU tests.
- ``synthetic_classification``: random x with integer labels.
- ``mnist``: real MNIST via torchvision (optional dependency).

All builders return a ``(dataset, loss_fn)`` tuple so the Trainer doesn't
need to know which loss matches which task.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, TensorDataset

from .config import DataConfig

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_dataset(cfg: DataConfig, seed: int = 42) -> tuple[Dataset, LossFn]:
    if cfg.name == "synthetic_regression":
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(cfg.num_samples, cfg.input_dim, generator=g)
        # Deterministic linear target + noise so loss actually decreases.
        weight = torch.randn(cfg.input_dim, 1, generator=g)
        y = x @ weight + 0.1 * torch.randn(cfg.num_samples, 1, generator=g)
        return TensorDataset(x, y), nn.MSELoss()

    if cfg.name == "synthetic_classification":
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(cfg.num_samples, cfg.input_dim, generator=g)
        y = torch.randint(0, cfg.num_classes, (cfg.num_samples,), generator=g)
        return TensorDataset(x, y), nn.CrossEntropyLoss()

    if cfg.name == "mnist":
        try:
            from torchvision import datasets, transforms
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "torchvision is required for the mnist dataset; "
                "install it with `pip install torchvision`."
            ) from e
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        ds = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
        # Flatten for MLP consumption.
        return _FlattenWrapper(ds), nn.CrossEntropyLoss()

    raise ValueError(f"Unknown dataset: {cfg.name}")


def build_dataloader(
    dataset: Dataset,
    cfg: DataConfig,
    rank: int,
    world_size: int,
    seed: int = 42,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler: DistributedSampler | None = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


class _FlattenWrapper(Dataset):
    """Flattens image tensors so MNIST can be fed to the MLP."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x.reshape(-1), y
