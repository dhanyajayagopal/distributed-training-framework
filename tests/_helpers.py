"""Top-level helper functions used by tests.

``mp.spawn`` pickles the target function, so these must be defined at module
scope rather than nested inside test functions.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from trainer import DistEnv, TrainConfig
from trainer.checkpoint import CheckpointManager
from trainer.engine import train_ddp, train_pipeline
from trainer.models import build_model


def hello_allreduce(env: DistEnv) -> None:
    """Rank i contributes (i+1); sum should equal world_size*(world_size+1)/2."""
    tensor = torch.tensor([env.rank + 1], dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, env.world_size + 1))
    assert abs(tensor.item() - expected) < 1e-6, (tensor.item(), expected)


def run_ddp(env: DistEnv, cfg: TrainConfig) -> None:
    train_ddp(env, cfg)


def run_pipeline(env: DistEnv, cfg: TrainConfig) -> None:
    train_pipeline(env, cfg)


def run_ddp_and_snapshot_final(env: DistEnv, cfg: TrainConfig, save_dir: str) -> None:
    """Each rank loads the final checkpoint written by rank 0 and saves it
    under its own filename, so the caller can compare cross-rank."""
    train_ddp(env, cfg)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    mgr = CheckpointManager(cfg.ckpt_dir)
    model = build_model(cfg.model)
    mgr.load(model=model, path="latest")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    torch.save(model.state_dict(), f"{save_dir}/rank_{env.rank}.pt")
