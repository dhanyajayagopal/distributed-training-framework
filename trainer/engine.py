"""Unified training engine.

Drives a training run from a ``TrainConfig``. Supports:

- Data-parallel (``strategy="ddp"``): wraps the model in
  ``DistributedDataParallel``; falls back to plain single-process for
  world_size == 1.
- Pipeline-parallel (``strategy="pipeline"``): splits an MLP across ranks
  and uses GPipe-style micro-batching.
- Mixed precision (``fp16`` / ``bf16``) with ``GradScaler`` when needed.
- Gradient accumulation.
- Gradient clipping.
- Checkpointing (save every N steps, resume from ``latest`` or explicit path).
"""
from __future__ import annotations

import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import CheckpointManager
from .config import OptimizerConfig, TrainConfig
from .data import build_dataloader, build_dataset
from .distributed import DistEnv
from .logger import TrainingLogger
from .models import build_model, build_pipeline_stages
from .pipeline import PipelineStage

# ---------------------------------------------------------------- helpers


def _seed_everything(seed: int, rank: int = 0) -> None:
    """Seed every RNG. Each rank gets an offset so they don't generate
    identical noise in places where that matters."""
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


def _build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    if cfg.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if cfg.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {cfg.name}")


def _amp_dtype(precision: str) -> torch.dtype | None:
    return {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


# ---------------------------------------------------------------- DDP path


def train_ddp(env: DistEnv, cfg: TrainConfig) -> dict:
    _seed_everything(cfg.seed, env.rank)

    dataset, loss_fn = build_dataset(cfg.data, seed=cfg.seed)
    loader, sampler = build_dataloader(
        dataset, cfg.data, rank=env.rank, world_size=env.world_size, seed=cfg.seed
    )

    model = build_model(cfg.model).to(env.device)
    if env.world_size > 1:
        model = DDP(
            model,
            device_ids=[env.local_rank] if torch.cuda.is_available() else None,
        )

    optimizer = _build_optimizer(model, cfg.optimizer)
    amp_dtype = _amp_dtype(cfg.precision)
    use_scaler = amp_dtype == torch.float16 and torch.cuda.is_available()
    # ``torch.amp.GradScaler`` (torch >= 2.3) is the new home; fall back on
    # the older path for older installations.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except TypeError:  # pragma: no cover - very old torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    ckpt = CheckpointManager(cfg.ckpt_dir)
    logger = TrainingLogger(cfg.log_dir, rank=env.rank, world_size=env.world_size)

    start_step = 0
    if cfg.resume_from is not None:
        info = ckpt.load(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            path=cfg.resume_from,
            map_location=env.device,
        )
        start_step = info.get("step", 0) + 1
        if env.is_main:
            print(f"Resumed from step {info.get('step')}")

    step = start_step
    done = False
    epoch = 0
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            x, y = batch
            x = x.to(env.device, non_blocking=True)
            y = y.to(env.device, non_blocking=True)

            is_accum_boundary = ((step + 1) % cfg.grad_accum_steps) == 0

            # DDP's no_sync disables gradient all-reduce on intermediate
            # accumulation substeps, matching what users expect from grad
            # accumulation under DDP.
            sync_ctx = (
                model.no_sync()
                if (not is_accum_boundary and isinstance(model, DDP))
                else nullcontext()
            )
            amp_ctx = (
                torch.autocast(device_type=env.device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )

            with sync_ctx, amp_ctx:
                out = model(x)
                loss = loss_fn(out, y) / cfg.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if is_accum_boundary:
                if cfg.grad_clip is not None:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_interval == 0:
                logger.log_step(
                    step,
                    loss.detach() * cfg.grad_accum_steps,
                    batch_size=x.size(0),
                    learning_rate=optimizer.param_groups[0]["lr"],
                )

            if cfg.ckpt_interval and step > 0 and step % cfg.ckpt_interval == 0:
                ckpt.save(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    rank=env.rank,
                )

            step += 1
            if step >= cfg.max_steps:
                done = True
                break

    # Final checkpoint so we can always resume from the end of a run.
    if cfg.ckpt_interval:
        ckpt.save(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step - 1,
            rank=env.rank,
        )
    logger.log_summary()
    return {"final_step": step - 1}


# ---------------------------------------------------------------- pipeline path


def train_pipeline(env: DistEnv, cfg: TrainConfig) -> dict:
    """Pipeline-parallel training.

    Only rank 0 iterates the dataloader; every other rank just receives
    activations / sends gradients through the ``PipelineStage`` protocol.
    """
    _seed_everything(cfg.seed)  # same seed across ranks so data agrees

    dataset, loss_fn = build_dataset(cfg.data, seed=cfg.seed)
    # For pipeline we want the same data on every rank; each step the first
    # stage reads a batch and sends activations downstream.
    loader, _sampler = build_dataloader(
        dataset, cfg.data, rank=0, world_size=1, seed=cfg.seed
    )

    stages = build_pipeline_stages(cfg.model, env.world_size)
    stage = PipelineStage(
        stages[env.rank],
        rank=env.rank,
        world_size=env.world_size,
        device=env.device,
    )
    optimizer = _build_optimizer(stage.module, cfg.optimizer)

    # In pipeline parallelism only the final stage has a real loss, so the
    # writer rank is world_size - 1.
    logger = TrainingLogger(
        cfg.log_dir,
        rank=env.rank,
        world_size=env.world_size,
        writer_rank=env.world_size - 1,
    )

    step = 0
    done = False
    while not done:
        for x, y in loader:
            num_micro = cfg.num_micro_batches
            if x.size(0) % num_micro != 0:
                raise ValueError(
                    f"batch_size={x.size(0)} must be divisible by num_micro_batches={num_micro}"
                )
            micro = []
            mb_size = x.size(0) // num_micro
            for i in range(num_micro):
                s = slice(i * mb_size, (i + 1) * mb_size)
                micro.append((x[s], y[s]))

            if env.rank == 0:
                loss_val = stage.step(micro, loss_fn, optimizer)
            else:
                loss_val = stage.step(None, loss_fn, optimizer)

            if step % cfg.log_interval == 0 and env.rank == env.world_size - 1:
                # Only the last pipeline rank has a loss; skip collective so
                # the other ranks don't need to call into the logger.
                logger.log_step(
                    step,
                    loss_val,
                    batch_size=x.size(0),
                    learning_rate=optimizer.param_groups[0]["lr"],
                    reduce=False,
                )

            step += 1
            if step >= cfg.max_steps:
                done = True
                break

    logger.log_summary()
    return {"final_step": step - 1}


# ---------------------------------------------------------------- entrypoint


def run_training(env: DistEnv, cfg: TrainConfig) -> dict:
    if cfg.strategy == "ddp":
        return train_ddp(env, cfg)
    if cfg.strategy == "pipeline":
        return train_pipeline(env, cfg)
    raise ValueError(f"Unknown strategy: {cfg.strategy}")
