"""Command-line entry point: ``python -m trainer --config path/to/config.yaml``.

Two launch modes:

- Default: spawns ``world_size`` processes on this machine using
  ``DistributedTrainer``. Good for local experiments and tests.
- ``--use-torchrun``: assume the process was launched by ``torchrun`` and
  read ``RANK``/``WORLD_SIZE`` etc. from the environment.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import TrainConfig, load_config
from .distributed import DistributedTrainer, init_from_env, shutdown
from .engine import run_training


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distributed training launcher")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    p.add_argument(
        "--use-torchrun",
        action="store_true",
        help="Assume torchrun set RANK/WORLD_SIZE; don't spawn.",
    )
    p.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Override world_size from the config file.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps from the config file (useful for smoke tests).",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint path or 'latest'.",
    )
    return p


def _apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.world_size is not None:
        cfg.world_size = args.world_size
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _apply_overrides(load_config(args.config), args)

    launched_by_torchrun = args.use_torchrun and "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if launched_by_torchrun:
        env = init_from_env(cfg.backend)
        try:
            run_training(env, cfg)
        finally:
            shutdown()
        return 0

    trainer = DistributedTrainer(world_size=cfg.world_size, backend=cfg.backend)
    trainer.run(run_training, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
