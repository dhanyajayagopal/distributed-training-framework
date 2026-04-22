"""Checkpoint save / load with safety guarantees.

Features:
- Atomic writes (tmp + os.replace) so a crash during save can't corrupt the file.
- Saves RNG state (python / numpy / torch / cuda) for deterministic resume.
- Saves optimizer, scheduler, AMP scaler, and arbitrary metadata.
- ``latest.pt`` symlink / copy pointer so callers don't need to know the step.
- Only rank 0 writes; all ranks barrier on save so no one races ahead.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

LATEST_NAME = "latest.pt"


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ save

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        step: int,
        extra: dict[str, Any] | None = None,
        rank: int = 0,
    ) -> Path | None:
        """Save a checkpoint. Returns the path on rank 0, ``None`` elsewhere."""
        path: Path | None = None
        if rank == 0:
            payload = {
                "step": step,
                "model_state_dict": _unwrap(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "rng": _gather_rng_state(),
                "extra": extra or {},
            }
            path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
            _atomic_torch_save(payload, path)
            _update_latest_pointer(self.checkpoint_dir, path)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return path

    # ------------------------------------------------------------------ load

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        path: str | Path | None = None,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Load a checkpoint.

        If ``path`` is ``None`` or ``"latest"``, the latest checkpoint in
        ``checkpoint_dir`` is used.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")

        # ``weights_only`` was added in torch 2.0. Fall back if unsupported.
        try:
            payload = torch.load(resolved, map_location=map_location, weights_only=False)
        except TypeError:  # pragma: no cover - very old torch
            payload = torch.load(resolved, map_location=map_location)

        _unwrap(model).load_state_dict(payload["model_state_dict"])
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])
        if restore_rng and payload.get("rng"):
            _set_rng_state(payload["rng"])
        return payload

    def _resolve_path(self, path: str | Path | None) -> Path | None:
        if path is None or str(path) == "latest":
            latest = self.checkpoint_dir / LATEST_NAME
            if latest.exists():
                if latest.is_symlink():
                    return latest.resolve()
                return latest
            # Fallback: pick highest step on disk
            candidates = sorted(
                self.checkpoint_dir.glob("checkpoint_step_*.pt"),
                key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
            )
            return candidates[-1] if candidates else None
        p = Path(path)
        return p if p.exists() else None


# ----------------------------------------------------------------- helpers


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module if wrapped in DDP."""
    return getattr(model, "module", model)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _update_latest_pointer(directory: Path, path: Path) -> None:
    latest = directory / LATEST_NAME
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        os.symlink(path.name, latest)
    except OSError:
        # On filesystems without symlink support (rare CI/Windows), copy.
        import shutil

        shutil.copy2(path, latest)


def _gather_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
