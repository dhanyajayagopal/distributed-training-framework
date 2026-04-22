"""Training metrics logger.

Aggregates loss across ranks (so what you log reflects the global mini-batch
loss, not just rank 0's shard), tracks throughput, and writes a JSONL file
on rank 0.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


class TrainingLogger:
    def __init__(
        self,
        log_dir: str | Path = "logs",
        rank: int = 0,
        world_size: int = 1,
        writer_rank: int = 0,
    ):
        """Only ``writer_rank`` writes to disk (default: rank 0).

        For pipeline parallelism set ``writer_rank = world_size - 1`` so the
        stage that actually computes the loss is the one that records it.
        """
        self.log_dir = Path(log_dir)
        self.rank = rank
        self.world_size = world_size
        self.writer_rank = writer_rank
        self.start_time = time.time()
        self._last_log_time = self.start_time
        self._last_log_samples = 0
        self._samples_seen = 0
        self.log_file: Path | None = None

        if rank == writer_rank:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / "training_log.jsonl"

    # -------------------------------------------------------------- logging

    def log_step(
        self,
        step: int,
        loss: float | torch.Tensor,
        *,
        batch_size: int = 0,
        learning_rate: float | None = None,
        extra: dict[str, Any] | None = None,
        reduce: bool = True,
    ) -> None:
        """Log a step.

        If ``reduce`` is True, ``loss`` is averaged across ranks using
        ``all_reduce`` — this requires *every* rank to call this method at
        the same step. When only a subset of ranks log (e.g. pipeline
        parallelism where only the last rank has a meaningful loss), pass
        ``reduce=False`` to skip the collective and avoid deadlock.
        """
        global_loss = self._average_across_ranks(loss) if reduce else float(
            loss.item() if isinstance(loss, torch.Tensor) else loss
        )
        self._samples_seen += batch_size * self.world_size

        if self.rank != self.writer_rank:
            return

        now = time.time()
        elapsed = now - self.start_time
        window = now - self._last_log_time
        window_samples = self._samples_seen - self._last_log_samples
        throughput = window_samples / window if window > 0 else 0.0
        self._last_log_time = now
        self._last_log_samples = self._samples_seen

        entry: dict[str, Any] = {
            "step": step,
            "loss": float(global_loss),
            "elapsed_s": round(elapsed, 3),
            "samples_per_s": round(throughput, 2),
            "world_size": self.world_size,
            "timestamp": now,
        }
        if learning_rate is not None:
            entry["lr"] = learning_rate
        if extra:
            entry.update(extra)

        assert self.log_file is not None
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(
            f"[step {step:>5}] loss={entry['loss']:.4f} "
            f"lr={learning_rate if learning_rate is not None else 'n/a'} "
            f"tp={entry['samples_per_s']:.1f} samples/s "
            f"elapsed={entry['elapsed_s']:.1f}s"
        )

    def log_summary(self) -> None:
        if self.rank != self.writer_rank:
            return
        total = time.time() - self.start_time
        print(
            f"[summary] done in {total:.1f}s, "
            f"saw {self._samples_seen} samples, "
            f"avg throughput={self._samples_seen / total:.1f} samples/s"
        )

    # -------------------------------------------------------------- helpers

    def _average_across_ranks(self, loss: float | torch.Tensor) -> float:
        if not (dist.is_available() and dist.is_initialized()) or self.world_size == 1:
            return float(loss.item() if isinstance(loss, torch.Tensor) else loss)
        tensor = loss.detach().float() if isinstance(loss, torch.Tensor) else torch.tensor(
            float(loss)
        )
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return float(tensor.item())
