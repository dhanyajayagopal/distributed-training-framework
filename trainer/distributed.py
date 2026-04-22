"""Process-group setup and launch helpers.

Two ways to start a distributed run:

1. Programmatically via ``DistributedTrainer.run(fn)`` which uses
   ``torch.multiprocessing.spawn`` on a single machine.
2. Externally via ``torchrun``, in which case ``init_from_env()`` should be
   called from inside the worker to read ``RANK`` / ``WORLD_SIZE`` /
   ``LOCAL_RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT`` from the environment.
"""
from __future__ import annotations

import os
import socket
import traceback
from contextlib import closing
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _auto_backend() -> str:
    return "nccl" if torch.cuda.is_available() else "gloo"


@dataclass
class DistEnv:
    """Distributed environment information for the current process."""
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_from_env(backend: str | None = None) -> DistEnv:
    """Initialise the process group from environment variables.

    Expected when launched via ``torchrun`` or ``DistributedTrainer``.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    backend = backend or _auto_backend()
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return DistEnv(rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class DistributedTrainer:
    """Spawns ``world_size`` worker processes on the local machine.

    Prefer ``torchrun`` for multi-node jobs; this class exists for simple
    single-machine experimentation and tests.
    """

    def __init__(
        self,
        world_size: int = 2,
        backend: str | None = None,
        master_addr: str = "127.0.0.1",
        master_port: int | None = None,
    ):
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        self.world_size = world_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port or _find_free_port()

    def run(
        self,
        train_fn: Callable[..., None],
        *fn_args,
        **fn_kwargs,
    ) -> None:
        """Spawn workers.

        ``train_fn`` must be a top-level (picklable) function accepting
        ``(env, *fn_args, **fn_kwargs)``. Lambdas will not work because
        ``mp.spawn`` uses pickle to ship the function to child processes.
        """
        if self.world_size == 1:
            self._worker(0, train_fn, fn_args, fn_kwargs)
            return
        mp.spawn(
            self._worker,
            args=(train_fn, fn_args, fn_kwargs),
            nprocs=self.world_size,
            join=True,
        )

    def _worker(
        self,
        rank: int,
        train_fn: Callable[..., None],
        fn_args: tuple = (),
        fn_kwargs: dict | None = None,
    ) -> None:
        os.environ.setdefault("MASTER_ADDR", self.master_addr)
        os.environ.setdefault("MASTER_PORT", str(self.master_port))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        env = init_from_env(self.backend)
        try:
            train_fn(env, *fn_args, **(fn_kwargs or {}))
        except Exception:
            # Make sure the traceback is visible; otherwise spawn can hang
            # or swallow the real error.
            traceback.print_exc()
            raise
        finally:
            shutdown()
