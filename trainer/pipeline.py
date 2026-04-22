"""Pipeline-parallel training with micro-batching.

The original implementation:
- sent a fixed-length shape tensor (breaks for tensors not exactly 4D);
- had no micro-batching, so every non-tail rank sat idle most of the time.

This rewrite fixes both. The shape protocol sends the rank first, then a
shape tensor of that rank, so any dimensionality works. Micro-batching
splits a batch into ``num_micro_batches`` chunks and processes them in a
simple GPipe schedule: forward all chunks in order, then backward in
reverse order. That keeps the code easy to follow while still filling the
pipeline.

The implementation is deliberately unopinionated about activation
checkpointing and 1F1B scheduling; it's intended as a clear, correct
reference that tests can exercise on CPU.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

# ---------------------------------------------------------------- tensor I/O


def send_tensor(tensor: torch.Tensor, dst: int) -> None:
    """Send a tensor to ``dst``, preceded by its shape.

    Only float dtypes are supported (which is all we need for activations
    and gradients). The receiver must know the dtype; we keep that simple
    by always using float32 on the wire.
    """
    tensor = tensor.detach().contiguous().to(torch.float32)
    rank = torch.tensor([tensor.dim()], dtype=torch.long)
    dist.send(rank, dst=dst)
    shape = torch.tensor(list(tensor.shape), dtype=torch.long)
    dist.send(shape, dst=dst)
    dist.send(tensor, dst=dst)


def recv_tensor(src: int, device: torch.device | None = None) -> torch.Tensor:
    rank = torch.zeros(1, dtype=torch.long)
    dist.recv(rank, src=src)
    shape = torch.zeros(int(rank.item()), dtype=torch.long)
    dist.recv(shape, src=src)
    out = torch.zeros([int(x) for x in shape.tolist()], dtype=torch.float32)
    dist.recv(out, src=src)
    if device is not None:
        out = out.to(device)
    return out


# ---------------------------------------------------------------- pipeline


class PipelineStage:
    """A single stage of a pipeline.

    ``stage_module`` is the ``nn.Module`` this rank owns. ``is_first`` and
    ``is_last`` determine whether this stage reads from the dataloader or
    computes the final loss, respectively.
    """

    def __init__(
        self,
        stage_module: nn.Module,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ):
        self.module = stage_module.to(device)
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.is_first = rank == 0
        self.is_last = rank == world_size - 1

    # The schedule is GPipe-style: all forwards, then all backwards.
    def step(
        self,
        micro_batches: list[tuple[torch.Tensor, torch.Tensor] | torch.Tensor] | None,
        loss_fn: nn.Module | None,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one optimiser step over a list of micro-batches.

        - On the first stage, each element of ``micro_batches`` must be
          ``(x, y)``; ``y`` is forwarded separately to the last stage.
        - On intermediate stages ``micro_batches`` is ignored.
        - On the last stage it is ignored too; labels arrive over the wire.

        Returns the mean loss across micro-batches on the last rank, or 0.0
        elsewhere.
        """
        optimizer.zero_grad(set_to_none=True)

        # Scratch storage for backward.
        inputs: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []

        ce_loss = loss_fn is not None and loss_fn.__class__.__name__ == "CrossEntropyLoss"

        if self.is_first:
            assert micro_batches is not None
            num_micro = len(micro_batches)
        else:
            count = torch.zeros(1, dtype=torch.long)
            dist.recv(count, src=self.rank - 1)
            num_micro = int(count.item())

        if not self.is_last:
            dist.send(torch.tensor([num_micro], dtype=torch.long), dst=self.rank + 1)

        # --- forward pass over all micro-batches -----------------------

        for mb_idx in range(num_micro):
            if self.is_first:
                assert micro_batches is not None
                x, y = micro_batches[mb_idx]  # type: ignore[misc]
                a = x.to(self.device)
                y = y.to(self.device)
            else:
                a = recv_tensor(self.rank - 1, device=self.device)
                a.requires_grad_(True)

            inputs.append(a)
            out = self.module(a)
            outputs.append(out)

            # Send activation downstream BEFORE sending labels so the order
            # of messages on the (rank, rank+1) edge is deterministic when
            # rank+1 is also the last rank (world_size == 2).
            if not self.is_last:
                send_tensor(out, dst=self.rank + 1)

            if self.is_first and self.world_size > 1:
                send_tensor(y.to(torch.float32), dst=self.world_size - 1)

            if self.is_last:
                if self.world_size == 1:
                    assert self.is_first
                    targets.append(y)
                else:
                    y_raw = recv_tensor(0, device=self.device)
                    targets.append(y_raw.to(torch.long) if ce_loss else y_raw)

        # --- backward pass in reverse ----------------------------------

        total_loss = torch.tensor(0.0, device=self.device)
        for mb_idx in reversed(range(num_micro)):
            out = outputs[mb_idx]
            a = inputs[mb_idx]
            if self.is_last:
                assert loss_fn is not None
                y = targets[mb_idx]
                loss = loss_fn(out, y) / num_micro
                total_loss = total_loss + loss.detach()
                loss.backward()
            else:
                grad_out = recv_tensor(self.rank + 1, device=self.device)
                out.backward(grad_out)

            if not self.is_first and a.grad is not None:
                send_tensor(a.grad.detach(), dst=self.rank - 1)

        optimizer.step()

        if self.is_last:
            return float(total_loss.item())
        return 0.0
