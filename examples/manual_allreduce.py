"""Teaching example: manual gradient all-reduce (no DDP).

This is what the framework used to do in every script — it's kept here to
make the "DDP does this for you, faster" point concrete.

Run:
    python examples/manual_allreduce.py
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from trainer import DistEnv, DistributedTrainer


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):  # type: ignore[override]
        return self.fc2(torch.relu(self.fc1(x)))


def train_fn(env: DistEnv) -> None:
    torch.manual_seed(42)
    model = SimpleNet().to(env.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(100, 10, device=env.device)
    y = torch.randn(100, 1, device=env.device)
    chunk = 100 // env.world_size
    x = x[env.rank * chunk : (env.rank + 1) * chunk]
    y = y[env.rank * chunk : (env.rank + 1) * chunk]

    for step in range(5):
        loss = nn.MSELoss()(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= env.world_size
        optimizer.step()
        if env.is_main:
            print(f"step {step}: loss={loss.item():.4f}")


if __name__ == "__main__":
    DistributedTrainer(world_size=2).run(train_fn)
