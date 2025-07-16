import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from trainer.distributed import DistributedTrainer
from trainer.checkpoint import CheckpointManager

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_function(rank, world_size):
    print(f"Process {rank}: Starting training")
    
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    checkpoint_manager = CheckpointManager()
    
    # Try to resume from step 5
    resume_step, resume_loss = checkpoint_manager.load_checkpoint(model, optimizer, 5)
    if resume_step is not None:
        print(f"Process {rank}: Resumed from step {resume_step}, loss was {resume_loss:.4f}")
        start_step = resume_step + 1
    else:
        print(f"Process {rank}: No checkpoint found, starting fresh")
        start_step = 0
    
    # Continue training from where we left off
    torch.manual_seed(42)
    full_x = torch.randn(200, 10)
    full_y = torch.randn(200, 1)
    
    chunk_size = 200 // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    x = full_x[start_idx:end_idx]
    y = full_y[start_idx:end_idx]
    
    for step in range(start_step, start_step + 5):
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
        
        optimizer.step()
        print(f"Process {rank}, Step {step}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    trainer = DistributedTrainer(world_size=2)
    trainer.run(train_function)