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
    
    # Create dataset
    torch.manual_seed(42)
    full_x = torch.randn(200, 10)
    full_y = torch.randn(200, 1)
    
    chunk_size = 200 // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    x = full_x[start_idx:end_idx]
    y = full_y[start_idx:end_idx]
    
    print(f"Process {rank}: Training on samples {start_idx} to {end_idx-1}")
    
    for step in range(10):  # More steps to see checkpointing
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Share gradients
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
        
        optimizer.step()
        
        print(f"Process {rank}, Step {step}: Loss = {loss.item():.4f}")
        
        # Save checkpoint every 3 steps
        if (step + 1) % 3 == 0:
            checkpoint_manager.save_checkpoint(model, optimizer, step, loss.item(), rank)

if __name__ == "__main__":
    trainer = DistributedTrainer(world_size=2)
    trainer.run(train_function)