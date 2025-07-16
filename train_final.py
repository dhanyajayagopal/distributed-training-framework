import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from trainer.distributed import DistributedTrainer
from trainer.checkpoint import CheckpointManager
from trainer.logger import TrainingLogger

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
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    checkpoint_manager = CheckpointManager()
    logger = TrainingLogger(rank=rank)
    
    # Dataset
    torch.manual_seed(42)
    full_x = torch.randn(200, 10)
    full_y = torch.randn(200, 1)
    
    chunk_size = 200 // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    x = full_x[start_idx:end_idx]
    y = full_y[start_idx:end_idx]
    
    for step in range(15):
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Distributed gradient sync
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
        
        optimizer.step()
        
        # Log and checkpoint
        logger.log_step(step, loss.item(), optimizer.param_groups[0]['lr'])
        
        if (step + 1) % 5 == 0:
            checkpoint_manager.save_checkpoint(model, optimizer, step, loss.item(), rank)
    
    logger.log_summary()

if __name__ == "__main__":
    trainer = DistributedTrainer(world_size=2)
    trainer.run(train_function)