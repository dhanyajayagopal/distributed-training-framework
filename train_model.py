import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from trainer.distributed import DistributedTrainer

# Simple neural network
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
    
    # Create model and move to device
    model = SimpleNet()
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create some fake training data
    x = torch.randn(100, 10)  # 100 samples, 10 features
    y = torch.randn(100, 1)   # 100 targets
    
    # Train for a few steps
    for step in range(5):
        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # HERE'S THE DISTRIBUTED PART:
        # Share gradients across all processes
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size  # Average the gradients
        
        optimizer.step()
        
        print(f"Process {rank}, Step {step}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    trainer = DistributedTrainer(world_size=2)
    trainer.run(train_function)