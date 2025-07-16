import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from trainer.distributed import DistributedTrainer

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

def train_function(rank, world_size):
    print(f"Process {rank}: Starting simplified model parallel training")
    
    # Simple model splitting: each process gets different layers
    if rank == 0:
        # First half of model
        model_part = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        print(f"Process {rank}: owns first 3 layers")
    else:
        # Second half of model  
        model_part = nn.Sequential(
            nn.Linear(512, 512),  # Note: input matches output of process 0
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        print(f"Process {rank}: owns last 3 layers")
    
    optimizer = optim.SGD(model_part.parameters(), lr=0.01)
    
    # Simple training without complex communication for now
    x = torch.randn(32, 784 if rank == 0 else 512)
    
    for step in range(5):
        output = model_part(x)
        
        # Simple loss (each part trains independently for now)
        if rank == 0:
            # First process: minimize output norm
            loss = torch.mean(output ** 2)
        else:
            # Second process: random target
            target = torch.randint(0, 10, (32,))
            loss = nn.CrossEntropyLoss()(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Process {rank}, Step {step}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    trainer = DistributedTrainer(world_size=2)
    trainer.run(train_function)