import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from trainer.distributed import DistributedTrainer

def safe_send_tensor(tensor, dst):
    """Safely send a tensor to another process"""
    # Send shape first as a list
    shape = list(tensor.shape)
    shape_tensor = torch.tensor(shape + [0] * (4 - len(shape)), dtype=torch.long)  # Pad to size 4
    dist.send(shape_tensor, dst=dst)
    
    # Send actual tensor
    dist.send(tensor.contiguous(), dst=dst)

def safe_recv_tensor(src, device='cpu'):
    """Safely receive a tensor from another process"""
    # Receive shape
    shape_tensor = torch.zeros(4, dtype=torch.long)
    dist.recv(shape_tensor, src=src)
    
    # Reconstruct actual shape (remove padding zeros)
    shape = [int(x) for x in shape_tensor if x > 0]
    
    # Receive tensor
    tensor = torch.zeros(shape, device=device)
    dist.recv(tensor, src=src)
    return tensor

def train_function(rank, world_size):
    print(f"Process {rank}: Starting pipeline parallel training")
    
    if rank == 0:
        # First stage: input processing
        model_part = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        print(f"Process {rank}: owns input layers (784→256)")
        
        # Create input data
        x = torch.randn(8, 784)  # Smaller batch for easier debugging
        target = torch.randint(0, 10, (8,))
        
    else:
        # Second stage: output processing
        model_part = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Linear(128, 10)
        )
        print(f"Process {rank}: owns output layers (256→10)")
        x = None
        target = torch.randint(0, 10, (8,))
    
    optimizer = optim.SGD(model_part.parameters(), lr=0.01)
    
    for step in range(3):
        print(f"\n--- Step {step} ---")
        
        if rank == 0:
            # Forward pass on first part
            intermediate = model_part(x)
            print(f"Process {rank}: intermediate shape = {intermediate.shape}")
            
            # Send to next process
            safe_send_tensor(intermediate, dst=1)
            print(f"Process {rank}: sent intermediate to process 1")
            
        else:
            # Receive from previous process
            intermediate = safe_recv_tensor(src=0)
            print(f"Process {rank}: received shape = {intermediate.shape}")
            
            # Forward pass on second part
            output = model_part(intermediate)
            loss = nn.CrossEntropyLoss()(output, target)
            
            print(f"Process {rank}: Loss = {loss.item():.4f}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    trainer = DistributedTrainer(world_size=2)
    trainer.run(train_function)