import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from trainer.distributed import DistributedTrainer

def safe_send_tensor(tensor, dst):
    shape = list(tensor.shape)
    shape_tensor = torch.tensor(shape + [0] * (4 - len(shape)), dtype=torch.long)
    dist.send(shape_tensor, dst=dst)
    dist.send(tensor.contiguous(), dst=dst)

def safe_recv_tensor(src, device='cpu'):
    shape_tensor = torch.zeros(4, dtype=torch.long)
    dist.recv(shape_tensor, src=src)
    shape = [int(x) for x in shape_tensor if x > 0]
    tensor = torch.zeros(shape, device=device)
    dist.recv(tensor, src=src)
    return tensor

def train_function(rank, world_size):
    print(f"Process {rank}: Starting full pipeline training")
    
    if rank == 0:
        model_part = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        x = torch.randn(8, 784, requires_grad=True)
        
    else:
        model_part = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Linear(128, 10)
        )
        x = None
        target = torch.randint(0, 10, (8,))
    
    optimizer = optim.SGD(model_part.parameters(), lr=0.01)
    
    for step in range(3):
        print(f"\n--- Step {step} ---")
        optimizer.zero_grad()
        
        if rank == 0:
            # Forward pass
            intermediate = model_part(x)
            intermediate.retain_grad()  # Keep gradients for backward pass
            
            print(f"Process {rank}: forward complete, sending to next")
            safe_send_tensor(intermediate, dst=1)
            
            # Wait for gradients from next process
            grad_from_next = safe_recv_tensor(src=1)
            print(f"Process {rank}: received gradients from next process")
            
            # Backward pass with received gradients
            intermediate.backward(grad_from_next)
            optimizer.step()
            
            print(f"Process {rank}: backward complete")
            
        else:
            # Receive intermediate results
            intermediate = safe_recv_tensor(src=0)
            intermediate.requires_grad_(True)
            
            # Forward pass
            output = model_part(intermediate)
            loss = nn.CrossEntropyLoss()(output, target)
            
            print(f"Process {rank}: Loss = {loss.item():.4f}")
            
            # Backward pass
            loss.backward()
            
            # Send gradients back to previous process
            if intermediate.grad is not None:
                print(f"Process {rank}: sending gradients back")
                safe_send_tensor(intermediate.grad, dst=0)
            
            optimizer.step()
            print(f"Process {rank}: backward complete")

if __name__ == "__main__":
    trainer = DistributedTrainer(world_size=2)
    trainer.run(train_function)