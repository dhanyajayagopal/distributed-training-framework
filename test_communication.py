import torch
import torch.distributed as dist
from trainer.distributed import DistributedTrainer

def communication_test(rank, world_size):
    # Each process creates a different tensor
    tensor = torch.tensor([rank * 10.0])
    print(f"Process {rank}: Before allreduce = {tensor}")
    
    # Sum all tensors across processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Process {rank}: After allreduce = {tensor}")

if __name__ == "__main__":
    trainer = DistributedTrainer(world_size=2)
    trainer.run(communication_test)