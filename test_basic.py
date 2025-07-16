import torch
from trainer.distributed import DistributedTrainer

def simple_train_fn(rank, world_size):
    print(f"Hello from process {rank} out of {world_size}")
    
    # Simple tensor operation to test distributed
    x = torch.tensor([rank * 2.0])
    print(f"Process {rank}: tensor = {x}")

if __name__ == "__main__":
    trainer = DistributedTrainer(world_size=2)
    trainer.run(simple_train_fn)