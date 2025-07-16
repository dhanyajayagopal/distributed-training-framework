import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

class DistributedTrainer:
    def __init__(self, world_size=2):
        self.world_size = world_size
        
    def run(self, train_fn):
        mp.spawn(self._worker, args=(train_fn,), nprocs=self.world_size, join=True)
    
    def _worker(self, rank, train_fn):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(backend='gloo', rank=rank, world_size=self.world_size)
        
        print(f"Process {rank} started")
        train_fn(rank, self.world_size)
        
        dist.destroy_process_group()