import torch
import os

class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, step, loss, rank):
        """Save training state"""
        if rank == 0:  # Only rank 0 saves to avoid conflicts
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'loss': loss
            }
            path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.pt")
            torch.save(checkpoint, path)
            print(f"Checkpoint saved at step {step}")
    
    def load_checkpoint(self, model, optimizer, step):
        """Load training state"""
        path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['step'], checkpoint['loss']
        return None, None