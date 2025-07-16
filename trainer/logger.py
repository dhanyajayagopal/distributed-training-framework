import time
import json
import os

class TrainingLogger:
    def __init__(self, log_dir="logs", rank=0):
        self.log_dir = log_dir
        self.rank = rank
        self.start_time = time.time()
        self.metrics = []
        
        if rank == 0:  # Only rank 0 logs to avoid conflicts
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, "training_log.jsonl")
    
    def log_step(self, step, loss, learning_rate=None):
        if self.rank == 0:
            elapsed = time.time() - self.start_time
            log_entry = {
                "step": step,
                "loss": loss,
                "elapsed_time": elapsed,
                "timestamp": time.time()
            }
            if learning_rate:
                log_entry["learning_rate"] = learning_rate
            
            # Save to file
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            print(f"Step {step}: Loss={loss:.4f}, Time={elapsed:.1f}s")
    
    def log_summary(self):
        if self.rank == 0:
            total_time = time.time() - self.start_time
            print(f"Training completed in {total_time:.1f} seconds")