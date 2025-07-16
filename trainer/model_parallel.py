import torch
import torch.nn as nn
import torch.distributed as dist

class ModelParallelManager:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
    
    def split_model(self, model, split_points):
        """
        Split model at specified layer indices
        split_points: list of layer indices where to split
        """
        layers = list(model.children())[0]  # Get the Sequential layers
        if isinstance(layers, nn.Sequential):
            layers = list(layers.children())
        
        total_layers = len(layers)
        print(f"Total layers to split: {total_layers}")
        
        # Calculate which layers this process should own
        layers_per_process = total_layers // self.world_size
        remainder = total_layers % self.world_size
        
        start_idx = self.rank * layers_per_process
        if self.rank < remainder:
            start_idx += self.rank
            end_idx = start_idx + layers_per_process + 1
        else:
            start_idx += remainder
            end_idx = start_idx + layers_per_process
        
        # Ensure we don't go out of bounds
        end_idx = min(end_idx, total_layers)
        
        print(f"Process {self.rank}: will own layers {start_idx} to {end_idx-1}")
        
        # Create sub-model for this process
        if start_idx < end_idx:
            sub_layers = layers[start_idx:end_idx]
            sub_model = nn.Sequential(*sub_layers)
        else:
            # Fallback: give at least one layer
            sub_model = nn.Sequential(layers[self.rank % total_layers])
            print(f"Process {self.rank}: got fallback layer {self.rank % total_layers}")
        
        return sub_model, start_idx, end_idx
    
    def forward_pass(self, x, is_first_process=False, is_last_process=False):
        """Handle forward pass with communication between processes"""
        if not is_first_process:
            # Receive input from previous process
            x = self._receive_tensor_from_prev()
        
        # Forward through local layers
        output = self.local_model(x)
        
        if not is_last_process:
            # Send output to next process
            self._send_tensor_to_next(output)
            return None  # Intermediate processes don't return final output
        
        return output  # Only last process returns final output
    
    def _send_tensor_to_next(self, tensor):
        """Send tensor to next process in pipeline"""
        if self.rank < self.world_size - 1:
            # Send tensor shape first
            shape = torch.tensor(tensor.shape, dtype=torch.long)
            dist.send(shape, dst=self.rank + 1)
            
            # Send the actual tensor
            dist.send(tensor.contiguous(), dst=self.rank + 1)

    def _receive_tensor_from_prev(self):
        """Receive tensor from previous process"""
        if self.rank > 0:
            # Receive tensor shape first
            shape = torch.zeros(4, dtype=torch.long)  # Assume max 4D tensors
            dist.recv(shape, src=self.rank - 1)
            
            # Only use non-zero dimensions
            actual_shape = []
            for dim in shape:
                if dim > 0:
                    actual_shape.append(int(dim.item()))
            
            # Receive actual tensor
            received_tensor = torch.zeros(actual_shape)
            dist.recv(received_tensor, src=self.rank - 1)
            return received_tensor