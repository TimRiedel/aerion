from typing import Dict, Any
import torch


class DecoderInputNoise:
    """
    Adds Gaussian noise to decoder input to simulate prediction errors during training.
    This helps the model be more robust to its own prediction errors during autoregressive generation.
    
    Noise is added per axis (x, y, altitude) separately, and the first point (current position) is kept unchanged.
    """
    
    def __init__(self, noise_std: torch.Tensor):
        if noise_std.dim() != 1:
            raise ValueError(f"noise_std must be a 1D tensor of shape [num_features], got {noise_std.shape}")
        self.noise_std = noise_std
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        dec_in = sample["dec_in"]
        
        noise = torch.randn_like(dec_in[1:]) * self.noise_std.to(dec_in.device) # Shape: [batch_size, seq_len-1, num_features]
        dec_in = torch.cat([
            dec_in[0:1],  # Keep first point unchanged
            dec_in[1:] + noise  # Add noise to rest
        ], dim=0)
        
        sample["dec_in"] = dec_in
        return sample
