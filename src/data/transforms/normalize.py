from typing import Any, Dict, Union, List
import torch
import torch.nn as nn


class FeatureSliceNormalizer:
    """
    Normalizes specific feature indices of input tensors.
    
    This allows composing multiple normalizers to operate on different feature subsets,
    making it easy to extend features without modifying normalization logic.
    """
    
    def __init__(
        self, 
        name: str,
        indices: Union[List[int], range, slice],
        mean: torch.Tensor, 
        std: torch.Tensor, 
        eps: float = 1e-6
    ):
        """
        Args:
            name: Name of tensor in sample dictionary
            indices: Feature indices to normalize. Can be:
                - List of integers: [0, 1, 2]
                - Tuple of integers: [min, max)
            mean: Mean for normalization (shape should match number of indices)
            std: Std for normalization (shape should match number of indices)
            eps: Small value to avoid division by zero
        """
        self.name = name
        self.mean = mean
        self.std = std
        self.eps = eps

        if isinstance(indices, tuple):
            self.indices = range(indices[0], indices[1])
        else:
            self.indices = list(indices)
            
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        x = sample[self.name]
        x_norm = x.clone()
        x_norm[..., self.indices] = (x[..., self.indices] - self.mean) / (self.std + self.eps)
        sample[self.name] = x_norm
        return sample


class Denormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        # register_buffer ensures these aren't updated by the optimizer
        # but are moved to the correct device/dtype automatically
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std) + self.mean