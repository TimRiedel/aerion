from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class FeatureSliceNormalizer:
    """
    Normalizes specific feature indices of input tensors.
    
    This allows composing multiple normalizers to operate on different feature subsets,
    making it easy to extend features without modifying normalization logic.

    Used for data normalization in the dataset.
    """
    
    def __init__(
        self, 
        path: Tuple[str, ...],
        indices: Union[List[int], range],
        mean: torch.Tensor, 
        std: torch.Tensor, 
        eps: float = 1e-6
    ):
        """
        Args:
            path: Path to the tensor in the sample (e.g. ("trajectory", "encoder_in"))
            indices: Feature indices to normalize. Can be:
                - List of integers: [0, 1, 2]
                - Tuple of integers: [min, max)
            mean: Mean for normalization (shape should match number of indices)
            std: Std for normalization (shape should match number of indices)
            eps: Small value to avoid division by zero
        """
        self.path = path
        self.mean = mean
        self.std = std
        self.eps = eps

        if isinstance(indices, tuple):
            self.indices = range(indices[0], indices[1])
        else:
            self.indices = list(indices)
            
    def _get_value_at_path(self, obj: Any, path: Tuple[str, ...]) -> Any:
        """Get value at path. Supports both dict (key) and object (attr) access."""
        for key in path:
            try:
                obj = getattr(obj, key)
            except AttributeError:
                obj = obj[key]
        return obj

    def _set_value_at_path(
        self, obj: Any, value: Any, path: Optional[Tuple[str, ...]] = None
    ) -> None:
        """Set value at path in place. Supports both dict and dataclass samples."""
        if path is None:
            path = self.path
        if not path:
            return
        if len(path) == 1:
            key = path[0]
            try:
                setattr(obj, key, value)
            except (AttributeError, TypeError):
                obj[key] = value
            return
        # Navigate to parent of target
        parent = obj
        for key in path[:-1]:
            try:
                parent = getattr(parent, key)
            except AttributeError:
                parent = parent[key]
        key = path[-1]
        try:
            setattr(parent, key, value)
        except (AttributeError, TypeError):
            parent[key] = value

    def __call__(self, sample: Any) -> Any:
        x = self._get_value_at_path(sample, self.path)
        x_norm = x.clone()
        x_norm[..., self.indices] = (x[..., self.indices] - self.mean) / (self.std + self.eps)
        self._set_value_at_path(sample, x_norm, self.path)
        return sample


class Normalizer(nn.Module):
    """
    Same as FeatureSliceNormalizer, but as a nn.Module for usage in on-demand normalization in models.
    User must ensure that x tensor has the same or a compatible shape as the mean and std tensors.
    Does not modify the sample dictionary, but returns the normalized tensor.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


class Denormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        # register_buffer ensures these aren't updated by the optimizer
        # but are moved to the correct device/dtype automatically
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std) + self.mean