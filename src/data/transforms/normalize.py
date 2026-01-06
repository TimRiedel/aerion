from typing import Dict, Any
import torch

class ZScoreNormalize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        """
        mean, std: tensors of shape (num_features,)
        """
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key in ["x", "y", "dec_in"]:
            sample[key] = (sample[key] - self.mean) / (self.std + self.eps)
        return sample