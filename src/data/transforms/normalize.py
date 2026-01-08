from typing import Dict, Any, Tuple, Union
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

class ZScoreDenormalize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def _validate_stats(self) -> None:
        if self.mean.dim() != 1 or self.std.dim() != 1:
            raise ValueError(f"mean and std must be 1D tensors of shape [num_features], got mean={tuple(self.mean.shape)} std={tuple(self.std.shape)}")
        if self.mean.numel() != self.std.numel():
            raise ValueError(f"mean and std must have the same number of features, got mean={self.mean.numel()} std={self.std.numel()}")

    def _validate_batched(self, t: torch.Tensor, batched: bool) -> None:
        expected_dim = 3 if batched else 2
        if t.dim() != expected_dim:
            kind = "batched [B, T, F]" if batched else "unbatched [T, F]"
            raise ValueError(f"Expected {kind} tensor, got shape {tuple(t.shape)} (dim={t.dim()})")

        if t.size(-1) != self.mean.numel():
            raise ValueError(
                f"Feature dimension mismatch: tensor has F={t.size(-1)} but mean/std have F={self.mean.numel()}"
            )

    def _denormalize_tensor(self, t: torch.Tensor, batched: bool = True) -> torch.Tensor:
        self._validate_batched(t, batched)
        mean = self.mean.to(device=t.device, dtype=t.dtype)
        std = self.std.to(device=t.device, dtype=t.dtype)
        return (t * std) + mean

    def __call__(
        self,
        tensor: torch.Tensor,
        batched: bool = True
    ) -> torch.Tensor:
        self._validate_stats()
        return self._denormalize_tensor(tensor, batched=batched)