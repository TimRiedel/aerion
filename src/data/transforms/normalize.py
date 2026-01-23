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
        for key in ["input_traj", "target_traj", "dec_in_traj"]:
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


class DeltaAwareNormalize:
    """
    Normalizes samples with separate stats for positions and deltas.
    
    - input_traj: 6 features [pos_x, pos_y, pos_alt, delta_x, delta_y, delta_alt]
      - First 3 features normalized with pos_mean/pos_std
      - Last 3 features normalized with delta_mean/delta_std
    - target_traj, dec_in_traj: 3 features [delta_x, delta_y, delta_alt]
      - Normalized with delta_mean/delta_std
    """
    
    def __init__(
        self, 
        pos_mean: torch.Tensor, 
        pos_std: torch.Tensor, 
        delta_mean: torch.Tensor, 
        delta_std: torch.Tensor,
        eps: float = 1e-6
    ):
        """
        Args:
            pos_mean: Mean for absolute positions [3]
            pos_std: Std for absolute positions [3]
            delta_mean: Mean for deltas [3]
            delta_std: Std for deltas [3]
            eps: Small value to avoid division by zero
        """
        self.pos_mean = pos_mean
        self.pos_std = pos_std
        self.delta_mean = delta_mean
        self.delta_std = delta_std
        self.eps = eps

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["input_traj"] = self.normalize_input_traj(sample["input_traj"])
        sample["target_traj"] = self.normalize_target_traj(sample["target_traj"])
        sample["dec_in_traj"] = self.normalize_dec_in_traj(sample["dec_in_traj"])

        if "flightinfo" in sample:
            sample["flightinfo"] = self.normalize_flightinfo(sample["flightinfo"])
        
        return sample

    def normalize_input_traj(self, input_traj: torch.Tensor) -> torch.Tensor:
        """ Input trajectory [T_in, 6]: 
            - Positions (x, y, alt) - normalized
            - Deltas (delta_x, delta_y, delta_alt) - normalized
        """
        input_traj_pos = input_traj[:, :3]
        input_traj_delta = input_traj[:, 3:6]
        input_traj_pos_norm = (input_traj_pos - self.pos_mean) / (self.pos_std + self.eps)
        input_traj_delta_norm = (input_traj_delta - self.delta_mean) / (self.delta_std + self.eps)
        return torch.cat([input_traj_pos_norm, input_traj_delta_norm], dim=1)

    def normalize_target_traj(self, target_traj: torch.Tensor) -> torch.Tensor:
        """ Target trajectory with only deltas [H, 3]"""
        return (target_traj - self.delta_mean) / (self.delta_std + self.eps)

    def normalize_dec_in_traj(self, dec_in_traj: torch.Tensor) -> torch.Tensor:
        """ Decoder input trajectory with only deltas [H, 3]"""
        return (dec_in_traj - self.delta_mean) / (self.delta_std + self.eps)

    def normalize_flightinfo(self, flightinfo: torch.Tensor) -> torch.Tensor:
        """ Flightinfo [num_features] (typically [4]): 
            - Runway position (x, y) at indices [0, 1] - normalized
            - Runway bearing (sin, cos) at indices [2, 3] - not normalized
        """
        x_y_mean = self.pos_mean[:2]
        x_y_std = self.pos_std[:2]

        flightinfo_rwy_pos = flightinfo[:2]
        remaining_features = flightinfo[2:]
        flightinfo_rwy_pos_norm = (flightinfo_rwy_pos - x_y_mean) / (x_y_std + self.eps)
        return torch.cat([flightinfo_rwy_pos_norm, remaining_features])
