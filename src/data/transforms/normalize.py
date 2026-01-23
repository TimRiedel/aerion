from typing import Dict, Any, Tuple, Union
import torch
import torch.nn as nn


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


class Denormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        # register_buffer ensures these aren't updated by the optimizer
        # but are moved to the correct device/dtype automatically
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std) + self.mean