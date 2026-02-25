import math

import torch
from torch import nn
from torch.nn import functional as F


class TurnRateLoss(nn.Module):
    """
    Penalizes predicted turn rates above the physically admissible maximum.

    Max turn rate is min(physics_from_bank, 3°/s):
    - Physics: turn_rate = (g * tan(bank_angle)) / V * (180/π), with bank_angle = 25°.
    - Capped at 3°/s (passenger comfort / operational limit).

    Only violations are penalized (ReLU); turns at or below the limit contribute zero.
    """

    def __init__(
        self,
        dt_seconds: float = 30.0,
        max_turn_rate_cap_deg_per_s: float = 3.0,
        bank_angle_deg: float = 25.0,
        speed_eps: float = 1e-3,
    ):
        super().__init__()
        self.dt = dt_seconds
        self.max_turn_rate_cap = max_turn_rate_cap_deg_per_s
        self.bank_angle_rad = math.radians(bank_angle_deg)
        self.speed_eps = speed_eps
        self.g = 9.81
        # deg/s per (m/s) when using 25° bank: (g * tan(25°)) * (180/π)
        self._physics_scale = (self.g * math.tan(self.bank_angle_rad)) * (180.0 / math.pi)

    def forward(
        self,
        pred_deltas_abs: torch.Tensor,
        target_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_deltas_abs: Predicted deltas in meters [B, H, 3] (dx, dy, dalt).
            target_pad_mask: Padding mask [B, H] (True = padded).

        Returns:
            Scalar loss: mean ReLU(actual_turn_rate - max_turn_rate) over valid steps.
        """
        dx = pred_deltas_abs[..., 0]  # [B, H]
        dy = pred_deltas_abs[..., 1]  # [B, H]

        # Groundspeed per segment (m/s), avoid division by zero
        segment_speed = torch.sqrt(dx * dx + dy * dy) / self.dt + self.speed_eps  # [B, H]

        # Heading per segment (radians)
        heading = torch.atan2(dy, dx)  # [B, H]

        # Heading change between consecutive segments (wrapped to [-π, π]
        delta_heading = torch.diff(heading, dim=1)  # [B, H-1]
        delta_heading = torch.atan2(
            torch.sin(delta_heading), torch.cos(delta_heading)
        )

        # Actual turn rate (deg/s) over one step
        actual_turn_rate = torch.abs(delta_heading) * (180.0 / math.pi) / self.dt  # [B, H-1]

        # Physics-based max turn rate: ψ̇_max = (g·tan(φ))/V, so lower V → higher allowed rate.
        # Use the minimum of the two segment speeds → higher max_turn_rate (less strict), avoiding over-penalty when one segment is fast.
        speed_before = segment_speed[:, :-1]  # [B, H-1]
        speed_after = segment_speed[:, 1:]    # [B, H-1]
        speed_at_turn = torch.minimum(speed_before, speed_after)
        physics_turn_rate = self._physics_scale / speed_at_turn  # [B, H-1]
        max_turn_rate = torch.clamp(physics_turn_rate, max=self.max_turn_rate_cap)  # [B, H-1]

        violation = F.relu(actual_turn_rate - max_turn_rate)  # [B, H-1]

        # Mask out padded waypoints: valid turn at i if both waypoint i and i+1 are valid
        valid = (~target_pad_mask[:, :-1]) & (~target_pad_mask[:, 1:])  # [B, H-1]
        if not valid.any():
            return torch.tensor(0.0, device=pred_deltas_abs.device)

        return violation.masked_fill(~valid, 0.0).sum() / valid.float().sum().clamp(min=1.0)
