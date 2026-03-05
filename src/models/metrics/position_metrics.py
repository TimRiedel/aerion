from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PositionResult:
    """Aggregated per-feature (X, Y, Altitude) MAE across all trajectories in an epoch."""
    mae_x_mean: torch.Tensor
    mae_y_mean: torch.Tensor
    altitude_mae_mean: torch.Tensor
    mae_x_trajectories: torch.Tensor        # [N] per-trajectory X MAE
    mae_y_trajectories: torch.Tensor        # [N] per-trajectory Y MAE
    altitude_mae_trajectories: torch.Tensor  # [N] per-trajectory altitude MAE


class PositionMetrics:
    """
    Accumulates per-trajectory mean absolute error for each spatial feature (X, Y, Altitude).

    Each MAE is the mean of absolute errors over all valid (non-padded) waypoints in a
    trajectory, so every trajectory contributes equally regardless of its length.
    Only trajectories with at least one valid waypoint are included.
    """

    def __init__(self):
        self.mae_x_traj_list: list[torch.Tensor] = []
        self.mae_y_traj_list: list[torch.Tensor] = []
        self.altitude_mae_traj_list: list[torch.Tensor] = []

    def update(
        self,
        pred_pos_abs: torch.Tensor,
        target_pos_abs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """
        Compute and accumulate per-trajectory per-feature MAE for one batch.

        Args:
            pred_pos_abs: Predicted absolute positions [B, H, 3].
            target_pos_abs: Target absolute positions [B, H, 3].
            valid_mask: Boolean mask [B, H], True for valid (non-padded) waypoints.
        """
        has_valid = valid_mask.any(dim=1)  # [B]
        valid_waypoints_per_traj = valid_mask.sum(dim=1).clamp(min=1)  # [B]

        abs_err = (pred_pos_abs - target_pos_abs).abs()  # [B, H, 3]
        abs_err_masked = abs_err * valid_mask.unsqueeze(-1)  # [B, H, 3]

        mae_x_traj = abs_err_masked[:, :, 0].sum(dim=1) / valid_waypoints_per_traj        # [B]
        mae_y_traj = abs_err_masked[:, :, 1].sum(dim=1) / valid_waypoints_per_traj        # [B]
        altitude_mae_traj = abs_err_masked[:, :, 2].sum(dim=1) / valid_waypoints_per_traj  # [B]

        self.mae_x_traj_list.append(mae_x_traj[has_valid])
        self.mae_y_traj_list.append(mae_y_traj[has_valid])
        self.altitude_mae_traj_list.append(altitude_mae_traj[has_valid])

    def compute(self) -> PositionResult:
        """
        Aggregate per-feature MAE over all accumulated trajectories.

        Returns:
            PositionResult with means and per-trajectory tensors for X, Y, and Altitude MAE.
        """
        mae_x_traj = torch.cat(self.mae_x_traj_list)          # [N]
        mae_y_traj = torch.cat(self.mae_y_traj_list)          # [N]
        altitude_mae_traj = torch.cat(self.altitude_mae_traj_list)  # [N]

        n = mae_x_traj.numel()
        denom = max(n, 1)

        return PositionResult(
            mae_x_mean=mae_x_traj.sum() / denom,
            mae_y_mean=mae_y_traj.sum() / denom,
            altitude_mae_mean=altitude_mae_traj.sum() / denom,
            mae_x_trajectories=mae_x_traj,
            mae_y_trajectories=mae_y_traj,
            altitude_mae_trajectories=altitude_mae_traj,
        )

    def dataframe_columns(self) -> dict[str, np.ndarray]:
        """Return per-trajectory per-feature MAE values as numpy arrays for parquet export."""
        return {
            "mae_x": torch.cat(self.mae_x_traj_list).detach().cpu().float().numpy(),
            "mae_y": torch.cat(self.mae_y_traj_list).detach().cpu().float().numpy(),
            "altitude_mae": torch.cat(self.altitude_mae_traj_list).detach().cpu().float().numpy(),
        }
