from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DisplacementResult:
    """Aggregated 2D displacement metrics across all trajectories in an epoch."""
    ade_mean: torch.Tensor
    fde_mean: torch.Tensor
    mde_mean: torch.Tensor
    ade_trajectories: torch.Tensor  # [N] per-trajectory ADE values
    fde_trajectories: torch.Tensor  # [N] per-trajectory FDE values
    mde_trajectories: torch.Tensor  # [N] per-trajectory MDE values


class DisplacementMetrics:
    """
    Accumulates per-trajectory 2D displacement metrics (ADE, FDE, MDE).

    Metrics are computed per trajectory on each update and aggregated in compute().
    Only trajectories with at least one valid (non-padded) waypoint are included.
    """

    def __init__(self):
        self.ade_traj_list: list[torch.Tensor] = []
        self.fde_traj_list: list[torch.Tensor] = []
        self.mde_traj_list: list[torch.Tensor] = []

    def update(
        self,
        pred_pos_abs: torch.Tensor,
        target_pos_abs: torch.Tensor,
        valid_mask: torch.Tensor,
        pred_valid_len: torch.Tensor,
        target_valid_len: torch.Tensor,
    ) -> None:
        """
        Compute and accumulate per-trajectory 2D displacement metrics for one batch.

        Args:
            pred_pos_abs: Predicted absolute positions [B, H, 3].
            target_pos_abs: Target absolute positions [B, H, 3].
            valid_mask: Boolean mask [B, H], True for valid positions in eval window.
            pred_valid_len: Number of valid prediction steps per sample [B].
            target_valid_len: Number of valid target steps per sample [B].
        """
        has_valid = valid_mask.any(dim=1)  # [B]
        valid_waypoints_per_traj = valid_mask.sum(dim=1).clamp(min=1)  # [B]
        batch_indices = torch.arange(pred_pos_abs.size(0), device=pred_pos_abs.device)


        # FDE: distance from last valid predicted position to last valid target position
        last_pred_idx = (pred_valid_len - 1).clamp(min=0)  # [B]
        last_target_idx = (target_valid_len - 1).clamp(min=0)  # [B]
        last_pred_pos = pred_pos_abs[batch_indices, last_pred_idx, :2]  # [B, 2]
        last_target_pos = target_pos_abs[batch_indices, last_target_idx, :2]  # [B, 2]
        fde_traj = torch.norm(last_pred_pos - last_target_pos, dim=-1)  # [B]

        # ADE: average distance over valid waypoints
        diff = pred_pos_abs - target_pos_abs  # [B, H, 3]
        dist_2d = torch.norm(diff[:, :, :2], dim=2)  # [B, H]
        dist_2d_masked = dist_2d * valid_mask  # [B, H]
        ade_traj = dist_2d_masked.sum(dim=1) / valid_waypoints_per_traj  # [B]

        # MDE: max distance over valid waypoints
        # Clone before setting padded positions to -inf so dist_2d_masked stays intact
        dist_2d_for_max = dist_2d_masked.clone()
        dist_2d_for_max[~valid_mask] = -torch.inf
        mde_traj = dist_2d_for_max.max(dim=1).values  # [B]

        self.ade_traj_list.append(ade_traj[has_valid])
        self.fde_traj_list.append(fde_traj[has_valid])
        self.mde_traj_list.append(mde_traj[has_valid])

    def compute(self) -> DisplacementResult:
        """
        Aggregate displacement metrics over all accumulated trajectories.

        Returns:
            DisplacementResult with means over all trajectories and concatenated per-trajectory tensors.
        """
        ade_traj = torch.cat(self.ade_traj_list)  # [N]
        fde_traj = torch.cat(self.fde_traj_list)  # [N]
        mde_traj = torch.cat(self.mde_traj_list)  # [N]

        n = ade_traj.numel()
        denom = max(n, 1)

        return DisplacementResult(
            ade_mean=ade_traj.sum() / denom,
            fde_mean=fde_traj.sum() / denom,
            mde_mean=mde_traj.sum() / denom,
            ade_trajectories=ade_traj,
            fde_trajectories=fde_traj,
            mde_trajectories=mde_traj,
        )

    def dataframe_columns(self) -> dict[str, np.ndarray]:
        """Return per-trajectory displacement values as numpy arrays for parquet export."""
        return {
            "ade": torch.cat(self.ade_traj_list).detach().cpu().float().numpy(),
            "fde": torch.cat(self.fde_traj_list).detach().cpu().float().numpy(),
            "mde": torch.cat(self.mde_traj_list).detach().cpu().float().numpy(),
        }
