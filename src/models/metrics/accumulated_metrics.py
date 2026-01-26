import torch
import numpy as np
from typing import Dict, Optional


class AccumulatedTrajectoryMetrics:
    """
    Accumulates and computes trajectory prediction metrics.
    
    Metrics computed:
    - ADE (Average Displacement Error): Average Euclidean distance across all waypoints
    - FDE (Final Displacement Error): Euclidean distance at the last valid waypoint
    - MDE (Max Displacement Error): Maximum Euclidean distance across all waypoints
    - ADE per horizon: ADE computed at each horizon step
    - MAE per horizon: MAE computed at each horizon step
    - RMSE per horizon: RMSE computed at each horizon step
    """
    
    def __init__(self, horizon_seq_len: int, device: torch.device):
        self.horizon_seq_len = horizon_seq_len
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all accumulators to zero."""
        H = self.horizon_seq_len
        F = 3  # Number of output features (x, y, altitude)
        
        # Per-horizon accumulators
        self.sum_abs_error = torch.zeros(H, F, device=self.device)
        self.sum_sq_error = torch.zeros(H, F, device=self.device)
        self.sum_dist_2d = torch.zeros(H, device=self.device)
        self.sum_dist_3d = torch.zeros(H, device=self.device)
        self.sum_fde_2d = torch.tensor(0.0, device=self.device)
        self.sum_fde_3d = torch.tensor(0.0, device=self.device)
        
        # Aggregate MDE accumulators (Max Displacement Error)
        self.sum_max_dist_2d = torch.tensor(0.0, device=self.device)
        self.sum_max_dist_3d = torch.tensor(0.0, device=self.device)
        
        # Count of valid waypoints and trajectories
        self.count_valid_waypoints = torch.zeros(H, device=self.device)  # Shape: [H]
        self.count_traj = torch.tensor(0.0, device=self.device)  # Shape: [1]
    
    def update(
        self,
        pred_abs: torch.Tensor,
        target_abs: torch.Tensor,
        target_pad_mask: torch.Tensor,
    ):
        """
        Accumulate metrics for a batch of predictions.
        
        Args:
            pred_abs: Model predictions (absolute positions) [batch_size, horizon_seq_len, 3]
            target_abs: Target values (absolute positions) [batch_size, horizon_seq_len, 3]
            target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
        """
        valid_mask = ~target_pad_mask  # [B, H]
        last_valid_index = (valid_mask.sum(dim=1) - 1).clamp(min=0)  # [B]
        batch_indices = torch.arange(pred_abs.size(0), device=pred_abs.device)  # [B]
        
        # 1. Feature-wise Errors on absolute positions [B, H, F]
        diff = pred_abs - target_abs
        abs_err = diff.abs()
        sq_err = diff ** 2
        
        # 2. Euclidean Distances [B, H]
        dist_2d = torch.norm(diff[:, :, :2], dim=2)  # Position distances (X, Y)
        dist_3d = torch.norm(diff[:, :, :3], dim=2)  # Position + altitude distances
        
        # 3. Masking
        # Zero out invalid entries for summation
        # Expand mask for features: [B, H] -> [B, H, 1]
        valid_mask_unsqueezed = valid_mask.unsqueeze(-1)
        abs_err_masked = abs_err * valid_mask_unsqueezed
        sq_err_masked = sq_err * valid_mask_unsqueezed
        dist_2d_masked = dist_2d * valid_mask
        dist_3d_masked = dist_3d * valid_mask
        fde_2d = dist_2d_masked[batch_indices, last_valid_index]
        fde_3d = dist_3d_masked[batch_indices, last_valid_index]
        
        # 4. Accumulate per horizon (Sum over batch dimension)
        self.sum_abs_error += abs_err_masked.sum(dim=0)
        self.sum_sq_error += sq_err_masked.sum(dim=0)
        self.sum_dist_2d += dist_2d_masked.sum(dim=0)
        self.sum_dist_3d += dist_3d_masked.sum(dim=0)
        self.sum_fde_2d += fde_2d.sum()
        self.sum_fde_3d += fde_3d.sum()
        
        # 5. MDE (Max Displacement Error) calculation
        # Use masked max: set padded positions to -inf so they're ignored in max()
        dist_2d_masked[~valid_mask] = -torch.inf
        dist_3d_masked[~valid_mask] = -torch.inf
        traj_max_dist_2d = dist_2d_masked.max(dim=1).values  # [B]
        traj_max_dist_3d = dist_3d_masked.max(dim=1).values  # [B]
        has_valid_points = valid_mask.any(dim=1)  # [B]
        self.sum_max_dist_2d += traj_max_dist_2d[has_valid_points].sum()
        self.sum_max_dist_3d += traj_max_dist_3d[has_valid_points].sum()
        
        self.count_valid_waypoints += valid_mask.sum(dim=0)
        self.count_traj += valid_mask.any(dim=1).sum()
    
    def compute(
        self,
        reduce_op: Optional[str] = None,
        strategy: Optional[object] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute final metrics from accumulators.
        
        Args:
            reduce_op: Reduction operation for distributed training (e.g., "sum")
            strategy: PyTorch Lightning strategy for distributed reduction
        
        Returns:
            Dictionary containing:
            - ade_2d_scalar: Scalar ADE in 2D
            - ade_3d_scalar: Scalar ADE in 3D
            - fde_2d_scalar: Scalar FDE in 2D
            - fde_3d_scalar: Scalar FDE in 3D
            - mde_2d_scalar: Scalar MDE in 2D
            - mde_3d_scalar: Scalar MDE in 3D
            - ade_2d_per_horizon: ADE per horizon step [horizon_seq_len]
            - ade_3d_per_horizon: ADE per horizon step [horizon_seq_len]
            - mae_per_horizon: MAE per horizon step [horizon_seq_len, 3] (X, Y, Altitude)
            - rmse_per_horizon: RMSE per horizon step [horizon_seq_len, 3] (X, Y, Altitude)
        """
        # Reduce tensors for distributed training if needed
        tensors_to_reduce = [
            self.sum_abs_error,
            self.sum_sq_error,
            self.sum_dist_2d,
            self.sum_dist_3d,
            self.sum_fde_2d,
            self.sum_fde_3d,
            self.sum_max_dist_2d,
            self.sum_max_dist_3d,
            self.count_valid_waypoints,
            self.count_traj,
        ]
        
        if reduce_op is not None and strategy is not None:
            for tensor in tensors_to_reduce:
                strategy.reduce(tensor, reduce_op=reduce_op)
        
        valid_points_per_horizon = self.count_valid_waypoints.clamp(min=1.0)  # Shape: [H]
        total_valid_points = self.count_valid_waypoints.sum().clamp(min=1.0)  # Scalar
        total_trajectories = self.count_traj.clamp(min=1.0)                   # Scalar
        
        # 1. ADE (Average of Euclidean Distance per valid waypoint)
        ade_2d_per_horizon = self.sum_dist_2d / valid_points_per_horizon  # Shape: [H]
        ade_3d_per_horizon = self.sum_dist_3d / valid_points_per_horizon  # Shape: [H]
        ade_2d_scalar = self.sum_dist_2d.sum() / total_valid_points       # Scalar
        ade_3d_scalar = self.sum_dist_3d.sum() / total_valid_points       # Scalar
        
        # 2. MDE (Average Max Displacement Error)
        mde_2d_scalar = self.sum_max_dist_2d / total_trajectories  # Scalar
        mde_3d_scalar = self.sum_max_dist_3d / total_trajectories  # Scalar
        
        # 3. FDE (Average Final Displacement Error)
        fde_2d_scalar = self.sum_fde_2d / total_trajectories  # Scalar
        fde_3d_scalar = self.sum_fde_3d / total_trajectories  # Scalar
        
        # 4. MAE (Mean Absolute Error) and RMSE (Root Mean Square Error) per horizon
        valid_points_per_horizon_unsqueezed = valid_points_per_horizon.unsqueeze(1)  # Shape: [H, 1]
        mae_per_horizon = self.sum_abs_error / valid_points_per_horizon_unsqueezed  # Shape: [H, F]
        rmse_per_horizon = torch.sqrt(self.sum_sq_error / valid_points_per_horizon_unsqueezed)  # Shape: [H, F]
        
        return {
            "ade_2d_scalar": ade_2d_scalar,
            "ade_3d_scalar": ade_3d_scalar,
            "fde_2d_scalar": fde_2d_scalar,
            "fde_3d_scalar": fde_3d_scalar,
            "mde_2d_scalar": mde_2d_scalar,
            "mde_3d_scalar": mde_3d_scalar,
            "ade_2d_per_horizon": ade_2d_per_horizon,
            "ade_3d_per_horizon": ade_3d_per_horizon,
            "mae_per_horizon": mae_per_horizon,
            "rmse_per_horizon": rmse_per_horizon,
        }
