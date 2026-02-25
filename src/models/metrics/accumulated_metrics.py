from typing import Dict, Optional

import torch


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
    - RTDE (Remaining Track Distance Error): Difference between predicted and target RTD
    - Relative RTDE: RTDE divided by actual RTD, expressed as percentage
    - Relative RTDE std: Standard deviation of relative RTDE across trajectories
    - MSE Altitude: Mean squared error for altitude (meters²) over all valid waypoints
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
        self.sum_ade_2d = torch.zeros(H, device=self.device)
        self.sum_ade_3d = torch.zeros(H, device=self.device)
        self.sum_fde_2d = torch.tensor(0.0, device=self.device)
        self.sum_fde_3d = torch.tensor(0.0, device=self.device)
        
        # Aggregate MDE accumulators (Max Displacement Error)
        self.sum_mde_2d = torch.tensor(0.0, device=self.device)
        self.sum_mde_3d = torch.tensor(0.0, device=self.device)
        
        # RTDE accumulators (per-trajectory metrics)
        self.sum_rtde = torch.tensor(0.0, device=self.device)  # Sum of RTDE across all trajectories
        self.sum_relative_rtde = torch.tensor(0.0, device=self.device)  # Sum of relative RTDE (%)
        
        # Count of valid waypoints and trajectories
        self.count_valid_waypoints = torch.zeros(H, device=self.device)  # Shape: [H]
        self.count_traj = torch.tensor(0.0, device=self.device)  # Shape: [1]
        
        # Per-trajectory ADE and FDE lists for histogram generation
        self.traj_ade_2d_list = []
        self.traj_ade_3d_list = []
        self.traj_fde_2d_list = []
        self.traj_fde_3d_list = []
        
        # Per-trajectory RTDE lists for histogram and scatter plots
        self.traj_rtde_list = []  # RTDE per trajectory
        self.traj_rtde_relative_list = []  # Relative RTDE per trajectory
        self.traj_rtd_pred_list = []  # Predicted RTD per trajectory (for scatter plot)
        self.traj_rtd_target_list = []  # Target RTD per trajectory (for scatter plot)
    
    def update(
        self,
        pred_pos_abs: torch.Tensor,
        target_pos_abs: torch.Tensor,
        target_pad_mask: torch.Tensor,
        pred_rtd: torch.Tensor,
        target_rtd: torch.Tensor,
    ):
        """
        Accumulate metrics for a batch of predictions.

        Args:
            pred_pos_abs: Model predictions (absolute positions) [batch_size, horizon_seq_len, 3]
            target_pos_abs: Target values (absolute positions) [batch_size, horizon_seq_len, 3]
            target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
            pred_rtd: Predicted RTD for the full trajectory [batch_size]
            target_rtd: Actual RTD for the full trajectory [batch_size]
        """
        valid_mask = ~target_pad_mask  # [B, H]
        has_traj_valid_points = valid_mask.any(dim=1)  # [B]
        valid_waypoints_per_traj = valid_mask.sum(dim=1).clamp(min=1)  # [B]
        last_valid_index = (valid_mask.sum(dim=1) - 1).clamp(min=0)  # [B]
        batch_indices = torch.arange(pred_pos_abs.size(0), device=pred_pos_abs.device)  # [B]
        
        # 1. Feature-wise Errors on absolute positions [B, H, F]
        diff = pred_pos_abs - target_pos_abs
        abs_err = diff.abs()
        sq_err = diff ** 2
        
        # 2. Euclidean Distances [B, H]
        ade_2d = torch.norm(diff[:, :, :2], dim=2)  # Position distances (X, Y)
        ade_3d = torch.norm(diff[:, :, :3], dim=2)  # Position + altitude distances
        
        # 3. Masking
        # Zero out invalid entries for summation
        # Expand mask for features: [B, H] -> [B, H, 1]
        valid_mask_unsqueezed = valid_mask.unsqueeze(-1)
        abs_err_masked = abs_err * valid_mask_unsqueezed
        sq_err_masked = sq_err * valid_mask_unsqueezed
        ade_2d_masked = ade_2d * valid_mask
        ade_3d_masked = ade_3d * valid_mask
        fde_2d = ade_2d_masked[batch_indices, last_valid_index]
        fde_3d = ade_3d_masked[batch_indices, last_valid_index]
        
        # 4. Accumulate per horizon (Sum over batch dimension)
        self.sum_abs_error += abs_err_masked.sum(dim=0)
        self.sum_sq_error += sq_err_masked.sum(dim=0)
        self.sum_ade_2d += ade_2d_masked.sum(dim=0)
        self.sum_ade_3d += ade_3d_masked.sum(dim=0)
        self.sum_fde_2d += fde_2d.sum()
        self.sum_fde_3d += fde_3d.sum()

        # 5. Per-trajectory ADE calculation for histograms
        # Compute ADE per trajectory: average distance across all valid waypoints
        traj_ade_2d = ade_2d_masked.sum(dim=1) / valid_waypoints_per_traj  # [B]
        traj_ade_3d = ade_3d_masked.sum(dim=1) / valid_waypoints_per_traj  # [B]
        
        self.traj_ade_2d_list.append(traj_ade_2d[has_traj_valid_points])
        self.traj_ade_3d_list.append(traj_ade_3d[has_traj_valid_points])
        self.traj_fde_2d_list.append(fde_2d[has_traj_valid_points])
        self.traj_fde_3d_list.append(fde_3d[has_traj_valid_points])
        
        # 6. MDE (Max Displacement Error) calculation
        # Use masked max: set padded positions to -inf so they're ignored in max()
        ade_2d_masked[~valid_mask] = -torch.inf
        ade_3d_masked[~valid_mask] = -torch.inf
        traj_mde_2d = ade_2d_masked.max(dim=1).values  # [B]
        traj_mde_3d = ade_3d_masked.max(dim=1).values  # [B]
        self.sum_mde_2d += traj_mde_2d[has_traj_valid_points].sum()
        self.sum_mde_3d += traj_mde_3d[has_traj_valid_points].sum()
        
        # 7. RTDE (Remaining Track Distance Error) calculation - per trajectory
        # Positive: overestimate (predicted more distance remaining than actual)
        # Negative: underestimate (predicted less distance remaining than actual)
        traj_rtde = pred_rtd - target_rtd  # [B]
        # Relative RTDE: RTDE / actual_rtd * 100 (percentage)
        traj_rtde_relative = traj_rtde / target_rtd * 100.0
        
        # Accumulate RTDE metrics (only for trajectories with valid points)
        self.sum_rtde += traj_rtde[has_traj_valid_points].sum()
        self.sum_relative_rtde += traj_rtde_relative[has_traj_valid_points].sum()
        
        # Store per-trajectory RTDE values for histograms and scatter plots
        self.traj_rtde_list.append(traj_rtde[has_traj_valid_points])
        self.traj_rtde_relative_list.append(traj_rtde_relative[has_traj_valid_points])
        self.traj_rtd_pred_list.append(pred_rtd[has_traj_valid_points])
        self.traj_rtd_target_list.append(target_rtd[has_traj_valid_points])
        
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
            - traj_ade_2d_values: List of per-trajectory ADE 2D values for histograms
            - traj_ade_3d_values: List of per-trajectory ADE 3D values for histograms
            - traj_fde_2d_values: List of per-trajectory FDE 2D values for histograms
            - traj_fde_3d_values: List of per-trajectory FDE 3D values for histograms
            - rtde_scalar: Scalar average RTDE (can be positive or negative)
            - rtde_relative_scalar: Scalar average relative RTDE (positive or negative %)
            - rtde_relative_std: Standard deviation of relative RTDE across trajectories (%)
            - altitude_mse_scalar: MSE for altitude over all valid waypoints (meters²)
            - traj_rtde_values: Per-trajectory RTDE values for histograms
            - traj_rtde_relative_values: Per-trajectory relative RTDE values (%)
            - traj_rtd_pred_values: Per-trajectory predicted RTD values for scatter plots
            - traj_rtd_target_values: Per-trajectory target RTD values for scatter plots
        """
        # Reduce tensors for distributed training if needed
        tensors_to_reduce = [
            self.sum_abs_error,
            self.sum_sq_error,
            self.sum_ade_2d,
            self.sum_ade_3d,
            self.sum_fde_2d,
            self.sum_fde_3d,
            self.sum_mde_2d,
            self.sum_mde_3d,
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
        ade_2d_per_horizon = self.sum_ade_2d / valid_points_per_horizon  # Shape: [H]
        ade_3d_per_horizon = self.sum_ade_3d / valid_points_per_horizon  # Shape: [H]
        ade_2d_scalar = self.sum_ade_2d.sum() / total_valid_points       # Scalar
        ade_3d_scalar = self.sum_ade_3d.sum() / total_valid_points       # Scalar
        
        # 2. MDE (Average Max Displacement Error)
        mde_2d_scalar = self.sum_mde_2d / total_trajectories  # Scalar
        mde_3d_scalar = self.sum_mde_3d / total_trajectories  # Scalar
        
        # 3. FDE (Average Final Displacement Error)
        fde_2d_scalar = self.sum_fde_2d / total_trajectories  # Scalar
        fde_3d_scalar = self.sum_fde_3d / total_trajectories  # Scalar
        
        # 4. MAE (Mean Absolute Error) and RMSE (Root Mean Square Error) per horizon
        valid_points_per_horizon_unsqueezed = valid_points_per_horizon.unsqueeze(1)  # Shape: [H, 1]
        mae_per_horizon = self.sum_abs_error / valid_points_per_horizon_unsqueezed  # Shape: [H, F]
        rmse_per_horizon = torch.sqrt(self.sum_sq_error / valid_points_per_horizon_unsqueezed)  # Shape: [H, F]

        # 5. RTDE metrics (per-trajectory)
        # Scalar RTDE metrics (average across all trajectories)
        rtde_scalar = self.sum_rtde / total_trajectories
        rtde_relative_scalar = self.sum_relative_rtde / total_trajectories

        # 5b. MSE for altitude (over all valid waypoints)
        altitude_mse_scalar = self.sum_sq_error[:, 2].sum() / total_valid_points

        # 5c. Concatenate per-trajectory ADE/FDE values for histograms
        traj_ade_2d_values = torch.cat(self.traj_ade_2d_list)
        traj_ade_3d_values = torch.cat(self.traj_ade_3d_list)
        traj_fde_2d_values = torch.cat(self.traj_fde_2d_list)
        traj_fde_3d_values = torch.cat(self.traj_fde_3d_list)
        traj_rtde_values = torch.cat(self.traj_rtde_list)
        traj_rtde_relative_values = torch.cat(self.traj_rtde_relative_list)
        traj_rtd_pred_values = torch.cat(self.traj_rtd_pred_list)
        traj_rtd_target_values = torch.cat(self.traj_rtd_target_list)

        # Standard deviation of relative RTDE (sample std; 0 if < 2 trajectories)
        n_rtde = traj_rtde_relative_values.numel()
        rtde_relative_std = torch.tensor(0.0, device=self.device, dtype=traj_rtde_relative_values.dtype)
        if n_rtde >= 2:
            rtde_relative_std = traj_rtde_relative_values.std()
        
        if strategy is not None and hasattr(strategy, "world_size") and strategy.world_size > 1:
            raise NotImplementedError("Gathering trajectory metrics across multiple GPUs is not implemented.")
        
        return {
            "ade_2d_scalar": ade_2d_scalar,
            "ade_3d_scalar": ade_3d_scalar,
            "fde_2d_scalar": fde_2d_scalar,
            "fde_3d_scalar": fde_3d_scalar,
            "mde_2d_scalar": mde_2d_scalar,
            "mde_3d_scalar": mde_3d_scalar,
            "rtde_scalar": rtde_scalar,
            "rtde_relative_scalar": rtde_relative_scalar,
            "rtde_relative_std": rtde_relative_std,
            "altitude_mse_scalar": altitude_mse_scalar,
            "ade_2d_per_horizon": ade_2d_per_horizon,
            "ade_3d_per_horizon": ade_3d_per_horizon,
            "mae_per_horizon": mae_per_horizon,
            "rmse_per_horizon": rmse_per_horizon,
            "traj_ade_2d_values": traj_ade_2d_values,
            "traj_ade_3d_values": traj_ade_3d_values,
            "traj_fde_2d_values": traj_fde_2d_values,
            "traj_fde_3d_values": traj_fde_3d_values,
            "traj_rtde_values": traj_rtde_values,
            "traj_rtde_relative_values": traj_rtde_relative_values,
            "traj_rtd_pred_values": traj_rtd_pred_values,
            "traj_rtd_target_values": traj_rtd_target_values,
        }
