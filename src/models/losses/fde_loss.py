import torch
from torch import nn


class FDELoss(nn.Module):
    def __init__(self, use_3d: bool = True, epsilon: float = 1e-6):
        """
        Initialize FDE loss.
        
        Args:
            use_3d: If True, compute 3D distance (x, y, altitude). If False, compute 2D distance (x, y only).
            epsilon: Small value added to distance calculation for numerical stability
        """
        super().__init__()
        self.use_3d = use_3d
        self.epsilon = epsilon

    def forward(
        self,
        pred_pos_norm: torch.Tensor,
        target_pos_norm: torch.Tensor,
        pred_valid_len: torch.Tensor,
        target_valid_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Final Displacement Error (FDE) loss in 2D or 3D normalized space.

        Computes the distance between the last valid predicted position and the
        last valid target position. Endpoints are determined independently.

        Args:
            pred_pos_norm: Predicted normalized positions [batch_size, horizon_seq_len, 3]
            target_pos_norm: Target normalized positions [batch_size, horizon_seq_len, 3]
            pred_valid_len: Number of valid prediction steps per sample [batch_size]
            target_valid_len: Number of valid target steps per sample [batch_size]

        Returns:
            Scalar loss value
        """
        batch_indices = torch.arange(pred_pos_norm.size(0), device=pred_pos_norm.device)

        # 1. Find the index of the last valid predicted/target waypoint for each trajectory in the batch
        pred_last_idx = (pred_valid_len - 1).clamp(min=0)
        target_last_idx = (target_valid_len - 1).clamp(min=0)

        # 2. Extract the last valid predicted and target positions for each trajectory
        last_pred = pred_pos_norm[batch_indices, pred_last_idx]  # [B, 3]
        last_target = target_pos_norm[batch_indices, target_last_idx]  # [B, 3]

        # 3. Calculate Euclidean distance for each trajectory
        diff_norm = last_pred - last_target  # [B, 3]
        if self.use_3d:
            dist_norm = torch.norm(diff_norm, dim=-1) + self.epsilon # [B]
        else:
            dist_norm = torch.norm(diff_norm[:, :2], dim=-1) + self.epsilon # [B]

        return dist_norm.mean()