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
        target_valid_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Final Displacement Error (FDE) loss in 2D or 3D normalized space.

        Computes the distance between the predicted position at target landing time
        and the target endpoint. Both positions are indexed by target_valid_len.

        Args:
            pred_pos_norm: Predicted normalized positions [batch_size, horizon_seq_len, 3]
            target_pos_norm: Target normalized positions [batch_size, horizon_seq_len, 3]
            target_valid_len: Number of valid target steps per sample [batch_size]

        Returns:
            Scalar loss value
        """
        batch_indices = torch.arange(pred_pos_norm.size(0), device=pred_pos_norm.device)

        # 1. Find the index of the target landing step for each trajectory in the batch
        last_idx = (target_valid_len - 1).clamp(min=0)

        # 2. Extract the predicted and target positions at the target landing step
        last_pred = pred_pos_norm[batch_indices, last_idx]    # [B, 3]
        last_target = target_pos_norm[batch_indices, last_idx]  # [B, 3]

        # 3. Calculate Euclidean distance for each trajectory
        diff_norm = last_pred - last_target  # [B, 3]
        if self.use_3d:
            dist_norm = torch.norm(diff_norm, dim=-1) + self.epsilon # [B]
        else:
            dist_norm = torch.norm(diff_norm[:, :2], dim=-1) + self.epsilon # [B]

        return dist_norm.mean()