import torch
from data.compute.trajectory import length_to_mask
from torch import nn


class ADELoss(nn.Module):

    def __init__(
        self,
        use_3d: bool = True,
        epsilon: float = 1e-6,
    ):
        """
        Initialize ADE loss.

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
        eval_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Average Displacement Error (ADE) loss in 2D or 3D normalized space.

        Computes the masked ADE loss by calculating Euclidean distance between
        normalized positions and averaging over valid positions in the evaluation window.

        Args:
            pred_pos_norm: Predicted normalized positions [batch_size, horizon_seq_len, 3]
            target_pos_norm: Target normalized positions [batch_size, horizon_seq_len, 3]
            eval_len: Number of valid evaluation steps per sample [batch_size]

        Returns:
            Scalar loss value
        """
        # 1. Create active mask (False for padded positions)
        H = pred_pos_norm.size(1)
        active_mask = length_to_mask(eval_len, H)  # [B, H]
        if not active_mask.any():
            return torch.tensor(0.0, device=pred_pos_norm.device, requires_grad=True)

        # 2. Calculate Distance Error (ADE) as Euclidean distance in meters
        diff_norm = pred_pos_norm - target_pos_norm
        if self.use_3d:
            dist_norm = torch.norm(diff_norm, dim=-1) + self.epsilon
        else:
            dist_norm = torch.norm(diff_norm[:, :, :2], dim=-1) + self.epsilon

        loss = dist_norm[active_mask].mean()
        return loss
