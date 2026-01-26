import torch
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
        pred_abs: torch.Tensor,
        target_abs: torch.Tensor,
        target_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Average Displacement Error (ADE) loss in 2D or 3D space (in meters).
        
        Computes the masked ADE loss by calculating Euclidean distance between
        absolute positions and averaging over valid (non-padded) positions.
        
        Args:
            pred_abs: Predicted absolute positions [batch_size, horizon_seq_len, 3] (in meters)
            target_abs: Target absolute positions [batch_size, horizon_seq_len, 3] (in meters)
            target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
            
        Returns:
            Scalar loss value
        """
        # 1. Create active mask (False for padded positions)
        active_mask = ~target_pad_mask  # [B, H]
        if not active_mask.any():
            return torch.tensor(0.0, device=pred_abs.device, requires_grad=True)
        
        # 2. Calculate Distance Error (ADE) as Euclidean distance in meters
        diff = pred_abs - target_abs
        if self.use_3d:
            dist = torch.norm(diff, dim=-1) + self.epsilon
        else:
            dist = torch.norm(diff[:, :, :2], dim=-1) + self.epsilon

        loss = dist[active_mask].mean()
        return loss
