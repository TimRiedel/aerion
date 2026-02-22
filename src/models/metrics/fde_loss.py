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

    def forward(self, pred_norm, target_norm, target_pad_mask):
        """
        Final Displacement Error (FDE) loss in 2D or 3D normalized space.
        
        Computes the masked FDE loss by calculating Euclidean distance between
        the last valid (non-padded) predicted and target normalized positions.
        
        Args:
            pred_norm: Predicted normalized positions [batch_size, horizon_seq_len, 3]
            target_norm: Target normalized positions [batch_size, horizon_seq_len, 3]
            target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
            
        Returns:
            Scalar loss value
        """
        # 1. Find the index of the last valid (unpadded) waypoint for each trajectory in the batch
        # target_pad_mask is True for padding. ~mask.sum(dim=1) - 1 gives the last valid index.
        lengths = (~target_pad_mask).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)

        # 2. Extract the last valid predicted and target positions for each trajectory
        batch_indices = torch.arange(pred_norm.size(0), device=pred_norm.device)
        last_pred = pred_norm[batch_indices, lengths]        # [batch_size, 3]
        last_target = target_norm[batch_indices, lengths]    # [batch_size, 3]

        # 3. Calculate Euclidean distance for each trajectory
        diff_norm = last_pred - last_target  # [batch_size, 3]
        if self.use_3d:
            dist_norm = torch.norm(diff_norm, dim=-1) + self.epsilon          # [batch_size]
        else:
            dist_norm = torch.norm(diff_norm[:, :2], dim=-1) + self.epsilon   # [batch_size]

        return dist_norm.mean()