import torch
from data.compute.trajectory import length_to_mask
from torch import nn


class AltitudeLoss(nn.Module):
    """
    MSE on normalized altitude over valid waypoints of the target trajectory.

    Used as a dedicated vertical loss so altitude is not dominated by X/Y
    when using a single Euclidean distance in normalized space.
    """

    def forward(
        self,
        pred_pos_norm: torch.Tensor,
        target_pos_norm: torch.Tensor,
        target_valid_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_pos_norm: Predicted normalized positions [B, H, 3]
            target_pos_norm: Target normalized positions [B, H, 3]
            target_valid_len: Number of valid target steps per sample [B]

        Returns:
            Scalar MSE on normalized altitude (index 2) over valid waypoints.
        """
        H = pred_pos_norm.size(1)
        active_mask = length_to_mask(target_valid_len, H)  # [B, H]
        if not active_mask.any():
            return torch.tensor(0.0, device=pred_pos_norm.device)

        pred_alt = pred_pos_norm[:, :, 2][active_mask]   # [N]
        target_alt = target_pos_norm[:, :, 2][active_mask]  # [N]
        return nn.functional.mse_loss(pred_alt, target_alt)
