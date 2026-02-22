import torch
from torch import nn


class AltitudeLoss(nn.Module):
    """
    MSE on normalized altitude over valid (non-padded) waypoints.

    Used as a dedicated vertical loss so altitude is not dominated by X/Y
    when using a single Euclidean distance in normalized space.
    """

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        target_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_norm: Predicted normalized positions [B, H, 3]
            target_norm: Target normalized positions [B, H, 3]
            target_pad_mask: Padding mask [B, H] (True for padded positions)

        Returns:
            Scalar MSE on normalized altitude (index 2) over valid waypoints.
        """
        active_mask = ~target_pad_mask  # [B, H]
        if not active_mask.any():
            return torch.tensor(0.0, device=pred_norm.device)

        pred_alt = pred_norm[:, :, 2][active_mask]   # [N]
        target_alt = target_norm[:, :, 2][active_mask]  # [N]
        return nn.functional.mse_loss(pred_alt, target_alt)
