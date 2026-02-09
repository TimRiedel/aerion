import torch
from torch import nn
from torch.nn import MSELoss
from .ade_loss import ADELoss
from .fde_loss import FDELoss
from .ils_alignment_loss import ILSAlignmentLoss


class CompositeApproachLoss(nn.Module):
    def __init__(
        self,
        weights: dict,
        use_3d: bool = True,
        epsilon: float = 1e-6,
        alignment_num_waypoints: int = 7,
    ):
        """
        Initialize Composite Approach Loss.
        
        Combines ADE (Average Displacement Error), FDE (Final Displacement Error), and
        optionally ILS Alignment losses with configurable weights.
        
        - ADE defines how to get there (overall trajectory accuracy)
        - FDE serves as a fixed target (get to the runway threshold)
        - Alignment ensures the approach direction matches the runway heading
        
        Args:
            weights: Dictionary containing weights for each loss component.
            use_3d: If True, compute 3D distance (x, y, altitude). If False, compute 2D distance (x, y only).
            epsilon: Small value added to distance calculation for numerical stability.
            alignment_num_waypoints: Number of final waypoints for alignment loss (default: 4).
                                     With 30s spacing, 4 waypoints = last 90 seconds.
        """
        super().__init__()
        if 'ade' not in weights or 'fde' not in weights:
            raise ValueError("weights dictionary must contain 'ade' and 'fde' keys")

        self.weight_mse = weights.get('mse', 0)
        self.weight_ade = weights.get('ade', 0)
        self.weight_fde = weights.get('fde', 0)
        self.weight_alignment = weights.get('alignment', 0)

        self.mse = MSELoss()
        self.ade = ADELoss(use_3d=use_3d, epsilon=epsilon)
        self.fde = FDELoss(use_3d=use_3d, epsilon=epsilon)
        self.alignment_loss = ILSAlignmentLoss(
            num_final_waypoints=alignment_num_waypoints,
            epsilon=epsilon,
        )

    def forward(
        self,
        pred_abs: torch.Tensor,
        target_abs: torch.Tensor,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        target_pad_mask: torch.Tensor,
        runway: dict,
    ):
        """
        Compute composite loss combining MSE, ADE, FDE, and optionally Alignment.
        
        Args:
            pred_abs: Predicted absolute positions [B, H, 3] (in meters)
            target_abs: Target absolute positions [B, H, 3] (in meters)
            pred_norm: Predicted normalized positions [B, H, 3]
            target_norm: Target normalized positions [B, H, 3]
            target_pad_mask: Padding mask [B, H] (True for padded positions)
            runway: Dictionary containing "xyz" coordinates and "bearing" in sin, cos format.
            
        Returns:
            Scalar loss value (weighted sum of all enabled loss components)
        """
        loss = 0
        if self.weight_mse > 0:
            active_mask = ~target_pad_mask
            loss_mse = self.mse(pred_norm[active_mask], target_norm[active_mask])
            loss += self.weight_mse * loss_mse

        if self.weight_ade > 0:
            loss_ade = self.ade(pred_norm, target_norm, target_pad_mask)
            loss += self.weight_ade * loss_ade

        if self.weight_fde > 0:
            loss_fde = self.fde(pred_norm, target_norm, target_pad_mask)
            loss += self.weight_fde * loss_fde

        if self.weight_alignment > 0:
            loss_alignment = self.alignment_loss(pred_abs, target_abs, target_pad_mask, runway)
            loss += self.weight_alignment * loss_alignment

        return loss