from torch import nn
from .ade_loss import ADELoss
from .fde_loss import FDELoss


class CompositeApproachLoss(nn.Module):
    def __init__(
        self,
        weights: dict,
        use_3d: bool = True,
        epsilon: float = 1e-6,
    ):
        """
        Initialize Composite Approach Loss.
        
        Combines ADE (Average Displacement Error) and FDE (Final Displacement Error) losses
        with configurable weights. This loss function allows balancing between overall trajectory
        accuracy (ADE) and final position accuracy (FDE).
        FDE serves as a fixed target (what to achieve, i.e. get to a runway threshold), while ADE
        defines how to get there.
        
        Args:
            weights: Dictionary containing weights for each loss component. Must contain 'ade' and 'fde' keys.
                     Example: {'ade': 3.0, 'fde': 1.0}
            use_3d: If True, compute 3D distance (x, y, altitude). If False, compute 2D distance (x, y only).
            epsilon: Small value added to distance calculation for numerical stability
        """
        super().__init__()
        if 'ade' not in weights or 'fde' not in weights:
            raise ValueError("weights dictionary must contain 'ade' and 'fde' keys")

        self.weights = weights
        self.ade = ADELoss(use_3d=use_3d, epsilon=epsilon)
        self.fde = FDELoss(use_3d=use_3d, epsilon=epsilon)

    def forward(self, pred_abs, target_abs, target_pad_mask):
        """
        Compute composite loss combining ADE and FDE.
        
        Args:
            pred_abs: Predicted absolute positions [batch_size, horizon_seq_len, 3] (in meters)
            target_abs: Target absolute positions [batch_size, horizon_seq_len, 3] (in meters)
            target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
            
        Returns:
            Scalar loss value (weighted sum of ADE and FDE losses)
        """
        loss_ade = self.ade(pred_abs, target_abs, target_pad_mask)
        loss_fde = self.fde(pred_abs, target_abs, target_pad_mask)
        
        total_loss = (
            self.weights['ade'] * loss_ade +
            self.weights['fde'] * loss_fde
        )
        
        return total_loss