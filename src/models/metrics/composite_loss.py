import torch
from torch import nn
from torch.nn import MSELoss
from .ade_loss import ADELoss
from .fde_loss import FDELoss
from .rtd_loss import RTDLoss
from .ils_alignment_loss import ILSAlignmentLoss

# Canonical order for loss names (used by GradNorm and config).
LOSS_ORDER = ["mse", "ade", "fde", "alignment", "rtd"]


class CompositeApproachLoss(nn.Module):
    """
    Computes individual trajectory losses. Weighting is done by GradNormBalancer;
    config only enables or disables each loss term.
    """

    def __init__(
        self,
        enabled: dict,
        use_3d: bool = True,
        epsilon: float = 1e-6,
        alignment_num_waypoints: int = 7,
    ):
        """
        Initialize Composite Approach Loss.
        
        Combines ADE (Average Displacement Error), FDE (Final Displacement Error), 
        RTD (Remaining Track Distance) and ILS Alignment losses.
        
        - FDE serves as a fixed target (get to the runway threshold)
        - Alignment complements FDE and ensures the approach direction matches the runway heading
        - ADE defines how to get to the target (overall trajectory accuracy)
        - RTD complements ADE by making sure the flown track distance is correct
        
        Args:
            enabled: Dict of loss name -> bool (e.g. ade: true, fde: true, alignment: true, rtd: true).
            use_3d: If True, compute 3D distance for ADE/FDE.
            epsilon: Small value for numerical stability.
            alignment_num_waypoints: Number of final waypoints for ILS alignment.
        """
        super().__init__()

        self.enabled = {k: bool(enabled.get(k, False)) for k in LOSS_ORDER}

        self.mse = MSELoss()
        self.ade = ADELoss(use_3d=use_3d, epsilon=epsilon)
        self.fde = FDELoss(use_3d=use_3d, epsilon=epsilon)
        self.alignment_loss = ILSAlignmentLoss(
            num_final_waypoints=alignment_num_waypoints,
            epsilon=epsilon,
        )
        self.rtd_loss = RTDLoss()
        
        if not any(self.enabled.values()):
            raise ValueError("At least one loss must be enabled")

    def __call__(
        self,
        pred_abs: torch.Tensor,
        target_abs: torch.Tensor,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        target_pad_mask: torch.Tensor,
        pred_rtd: torch.Tensor,
        target_rtd: torch.Tensor,
        runway: dict,
    ) -> torch.Tensor:
        """
        Compute unweighted scalar loss for each enabled task.

        Returns:
            Tensor of shape [num_enabled_tasks].
        """
        losses = []
        if self.enabled.get("mse", False):
            active_mask = ~target_pad_mask
            mse_loss = self.mse(pred_norm[active_mask], target_norm[active_mask])   
            losses.append(mse_loss)
        if self.enabled.get("ade", False):
            ade_loss = self.ade(pred_norm, target_norm, target_pad_mask)
            losses.append(ade_loss)
        if self.enabled.get("fde", False):
            fde_loss = self.fde(pred_norm, target_norm, target_pad_mask)
            losses.append(fde_loss)
        if self.enabled.get("alignment", False):
            alignment_loss = self.alignment_loss(pred_abs, target_abs, target_pad_mask, runway)
            losses.append(alignment_loss)
        if self.enabled.get("rtd", False):
            rtd_loss = self.rtd_loss(pred_rtd, target_rtd)
            losses.append(rtd_loss)

        return torch.stack(losses)
