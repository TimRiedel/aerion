import torch
from torch import nn
from torch.nn import MSELoss
from .ade_loss import ADELoss
from .fde_loss import FDELoss
from .altitude_loss import AltitudeLoss
from .rtd_loss import RTDLoss
from .ils_alignment_loss import ILSAlignmentLoss

LOSS_ORDER = ["ade", "fde", "rtd", "altitude", "alignment"]


class CompositeApproachLoss(nn.Module):
    """
    Computes individual trajectory losses with uncertainty weighting.
    Combines ADE, FDE, RTD, Altitude and ILS Alignment losses.

    - ADE (2D, normalized): lateral trajectory shape; avoids X/Y dominating in 3D.
    - FDE (3D, normalized): endpoint (x,y,z) correct so aircraft arrives at threshold and altitude.
    - Altitude (MSE, normalized): dedicated vertical profile loss.
    - FDE + Alignment: get to runway and match approach heading/glideslope.
    - RTD: flown track distance correct.

    Losses are always in LOSS_ORDER. enabled_losses, stacked losses, and loss_dict
    use the same index so they align with log_vars (weights) for logging.

    The loss weights are computed using uncertainty weighting, as described in:
    R. Cipolla, Y. Gal, and A. Kendall, "Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics,"
    2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7482–7491, Jun. 2018, doi: https://doi.org/10.1109/cvpr.2018.00781.
    """

    def __init__(self, loss_configs: dict):
        """
        Args:
            loss_configs: Dict of loss configs. Each key (e.g. "ade", "fde") must have an "enabled" boolean.
        """
        super().__init__()
        self.loss_configs = loss_configs
        for name in self.loss_configs.keys():
            if name not in LOSS_ORDER:
                raise NotImplementedError(f"Loss '{name}' not implemented.")

        # Fixed order: only enabled losses, in LOSS_ORDER (same index as log_vars and loss_dict)
        self.enabled_losses = [name for name in LOSS_ORDER if (loss_configs.get(name) or {}).get("enabled", False)]
        if not self.enabled_losses:
            raise ValueError("At least one loss must be enabled.")
        self.num_tasks = len(self.enabled_losses)

        if "ade" in self.enabled_losses:
            self.ade_loss = ADELoss(use_3d=loss_configs["ade"]["use_3d"])
        if "fde" in self.enabled_losses:
            self.fde_loss = FDELoss(use_3d=loss_configs["fde"]["use_3d"])
        if "rtd" in self.enabled_losses:
            self.rtd_loss = RTDLoss()
        if "altitude" in self.enabled_losses:
            self.altitude_loss = AltitudeLoss()
        if "alignment" in self.enabled_losses:
            self.alignment_loss = ILSAlignmentLoss(
                num_final_waypoints=loss_configs["alignment"].get("num_waypoints", 7)
            )

        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))

    def forward(
        self,
        pred_abs: torch.Tensor,
        target_abs: torch.Tensor,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        target_pad_mask: torch.Tensor,
        pred_rtd: torch.Tensor,
        target_rtd: torch.Tensor,
        runway: dict,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute composite loss: ADE (2D norm), FDE (3D norm), RTD, Altitude (MSE norm), Alignment.
        
        Args:
            pred_abs: Predicted absolute positions [B, H, 3] (in meters)
            target_abs: Target absolute positions [B, H, 3] (in meters)
            pred_norm: Predicted normalized positions [B, H, 3]
            target_norm: Target normalized positions [B, H, 3]
            target_pad_mask: Padding mask [B, H] (True for padded positions)
            pred_rtd: Predicted remaining track distance [B]
            target_rtd: Target remaining track distance [B]
            runway: Dictionary containing "xyz" coordinates and "bearing" in sin, cos format.
        
        Returns:
            total: Scalar total loss
            loss_info: Dictionary containing loss weights and sigmas
        """
        losses = []
        for name in self.enabled_losses:
            if name == "ade":
                losses.append(self.ade_loss(pred_norm, target_norm, target_pad_mask))
            if name == "fde":
                losses.append(self.fde_loss(pred_norm, target_norm, target_pad_mask))
            if name == "rtd":
                losses.append(self.rtd_loss(pred_rtd, target_rtd))
            if name == "altitude":
                losses.append(self.altitude_loss(pred_norm, target_norm, target_pad_mask))
            if name == "alignment":
                losses.append(self.alignment_loss(pred_abs, target_abs, target_pad_mask, runway))

        task_losses = torch.stack(losses)
        weights = torch.exp(-self.log_vars)
        weighted = 0.5 * weights * task_losses
        regularizer = 0.5 * self.log_vars
        total = (weighted + regularizer).sum()

        sigmas = torch.exp(0.5 * self.log_vars)
        loss_info = {
            "loss_weights": {
                name: weights[i].item()
                for i, name in enumerate(self.enabled_losses)
            },
            "loss_sigmas": {
                name: sigmas[i].item()
                for i, name in enumerate(self.enabled_losses)
            },
        }
        return total, loss_info
