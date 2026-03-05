from dataclasses import dataclass

import torch


@dataclass
class HorizonResult:
    """Per-horizon displacement metrics for one epoch."""
    ade: torch.Tensor   # [H] mean 2D Euclidean distance at each horizon step
    mae: torch.Tensor   # [H, 3] mean absolute error per feature (X, Y, Altitude)
    rmse: torch.Tensor  # [H, 3] root mean squared error per feature


class HorizonMetrics:
    """
    Accumulates running sums for per-horizon ADE2D, MAE, and RMSE.

    Uses running sums (rather than per-trajectory storage) because storing
    [H]-shaped tensors per trajectory would be prohibitively wide for export.
    """

    def __init__(self, horizon_seq_len: int, device: torch.device):
        self._horizon_seq_len = horizon_seq_len
        self._device = device
        self._reset()

    def _reset(self) -> None:
        H, F = self._horizon_seq_len, 3
        self._sum_ade_2d = torch.zeros(H, device=self._device)        # [H]
        self._sum_abs_error = torch.zeros(H, F, device=self._device)  # [H, F]
        self._sum_sq_error = torch.zeros(H, F, device=self._device)   # [H, F]
        self._valid_count = torch.zeros(H, device=self._device)        # [H]

    def update(
        self,
        pred_pos_abs: torch.Tensor,
        target_pos_abs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """
        Accumulate running sums for per-horizon metrics.

        Args:
            pred_pos_abs: Predicted absolute positions [B, H, 3].
            target_pos_abs: Target absolute positions [B, H, 3].
            valid_mask: Boolean mask [B, H], True for valid (non-padded) waypoints.
        """
        diff = pred_pos_abs - target_pos_abs                    # [B, H, 3]
        abs_err = diff.abs()                                    # [B, H, 3]
        sq_err = diff ** 2                                      # [B, H, 3]
        dist_2d = torch.norm(diff[:, :, :2], dim=2)            # [B, H]

        valid_mask_feat = valid_mask.unsqueeze(-1)              # [B, H, 1]
        self._sum_abs_error += (abs_err * valid_mask_feat).sum(dim=0)
        self._sum_sq_error += (sq_err * valid_mask_feat).sum(dim=0)
        self._sum_ade_2d += (dist_2d * valid_mask).sum(dim=0)
        self._valid_count += valid_mask.sum(dim=0)

    def compute(self) -> HorizonResult:
        """
        Compute per-horizon metrics from running sums.

        Returns:
            HorizonResult with per-horizon ADE2D, MAE, and RMSE arrays.
        """
        valid_count = self._valid_count.clamp(min=1.0)   # [H]
        valid_count_feat = valid_count.unsqueeze(1)       # [H, 1]

        return HorizonResult(
            ade=self._sum_ade_2d / valid_count,
            mae=self._sum_abs_error / valid_count_feat,
            rmse=torch.sqrt(self._sum_sq_error / valid_count_feat),
        )
