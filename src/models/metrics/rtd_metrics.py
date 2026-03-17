from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RtdResult:
    """Aggregated Remaining Track Distance error metrics across all trajectories in an epoch."""
    mean_error: torch.Tensor            # Mean signed error (ME), same unit as RTD
    mean_abs_error: torch.Tensor        # Mean absolute error (MAE)
    mean_pct_error: torch.Tensor        # Mean signed percentage error (MPE, %)
    mean_abs_pct_error: torch.Tensor    # Mean absolute percentage error (MAPE, %)
    std_error: torch.Tensor             # Standard deviation of signed error
    std_abs_error: torch.Tensor         # Standard deviation of absolute error
    std_pct_error: torch.Tensor         # Standard deviation of signed percentage error
    std_abs_pct_error: torch.Tensor     # Standard deviation of absolute percentage error
    rtde_trajectories: torch.Tensor             # [N] per-trajectory signed error
    rtdae_trajectories: torch.Tensor            # [N] per-trajectory absolute error
    rtdpe_trajectories: torch.Tensor            # [N] per-trajectory signed percentage error
    rtdape_trajectories: torch.Tensor           # [N] per-trajectory absolute percentage error
    rtd_pred_trajectories: torch.Tensor         # [N] per-trajectory predicted RTD
    rtd_target_trajectories: torch.Tensor       # [N] per-trajectory target RTD


def _std_or_zero(values: torch.Tensor) -> torch.Tensor:
    """Return sample standard deviation, or 0.0 if fewer than two values are present."""
    if values.numel() < 2:
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)
    return values.std()


class RtdMetrics:
    """
    Accumulates per-trajectory Remaining Track Distance (RTD) error metrics.

    Sign convention: positive error means the model overestimated remaining distance.
    Percentage errors are relative to the target RTD and expressed in percent (×100).
    Only trajectories with at least one valid (non-padded) waypoint are included.
    """

    def __init__(self):
        self.rtde_trajectories: list[torch.Tensor] = []
        self.rtdae_trajectories: list[torch.Tensor] = []
        self.rtdpe_trajectories: list[torch.Tensor] = []
        self.rtdape_trajectories: list[torch.Tensor] = []
        self.rtd_pred_trajectories: list[torch.Tensor] = []
        self.rtd_target_trajectories: list[torch.Tensor] = []

    def update(
        self,
        pred_rtd: torch.Tensor,
        target_rtd: torch.Tensor,
        has_valid_points: torch.Tensor,
    ) -> None:
        """
        Compute and accumulate per-trajectory RTD error metrics for one batch.

        Args:
            pred_rtd: Predicted RTD per trajectory [B].
            target_rtd: Target RTD per trajectory [B].
            has_valid_points: Boolean mask [B], True if the trajectory has valid waypoints.
        """
        rtde = pred_rtd - target_rtd       # [B] signed error
        rtdpe = rtde / target_rtd * 100.0  # [B] signed percentage error
        rtdae = rtde.abs()                 # [B] absolute error
        rtdape = rtdpe.abs()               # [B] absolute percentage error

        self.rtde_trajectories.append(rtde[has_valid_points])
        self.rtdae_trajectories.append(rtdae[has_valid_points])
        self.rtdpe_trajectories.append(rtdpe[has_valid_points])
        self.rtdape_trajectories.append(rtdape[has_valid_points])
        self.rtd_pred_trajectories.append(pred_rtd[has_valid_points])
        self.rtd_target_trajectories.append(target_rtd[has_valid_points])

    def compute(self) -> RtdResult:
        """
        Aggregate RTD error metrics over all accumulated trajectories.

        Returns:
            RtdResult with means, standard deviations, and per-trajectory tensors.
        """
        rtde_trajectories = torch.cat(self.rtde_trajectories)
        rtdae_trajectories = torch.cat(self.rtdae_trajectories)
        rtdpe_trajectories = torch.cat(self.rtdpe_trajectories)
        rtdape_trajectories = torch.cat(self.rtdape_trajectories)
        rtd_pred_trajectories = torch.cat(self.rtd_pred_trajectories)
        rtd_target_trajectories = torch.cat(self.rtd_target_trajectories)

        n = rtde_trajectories.numel()
        denom = max(n, 1)

        return RtdResult(
            mean_error=rtde_trajectories.sum() / denom,
            mean_abs_error=rtdae_trajectories.sum() / denom,
            mean_pct_error=rtdpe_trajectories.sum() / denom,
            mean_abs_pct_error=rtdape_trajectories.sum() / denom,
            std_error=_std_or_zero(rtde_trajectories),
            std_abs_error=_std_or_zero(rtdae_trajectories),
            std_pct_error=_std_or_zero(rtdpe_trajectories),
            std_abs_pct_error=_std_or_zero(rtdape_trajectories),
            rtde_trajectories=rtde_trajectories,
            rtdae_trajectories=rtdae_trajectories,
            rtdpe_trajectories=rtdpe_trajectories,
            rtdape_trajectories=rtdape_trajectories,
            rtd_pred_trajectories=rtd_pred_trajectories,
            rtd_target_trajectories=rtd_target_trajectories,
        )

    def dataframe_columns(self) -> dict[str, np.ndarray]:
        """Return per-trajectory RTD error values as numpy arrays for parquet export."""
        def to_numpy(tensors: list[torch.Tensor]) -> np.ndarray:
            return torch.cat(tensors).detach().cpu().float().numpy()

        return {
            "rtd_pred": to_numpy(self.rtd_pred_trajectories),
            "rtd_target": to_numpy(self.rtd_target_trajectories),
            "rtde": to_numpy(self.rtde_trajectories),
            "rtdae": to_numpy(self.rtdae_trajectories),
            "rtdpe": to_numpy(self.rtdpe_trajectories),
            "rtdape": to_numpy(self.rtdape_trajectories),
        }
