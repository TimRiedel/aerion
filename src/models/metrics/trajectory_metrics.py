from dataclasses import dataclass
from typing import Optional  # used for flight_id kwarg

import pandas as pd
import torch

from data.compute.trajectory import TrajectoryLengths, length_to_mask
from models.metrics.displacement_metrics import DisplacementMetrics, DisplacementResult
from models.metrics.horizon_metrics import HorizonMetrics, HorizonResult
from models.metrics.position_metrics import PositionMetrics, PositionResult
from models.metrics.rtd_metrics import RtdMetrics, RtdResult


@dataclass
class TrajectoryMetricsResult:
    """Composed result from all trajectory metric groups for one epoch."""
    displacement: DisplacementResult
    position: PositionResult
    rtd: RtdResult
    horizon: HorizonResult


class TrajectoryMetrics:
    """
    Orchestrator that coordinates DisplacementMetrics, RtdMetrics, and HorizonMetrics.

    Each call to update() delegates to the three metric classes. compute() returns
    a composed TrajectoryMetricsResult. to_dataframe() produces a per-trajectory
    DataFrame suitable for parquet export.
    """

    def __init__(self, horizon_seq_len: int, device: torch.device):
        self.displacement = DisplacementMetrics()
        self.position = PositionMetrics()
        self.rtd = RtdMetrics()
        self.horizon = HorizonMetrics(horizon_seq_len, device)
        self._flight_ids: list[str] = []

    def update(
        self,
        pred_pos_abs: torch.Tensor,
        target_pos_abs: torch.Tensor,
        lengths: TrajectoryLengths,
        pred_rtd: torch.Tensor,
        target_rtd: torch.Tensor,
        flight_id: Optional[list[str]] = None,
    ) -> None:
        """
        Accumulate metrics for one batch of predictions.

        Args:
            pred_pos_abs: Predicted absolute positions [B, H, 3].
            target_pos_abs: Target absolute positions [B, H, 3].
            lengths: TrajectoryLengths with pred_valid_len, target_valid_len.
            pred_rtd: Predicted RTD per trajectory [B].
            target_rtd: Target RTD per trajectory [B].
            flight_id: Optional list of B flight ID strings.
        """
        H = pred_pos_abs.size(1)
        target_valid_mask = length_to_mask(lengths.target_valid_len, H)  # [B, H]
        has_valid = target_valid_mask.any(dim=1)                  # [B]

        self.displacement.update(pred_pos_abs, target_pos_abs, target_valid_mask, lengths.pred_valid_len, lengths.target_valid_len)
        self.position.update(pred_pos_abs, target_pos_abs, target_valid_mask)
        self.rtd.update(pred_rtd, target_rtd, has_valid)
        self.horizon.update(pred_pos_abs, target_pos_abs, target_valid_mask)

        if flight_id is not None:
            has_valid_list = has_valid.tolist()
            self._flight_ids.extend(fid for fid, valid in zip(flight_id, has_valid_list) if valid)

    def compute(self) -> TrajectoryMetricsResult:
        """
        Aggregate all metric groups and return a composed result.

        Returns:
            TrajectoryMetricsResult with displacement, rtd, and horizon sub-results.
        """
        return TrajectoryMetricsResult(
            displacement=self.displacement.compute(),
            position=self.position.compute(),
            rtd=self.rtd.compute(),
            horizon=self.horizon.compute(),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Build a per-trajectory DataFrame for parquet export.

        Columns include displacement (ADE2D, FDE2D, MDE2D) and RTD error metrics.
        Flight IDs are included as the first column when they were provided via update().

        Returns:
            DataFrame with one row per trajectory across all accumulated batches.
        """
        cols: dict = {}

        if self._flight_ids:
            cols["flight_id"] = self._flight_ids

        cols.update(self.displacement.dataframe_columns())
        cols.update(self.position.dataframe_columns())
        cols.update(self.rtd.dataframe_columns())

        return pd.DataFrame(cols)
