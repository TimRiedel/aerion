from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from data.compute.runway import get_distances_to_centerline
from data.interface import RunwayData
from data.utils import Normalizer
from torch import Tensor


class FeatureGroup(ABC):
    """A named group of related features that are computed together."""

    def __init__(self, start_idx: int, params: Dict[str, Any] | None = None):
        self.start_idx = start_idx
        self._params = params or {}
        self.end_idx = start_idx + self.width
        self._normalizer = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Config-level identifier (must match YAML feature name)."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Number of scalar features this group contributes to the tensor."""

    @property
    def required_df_cols(self) -> List[str]:
        """DataFrame columns this group needs from the parquet file.
        Declared columns are extracted by the dataset."""
        return []

    @property
    def normalizer(self):
        if self._normalizer is None:
            raise ValueError(f"Normalizer not created for group {self.name}")
        return self._normalizer

    @abstractmethod
    def compute(
        self,
        xyz_positions: torch.Tensor,
        xyz_deltas: torch.Tensor,
        runway: RunwayData,
    ) -> Tensor:
        """
        Compute RAW (unnormalized) features from trajectory data.
        Called by the dataset during __getitem__.

        Args:
            xyz_positions: absolute trajectory positions [T, 3]
            xyz_deltas: absolute trajectory deltas [T, 3]
            runway: RunwayData

        Returns:
            Feature tensor [T, width]
        """

    @abstractmethod
    def build_next_decoder_input(
        self,
        current_position_abs: Tensor,
        pred_delta_abs: Tensor,
        runway: RunwayData,
    ) -> Tensor:
        """
        Build the next absolute decoder input token during autoregressive inference.
        Both input and output are in absolute space (not normalized).
        Works for both single-agent and multi-agent tensors.

        Args:
            current_position_abs: [B, 3] or [B, N, 3].
            pred_delta_abs: [B, 3] or [B, N, 3].
            runway: RunwayData (fields are [B, ...] or [B, N, ...]).

        Returns:
            Next decoder input token [B, width] or [B, N, width].
        """
    
    def get_data(self, trajectory: torch.Tensor) -> Tensor:
        """
        Args:
            trajectory: [T, F] (single-agent) or [T, N, F] (multi-agent scene).

        Returns:
            Feature tensor [..., width] — same leading dims as input.
        """
        return trajectory[..., self.start_idx:self.end_idx]

    def create_normalizer(self, mean: Tensor, std: Tensor) -> None:
        self._normalizer = Normalizer(mean=mean, std=std)
        self.mean = mean
        self.std = std

    def normalize(self, x: Tensor) -> Tensor:
        return self.normalizer(x)


class XYPosition(FeatureGroup):
    name = "xy_position"
    required_df_cols = ["x_coord", "y_coord"]
    width = 2

    def compute(
        self,
        xyz_positions: torch.Tensor,
        xyz_deltas: torch.Tensor,
        runway: RunwayData,
    ) -> Tensor:
        return xyz_positions[:, :2]

    def build_next_decoder_input(
        self,
        current_position_abs: Tensor,
        pred_delta_abs: Tensor,
        runway: RunwayData,
    ) -> Tensor:
        return current_position_abs[..., :2]


class Altitude(FeatureGroup):
    name = "altitude"
    required_df_cols = ["altitude"]
    width = 1

    def compute(
        self,
        xyz_positions: torch.Tensor,
        xyz_deltas: torch.Tensor,
        runway: RunwayData,
    ) -> Tensor:
        return xyz_positions[:, 2:3]

    def build_next_decoder_input(
        self,
        current_position_abs: Tensor,
        pred_delta_abs: Tensor,
        runway: RunwayData,
    ) -> Tensor:
        return current_position_abs[..., 2:3]


class DeltaXYZ(FeatureGroup):
    name = "delta_xyz"
    required_df_cols = ["x_coord", "y_coord", "altitude"]
    width = 3

    def compute(
        self,
        xyz_positions: torch.Tensor,
        xyz_deltas: torch.Tensor,
        runway: RunwayData,
    ) -> Tensor:
        return xyz_deltas

    def build_next_decoder_input(
        self,
        current_position_abs: Tensor,
        pred_delta_abs: Tensor,
        runway: RunwayData,
    ) -> Tensor:
        return pred_delta_abs


class DistanceToThresholdXY(FeatureGroup):
    name = "distance_to_threshold_xy"
    required_df_cols = ["x_coord", "y_coord"]
    width = 2

    def compute(
        self,
        xyz_positions: torch.Tensor,
        xyz_deltas: torch.Tensor,
        runway: RunwayData,
    ) -> Tensor:
        threshold_xy = runway.xyz[:2]
        return get_distances_to_centerline(xyz_positions[:, :2], [threshold_xy])

    def build_next_decoder_input(
        self,
        current_position_abs: Tensor,
        pred_delta_abs: Tensor,
        runway: RunwayData,
    ) -> Tensor:
        threshold_xy = runway.xyz[..., :2]
        return get_distances_to_centerline(current_position_abs[..., :2], [threshold_xy])


class DistancesToCenterlineXY(FeatureGroup):
    name = "distances_to_centerline_xy"
    required_df_cols = ["x_coord", "y_coord"]

    @property
    def width(self) -> int:
        n_points = len(
            self._params.get("centerline_distances", [4, 8, 16, 32])
        )
        return 2 * n_points

    def compute(
        self,
        xyz_positions: torch.Tensor,
        xyz_deltas: torch.Tensor,
        runway: RunwayData,
    ) -> Tensor:
        return get_distances_to_centerline(
            xyz_positions[:, :2], runway.centerline_points_xy
        )

    def build_next_decoder_input(
        self,
        current_position_abs: Tensor,
        pred_delta_abs: Tensor,
        runway: RunwayData,
    ) -> Tensor:
        return get_distances_to_centerline(
            current_position_abs[..., :2], runway.centerline_points_xy
        )


class TrafficCountCosequenced(FeatureGroup):

    name = "traffic_count_cosequenced"
    required_df_cols = ["traffic_count_cosequenced"]
    width = 1

    def compute(
        self,
        xyz_positions: torch.Tensor,
        xyz_deltas: torch.Tensor,
        runway: RunwayData,
    ) -> Tensor:
        raise NotImplementedError("TrafficCountCosequenced.compute is not yet implemented.")

    def build_next_decoder_input(
        self,
        current_position_abs: Tensor,
        pred_delta_abs: Tensor,
        runway: RunwayData,
    ) -> Tensor:
        raise NotImplementedError(
            "build_next_decoder_input is not yet implemented for TrafficCountCosequenced."
        )


FEATURE_REGISTRY: Dict[str, type[FeatureGroup]] = {
    "xy_position": XYPosition,
    "altitude": Altitude,
    "delta_xyz": DeltaXYZ,
    "distance_to_threshold_xy": DistanceToThresholdXY,
    "distances_to_centerline_xy": DistancesToCenterlineXY,
    "traffic_count_cosequenced": TrafficCountCosequenced,
}
