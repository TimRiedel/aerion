from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import Tensor

from data.compute.runway import get_distances_to_centerline
from data.interface import RunwayData
from data.transforms.normalize import Normalizer


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
    def normalizer(self):
        if self._normalizer is None:
            raise ValueError(f"Normalizer not created for group {self.name}")
        return self._normalizer

    @abstractmethod
    def compute(
        self, xyz_positions: torch.Tensor, xyz_deltas: torch.Tensor, runway: RunwayData
    ) -> Tensor:
        """
        Compute RAW (unnormalized) features from trajectory data.
        Called by the dataset during __getitem__.

        Args:
            xyz_positions: absolute trajectory positions [B, T, 3]
            xyz_deltas: absolute trajectory deltas [B, T, 3]
            runway: RunwayData

        Returns:
            Feature tensor [B, T, width]
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

        Args:
            current_position_abs: current absolute position [B, 3]
            pred_delta_abs: predicted delta in absolute coordinates [B, 3]
            runway: RunwayData

        Returns:
            Next decoder input token [B, width]
        """
    
    def get_data(self, trajectory: torch.Tensor) -> Tensor:
        """
        Args:
            trajectory: Unbatched tensor [T, F]

        Returns:
            Feature tensor [T, width]
        """
        return trajectory[:, self.start_idx:self.end_idx]

    def create_normalizer(self, mean: Tensor, std: Tensor) -> None:
        self._normalizer = Normalizer(mean=mean, std=std)
        self.mean = mean
        self.std = std

    def normalize(self, x: Tensor) -> Tensor:
        return self.normalizer(x)


class XYPosition(FeatureGroup):
    name = "xy_position"
    width = 2

    def compute(
        self, xyz_positions: torch.Tensor, xyz_deltas: torch.Tensor, runway: RunwayData
    ) -> Tensor:
        return xyz_positions[:, :2]

    def build_next_decoder_input(
        self,
        current_position_abs: Tensor,
        pred_delta_abs: Tensor,
        runway: RunwayData,
    ) -> Tensor:
        return current_position_abs[:, :2]


class Altitude(FeatureGroup):
    name = "altitude"
    width = 1

    def compute(
        self, xyz_positions: torch.Tensor, xyz_deltas: torch.Tensor, runway: RunwayData
    ) -> Tensor:
        return xyz_positions[:, 2:3]

    def build_next_decoder_input(
        self,
        current_position_abs: Tensor,
        pred_delta_abs: Tensor,
        runway: RunwayData,
    ) -> Tensor:
        return current_position_abs[:, 2:3]


class DeltaXYZ(FeatureGroup):
    name = "delta_xyz"
    width = 3

    def compute(
        self, xyz_positions: torch.Tensor, xyz_deltas: torch.Tensor, runway: RunwayData
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
    width = 2

    def compute(
        self, xyz_positions: torch.Tensor, xyz_deltas: torch.Tensor, runway: RunwayData
    ) -> Tensor:
        threshold_xy = runway.xyz[:2]
        return get_distances_to_centerline(xyz_positions[:, :2], [threshold_xy])

    def build_next_decoder_input(
        self,
        current_position_abs: Tensor,
        pred_delta_abs: Tensor,
        runway: RunwayData,
    ) -> Tensor:
        threshold_xy = runway.xyz[:, :2]  # [B, 2]
        return get_distances_to_centerline(current_position_abs[:, :2], [threshold_xy])


class DistancesToCenterlineXY(FeatureGroup):
    name = "distances_to_centerline_xy"

    @property
    def width(self) -> int:
        n_points = len(
            self._params.get("centerline_distances", [4, 8, 16, 32])
        )
        return 2 * n_points

    def compute(
        self, xyz_positions: torch.Tensor, xyz_deltas: torch.Tensor, runway: RunwayData
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
            current_position_abs[:, :2], runway.centerline_points_xy
        )


FEATURE_REGISTRY: Dict[str, type[FeatureGroup]] = {
    "xy_position": XYPosition,
    "altitude": Altitude,
    "delta_xyz": DeltaXYZ,
    "distance_to_threshold_xy": DistanceToThresholdXY,
    "distances_to_centerline_xy": DistancesToCenterlineXY,
}
