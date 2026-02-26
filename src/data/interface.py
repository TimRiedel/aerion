from dataclasses import dataclass
from typing import List, Union

import torch


@dataclass
class TrajectoryData:
    """Trajectory tensors: encoder input, decoder input, and target.

    Tensor shapes depend on dataset:
    - Single agent (e.g. ApproachDataset): [T, F]
    - Multi-agent scene (e.g. TrafficDataset): [T, N, F]
    """

    encoder_in: torch.Tensor
    dec_in: torch.Tensor
    target: torch.Tensor


@dataclass
class RunwayData:
    """Runway coordinates (xyz), bearing, and centerline reference points."""

    xyz: torch.Tensor
    bearing: torch.Tensor
    length: torch.Tensor
    centerline_points_xy: List[torch.Tensor]


@dataclass
class PredictionSample:
class Sample:
    """Single sample: trajectory interface, target padding mask, runway data, RTD, and flight ID."""

    xyz_positions: TrajectoryData
    xyz_deltas: TrajectoryData
    trajectory: TrajectoryData
    target_padding_mask: torch.Tensor
    target_rtd: torch.Tensor
    last_input_pos_abs: torch.Tensor
    runway: RunwayData
    flight_id: Union[str, List[str]]  # str for single sample, List[str] when batched
