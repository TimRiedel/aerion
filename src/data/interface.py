from dataclasses import dataclass
from typing import List, Optional, Union

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
    """
    Single-flight prediction or multi-agent scene prediction data.

    Tensor shapes depend on the dataset:
    - Single agent (e.g. ApproachDataset): 
        * trajectory tensors [T, F]
        * target_padding_mask [H]
        * target_rtd scalar
        * last_input_pos_abs [3]
        * flight_id str.
    - Multi-agent scene (e.g. TrafficDataset): 
        * trajectory tensors [T, N, F]
        * target_padding_mask [H, N]
        * target_rtd [N]
        * last_input_pos_abs [N, 3]
        * flight_id List[str]
        * agent_padding_mask [N] or [B, N_max] when batched; True = padded slot.
    """

    xyz_positions: TrajectoryData
    xyz_deltas: TrajectoryData
    trajectory: TrajectoryData
    target_padding_mask: torch.Tensor
    target_rtd: torch.Tensor
    last_input_pos_abs: torch.Tensor
    runway: RunwayData
    flight_id: Union[str, List[str]]  # str for single flight, List[str] when batched
    agent_padding_mask: Optional[torch.Tensor] = None
