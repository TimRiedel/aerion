from typing import List

import torch

from data.interface import RunwayData, PredictionSample, TrajectoryData


def stack_trajectory_data(items: List[TrajectoryData]) -> TrajectoryData:
    """Stack a list of TrajectoryData into a single batched TrajectoryData."""
    return TrajectoryData(
        encoder_in=torch.stack([x.encoder_in for x in items]),
        dec_in=torch.stack([x.dec_in for x in items]),
        target=torch.stack([x.target for x in items]),
    )


def stack_runway_data(items: List[RunwayData]) -> RunwayData:
    """Stack a list of RunwayData into a single batched RunwayData."""
    n_points = len(items[0].centerline_points_xy)
    centerline_batched = [
        torch.stack([r.centerline_points_xy[i] for r in items])
        for i in range(n_points)
    ]
    return RunwayData(
        xyz=torch.stack([r.xyz for r in items]),
        bearing=torch.stack([r.bearing for r in items]),
        length=torch.stack([r.length for r in items]),
        centerline_points_xy=centerline_batched,
    )


def collate_samples(samples: List[PredictionSample]) -> PredictionSample:
    """Collate a list of PredictionSample dataclasses into a single batched PredictionSample.

    Stacks all tensor fields along a new batch dimension. flight_id becomes
    a list of strings.
    """
    if not samples:
        raise ValueError("Cannot collate empty list of samples")

    return PredictionSample(
        xyz_positions=stack_trajectory_data([s.xyz_positions for s in samples]),
        xyz_deltas=stack_trajectory_data([s.xyz_deltas for s in samples]),
        trajectory=stack_trajectory_data([s.trajectory for s in samples]),
        target_padding_mask=torch.stack([s.target_padding_mask for s in samples]),
        target_rtd=torch.stack([s.target_rtd for s in samples]),
        last_input_pos_abs=torch.stack([s.last_input_pos_abs for s in samples]),
        runway=stack_runway_data([s.runway for s in samples]),
        flight_id=[s.flight_id for s in samples],
    )
