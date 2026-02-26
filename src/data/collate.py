from typing import List, Optional

import torch

from data.interface import RunwayData, PredictionSample, TrajectoryData


def pad_trajectory_data_agents(td: TrajectoryData, n_agents: int, max_n_agents: int) -> TrajectoryData:
    """Pad TrajectoryData agent dimension from n_agents to max_n_agents with zeros."""
    if n_agents >= max_n_agents:
        return td
    pad_size = max_n_agents - n_agents
    return TrajectoryData(
        encoder_in=torch.nn.functional.pad(td.encoder_in, (0, 0, 0, pad_size), value=0.0),
        dec_in=torch.nn.functional.pad(td.dec_in, (0, 0, 0, pad_size), value=0.0),
        target=torch.nn.functional.pad(td.target, (0, 0, 0, pad_size), value=0.0),
    )

def pad_runway_data_agents(rw: RunwayData, n_agents: int, max_n_agents: int) -> RunwayData:
    """Pad RunwayData agent dimension from n_agents to max_n_agents with zeros."""
    if n_agents >= max_n_agents:
        return rw
    pad_size = max_n_agents - n_agents
    return RunwayData(
        xyz=torch.nn.functional.pad(rw.xyz, (0, 0, 0, pad_size), value=0.0),
        bearing=torch.nn.functional.pad(rw.bearing, (0, 0, 0, pad_size), value=0.0),
        length=torch.nn.functional.pad(rw.length, (0, pad_size), value=0.0),
        centerline_points_xy=[
            torch.nn.functional.pad(cp, (0, 0, 0, pad_size), value=0.0)
            for cp in rw.centerline_points_xy
        ],
    )

def pad_sample_to_max_agents(
    sample: PredictionSample, max_n_agents: int
) -> tuple[PredictionSample, torch.Tensor]:
    """
    Pad a multi-agent PredictionSample to max_n_agents. Returns (padded_sample, agent_padding_mask).
    agent_padding_mask: [N_max], True = padded slot.
    """
    n_agents = sample.trajectory.encoder_in.size(1)
    if n_agents >= max_n_agents:
        mask = torch.zeros(max_n_agents, dtype=torch.bool, device=sample.trajectory.encoder_in.device)
        return sample, mask

    padded_xyz = pad_trajectory_data_agents(sample.xyz_positions, n_agents, max_n_agents)
    padded_deltas = pad_trajectory_data_agents(sample.xyz_deltas, n_agents, max_n_agents)
    padded_traj = pad_trajectory_data_agents(sample.trajectory, n_agents, max_n_agents)

    # target_padding_mask [H, N]: pad with True (invalid)
    target_mask = sample.target_padding_mask
    target_mask_padded = torch.nn.functional.pad(target_mask, (0, max_n_agents - n_agents), value=True)

    # target_rtd [N], last_input_pos_abs [N, 3]: pad with 0
    target_rtd_padded = torch.nn.functional.pad(sample.target_rtd, (0, max_n_agents - n_agents), value=0.0)
    last_pos_padded = torch.nn.functional.pad(sample.last_input_pos_abs, (0, 0, 0, max_n_agents - n_agents), value=0.0)

    padded_runway = pad_runway_data_agents(sample.runway, n_agents, max_n_agents)

    flight_ids = sample.flight_id if isinstance(sample.flight_id, list) else [sample.flight_id]
    flight_ids_padded = flight_ids + [""] * (max_n_agents - n_agents)

    agent_padding_mask = torch.zeros(max_n_agents, dtype=torch.bool, device=sample.trajectory.encoder_in.device)
    agent_padding_mask[n_agents:] = True

    padded_sample = PredictionSample(
        xyz_positions=padded_xyz,
        xyz_deltas=padded_deltas,
        trajectory=padded_traj,
        target_padding_mask=target_mask_padded,
        target_rtd=target_rtd_padded,
        last_input_pos_abs=last_pos_padded,
        runway=padded_runway,
        flight_id=flight_ids_padded,
        agent_padding_mask=None,
    )
    return padded_sample, agent_padding_mask


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


def is_multi_agent(sample: PredictionSample) -> bool:
    """True if sample has agent dimension (shape [T, N, F])."""
    return sample.trajectory.encoder_in.ndim == 3


def collate_samples(
    samples: List[PredictionSample],
    max_n_agents: Optional[int] = None,
) -> PredictionSample:
    """Collate a list of PredictionSample dataclasses into a single batched PredictionSample.

    Stacks all tensor fields along a new batch dimension. flight_id becomes
    a list of strings.

    When max_n_agents is set and samples are multi-agent, pads each sample to
    max_n_agents along the agent dimension (zero padding) and adds
    agent_padding_mask [B, N_max] with True for padded slots.
    """
    if not samples:
        raise ValueError("Cannot collate empty list of samples")

    padded_samples, agent_masks = [], None
    if max_n_agents is not None and is_multi_agent(samples[0]):
        agent_masks = []
        for sample in samples:
            padded_sample, agent_mask = pad_sample_to_max_agents(sample, max_n_agents)
            padded_samples.append(padded_sample)
            agent_masks.append(agent_mask)
        samples = padded_samples
        agent_masks = torch.stack(agent_masks, dim=0)

    return PredictionSample(
        xyz_positions=stack_trajectory_data([s.xyz_positions for s in samples]),
        xyz_deltas=stack_trajectory_data([s.xyz_deltas for s in samples]),
        trajectory=stack_trajectory_data([s.trajectory for s in samples]),
        target_padding_mask=torch.stack([s.target_padding_mask for s in samples]),
        agent_padding_mask=agent_masks,
        target_rtd=torch.stack([s.target_rtd for s in samples]),
        last_input_pos_abs=torch.stack([s.last_input_pos_abs for s in samples]),
        runway=stack_runway_data([s.runway for s in samples]),
        flight_id=[s.flight_id for s in samples],
    )
