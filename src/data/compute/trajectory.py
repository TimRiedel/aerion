from dataclasses import dataclass

import torch


@dataclass
class TrajectoryLengths:
    """Trajectory length information for variable-length evaluation.

    Instead of passing mask tensors [B, H] through the system, we pass three
    scalar-per-sample lengths. Losses and metrics receive only the length(s) they need.
    - pred_valid_len: How many steps of the predicted trajectory are valid (not ended)
    - target_valid_len: How many steps of the target trajectory are valid (not padded)
    """
    pred_valid_len: torch.Tensor    # [B] number of valid prediction steps
    target_valid_len: torch.Tensor  # [B] number of valid target steps


def length_to_mask(length: torch.Tensor, max_len: int) -> torch.Tensor:
    """Convert per-sample lengths to a boolean validity mask.

    Args:
        length: Valid length per sample [B]
        max_len: Maximum sequence length (H)

    Returns:
        Mask [B, H] where True = valid position, False = padded
    """
    return torch.arange(max_len, device=length.device).unsqueeze(0) < length.unsqueeze(1)


def compute_pred_valid_len(
    pred_pos_abs: torch.Tensor,
    runway_xy: torch.Tensor,
    arrival_threshold_m: float,
) -> torch.Tensor:
    """Determine where each predicted trajectory has arrived at the runway threshold.

    A prediction is considered ended at step t if the 2D distance to the runway
    threshold first drops below arrival_threshold_m.

    Args:
        pred_pos_abs: Predicted absolute positions [B, H, 3] (x, y, altitude)
        runway_xy: Runway threshold XY position [B, 2]
        arrival_threshold_m: Distance threshold in meters to declare arrival

    Returns:
        pred_valid_len: [B] number of valid prediction steps per sample
    """
    B, H, _ = pred_pos_abs.shape
    dist_to_runway = torch.norm(pred_pos_abs[:, :, :2] - runway_xy.unsqueeze(1), dim=-1)  # [B, H]
    arrived = dist_to_runway < arrival_threshold_m  # [B, H]
    has_arrived = arrived.any(dim=1)  # [B]
    first_arrival = arrived.float().argmax(dim=1)  # [B]
    pred_valid_len = torch.where(
        has_arrived, first_arrival, torch.tensor(H, device=pred_pos_abs.device)
    )
    return pred_valid_len


def compute_trajectory_lengths(
    pred_pos_abs: torch.Tensor,
    runway_xy: torch.Tensor,
    target_pad_mask: torch.Tensor,
    arrival_threshold_m: float = 750.0,
) -> TrajectoryLengths:
    """Compute all trajectory lengths from predicted positions and target padding.

    Args:
        pred_pos_abs: Predicted absolute positions [B, H, 3]
        runway_xy: Runway threshold XY position [B, 2]
        target_pad_mask: Target padding mask [B, H] (True = padded)
        arrival_threshold_m: Distance threshold in meters to declare arrival

    Returns:
        TrajectoryLengths with pred_valid_len, target_valid_len
    """
    pred_valid_len = compute_pred_valid_len(pred_pos_abs, runway_xy, arrival_threshold_m)
    target_valid_len = (~target_pad_mask).sum(dim=1)  # [B]

    return TrajectoryLengths(
        pred_valid_len=pred_valid_len,
        target_valid_len=target_valid_len,
    )


def reconstruct_positions_from_deltas(
    input_pos: torch.Tensor,
    deltas: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct positions from deltas. Works with normalized or absolute input positions or deltas.

    The model predicts freely for all steps — no masking is applied to deltas.

    Args:
        input_pos: Normalized or absolute input positions [B, T, 3]
        deltas: Normalized or absolute deltas [B, T, 3]

    Returns:
        Reconstructed positions [B, T, 3]
    """
    last_position = input_pos[:, -1, :3].unsqueeze(1)
    positions = last_position + torch.cumsum(deltas, dim=1)
    return positions


def compute_rtd(
    pos_abs: torch.Tensor,
    valid_len: torch.Tensor,
    runway_xyz: torch.Tensor,
    runway_bearing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Remaining Track Distance (RTD) for batched trajectories.

    RTD is the cumulative 2D distance between consecutive valid waypoints, adjusted
    for the last valid point relative to the runway threshold:
    - If the last point is on the approach side (before the threshold), the along-track
      distance to the threshold is added.
    - If the last point has overshot the threshold (landing side), the along-track
      distance past the threshold is subtracted.

    Args:
        pos_abs: Trajectory positions [B, H, 3] (x, y, altitude) - absolute positions
        valid_len: Number of valid waypoints per sample [B]
        runway_xyz: Runway threshold position [B, 3] (x, y, altitude)
        runway_bearing: Runway bearing [B, 2] (sin(θ), cos(θ))

    Returns:
        traj_distance: Cumulative trajectory distance [B]
        rtd: Remaining track distance, including distance to threshold [B]
    """
    H = pos_abs.size(1)
    valid_mask = length_to_mask(valid_len, H)  # [B, H]

    # 1. Cumulative inter-waypoint 2D distances
    deltas = pos_abs[:, 1:, :2] - pos_abs[:, :-1, :2]  # [B, H-1, 2]
    segment_dists = torch.norm(deltas, dim=-1)  # [B, H-1]

    # A segment is valid up to the last valid point (padding is always trailing)
    segment_valid = valid_mask[:, 1:]  # [B, H-1]
    cumulative_dist = (segment_dists * segment_valid).sum(dim=1)  # [B]

    # 2. Find last valid point position
    last_valid_idx = (valid_len - 1).clamp(min=0)  # [B]
    batch_indices = torch.arange(pos_abs.size(0), device=pos_abs.device)
    last_valid_pos_xy = pos_abs[batch_indices, last_valid_idx, :2]  # [B, 2]

    # 3. Compute along-track distance from threshold to last valid point
    #    in runway-relative coordinates (positive = landing side, negative = approach side)
    runway_xy = runway_xyz[:, :2]  # [B, 2]
    translated = last_valid_pos_xy - runway_xy  # [B, 2]
    sin_theta = runway_bearing[:, 0]  # [B]
    cos_theta = runway_bearing[:, 1]  # [B]
    # Approach side (along_track < 0): adds distance to threshold
    # Landing side  (along_track > 0): subtracts overshoot past threshold
    along_track = sin_theta * translated[:, 0] + cos_theta * translated[:, 1]  # [B]

    # 4. Compute RTD
    rtd = cumulative_dist - along_track  # [B]

    return cumulative_dist, rtd
