from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TrajectoryLengths:
    """Trajectory length information for variable-length evaluation.

    Instead of passing mask tensors [B, H] through the system, we pass three
    scalar-per-sample lengths. Losses and metrics receive only the length(s) they need.
    - pred_valid_len: How many steps of the predicted trajectory are valid (not ended)
    - target_valid_len: How many steps of the target trajectory are valid (not padded)
    - eval_valid_len: min(pred_valid_len, target_valid_len) — for displacement/position/horizon metrics
    """
    pred_valid_len: torch.Tensor    # [B] number of valid prediction steps
    target_valid_len: torch.Tensor  # [B] number of valid target steps

    @property
    def eval_valid_len(self) -> torch.Tensor:
        return torch.minimum(self.pred_valid_len, self.target_valid_len)


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
    pred_deltas_abs: torch.Tensor,    # [B, H, 3]
    pred_pos_abs: torch.Tensor,       # [B, H, 3]
    runway_xy: torch.Tensor,          # [B, 2]
    delta_epsilon: float = 1100.0,    # ~70 knots (logical flight speed floor)
    terminal_area_m: float = 5000.0,  # short-final boundary radius in meters
    min_consecutive_steps: int = 3,   # steps below delta_epsilon required to trigger
) -> torch.Tensor:
    """Determine where each predicted trajectory has ended.

    A prediction is considered ended at step t if either of two triggers fires
    while the aircraft is within the terminal area (``terminal_area_m`` of the runway):

    - **Monotonicity trigger**: The 2D distance to the runway starts *increasing*
      (i.e. the aircraft has passed the threshold and is moving away).
    - **Delta-floor trigger**: ``min_consecutive_steps`` consecutive steps all have
      a 2D speed below ``delta_epsilon`` (model has effectively stopped predicting
      meaningful motion).

    Requiring proximity to the runway prevents spurious end-detection far from the
    airport (e.g. early in training when the model outputs near-zero deltas).

    Args:
        pred_deltas_abs: Predicted absolute position deltas [B, H, 3] (dx, dy, dalt)
        pred_pos_abs: Predicted absolute positions [B, H, 3] (x, y, altitude)
        runway_xy: Runway threshold XY position [B, 2]
        delta_epsilon: Speed threshold in m/step below which a step counts as
            "stopped" (default 1100.0 ≈ 70 knots at 30-second intervals)
        terminal_area_m: Radius around the runway threshold that defines the
            short-final zone; triggers are only active inside this zone (default 5000.0 m)
        min_consecutive_steps: Number of consecutive sub-epsilon steps required for
            the delta-floor trigger to fire (default 3)

    Returns:
        pred_valid_len: [B] number of valid prediction steps per sample; equals
            ``first_arrival_idx + 1`` if a trigger fired, else the full horizon H
    """
    B, H, _ = pred_pos_abs.shape
    device = pred_pos_abs.device

    # 1. Compute 2D distance to threshold
    dist_to_runway = torch.norm(pred_pos_abs[:, :, :2] - runway_xy.unsqueeze(1), dim=-1)  # [B, H]

    # 2. Trigger: Monotonicity (distance starts increasing while inside terminal area)
    dist_increased = torch.zeros((B, H), device=device, dtype=torch.bool)
    dist_increased[:, 1:] = dist_to_runway[:, 1:] > dist_to_runway[:, :-1]
    arrived_monotonic = (dist_to_runway < terminal_area_m) & dist_increased

    # 3. Trigger: Delta Floor (min_consecutive_steps consecutive steps below delta_epsilon)
    speed_2d = torch.norm(pred_deltas_abs[:, :, :2], dim=-1)  # [B, H]
    below_epsilon = (speed_2d < delta_epsilon).float()  # [B, H]

    # Use 1D convolution to count consecutive sub-epsilon steps
    if H >= min_consecutive_steps:
        kernel = torch.ones(1, 1, min_consecutive_steps, device=device)
        # Shape: [B, H - min_consecutive_steps + 1]
        consecutive_sum = F.conv1d(below_epsilon.unsqueeze(1), kernel).squeeze(1)
        has_consecutive = consecutive_sum >= min_consecutive_steps

        # Pad back to length H so indices align with pred_pos_abs.
        # Padding at the end: a run starting at step t is detected at step t.
        consecutive_trigger = F.pad(has_consecutive, (0, min_consecutive_steps - 1), value=False)
    else:
        consecutive_trigger = torch.zeros_like(below_epsilon, dtype=torch.bool)

    arrived_delta = (dist_to_runway < terminal_area_m) & consecutive_trigger

    # 4. Combine triggers and find first arrival step
    any_arrival_mask = arrived_monotonic | arrived_delta
    has_arrived = any_arrival_mask.any(dim=1)

    # argmax returns the first index where True occurs
    first_arrival_idx = any_arrival_mask.float().argmax(dim=1)

    # If arrived, valid length = first arrival index + 1; otherwise full horizon H.
    return torch.where(
        has_arrived,
        first_arrival_idx + 1,
        torch.tensor(H, device=device),
    )


def compute_trajectory_lengths(
    pred_deltas_abs: torch.Tensor,
    pred_pos_abs: torch.Tensor,
    runway_xy: torch.Tensor,
    target_pad_mask: torch.Tensor,
    delta_epsilon: float = 1100.0,
    terminal_area_m: float = 5000.0,
    min_consecutive_steps: int = 3,
) -> TrajectoryLengths:
    """Compute pred and target valid lengths and return them as a :class:`TrajectoryLengths`.

    Args:
        pred_deltas_abs: Predicted absolute position deltas [B, H, 3]
        pred_pos_abs: Predicted absolute positions [B, H, 3]
        runway_xy: Runway threshold XY position [B, 2]
        target_pad_mask: Boolean padding mask for target sequence [B, H],
            True = padded (invalid) position
        delta_epsilon: Speed threshold for the delta-floor trigger (m/step)
        terminal_area_m: Terminal area radius around the runway (m)
        min_consecutive_steps: Consecutive sub-epsilon steps to trigger end detection

    Returns:
        TrajectoryLengths with pred_valid_len and target_valid_len [B] each
    """
    pred_valid_len = compute_pred_valid_len(
        pred_deltas_abs,
        pred_pos_abs,
        runway_xy,
        delta_epsilon=delta_epsilon,
        terminal_area_m=terminal_area_m,
        min_consecutive_steps=min_consecutive_steps,
    )

    # target_pad_mask: True = padded; valid length = number of non-padded steps
    target_valid_len = (~target_pad_mask).sum(dim=1)

    return TrajectoryLengths(pred_valid_len=pred_valid_len, target_valid_len=target_valid_len)


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
