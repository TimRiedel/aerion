import torch

def reconstruct_positions_from_deltas(
    input_pos: torch.Tensor,
    deltas: torch.Tensor,
    target_pad_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct positions from deltas. Works with normalized or absolute input positions or deltas.

    Args:
        input_pos: Normalized or absolute input positions [B, T, 3]
        deltas: Normalized or absolute deltas [B, T, 3]
        target_pad_mask: Padding mask [B, T]

    Returns:
        Reconstructed positions [B, T, 3]
    """
    active_mask = ~target_pad_mask
    active_mask_expanded = active_mask.unsqueeze(-1)
    deltas_masked = deltas * active_mask_expanded
    
    last_position = input_pos[:, -1, :3].unsqueeze(1)
    positions = last_position + torch.cumsum(deltas_masked, dim=1)
    
    return positions


def compute_rtd(
    target_pos_abs: torch.Tensor,
    target_pad_mask: torch.Tensor,
    runway_xyz: torch.Tensor,
    runway_bearing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Remaining Track Distance (RTD) for batched or unbatched trajectories.

    RTD is the cumulative 2D distance between consecutive valid waypoints, adjusted
    for the last valid point relative to the runway threshold:
    - If the last point is on the approach side (before the threshold), the along-track
      distance to the threshold is added.
    - If the last point has overshot the threshold (landing side), the along-track
      distance past the threshold is subtracted.

    Args:
        target_pos_abs: Trajectory positions [H, 3] or [B, H, 3] (x, y, altitude) - absolute positions
        target_pad_mask: Padding mask [H] or [B, H] (True for padded/invalid positions)
        runway_xyz: Runway threshold position [3] or [B, 3] (x, y, altitude)
        runway_bearing: Runway bearing [2] or [B, 2] (sin(θ), cos(θ))

    Returns:
        traj_distance: Cumulative trajectory distance [B]
        rtd: Remaining track distance, including distance to threshold [B]
    """
    unbatched = target_pos_abs.dim() == 2
    if unbatched:
        target_pos_abs = target_pos_abs.unsqueeze(0)
        target_pad_mask = target_pad_mask.unsqueeze(0)
        runway_xyz = runway_xyz.unsqueeze(0)
        runway_bearing = runway_bearing.unsqueeze(0)

    valid_mask = ~target_pad_mask  # [B, H]

    # 1. Cumulative inter-waypoint 2D distances
    deltas = target_pos_abs[:, 1:, :2] - target_pos_abs[:, :-1, :2]  # [B, H-1, 2]
    segment_dists = torch.norm(deltas, dim=-1)  # [B, H-1]

    # A segment is valid up to the last valid point (padding is always trailing)
    segment_valid = valid_mask[:, 1:]  # [B, H-1]
    cumulative_dist = (segment_dists * segment_valid).sum(dim=1)  # [B]

    # 2. Find last valid point position
    last_valid_idx = (valid_mask.sum(dim=1) - 1).clamp(min=0)  # [B]
    batch_indices = torch.arange(target_pos_abs.size(0), device=target_pos_abs.device)
    last_valid_pos_xy = target_pos_abs[batch_indices, last_valid_idx, :2]  # [B, 2]

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

    if unbatched:
        cumulative_dist = cumulative_dist.squeeze(0)
        rtd = rtd.squeeze(0)
    return cumulative_dist, rtd

