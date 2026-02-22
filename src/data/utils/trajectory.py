import torch
from data.transforms.normalize import Denormalizer


def reconstruct_absolute_from_deltas(
    input_traj: torch.Tensor,
    target_deltas: torch.Tensor,
    pred_deltas: torch.Tensor,
    denormalize_inputs: Denormalizer,
    denormalize_target_deltas: Denormalizer,
    target_pad_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reconstruct absolute positions from normalized deltas.
    
    This function:
    1. Denormalizes input trajectory and deltas
    2. Masks deltas before cumulative sum
    3. Reconstructs absolute positions by integrating deltas from last known position
    
    Args:
        input_traj: Normalized input trajectory [batch_size, input_seq_len, 8]
        target_deltas: Normalized target deltas [batch_size, horizon_seq_len, 3]
        pred_deltas: Normalized prediction deltas [batch_size, horizon_seq_len, 3]
        denormalize_inputs: Denormalizer for input trajectories
        denormalize_target_deltas: Denormalizer for target/prediction deltas
        target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
        
    Returns:
        input_abs: Denormalized input absolute positions [batch_size, input_seq_len, 8]
        target_abs: Reconstructed target absolute positions [batch_size, horizon_seq_len, 3]
        pred_abs: Reconstructed prediction absolute positions [batch_size, horizon_seq_len, 3]
    """
    # 1. Denormalize inputs and deltas
    input_abs = denormalize_inputs(input_traj)
    pred_deltas_m = denormalize_target_deltas(pred_deltas)
    target_deltas_m = denormalize_target_deltas(target_deltas)
    
    # 2. Mask deltas before cumulative sum
    active_mask = ~target_pad_mask  # [B, H]
    active_mask_expanded = active_mask.unsqueeze(-1)  # [B, H, 1]
    pred_deltas_m = pred_deltas_m * active_mask_expanded
    target_deltas_m = target_deltas_m * active_mask_expanded
    
    # 3. Get last known absolute position (p0)
    p0 = input_abs[:, -1, :3].unsqueeze(1)  # Shape: [B, 1, 3]
    
    # 4. Integrate: Reconstruct absolute positions from deltas
    pred_abs = p0 + torch.cumsum(pred_deltas_m, dim=1)
    target_abs = p0 + torch.cumsum(target_deltas_m, dim=1)
    
    return input_abs, target_abs, pred_abs

def compute_rtd(
    horizon_traj: torch.Tensor,
    padding_mask: torch.Tensor,
    runway_xyz: torch.Tensor,
    runway_bearing: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Remaining Track Distance (RTD) for batched or unbatched trajectories.
    
    RTD is the cumulative 2D distance between consecutive valid waypoints. If
    add_distance_to_threshold is True, the result is adjusted for the last valid point
    relative to the runway threshold:
    - If the last point is on the approach side (before the threshold), the along-track
      distance to the threshold is added.
    - If the last point has overshot the threshold (landing side), the along-track
      distance past the threshold is subtracted.
    
    Args:
        horizon_traj: Trajectory positions [H, 3] or [B, H, 3] (x, y, altitude) - absolute positions
        padding_mask: Padding mask [H] or [B, H] (True for padded/invalid positions)
        runway_xyz: Runway threshold position [3] or [B, 3] (x, y, altitude)
        runway_bearing: Runway bearing [2] or [B, 2] (sin(θ), cos(θ))
        
    Returns:
        traj_distance: Cumulative trajectory distance [B]
        rtd: Remaining track distance, including distance to threshold [B]
    """
    unbatched = horizon_traj.dim() == 2
    if unbatched:
        horizon_traj = horizon_traj.unsqueeze(0)
        padding_mask = padding_mask.unsqueeze(0)
        runway_xyz = runway_xyz.unsqueeze(0)
        runway_bearing = runway_bearing.unsqueeze(0)

    valid_mask = ~padding_mask  # [B, H]

    # 1. Cumulative inter-waypoint 2D distances
    deltas = horizon_traj[:, 1:, :2] - horizon_traj[:, :-1, :2]  # [B, H-1, 2]
    segment_dists = torch.norm(deltas, dim=-1)  # [B, H-1]

    # A segment is valid up to the last valid point (padding is always trailing)
    segment_valid = valid_mask[:, 1:]  # [B, H-1]
    cumulative_dist = (segment_dists * segment_valid).sum(dim=1)  # [B]

    # 2. Find last valid point position
    last_valid_idx = (valid_mask.sum(dim=1) - 1).clamp(min=0)  # [B]
    batch_indices = torch.arange(horizon_traj.size(0), device=horizon_traj.device)
    last_valid_pos_xy = horizon_traj[batch_indices, last_valid_idx, :2]  # [B, 2]

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

