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
