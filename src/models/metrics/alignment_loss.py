import torch
from torch import nn


class RunwayAlignmentLoss(nn.Module):
    """
    Runway Alignment Loss using cosine similarity.
    
    Penalizes trajectories that are not aligned with the runway direction during
    the final approach phase (last N waypoints before touchdown).
    
    The loss uses cosine similarity between:
    - The direction vectors of the predicted trajectory segments
    - The runway direction vector (derived from bearing sin/cos)
    
    Loss formulation:
    - Perfect alignment (cos_sim = 1): loss = 0
    - Perpendicular (cos_sim = 0): loss = 0.5  
    - Opposite direction (cos_sim = -1): loss = 1 (strongly penalized)
    
    Formula: loss = (1 - cos_sim) / 2.0 (per segment), averaged over valid segments
    """
    
    def __init__(
        self,
        num_final_waypoints: int = 4,
        scale_factor: float = 1.0,
        epsilon: float = 1e-6,
    ):
        """
        Initialize Runway Alignment Loss.
        
        Args:
            num_final_waypoints: Number of final waypoints to consider for alignment (default: 4).
                                 With 30s spacing, 4 waypoints = last 90 seconds of approach =~ 3.4NM.
            scale_factor: Scaling factor to balance with ADE/FDE losses.
            epsilon: Small value for numerical stability in normalization.
        """
        super().__init__()
        self.num_final_waypoints = num_final_waypoints
        self.scale_factor = scale_factor
        self.epsilon = epsilon
    
    def forward(
        self,
        pred_abs: torch.Tensor,
        target_abs: torch.Tensor,
        target_pad_mask: torch.Tensor,
        runway: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute runway alignment loss.
        
        Args:
            pred_abs: Predicted absolute positions [B, H, 3] (x, y, altitude in meters)
            target_abs: Target absolute positions [B, H, 3] (not used, kept for API consistency)
            target_pad_mask: Padding mask [B, H] (True for padded positions)
            runway: Tensor [B, 4] with [threshold_x, threshold_y, bearing_sin, bearing_cos]
            
        Returns:
            Scalar loss value (scaled)
        """
        batch_size = pred_abs.size(0)
        losses = []
        
        # Runway direction vector [B, 2]
        rwy_bearing_sin = runway[:, 2]
        rwy_bearing_cos = runway[:, 3]
        runway_dirs = torch.stack([rwy_bearing_sin, rwy_bearing_cos], dim=-1)
        runway_dirs = runway_dirs / (torch.norm(runway_dirs, dim=-1, keepdim=True) + self.epsilon)

        # Find valid trajectory lengths (excluding padding)
        valid_lengths = (~target_pad_mask).sum(dim=1)  # [B]
        
        # Compute alignment loss for each trajectory in batch
        for b in range(batch_size):
            end_idx = valid_lengths[b].item()
            start_idx = end_idx - self.num_final_waypoints
            final_positions = pred_abs[b, start_idx:end_idx, :2]  # [N, 2]

            dirs = final_positions[1:] - final_positions[:-1]  # [N-1, 2]
            dirs_norm = torch.norm(dirs, dim=-1, keepdim=True) + self.epsilon  # [N-1, 1]
            dirs_unit = dirs / dirs_norm  # [N-1, 2]
            
            # Cosine similarity: dot product of normalized vectors
            flight_runway_dir = runway_dirs[b:b+1, :]  # [1, 2]
            cos_sim = (dirs_unit * flight_runway_dir).sum(dim=-1)  # [N-1]
            # Loss: squared difference between cosine similarity and 1
            alignment_loss = (1.0 - cos_sim)**2
            
            # Average over all waypoints
            losses.append(alignment_loss.mean())
        
        total_loss = torch.stack(losses).mean() * self.scale_factor
        return total_loss
