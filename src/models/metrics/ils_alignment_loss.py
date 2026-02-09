import torch
import math
from torch import nn
from torch.nn import functional as F


class ILSAlignmentLoss(nn.Module):
    """
    ILS Alignment Loss for aircraft approach trajectories.
    
    Computes alignment penalties based on Instrument Landing System (ILS) constraints
    for the final waypoints of predicted trajectories. The loss penalizes deviations from:
    1. Localizer (horizontal centerline): Maximum deviation ±2.5° from runway centerline
    2. Glideslope (vertical): Maximum deviation ±0.7° from standard 3° glideslope
    
    The loss is computed only for the last `num_final_waypoints` waypoints of each trajectory
    and uses Huber loss to penalize deviations beyond the maximum allowed angles.
    """
    
    def __init__(
        self,
        num_final_waypoints: int = 4,
        epsilon: float = 1e-6,
        localizer_max_deviation_deg: float = 2.5,
        glideslope_standard_deg: float = 3.0,
        glideslope_max_deviation_deg: float = 0.7,
        heading_max_deviation_deg: float = 5.0,
    ):
        """
        Initialize ILS Alignment Loss.
        
        Args:
            num_final_waypoints: Number of final waypoints to consider for alignment (default: 4).
                                 Only the last N waypoints of each trajectory are evaluated.
            epsilon: Small value for numerical stability (currently unused, kept for future use).
            localizer_max_deviation_deg: Maximum allowed horizontal deviation from centerline in degrees (default: 2.5°).
            glideslope_standard_deg: Standard glideslope angle in degrees (default: 3.0°).
            glideslope_max_deviation_deg: Maximum allowed deviation from standard glideslope in degrees (default: 0.7°).
        """
        super().__init__()
        self.num_final_waypoints = num_final_waypoints
        self.epsilon = epsilon
        self.localizer_max_deviation_rad = math.radians(localizer_max_deviation_deg)
        self.glideslope_standard_rad = math.radians(glideslope_standard_deg)
        self.glideslope_max_deviation_rad = math.radians(glideslope_max_deviation_deg)
        self.heading_max_deviation_rad = math.radians(heading_max_deviation_deg)

    def _compute_localizer_deviation(
        self,
        positions_xy: torch.Tensor,
        threshold_xy: torch.Tensor,
        runway_bearing: torch.Tensor,
        runway_length_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute angular deviation from localizer centerline.
        
        The localizer antenna is positioned at the far end of the runway (runway_length_m away
        from threshold in the runway direction). The angular deviation is computed as the angle
        between the approach direction and the vector from antenna to aircraft position.
        
        Args:
            positions_xy: Aircraft positions [N, 2] (x, y coordinates in meters)
            threshold_xy: Runway threshold position [2] (x, y coordinates in meters)
            runway_bearing: Runway bearing [2] (sin(bearing), cos(bearing))
            runway_length_m: Runway length in meters (scalar)
            
        Returns:
            Angular deviations in radians [N] (positive = right of centerline, negative = left)
        """
        # Compute localizer antenna position
        # Antenna is positioned at the far end of the runway (runway_length_m away from threshold)
        runway_direction_xy = torch.stack([
            runway_bearing[0],  # sin(bearing)
            runway_bearing[1],  # cos(bearing)
        ])  # [2]
        antenna_offset = runway_length_m * runway_direction_xy  # [2]
        antenna_xy = threshold_xy + antenna_offset  # [2]
        
        antenna_to_aircraft = positions_xy - antenna_xy.unsqueeze(0) 
        
        # Approach direction is opposite to runway bearing
        approach_direction = torch.stack([
            -runway_bearing[0], # -sin(bearing)
            -runway_bearing[1]  # -cos(bearing)
        ]) # [2]

        # Calculate Along-track (x) and Cross-track (y) components
        # Using the dot product for 'x' and the 2D cross-product for 'y'
        # x: how far "out" the plane is on the approach line
        # y: how far "left/right" it is from that line
        x = (antenna_to_aircraft[:, 0] * approach_direction[0] + 
            antenna_to_aircraft[:, 1] * approach_direction[1])
        
        y = (antenna_to_aircraft[:, 0] * approach_direction[1] - 
            antenna_to_aircraft[:, 1] * approach_direction[0])

        # Use atan2 to get the stable angle in radians (-pi to pi)
        angular_deviation = torch.atan2(y, x)
        return angular_deviation

    def _compute_glideslope_deviation(
        self,
        positions_xyz: torch.Tensor,
        threshold_xyz: torch.Tensor,
        runway_bearing: torch.Tensor,
        glideslope_standard_rad: float,
    ) -> torch.Tensor:
        """
        Compute angular deviation from standard glideslope angle.
        
        The glideslope antenna is positioned 304.8m away from threshold in the runway direction
        and 15.25m below the threshold altitude (screen height). The deviation is computed as the
        difference between the actual approach angle and the standard glideslope angle.
        
        Args:
            positions_xyz: Aircraft positions [N, 3] (x, y, altitude in meters)
            threshold_xyz: Runway threshold position [3] (x, y, altitude in meters)
            runway_bearing: Runway bearing [2] (sin(bearing), cos(bearing))
            glideslope_standard_rad: Standard glideslope angle in radians
            
        Returns:
            Angular deviations in radians [N] (positive = above glideslope, negative = below)
        """
        # Compute glideslope antenna position
        # Antenna is 304.8m away from threshold in runway direction and 15.25m down
        runway_direction_xy = torch.stack([
            runway_bearing[0],  # sin(bearing)
            runway_bearing[1],  # cos(bearing)
        ])  # [2]
        
        # Antenna position: threshold + 304.8m in runway direction, -15.25m in altitude (screen height)
        screen_height_m = torch.tensor(15.25, device=positions_xyz.device)
        antenna_distance_m = torch.tensor(304.8, device=positions_xyz.device)
        antenna_offset = torch.stack([
            antenna_distance_m * runway_direction_xy[0],  # x offset
            antenna_distance_m * runway_direction_xy[1],  # y offset
            -screen_height_m,  # z offset (down)
        ])  # [3]
        antenna_xyz = threshold_xyz + antenna_offset  # [3]
        
        # Compute deviation from antenna position
        antenna_to_aircraft = positions_xyz - antenna_xyz.unsqueeze(0)
        approach_direction = torch.stack([
            -runway_bearing[0], # -sin(bearing)
            -runway_bearing[1],  # -cos(bearing)
            torch.tensor(0.0, device=antenna_to_aircraft.device),
        ]) # [3]

        along_track = torch.sum(antenna_to_aircraft * approach_direction, dim=1) # [N]
        height = antenna_to_aircraft[:, 2] # [N]

        # atan2(height, distance) gives the angle relative to the horizon
        actual_gs_rad = torch.atan2(height, along_track) # [N]
        angular_deviation = actual_gs_rad - glideslope_standard_rad # [N]
        return angular_deviation

    def _compute_track_alignment_deviation(
        self,
        positions_xy: torch.Tensor,
        runway_bearing: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute heading alignment angular deviations using atan2.

        Args:
            positions_xy: Aircraft positions [N, 2] (x, y coordinates in meters)
            runway_bearing: Runway bearing [2] (sin(bearing), cos(bearing))

        Returns:
            Angular deviations in radians [N-1] (positive = right of centerline, negative = left)
        """
        directions = positions_xy[1:] - positions_xy[:-1]  # [N-1, 2]
        
        # Normalize directions to unit vectors
        dir_lengths = torch.norm(directions, dim=-1, keepdim=True) + self.epsilon
        dir_normalized = directions / dir_lengths  # [N-1, 2]
        
        # Compute cross and dot products with runway bearing
        # runway_bearing is [sin(θ), cos(θ)]
        dot_product = (dir_normalized * runway_bearing).sum(dim=-1)  # Along-track
        cross_product = (
            dir_normalized[:, 0] * runway_bearing[1] - 
            dir_normalized[:, 1] * runway_bearing[0]
        )  # Cross-track (2D cross product)
        
        # atan2 gives signed angle in radians [-π, π]
        angular_deviation = torch.atan2(cross_product, dot_product)  # [N-1]
        return angular_deviation
        

    def _compute_huber_deviation_loss(
        self,
        angular_deviation_rad: torch.Tensor, 
        max_angle_rad: float,
    ) -> torch.Tensor:
        """
        Compute Huber loss for angular deviations.
        
        Uses standard Huber loss formulation where delta is the maximum allowable deviation.
        Loss is quadratic within the safe boundary and linear beyond it.
        
        Args:
            angular_deviation_rad: Angular deviations in radians [N]
            max_angle_rad: Maximum allowed deviation angle in radians (delta parameter)
            
        Returns:
            Huber loss values [N]
        """
        delta = max_angle_rad
        abs_dev = torch.abs(angular_deviation_rad)
        
        quadratic = torch.min(abs_dev, torch.tensor(delta))
        linear = abs_dev - quadratic
        
        # Huber formula: 0.5 * sq(quadratic) + delta * linear
        return 0.5 * quadratic**2 + delta * linear
    

    def forward(
        self,
        pred_abs: torch.Tensor,
        target_abs: torch.Tensor,
        target_pad_mask: torch.Tensor,
        runway: dict,
    ) -> torch.Tensor:
        """
        Compute ILS alignment loss for predicted trajectories.
        
        Computes localizer and glideslope alignment penalties for the last `num_final_waypoints`
        waypoints of each trajectory in the batch. The loss is averaged across waypoints and batches,
        then scaled by `scale_factor`.
        
        Args:
            pred_abs: Predicted absolute positions [B, H, 3] (x, y, altitude in meters)
            target_abs: Target absolute positions [B, H, 3] (not used, kept for API consistency)
            target_pad_mask: Padding mask [B, H] (True for padded positions, False for valid positions)
            runway: Dictionary containing:
                - "bearing": tensor [B, 2] with [sin(bearing), cos(bearing)] for runway direction
                - "xyz": tensor [B, 3] with threshold position [x, y, altitude] in meters
                - "length": tensor [B] with runway length in meters
            
        Returns:
            Scalar loss value (averaged across batch and waypoints, scaled by scale_factor)
        """
        runway_bearing = runway["bearing"]  # [B, 2]
        threshold_xyz = runway["xyz"]  # [B, 3]
        runway_length_m = runway["length"]  # [B]
        batch_size = pred_abs.size(0)
        losses = []
        
        # Find valid trajectory lengths (excluding padding)
        valid_lengths = (~target_pad_mask).sum(dim=1)  # [B]
        
        # Compute alignment loss for each trajectory in batch
        for b in range(batch_size):
            end_idx = valid_lengths[b].item()
            start_idx = max(0, end_idx - self.num_final_waypoints)
            
            final_positions_xyz = pred_abs[b, start_idx:end_idx, :]  # [N, 3]
            final_positions_xy = final_positions_xyz[:, :2]  # [N, 2]
            
            rwy_bearing = runway_bearing[b]  # [2] (sin, cos)
            thresh_xyz = threshold_xyz[b]  # [3]
            thresh_xy = thresh_xyz[:2]
            rwy_length = runway_length_m[b]  # scalar
            
            localizer_deviations = self._compute_localizer_deviation(
                final_positions_xy,
                thresh_xy,
                rwy_bearing,
                rwy_length,
            )  # [N]
            
            glideslope_deviations = self._compute_glideslope_deviation(
                final_positions_xyz,
                thresh_xyz,
                rwy_bearing,
                self.glideslope_standard_rad,
            )  # [N]

            track_deviations = self._compute_track_alignment_deviation(
                final_positions_xy,
                rwy_bearing,
            )  # [N-1] - computed between consecutive positions
            
            localizer_penalties = self._compute_huber_deviation_loss(
                localizer_deviations,
                self.localizer_max_deviation_rad,
            )  # [N]
            
            glideslope_penalties = self._compute_huber_deviation_loss(
                glideslope_deviations,
                self.glideslope_max_deviation_rad,
            )  # [N]

            track_penalties = self._compute_huber_deviation_loss(
                track_deviations,
                self.heading_max_deviation_rad,
            )  # [N-1]
            # Pad track penalties at the beginning to match size [N-1] -> [N]
            track_penalties_padded = torch.cat([
                torch.zeros(1, device=track_penalties.device),
                track_penalties
            ])  # [N]
            
            # Combined loss: sum of localizer, glideslope, and track penalties
            combined_loss = localizer_penalties + glideslope_penalties  # [N]
            losses.append(combined_loss.mean())
        
        total_loss = torch.stack(losses).mean()
        return total_loss
