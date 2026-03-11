import math

import torch
from torch import nn


def _squared_deviation_loss(angular_deviation_rad: torch.Tensor, max_angle_rad: float) -> torch.Tensor:
    return angular_deviation_rad**2


def _huber_deviation_loss(angular_deviation_rad: torch.Tensor, max_angle_rad: float) -> torch.Tensor:
    delta = max_angle_rad
    abs_dev = torch.abs(angular_deviation_rad)
    quadratic = torch.min(abs_dev, torch.tensor(delta))
    linear = abs_dev - quadratic
    return 0.5 * quadratic**2 + delta * linear


def _get_loss_fn(loss_type: str):
    if loss_type == "quadratic":
        return _squared_deviation_loss
    elif loss_type == "huber":
        return _huber_deviation_loss
    raise ValueError(f"Invalid loss type: {loss_type}")


class LocalizerAlignmentLoss(nn.Module):
    """
    Localizer Alignment Loss for aircraft approach trajectories.

    Computes alignment penalties based on ILS localizer (horizontal centerline) and
    track heading constraints for the final waypoints of predicted trajectories.
    Penalizes deviations from:
    1. Localizer (horizontal centerline): Maximum deviation ±2.5° from runway centerline
    2. Track alignment: Maximum heading deviation from runway bearing

    The loss is computed only for the last `num_final_waypoints` waypoints of each trajectory.
    """

    def __init__(
        self,
        num_final_waypoints: int = 4,
        epsilon: float = 1e-6,
        localizer_max_deviation_deg: float = 2.5,
        heading_max_deviation_deg: float = 10.0,
        loss_type: str = "quadratic",
    ):
        super().__init__()
        self.num_final_waypoints = num_final_waypoints
        self.epsilon = epsilon
        self.localizer_max_deviation_rad = math.radians(localizer_max_deviation_deg)
        self.heading_max_deviation_rad = math.radians(heading_max_deviation_deg)
        self.loss_fn = _get_loss_fn(loss_type)

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
        runway_direction_xy = torch.stack([runway_bearing[0], runway_bearing[1]])  # [2]
        antenna_xy = threshold_xy + runway_length_m * runway_direction_xy  # [2]

        antenna_to_aircraft = positions_xy - antenna_xy.unsqueeze(0)
        approach_direction = torch.stack([-runway_bearing[0], -runway_bearing[1]])  # [2]

        x = (antenna_to_aircraft[:, 0] * approach_direction[0] +
             antenna_to_aircraft[:, 1] * approach_direction[1])
        y = (antenna_to_aircraft[:, 0] * approach_direction[1] -
             antenna_to_aircraft[:, 1] * approach_direction[0])

        return torch.atan2(y, x)

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
        dir_normalized = directions / (torch.norm(directions, dim=-1, keepdim=True) + self.epsilon)

        dot_product = (dir_normalized * runway_bearing).sum(dim=-1)
        cross_product = (
            dir_normalized[:, 0] * runway_bearing[1] -
            dir_normalized[:, 1] * runway_bearing[0]
        )
        return torch.atan2(cross_product, dot_product)  # [N-1]

    def forward(
        self,
        pred_pos_abs: torch.Tensor,
        target_pos_abs: torch.Tensor,
        target_pad_mask: torch.Tensor,
        runway: dict,
    ) -> torch.Tensor:
        """
        Compute localizer + track alignment loss for predicted trajectories.

        Args:
            pred_pos_abs: Predicted absolute positions [B, H, 3] (x, y, altitude in meters)
            target_pos_abs: Target absolute positions [B, H, 3] (not used, kept for API consistency)
            target_pad_mask: Padding mask [B, H] (True for padded positions, False for valid positions)
            runway: RunwayData object containing:
                - bearing: tensor [B, 2] with [sin(bearing), cos(bearing)] for runway direction
                - xyz: tensor [B, 3] with threshold position [x, y, altitude] in meters
                - length: tensor [B] with runway length in meters

        Returns:
            Scalar loss value (averaged across batch and waypoints)
        """
        runway_bearing = runway.bearing  # [B, 2]
        threshold_xyz = runway.xyz  # [B, 3]
        runway_length_m = runway.length  # [B]
        valid_lengths = (~target_pad_mask).sum(dim=1)  # [B]
        losses = []

        for b in range(pred_pos_abs.size(0)):
            end_idx = valid_lengths[b].item()
            start_idx = max(0, end_idx - self.num_final_waypoints)

            final_positions_xy = pred_pos_abs[b, start_idx:end_idx, :2]  # [N, 2]
            rwy_bearing = runway_bearing[b]
            thresh_xy = threshold_xyz[b, :2]
            rwy_length = runway_length_m[b]

            localizer_dev = self._compute_localizer_deviation(final_positions_xy, thresh_xy, rwy_bearing, rwy_length)
            track_dev = self._compute_track_alignment_deviation(final_positions_xy, rwy_bearing)

            localizer_penalties = self.loss_fn(localizer_dev, self.localizer_max_deviation_rad)  # [N]
            track_penalties = self.loss_fn(track_dev, self.heading_max_deviation_rad)  # [N-1]

            # Pad track penalties at the beginning to match size [N-1] -> [N]
            track_penalties_padded = torch.cat([track_penalties[:1], track_penalties])  # [N]
            losses.append((localizer_penalties + track_penalties_padded).mean())

        return torch.stack(losses).mean()


class GlideslopeAlignmentLoss(nn.Module):
    """
    Glideslope Alignment Loss for aircraft approach trajectories.

    Computes alignment penalties based on ILS glideslope (vertical profile) constraints
    for the final waypoints of predicted trajectories. Penalizes deviations from:
    - Glideslope (vertical): Maximum deviation ±0.7° from standard 3° glideslope

    The loss is computed only for the last `num_final_waypoints` waypoints of each trajectory.
    """

    def __init__(
        self,
        num_final_waypoints: int = 4,
        glideslope_standard_deg: float = 3.0,
        glideslope_max_deviation_deg: float = 0.7,
        loss_type: str = "quadratic",
    ):
        super().__init__()
        self.num_final_waypoints = num_final_waypoints
        self.glideslope_standard_rad = math.radians(glideslope_standard_deg)
        self.glideslope_max_deviation_rad = math.radians(glideslope_max_deviation_deg)
        self.loss_fn = _get_loss_fn(loss_type)

    def _compute_glideslope_deviation(
        self,
        positions_xyz: torch.Tensor,
        threshold_xyz: torch.Tensor,
        runway_bearing: torch.Tensor,
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

        Returns:
            Angular deviations in radians [N] (positive = above glideslope, negative = below)
        """
        antenna_distance_m = torch.tensor(304.8, device=positions_xyz.device)
        screen_height_m = torch.tensor(15.25, device=positions_xyz.device)
        antenna_offset = torch.stack([
            antenna_distance_m * runway_bearing[0],
            antenna_distance_m * runway_bearing[1],
            -screen_height_m,
        ])  # [3]
        antenna_xyz = threshold_xyz + antenna_offset  # [3]

        antenna_to_aircraft = positions_xyz - antenna_xyz.unsqueeze(0)
        approach_direction = torch.stack([
            -runway_bearing[0],
            -runway_bearing[1],
            torch.tensor(0.0, device=positions_xyz.device),
        ])  # [3]

        along_track = torch.sum(antenna_to_aircraft * approach_direction, dim=1)  # [N]
        height = antenna_to_aircraft[:, 2]  # [N]

        actual_gs_rad = torch.atan2(height, along_track)  # [N]
        return actual_gs_rad - self.glideslope_standard_rad  # [N]

    def forward(
        self,
        pred_pos_abs: torch.Tensor,
        target_pos_abs: torch.Tensor,
        target_pad_mask: torch.Tensor,
        runway: dict,
    ) -> torch.Tensor:
        """
        Compute glideslope alignment loss for predicted trajectories.

        Args:
            pred_pos_abs: Predicted absolute positions [B, H, 3] (x, y, altitude in meters)
            target_pos_abs: Target absolute positions [B, H, 3] (not used, kept for API consistency)
            target_pad_mask: Padding mask [B, H] (True for padded positions, False for valid positions)
            runway: RunwayData object containing:
                - bearing: tensor [B, 2] with [sin(bearing), cos(bearing)] for runway direction
                - xyz: tensor [B, 3] with threshold position [x, y, altitude] in meters

        Returns:
            Scalar loss value (averaged across batch and waypoints)
        """
        runway_bearing = runway.bearing  # [B, 2]
        threshold_xyz = runway.xyz  # [B, 3]
        valid_lengths = (~target_pad_mask).sum(dim=1)  # [B]
        losses = []

        for b in range(pred_pos_abs.size(0)):
            end_idx = valid_lengths[b].item()
            start_idx = max(0, end_idx - self.num_final_waypoints)

            final_positions_xyz = pred_pos_abs[b, start_idx:end_idx, :]  # [N, 3]
            rwy_bearing = runway_bearing[b]
            thresh_xyz = threshold_xyz[b]

            glideslope_dev = self._compute_glideslope_deviation(final_positions_xyz, thresh_xyz, rwy_bearing)
            losses.append(self.loss_fn(glideslope_dev, self.glideslope_max_deviation_rad).mean())

        return torch.stack(losses).mean()


# Backwards-compatible alias
ILSAlignmentLoss = LocalizerAlignmentLoss
