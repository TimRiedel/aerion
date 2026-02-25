from typing import Dict, List, Tuple

import numpy as np
import torch
from traffic.data import airports

from data.interface import RunwayData
from data.compute.projections import get_transformer_wgs84_to_aeqd

def compute_dx_dy_bearing(
    positions_xy: torch.Tensor,
    reference_xy: torch.Tensor,
) -> torch.Tensor:
    """
    Compute displacement (dx, dy) from positions to a reference point.

    Args:
        positions_xy: Position coordinates [*, 2] (x, y)
        reference_xy: Reference coordinates [2] or broadcastable shape (e.g. runway threshold)

    Returns:
        Displacement features (dx, dy) as [*, 2]
    """
    return reference_xy - positions_xy

def construct_runway_features(unique_airport_runways: List[Tuple[str, str]], distance_nm: List[float] = [4, 8, 16, 32]) -> Dict[str, RunwayData]:
    runway_features = {}

    for airport_runway in unique_airport_runways:
        airport = airport_runway[0]
        rwy_name = airport_runway[1]
        thresholds = airports[airport].runways[rwy_name].tuple_runway
        runway = next(t for t in thresholds if t.name == rwy_name)

        airport_lat, airport_lon = airports[airport].latlon
        rwy_lat, rwy_lon, rwy_bearing = runway.latitude, runway.longitude, runway.bearing
        rwy_x, rwy_y = get_transformer_wgs84_to_aeqd(ref_lat=airport_lat, ref_lon=airport_lon).transform(rwy_lon, rwy_lat)

        # Compute runway length in meters by finding distance between thresholds
        # Find the opposite threshold (other end of the runway)
        threshold_opposite = next(t for t in thresholds if t.name != rwy_name)
        opp_lat, opp_lon = threshold_opposite.latitude, threshold_opposite.longitude
        opp_x, opp_y = get_transformer_wgs84_to_aeqd(ref_lat=airport_lat, ref_lon=airport_lon).transform(opp_lon, opp_lat)
        length_m = torch.norm(torch.tensor([rwy_x - opp_x, rwy_y - opp_y]))
        
        airport_altitude_ft = airports[airport].altitude
        airport_altitude_m = airport_altitude_ft * 0.3048  # Convert feet to meters
        threshold_altitude_m = airport_altitude_m + 15.24  # Add 50ft screen height over threshold
        threshold_xyz = torch.tensor([rwy_x, rwy_y, threshold_altitude_m], dtype=torch.float32)

        bearing_sin = torch.sin(torch.tensor(rwy_bearing * 2 * np.pi / 360))
        bearing_cos = torch.cos(torch.tensor(rwy_bearing * 2 * np.pi / 360))
        bearing = torch.stack([bearing_sin, bearing_cos], dim=-1)

        centerline_points_xy = []
        for dist in distance_nm:
            centerline_xy = compute_extended_centerline_point(threshold_xyz[:2], bearing, dist)
            centerline_points_xy.append(centerline_xy)

        runway_features[f"{airport}-{rwy_name}"] = RunwayData(
            xyz=threshold_xyz,
            bearing=bearing,
            length=length_m,
            centerline_points_xy=centerline_points_xy,
        )

    return runway_features

def compute_extended_centerline_point(
    threshold_xy: torch.Tensor,
    bearing: torch.Tensor,
    distance_nm: float,
) -> torch.Tensor:
    """
    Compute a point on the extended centerline backward from the threshold.

    This extends the centerline in the approach direction (opposite to runway bearing).

    Args:
        threshold_xy: Threshold coordinates [2] or [3] (x, y); altitude ignored if present
        bearing: Runway bearing [*, 2] (sin θ, cos θ)
        distance_nm: Distance to extend backward from threshold in nautical miles

    Returns:
        Point on extended centerline [*, 2] (x, y)
    """
    threshold_xy = threshold_xy[:2]
    distance_m = distance_nm * 1852
    delta = -distance_m * bearing
    return threshold_xy + delta

def get_distances_to_centerline(
    traj_pos_xy: torch.Tensor,
    runway_points_xy: List[torch.Tensor],
    clamp_m: float = 20_000.0,
) -> torch.Tensor:
    """
    Compute (dx, dy) from trajectory positions to each runway centerline point.
    Values are clamped to [-clamp_m, +clamp_m] meters so normalization
    preserves resolution where the aircraft is close to the centerline.
    """
    distances = []
    for runway_point_xy in runway_points_xy:
        dist = compute_dx_dy_bearing(traj_pos_xy, runway_point_xy)
        dist = dist.clamp(min=-clamp_m, max=clamp_m)
        distances.append(dist)
    return torch.cat(distances, dim=-1)

def convert_pos_to_rwy_coordinates(traj_pos_xy: torch.Tensor, runway_xy: torch.Tensor, runway_bearing: torch.Tensor) -> torch.Tensor:
    """
    Transform airport-aligned coordinates (ENU) to runway-relative coordinates.

    The transformation consists of:
    1. Translation: subtract runway position to move origin to runway
    2. Rotation: rotate so that positive x-axis aligns with runway direction

    Args:
        traj_pos_xy: Trajectory positions [H, 2] or [B, H, 2] (x, y)
        runway_xy: Runway position [2] or [B, 2] (x, y)
        runway_bearing: Runway bearing [2] or [B, 2] as (sin θ, cos θ)

    Returns:
        Transformed positions in runway-relative frame [H, 2] or [B, H, 2]
    """
    # Translation - move origin to runway position
    translated = traj_pos_xy - runway_xy

    # Add dimensions to match translated shape for broadcasting
    # For [H, 2]: sin/cos are scalars, broadcasting works automatically
    # For [B, H, 2]: need [B, 1] to broadcast with [B, H, 2]
    sin_theta = runway_bearing[..., 0]  # [2] -> scalar, [B, 2] -> [B]
    cos_theta = runway_bearing[..., 1]  # [2] -> scalar, [B, 2] -> [B]
    while sin_theta.dim() < translated.dim() - 1:
        sin_theta = sin_theta.unsqueeze(-1)
        cos_theta = cos_theta.unsqueeze(-1)
    
    # Apply rotation matrix
    # [sin(θ)   cos(θ) ]
    # [-cos(θ)  sin(θ) ]
    rotated_x = sin_theta * translated[..., 0] + cos_theta * translated[..., 1]
    rotated_y = -cos_theta * translated[..., 0] + sin_theta * translated[..., 1]
    
    return torch.stack([rotated_x, rotated_y], dim=-1)

