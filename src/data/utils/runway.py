import torch
import numpy as np
from typing import List, Tuple
from traffic.data import airports

from .projections import get_transformer_wgs84_to_aeqd

def compute_dx_dy_bearing(
    positions_xy: torch.Tensor,
    reference_xy: torch.Tensor,
) -> torch.Tensor:
    """
    Compute displacement features from positions to threshold.
    
    Args:
        positions_xy: Position coordinates [*, 2] (x, y)
        reference_xy: Reference coordinates [2] or broadcastable shape (e.g. runway threshold)
        
    Returns:
        Displacement features (dx, dy) as [*, 2]
    """
    return reference_xy - positions_xy

def construct_runway_features(unique_airport_runways: List[Tuple[str, str]], distance_nm: List[float] = [4, 8, 32]) -> torch.Tensor:
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

        runway_features[f"{airport}-{rwy_name}"] = {
            "xyz": threshold_xyz,
            "bearing": bearing,
            "length": length_m,
            "centerline_points_xy": centerline_points_xy,
        }

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
        threshold_xyz: Threshold coordinates [3] (x, y, altitude)
        bearing: Runway bearing [*, 2] (sin, cos)
        distance: Distance to extend backward from threshold (in NM) [float]
        
    Returns:
        Point on extended centerline [*, 2] (x, y)
    """
    threshold_xy = threshold_xy[:2]
    distance_m = distance_nm * 1852
    delta = -distance_m * bearing
    return threshold_xy + delta

def get_distances_to_centerline(traj_pos_xy: torch.Tensor, runway_points_xy: List[torch.Tensor]) -> torch.Tensor:
    distances = []
    for runway_point_xy in runway_points_xy:
        dist = compute_dx_dy_bearing(traj_pos_xy, runway_point_xy)
        distances.append(dist)
    return torch.cat(distances, dim=-1)

