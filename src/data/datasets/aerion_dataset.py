import torch
import pandas as pd
from torch.utils.data import Dataset
from omegaconf import DictConfig
from typing import Dict, Any, Optional, Callable, List, Tuple
from traffic.data import airports

from data.utils import *
from .approach_dataset import ApproachDataset


class AerionDataset(ApproachDataset):
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        flight_id = self.flight_ids[idx]
        input_traj_pos, input_traj_deltas, target_traj_pos, target_traj_deltas, dec_in_pos, dec_in_deltas, mask_traj = self._compute_inputs_outputs(idx)

        runway_data = self._get_runway_data(flight_id)
        threshold_xy = runway_data["xyz"][:2]  # [2]
        centerline_points_xy = runway_data["centerline_points_xy"]

        # Input: Append distances to runway threshold and centerline points to input trajectory
        input_traj_pos_xy = input_traj_pos[:, :2]
        input_dist_runway = get_distances_to_centerline(input_traj_pos_xy, [threshold_xy])
        input_dist_centerline = get_distances_to_centerline(input_traj_pos_xy, centerline_points_xy)
        input_traj = torch.cat([input_traj_pos, input_traj_deltas, input_dist_runway, input_dist_centerline], dim=1)
        
        # Decoder input: altitude (1) + shifted deltas (3) + distance features (2 * (1 + num_centerline_points))
        dec_in_pos_xy = dec_in_pos[:, :2]
        dec_in_alt = dec_in_pos[:, 2:3]  # [H, 1]
        dec_in_dist_runway = get_distances_to_centerline(dec_in_pos_xy, [threshold_xy])
        dec_in_dist_centerline = get_distances_to_centerline(dec_in_pos_xy, centerline_points_xy)
        dec_in_traj = torch.cat([dec_in_alt, dec_in_deltas, dec_in_dist_runway, dec_in_dist_centerline], dim=1)
        
        # Because target trajectories end at the runway threshold, the RTD is the same as the trajectory distance.
        traj_distance, _ = compute_rtd(target_traj_pos, mask_traj, runway_data["xyz"], runway_data["bearing"])

        sample = {
            "input_traj": input_traj,            # [T_in, 3 + 3 + 2 * (1 + num_centerline_points)] positions + deltas + distance features
            "target_traj": target_traj_deltas,   # [H, 3] target deltas
            "target_rtd": traj_distance,         # scalar
            "dec_in_traj": dec_in_traj,          # [H, 1 alt + 3 deltas + 2 * (1 + num_centerline_points)] decoder input
            "mask_traj": mask_traj,              # [H] mask for padded positions
            "runway": runway_data,
            "flight_id": flight_id
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample