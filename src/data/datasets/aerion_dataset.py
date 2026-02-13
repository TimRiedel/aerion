import torch
import numpy as np
import re
import pandas as pd
from torch.utils.data import Dataset
from omegaconf import DictConfig
from typing import Dict, Any, Optional, Callable, List, Tuple
from traffic.data import airports

from data.utils import *
from .approach_dataset import ApproachDataset


class AerionDataset(ApproachDataset):
    def __init__(
        self,
        inputs_path: str,
        horizons_path: str,
        flightinfo_path: str,
        input_time_minutes: int,
        horizon_time_minutes: int,
        resampling_rate_seconds: int,
        num_trajectories_to_predict: int = None,
        num_waypoints_to_predict: int = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        contexts_cfg: Optional[DictConfig] = None,
    ):
        super().__init__(
            inputs_path=inputs_path,
            horizons_path=horizons_path,
            flightinfo_path=flightinfo_path,
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            resampling_rate_seconds=resampling_rate_seconds,
            num_trajectories_to_predict=num_trajectories_to_predict,
            num_waypoints_to_predict=num_waypoints_to_predict,
            transform=transform,
        )
        self.contexts_cfg = contexts_cfg or {}


    def _is_context_enabled(self, name: str) -> bool:
        """Check if a context is enabled in the configuration."""
        return self.contexts_cfg.get(name, {}).get("enabled", False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        flight_id = self.flight_ids[idx]
        input_traj_pos, input_traj_deltas, target_traj_pos, target_traj_deltas, dec_in_pos, dec_in_deltas, mask_traj = self._compute_inputs_outputs(idx)

        runway_data = self._get_runway_data(flight_id)
        threshold_xy = runway_data["xyz"][:2]  # [2]
        centerline_points_xy = runway_data["centerline_points_xy"]

        # Input: Append distances to runway and centerline points to input trajectory
        input_traj_pos_xy = input_traj_pos[:, :2]
        input_dist_runway = get_distances_to_centerline(input_traj_pos_xy, [threshold_xy])
        input_dist_centerline = get_distances_to_centerline(input_traj_pos_xy, centerline_points_xy)
        input_traj = torch.cat([input_traj_pos, input_traj_deltas, input_dist_runway, input_dist_centerline], dim=1)
        
        # Decoder input: shifted deltas (3) + centerline features (8)
        dec_in_pos_xy = dec_in_pos[:, :2]
        dec_in_dist_runway = get_distances_to_centerline(dec_in_pos_xy, [threshold_xy])
        dec_in_dist_centerline = get_distances_to_centerline(dec_in_pos_xy, centerline_points_xy)
        dec_in_traj = torch.cat([dec_in_deltas, dec_in_dist_runway, dec_in_dist_centerline], dim=1)
        
        # Compute remaining track distance
        target_rtd = compute_rtd(target_traj_pos, mask_traj, runway_data["xyz"], runway_data["bearing"])

        sample = {
            "input_traj": input_traj,            # [T_in, 3 + 3 + num_centerline_points * 2] positions + deltas + centerline features
            "target_traj": target_traj_deltas,   # [H, 3] target deltas
            "target_rtd": target_rtd,            # scalar
            "dec_in_traj": dec_in_traj,          # [H, 3 + num_centerline_points * 2] decoder input deltas + centerline features
            "mask_traj": mask_traj,              # [H] mask for padded positions
            "runway": runway_data,
            "flight_id": flight_id
        }
        
        if self._is_context_enabled("flightinfo"):
            sample["flightinfo"] = self._get_flightinfo(flight_id)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _get_flightinfo(self, flight_id: str) -> torch.Tensor:
        flight_id_without_sample_index = re.sub(r"_S\d+$", "", flight_id) # Remove trailing _S+digits
        flightinfo = self.flightinfo_df.loc[flight_id_without_sample_index]

        # Create one-hot encoding for runway
        one_hot = np.zeros(len(self.runway_categories), dtype=np.float32)
        runway_idx = self.runway_categories.index(flightinfo["runway"])
        one_hot[runway_idx] = 1.0
        return torch.from_numpy(one_hot)