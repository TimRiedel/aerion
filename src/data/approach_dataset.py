import torch
import numpy as np
import re
import pandas as pd
from torch.utils.data import Dataset
from omegaconf import DictConfig
from typing import Dict, Any, Optional, Callable, List

from data.utils.trajectory import compute_threshold_features


class ApproachDataset(Dataset):
    def __init__(
        self,
        inputs_path: str,
        horizons_path: str,
        input_time_minutes: int,
        horizon_time_minutes: int,
        resampling_rate_seconds: int,
        feature_cols: Optional[List[str]] = ["x_coord", "y_coord", "altitude"],
        num_trajectories_to_predict: int = None,
        num_waypoints_to_predict: int = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        flightinfo_path: Optional[str] = None,
        contexts_cfg: Optional[DictConfig] = None,
    ):
        self.inputs_path = inputs_path
        self.horizons_path = horizons_path
        self.input_time_minutes = input_time_minutes
        self.horizon_time_minutes = horizon_time_minutes
        self.resampling_rate_seconds = resampling_rate_seconds
        self.transform = transform
        self.contexts_cfg = contexts_cfg or {}

        self.input_seq_len = input_time_minutes * 60 // resampling_rate_seconds
        base_horizon_seq_len = horizon_time_minutes * 60 // resampling_rate_seconds
        
        if num_waypoints_to_predict is not None:
            self.horizon_seq_len = min(base_horizon_seq_len, num_waypoints_to_predict)
        else:
            self.horizon_seq_len = base_horizon_seq_len

        self.inputs_df = pd.read_parquet(inputs_path).sort_values(["flight_id", "timestamp"]).reset_index(drop=True)
        self.horizons_df = pd.read_parquet(horizons_path).sort_values(["flight_id", "timestamp"]).reset_index(drop=True)
        self.grouped_inputs_df = self.inputs_df.groupby("flight_id")
        self.grouped_horizons_df = self.horizons_df.groupby("flight_id")

        self._validate_feature_cols(feature_cols)
        self.feature_cols = feature_cols

        self.flightinfo_df = None
        if flightinfo_path is not None:
            self.flightinfo_df = pd.read_parquet(flightinfo_path).set_index("flight_id")

        self.flight_ids = sorted(self.inputs_df["flight_id"].unique().tolist())
        if num_trajectories_to_predict is not None:
            self.flight_ids = self.flight_ids[:num_trajectories_to_predict]
        self.size = len(self.flight_ids)
    
    def _is_context_enabled(self, name: str) -> bool:
        """Check if a context is enabled in the configuration."""
        return self.contexts_cfg.get(name, {}).get("enabled", False)

    def _validate_feature_cols(self, feature_cols: List[str]) -> None:
        missing_in_inputs = set(feature_cols) - set(self.inputs_df.columns)
        missing_in_horizons = set(feature_cols) - set(self.horizons_df.columns)
        
        if missing_in_inputs:
            raise ValueError(
                f"Feature columns {missing_in_inputs} not found in inputs dataframe. "
                f"Available columns: {list(self.inputs_df.columns)}"
            )
        if missing_in_horizons:
            raise ValueError(
                f"Feature columns {missing_in_horizons} not found in horizons dataframe. "
                f"Available columns: {list(self.horizons_df.columns)}"
            )


    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        flight_id = self.flight_ids[idx]
        input_df = self.grouped_inputs_df.get_group(flight_id)[self.feature_cols].head(self.input_seq_len)
        horizon_df = self.grouped_horizons_df.get_group(flight_id)[self.feature_cols].head(self.horizon_seq_len)
        
        horizon_len = len(horizon_df)
        if horizon_len == 0:
            raise ValueError(f"No horizon data found for flight {flight_id}")

        input_traj_pos = torch.from_numpy(input_df.astype("float32").values) # [T_in, 3]
        input_traj_delta_computed = torch.diff(input_traj_pos, dim=0)  # [T_in-1, 3]
        input_traj_delta = torch.cat([input_traj_delta_computed[0:1], input_traj_delta_computed], dim=0)  # [T_in, 3] -> backfilled first delta
        
        target_traj_pos = torch.from_numpy(horizon_df.astype("float32").values)
        last_position = input_traj_pos[-1]  # [3]
        
        threshold_xy = target_traj_pos[-1, :2] # last valid horizon position
        input_thr_features = compute_threshold_features(input_traj_pos[:, :2], threshold_xy)  # [T_in, 2]
        input_traj = torch.cat([input_traj_pos, input_traj_delta, input_thr_features], dim=1)  # [T_in, 8]
        
        # Target: deltas between consecutive horizon positions [H, 3]
        target_traj_shifted_pos = torch.cat([last_position.unsqueeze(0), target_traj_pos[:-1]], dim=0)
        target_traj_deltas = target_traj_pos - target_traj_shifted_pos  # [H, 3]
        
        # Decoder input: shifted deltas (3) + threshold direction vector (2) [H, 5]
        last_observed_delta = input_traj_delta[-1]  # [3]
        dec_in_deltas = torch.cat([last_observed_delta.unsqueeze(0), target_traj_deltas[:-1]], dim=0)  # [H, 3]
        dec_in_positions = torch.cat([last_position.unsqueeze(0), target_traj_pos[:-1]], dim=0)  # [H, 3]
        dec_in_thr_features = compute_threshold_features(dec_in_positions[:, :2], threshold_xy)  # [H, 2]
        dec_in_traj = torch.cat([dec_in_deltas, dec_in_thr_features], dim=1)  # [H, 5]

        target_traj_deltas, dec_in_traj, mask_traj = self._pad_horizons(target_traj_deltas, dec_in_traj, horizon_len)

        sample = {
            "input_traj": input_traj,            # [T_in, 8] positions + deltas + threshold direction vector
            "target_traj": target_traj_deltas,   # [H, 3] target deltas
            "dec_in_traj": dec_in_traj,          # [H, 5] decoder input deltas + threshold direction vector
            "mask_traj": mask_traj,              # [H] mask for padded positions
            "runway": self._get_runway_data(flight_id, threshold_xy),
            "flight_id": flight_id
        }
        
        if self._is_context_enabled("flightinfo"):
            sample["flightinfo"] = self._get_flightinfo(flight_id)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
    def _get_runway_data(self, flight_id: str, threshold_xy: torch.Tensor) -> torch.Tensor:
        """Build runway tensor: [threshold_x, threshold_y] or [threshold_x, threshold_y, bearing_sin, bearing_cos]."""
        if self.flightinfo_df is not None:
            flight_id_without_sample_index = re.sub(r"_S\d+$", "", flight_id)
            row = self.flightinfo_df.loc[flight_id_without_sample_index]
            bearing_sin = torch.tensor(row["rwy_bearing_sin"], dtype=torch.float32)
            bearing_cos = torch.tensor(row["rwy_bearing_cos"], dtype=torch.float32)
            return torch.cat([threshold_xy, bearing_sin.unsqueeze(0), bearing_cos.unsqueeze(0)])  # [4]
        else:
            return threshold_xy  # [2]

    def _get_flightinfo(self, flight_id: str) -> torch.Tensor:
        if self.flightinfo_df is None:
            raise ValueError("Flightinfo context is enabled but flightinfo_df is not loaded")
        
        flight_id_without_sample_index = re.sub(r"_S\d+$", "", flight_id) # Remove trailing _S+digits
        row = self.flightinfo_df.loc[flight_id_without_sample_index]
        features = self.contexts_cfg["flightinfo"]["features"]
        values = row[features].values.astype(np.float32)
        return torch.from_numpy(values)


    def _pad_horizons(
        self,
        target_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        horizon_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        if horizon_len < self.horizon_seq_len:
            padding_len = self.horizon_seq_len - horizon_len
            
            # Pad target_traj (deltas only)
            target_padding = target_traj[-1:].repeat(padding_len, 1)
            target_traj = torch.cat([target_traj, target_padding], dim=0)
            
            # Pad dec_in_traj (deltas + threshold features)
            dec_in_padding = dec_in_traj[-1:].repeat(padding_len, 1)
            dec_in_traj = torch.cat([dec_in_traj, dec_in_padding], dim=0)
            
            mask_traj = torch.cat([
                torch.zeros(horizon_len, dtype=torch.bool),
                torch.ones(padding_len, dtype=torch.bool)
            ])
        else:
            target_traj = target_traj[:self.horizon_seq_len]
            dec_in_traj = dec_in_traj[:self.horizon_seq_len]
            mask_traj = torch.zeros(self.horizon_seq_len, dtype=torch.bool)

        return target_traj, dec_in_traj, mask_traj