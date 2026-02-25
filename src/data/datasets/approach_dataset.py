import re
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.compute import compute_rtd, construct_runway_features
from data.features import FeatureSchema
from data.interface import RunwayData, Sample, TrajectoryData

class ApproachDataset(Dataset):
    def __init__(
        self,
        inputs_path: str,
        horizons_path: str,
        flightinfo_path: str,
        input_time_minutes: int,
        horizon_time_minutes: int,
        resampling_rate_seconds: int,
        feature_schema: FeatureSchema,
        num_trajectories_to_predict: int = None,
        num_waypoints_to_predict: int = None,
        transform: Optional[Callable[[Sample], Sample]] = None,
    ):
        self.inputs_path = inputs_path
        self.horizons_path = horizons_path
        self.flightinfo_path = flightinfo_path
        self.input_time_minutes = input_time_minutes
        self.horizon_time_minutes = horizon_time_minutes
        self.resampling_rate_seconds = resampling_rate_seconds
        self.feature_schema = feature_schema
        self.transform = transform

        # Set horizon sequence length
        self.input_seq_len = input_time_minutes * 60 // resampling_rate_seconds
        base_horizon_seq_len = horizon_time_minutes * 60 // resampling_rate_seconds
        if num_waypoints_to_predict is not None:
            self.horizon_seq_len = min(base_horizon_seq_len, num_waypoints_to_predict)
        else:
            self.horizon_seq_len = base_horizon_seq_len

        # Read input and horizon data
        self.inputs_df = pd.read_parquet(inputs_path).sort_values(["flight_id", "timestamp"]).reset_index(drop=True)
        self.horizons_df = pd.read_parquet(horizons_path).sort_values(["flight_id", "timestamp"]).reset_index(drop=True)
        self.grouped_inputs_df = self.inputs_df.groupby("flight_id")
        self.grouped_horizons_df = self.horizons_df.groupby("flight_id")

        # Validate feature columns
        self.position_cols = ["x_coord", "y_coord", "altitude"]
        self._validate_feature_cols(self.position_cols)

        # Construct flight ids
        self.flight_ids = sorted(self.inputs_df["flight_id"].unique().tolist())
        if num_trajectories_to_predict is not None:
            self.flight_ids = self.flight_ids[:num_trajectories_to_predict]
        self.size = len(self.flight_ids)

        # Construct runway features
        self.flightinfo_df = pd.read_parquet(flightinfo_path).set_index("flight_id")
        unique_airport_runways = self.flightinfo_df[["airport", "runway"]].dropna().drop_duplicates().values.tolist()
        self.runway_features = construct_runway_features(unique_airport_runways)
    

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Sample:
        flight_id = self.flight_ids[idx]
        runway_data = self._get_runway_data(flight_id)
        xyz_positions, xyz_deltas, target_padding_mask = self._compute_inputs_outputs(idx)
        last_input_pos_abs = xyz_positions.encoder_in[-1, :3]

        input_traj = self.feature_schema.build_encoder_input(
            xyz_positions.encoder_in, xyz_deltas.encoder_in, runway_data
        )
        dec_in_traj = self.feature_schema.build_decoder_input(
            xyz_positions.dec_in, xyz_deltas.dec_in, runway_data
        )
        target_traj = self.feature_schema.build_target(
            xyz_positions.target, xyz_deltas.target, runway_data
        )
        trajectory = TrajectoryData(encoder_in=input_traj, dec_in=dec_in_traj, target=target_traj)

        # Because target trajectories end at the runway threshold, the RTD is the same as the trajectory distance and we can ignore the second return value.
        target_rtd, _ = compute_rtd(xyz_positions.target, target_padding_mask, runway_data.xyz, runway_data.bearing)

        sample = Sample(
            xyz_positions=xyz_positions,
            xyz_deltas=xyz_deltas,
            trajectory=trajectory,
            target_padding_mask=target_padding_mask,
            target_rtd=target_rtd,
            last_input_pos_abs=last_input_pos_abs,
            runway=runway_data,
            flight_id=flight_id,
        )

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _compute_inputs_outputs(self, idx: int) -> tuple[TrajectoryData, TrajectoryData, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        flight_id = self.flight_ids[idx]
        input_df = self.grouped_inputs_df.get_group(flight_id)[self.position_cols].head(self.input_seq_len)
        horizon_df = self.grouped_horizons_df.get_group(flight_id)[self.position_cols].head(self.horizon_seq_len)
        
        horizon_len = len(horizon_df)
        if horizon_len == 0:
            raise ValueError(f"No horizon data found for flight {flight_id}")
        input_traj_pos, input_traj_deltas, last_position, last_delta = self._compute_input_traj(input_df)
        target_traj_pos, target_traj_deltas = self._compute_target_traj(horizon_df, last_position)
        dec_in_deltas = self._compute_dec_in_deltas(target_traj_deltas, last_delta)
        dec_in_pos = torch.cat([last_position.unsqueeze(0), target_traj_pos[:-1]], dim=0)

        target_traj_pos, target_traj_deltas, dec_in_pos, dec_in_deltas, target_padding_mask = self._pad_horizons(
            target_traj_pos, target_traj_deltas, dec_in_pos, dec_in_deltas, horizon_len
        )

        xyz_positions = TrajectoryData(encoder_in=input_traj_pos, target=target_traj_pos, dec_in=dec_in_pos)
        xyz_deltas = TrajectoryData(encoder_in=input_traj_deltas, target=target_traj_deltas, dec_in=dec_in_deltas)
        return xyz_positions, xyz_deltas, target_padding_mask

    def _compute_input_traj(self, input_df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_traj_pos = torch.from_numpy(input_df.astype("float32").values) # [T_in, 3]
        input_traj_deltas_computed = torch.diff(input_traj_pos, dim=0)        # [T_in-1, 3]
        input_traj_deltas = torch.cat([input_traj_deltas_computed[0:1], input_traj_deltas_computed], dim=0)  # [T_in, 3] -> backfilled first delta
        last_position = input_traj_pos[-1]  # [3]
        last_delta = input_traj_deltas[-1]   # [3]
        return input_traj_pos, input_traj_deltas, last_position, last_delta

    def _compute_target_traj(self, horizon_df: pd.DataFrame, last_position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_traj_pos = torch.from_numpy(horizon_df.astype("float32").values)
        target_traj_shifted_pos = torch.cat([last_position.unsqueeze(0), target_traj_pos[:-1]], dim=0)
        target_traj_deltas = target_traj_pos - target_traj_shifted_pos  # [H, 3]
        return target_traj_pos, target_traj_deltas

    def _compute_dec_in_deltas(self, target_traj_deltas: torch.Tensor, last_observed_delta: torch.Tensor) -> torch.Tensor:
        return torch.cat([last_observed_delta.unsqueeze(0), target_traj_deltas[:-1]], dim=0)

    def _get_runway_data(self, flight_id: str) -> RunwayData:
        flight_id_without_sample_index = re.sub(r"_S\d+$", "", flight_id)
        flight_row = self.flightinfo_df.loc[flight_id_without_sample_index]
        airport = flight_row["airport"]
        runway = flight_row["runway"]
        return self.runway_features[f"{airport}-{runway}"]

    def _pad_horizons(
        self,
        target_traj_pos: torch.Tensor,
        target_traj_deltas: torch.Tensor,
        dec_in_pos: torch.Tensor,
        dec_in_deltas: torch.Tensor,
        horizon_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
        if horizon_len < self.horizon_seq_len:
            padding_len = self.horizon_seq_len - horizon_len

            # Pad target_traj
            target_pos_padding = target_traj_pos[-1:].repeat(padding_len, 1)
            target_traj_pos = torch.cat([target_traj_pos, target_pos_padding], dim=0)
            
            target_delta_padding = target_traj_deltas[-1:].repeat(padding_len, 1)
            target_traj_deltas = torch.cat([target_traj_deltas, target_delta_padding], dim=0)
            
            # Pad dec_in
            dec_in_pos_padding = dec_in_pos[-1:].repeat(padding_len, 1)
            dec_in_pos = torch.cat([dec_in_pos, dec_in_pos_padding], dim=0)

            dec_in_delta_padding = dec_in_deltas[-1:].repeat(padding_len, 1)
            dec_in_deltas = torch.cat([dec_in_deltas, dec_in_delta_padding], dim=0)
            
            mask_traj = torch.cat([
                torch.zeros(horizon_len, dtype=torch.bool),
                torch.ones(padding_len, dtype=torch.bool)
            ])
        else:
            target_traj_pos = target_traj_pos[:self.horizon_seq_len]
            target_traj_deltas = target_traj_deltas[:self.horizon_seq_len]
            dec_in_pos = dec_in_pos[:self.horizon_seq_len]
            dec_in_deltas = dec_in_deltas[:self.horizon_seq_len]
            mask_traj = torch.zeros(self.horizon_seq_len, dtype=torch.bool)

        return target_traj_pos, target_traj_deltas, dec_in_pos, dec_in_deltas, mask_traj

    def _validate_feature_cols(self, columns: List[str]) -> None:
        missing_in_inputs = set(columns) - set(self.inputs_df.columns)
        missing_in_horizons = set(columns) - set(self.horizons_df.columns)
        
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