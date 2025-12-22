from typing import Dict, Any, Optional, Callable, List
import torch
from torch.utils.data import Dataset
import pandas as pd


class ApproachDataset(Dataset):
    def __init__(
        self,
        inputs_path: str,
        horizons_path: str,
        input_time_minutes: int,
        horizon_time_minutes: int,
        resampling_rate_seconds: int,
        feature_cols: Optional[List[str]] = ["latitude", "longitude", "altitude", "groundspeed", "track", "vertical_rate"],
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ):
        self.inputs_path = inputs_path
        self.horizons_path = horizons_path
        self.input_time_minutes = input_time_minutes
        self.horizon_time_minutes = horizon_time_minutes
        self.resampling_rate_seconds = resampling_rate_seconds
        self.transform = transform

        self.input_seq_len = input_time_minutes * 60 // resampling_rate_seconds
        self.horizon_seq_len = horizon_time_minutes * 60 // resampling_rate_seconds

        self.inputs_df = pd.read_parquet(inputs_path).sort_values(["flight_id", "timestamp"]).reset_index(drop=True)
        self.horizons_df = pd.read_parquet(horizons_path).sort_values(["flight_id", "timestamp"]).reset_index(drop=True)
        self.grouped_inputs_df = self.inputs_df.groupby("flight_id")
        self.grouped_horizons_df = self.horizons_df.groupby("flight_id")

        self._validate_feature_cols(feature_cols)
        self.feature_cols = feature_cols
        self.flight_ids = sorted(self.inputs_df["flight_id"].unique().tolist())
        self.size = len(self.flight_ids)

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

        input_df = self.grouped_inputs_df.get_group(flight_id)[self.feature_cols]
        horizon_df = self.grouped_horizons_df.get_group(flight_id)[self.feature_cols]
        
        horizon_len = len(horizon_df)
        if horizon_len == 0:
            raise ValueError(f"No horizon data found for flight {flight_id}")

        x = torch.from_numpy(input_df.astype("float32").values)
        y = torch.from_numpy(horizon_df.astype("float32").values)
        current_position = x[-1:]
        y_in = torch.cat([current_position, y[:-1]], dim=0)

        y, y_in, mask = self._pad_horizons(y, y_in, horizon_len)

        sample = {
            "x": x,
            "y": y,
            "y_in": y_in,
            "mask": mask,
            "flight_id": flight_id
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


    def _pad_horizons(
        self,
        y: torch.Tensor,
        y_in: torch.Tensor,
        horizon_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        if horizon_len < self.horizon_seq_len:
            padding_len = self.horizon_seq_len - horizon_len
            last_value = y[-1:]
            padding = last_value.repeat(padding_len, 1)

            y = torch.cat([y, padding], dim=0)
            y_in = torch.cat([y_in, padding], dim=0)
            mask = torch.cat([
                torch.zeros(horizon_len, dtype=torch.bool),
                torch.ones(padding_len, dtype=torch.bool)
            ])
        else:
            y = y[:self.horizon_seq_len]
            y_in = y_in[:self.horizon_seq_len]
            mask = torch.zeros(self.horizon_seq_len, dtype=torch.bool)

        return y, y_in, mask