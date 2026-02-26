from typing import Callable, Optional

import pandas as pd

from data.features import FeatureSchema
from data.interface import Sample
from data.datasets.base_dataset import BaseDataset


class ApproachDataset(BaseDataset):
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
        super().__init__(
            flightinfo_path=flightinfo_path,
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            resampling_rate_seconds=resampling_rate_seconds,
            feature_schema=feature_schema,
            num_trajectories_to_predict=num_trajectories_to_predict,
            num_waypoints_to_predict=num_waypoints_to_predict,
            transform=transform,
        )
        self.inputs_path = inputs_path
        self.horizons_path = horizons_path

        # Read input and horizon data
        self.inputs_df = pd.read_parquet(inputs_path).sort_values(["flight_id", "timestamp"]).reset_index(drop=True)
        self.horizons_df = pd.read_parquet(horizons_path).sort_values(["flight_id", "timestamp"]).reset_index(drop=True)
        self.grouped_inputs_df = self.inputs_df.groupby("flight_id")
        self.grouped_horizons_df = self.horizons_df.groupby("flight_id")

        # Construct flight ids
        self.flight_ids = sorted(self.inputs_df["flight_id"].unique().tolist())
        if num_trajectories_to_predict is not None:
            self.flight_ids = self.flight_ids[:num_trajectories_to_predict]
        self.size = len(self.flight_ids)


    def __getitem__(self, idx: int) -> Sample:
        flight_id = self.flight_ids[idx]
        input_df = self.grouped_inputs_df.get_group(flight_id)
        horizon_df = self.grouped_horizons_df.get_group(flight_id)
        sample = self.get_trajectory_data(input_df, horizon_df, flight_id)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample