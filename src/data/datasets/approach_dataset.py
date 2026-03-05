from typing import Callable, Optional

import pandas as pd

from data.features import FeatureSchema
from data.interface import PredictionSample
from data.datasets.base_dataset import BaseDataset


class ApproachDataset(BaseDataset):
    def __init__(
        self,
        trajectories_path: str,
        scenes_path: str,
        max_num_agents: int,
        flightinfo_path: str,
        input_time_minutes: int,
        horizon_time_minutes: int,
        resampling_rate_seconds: int,
        feature_schema: FeatureSchema,
        num_trajectories_to_predict: int = None,
        num_waypoints_to_predict: int = None,
        transform: Optional[Callable[[PredictionSample], PredictionSample]] = None,
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
        self.trajectories_path = trajectories_path
        self.scenes_path = scenes_path
        self.max_num_agents = max_num_agents

        # Load resampled trajectories and index them by flight_id and timestamp
        traffic_df = pd.read_parquet(trajectories_path).sort_values(["flight_id", "timestamp"]).reset_index(drop=True)
        traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"])
        self.flight_dict = {
            flight_id: flight_df.set_index("timestamp")
            for flight_id, flight_df in traffic_df.groupby("flight_id", sort=False)
        }

        # Load scene definitions. For the single-agent approach case, we treat each row
        # as an independent sample, ignoring the multi-flight nature of scenes.
        scenes_df = pd.read_parquet(scenes_path)
        scenes_df["input_start_time"] = pd.to_datetime(scenes_df["input_start_time"])
        scenes_df["prediction_start_time"] = pd.to_datetime(scenes_df["prediction_start_time"])
        scenes_df["prediction_end_time"] = pd.to_datetime(scenes_df["prediction_end_time"])

        # Filter out scenes with more than max_num_agents flights to mirror TrafficDataset behavior.
        valid_scene_mask = (
            scenes_df.groupby("scene_id")["flight_id"].transform("size") <= self.max_num_agents
        )
        scenes_df = scenes_df[valid_scene_mask]

        # Each row corresponds to one (flight_id, time window) sample.
        self.samples_df = scenes_df[["flight_id", "input_start_time", "prediction_start_time", "prediction_end_time"]].reset_index(drop=True)
        if num_trajectories_to_predict is not None:
            self.samples_df = self.samples_df.head(num_trajectories_to_predict)
        self.size = len(self.samples_df)

    def __getitem__(self, idx: int) -> PredictionSample:
        row = self.samples_df.iloc[idx]
        flight_id = row["flight_id"]
        input_start_time = row["input_start_time"]
        prediction_start_time = row["prediction_start_time"]
        prediction_end_time = row["prediction_end_time"]

        flight_df = self.flight_dict[flight_id]
        input_df = flight_df.loc[input_start_time:prediction_start_time]
        horizon_df = flight_df.loc[prediction_start_time:prediction_end_time]
        prediction_sample = self.get_flight_data(input_df, horizon_df, flight_id)

        if self.transform is not None:
            prediction_sample = self.transform(prediction_sample)

        return prediction_sample