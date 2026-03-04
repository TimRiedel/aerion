import random
from typing import Callable, List, Optional
from dataclasses import dataclass

import pandas as pd
import torch

from data.features import FeatureSchema
from data.interface import PredictionSample, TrajectoryData
from data.datasets import BaseDataset
from data.collate import stack_runway_data


@dataclass
class Scene:
    """
    Lightweight representation of a multi-flight scene for trajectory prediction.

    Stores the scene's time windows and the IDs of all flights present at the
    input start time. Flight data is retrieved on demand from a pre-built flight
    dict using timestamp slicing.
    """
    scene_id: int
    flight_ids: list[str]
    input_start_time: pd.Timestamp
    prediction_start_time: pd.Timestamp
    prediction_end_time: pd.Timestamp


class TrafficDataset(BaseDataset):
    def __init__(
        self,
        resampled_path: str,
        scenes_path: str,
        flightinfo_path: str,
        input_time_minutes: int,
        horizon_time_minutes: int,
        resampling_rate_seconds: int,
        max_num_agents: int,
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
        self.resampled_path = resampled_path
        self.scenes_path = scenes_path
        self.max_num_agents = max_num_agents

        traffic_df = pd.read_parquet(resampled_path).sort_values(["flight_id", "timestamp"]).reset_index(drop=True)
        traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"])
        self.flight_dict = self.build_flight_dict(traffic_df)

        self.scenes = self.load_scene_index(scenes_path, max_num_agents=self.max_num_agents)
        self.size = len(self.scenes)

    def __getitem__(self, idx: int) -> PredictionSample:
        scene = self.scenes[idx]

        rng = random.Random(scene.scene_id)
        flight_ids = rng.sample(scene.flight_ids, len(scene.flight_ids))

        flight_data_list = []
        for flight_id in flight_ids:
            flight_df = self.flight_dict[flight_id]
            input_df = flight_df.loc[scene.input_start_time:scene.prediction_start_time]
            horizon_df = flight_df.loc[scene.prediction_start_time:scene.prediction_end_time]
            flight_data = self.get_flight_data(input_df, horizon_df, flight_id)
            flight_data_list.append(flight_data)

        sample = self.stack_flight_data_as_scene(flight_data_list)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def build_flight_dict(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Builds a dict mapping flight_id to a timestamp-indexed DataFrame for fast slicing.

        Args:
            df: Resampled trajectories DataFrame, sorted by flight_id and timestamp.

        Returns:
            Dict mapping each flight_id to its waypoints DataFrame with a DatetimeIndex.
        """
        return {
            fid: flight_df.set_index("timestamp")
            for fid, flight_df in df.groupby("flight_id", sort=False)
        }

    def load_scene_index(self, scenes_path: str, max_num_agents: int) -> list[Scene]:
        """
        Loads the scene manifest and builds a list of Scene objects.

        Args:
            scene_manifest_path: Path to the scene manifest parquet file.

        Returns:
            List of Scene objects ordered by scene_id.
        """
        scenes_df = pd.read_parquet(scenes_path)
        scenes_df["input_start_time"] = pd.to_datetime(scenes_df["input_start_time"])
        scenes_df["prediction_start_time"] = pd.to_datetime(scenes_df["prediction_start_time"])
        scenes_df["prediction_end_time"] = pd.to_datetime(scenes_df["prediction_end_time"])

        scenes = []
        for scene_id, group in scenes_df.groupby("scene_id", sort=True):
            if len(group["flight_id"]) > max_num_agents: # Skip scenes with more than max_num_agents
                continue

            row = group.iloc[0]
            scenes.append(Scene(
                scene_id=int(scene_id),
                flight_ids=group["flight_id"].tolist(),
                input_start_time=row["input_start_time"],
                prediction_start_time=row["prediction_start_time"],
                prediction_end_time=row["prediction_end_time"],
            ))
        return scenes

    def stack_flight_data_as_scene(self, flight_data: List[PredictionSample]) -> PredictionSample:
        """Stack a list of PredictionSample along the agent dimension into a single multi-agent PredictionSample.

        Produces tensors with shape [T_in, N, F], [H, N, F], etc., where N is the number of agents in the scene.
        """
        if not flight_data:
            raise ValueError("Cannot combine empty list of flights")

        def _stack_trajectory_as_scene(items: List[TrajectoryData]) -> TrajectoryData:
            # Stack [N, T, F] or [N, H, F], then permute to [T, N, F] or [H, N, F]
            enc = torch.stack([x.encoder_in for x in items], dim=0)
            dec = torch.stack([x.dec_in for x in items], dim=0)
            tgt = torch.stack([x.target for x in items], dim=0)
            return TrajectoryData(
                encoder_in=enc.permute(1, 0, 2),
                dec_in=dec.permute(1, 0, 2),
                target=tgt.permute(1, 0, 2),
            )

        return PredictionSample(
            xyz_positions=_stack_trajectory_as_scene([f.xyz_positions for f in flight_data]),
            xyz_deltas=_stack_trajectory_as_scene([f.xyz_deltas for f in flight_data]),
            trajectory=_stack_trajectory_as_scene([f.trajectory for f in flight_data]),
            target_padding_mask=torch.stack([f.target_padding_mask for f in flight_data], dim=1),
            target_rtd=torch.stack([f.target_rtd for f in flight_data]),
            last_input_pos_abs=torch.stack([f.last_input_pos_abs for f in flight_data]),
            runway=stack_runway_data([f.runway for f in flight_data]),
            flight_id=[f.flight_id for f in flight_data],
        )
