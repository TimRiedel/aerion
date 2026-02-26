from typing import Callable, List, Optional

import pandas as pd
import torch
from traffic.core import Traffic

from data.features import FeatureSchema
from data.interface import PredictionSample, TrajectoryData
from data.scenes import SceneCreationStrategy
from data.datasets import BaseDataset
from data.collate import stack_runway_data

class TrafficDataset(BaseDataset):
    def __init__(
        self,
        resampled_path: str,
        flightinfo_path: str,
        input_time_minutes: int,
        horizon_time_minutes: int,
        resampling_rate_seconds: int,
        feature_schema: FeatureSchema,
        scene_creation_strategy: SceneCreationStrategy,
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

        # Read traffic data
        traffic_df = pd.read_parquet(resampled_path).sort_values(["flight_id", "timestamp"]).reset_index(drop=True)
        traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"])
        self.traffic = Traffic(traffic_df)

        # Create scenes
        self.scene_creation_strategy = scene_creation_strategy
        self.scenes = self.scene_creation_strategy.create_scenes(self.traffic)
        self.size = len(self.scenes)

    def __getitem__(self, idx: int) -> PredictionSample:
        scene = self.scenes[idx]
        flight_data_list = []
        for input_flight, horizon_flight in zip(scene.input_flights, scene.horizon_flights):
            input_df = input_flight.data
            horizon_df = horizon_flight.data
            flight_data = self.get_flight_data(input_df, horizon_df)
            flight_data_list.append(flight_data)

        sample = self.stack_flight_data_as_scene(flight_data_list)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

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