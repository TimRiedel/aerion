from abc import ABC, abstractmethod

from traffic.core import Traffic

from .scene import Scene


class SceneCreationStrategy(ABC):
    def __init__(self, input_time_minutes: int = 5, horizon_time_minutes: int = 40, min_horizon_time_minutes: int = 8):
        self.input_time_minutes = input_time_minutes
        self.horizon_time_minutes = horizon_time_minutes
        self.min_trajectory_length_minutes = input_time_minutes + min_horizon_time_minutes

    @abstractmethod
    def create_scenes(self, traffic: Traffic) -> list[Scene]:
        """
        Creates scenes based on the scene strategy.

        Args:
            traffic: Traffic object containing all flights.

        Returns:
            List of Scene objects containing the input and horizon flights.
        """
        pass


class FlightAppearsSceneCreationStrategy(SceneCreationStrategy):
    """
    Creates scenes when a new flight enters the region of interest.
    """
    def create_scenes(self, traffic: Traffic) -> list[Scene]:
        scenes = []
        for flight in traffic:
            start_time = flight.start
            if flight.duration.components.minutes < self.min_trajectory_length_minutes:
                continue
            scene = Scene(traffic, start_time, self.input_time_minutes, self.horizon_time_minutes)
            scenes.append(scene)
        return scenes


class SamplingSceneCreationStrategy(SceneCreationStrategy):
    """
    Creates scenes by drawing multiple samples from each flight.
    """
    def __init__(self, input_time_minutes: int = 5, horizon_time_minutes: int = 40, min_horizon_time_minutes: int = 8, data_period_minutes: int = 3):
        super().__init__(input_time_minutes, horizon_time_minutes, min_horizon_time_minutes)
        self.data_period_minutes = data_period_minutes

    def create_scenes(self, traffic: Traffic) -> list[Scene]:
        scenes = []
        for flight in traffic:
            start_time = flight.start
            if flight.duration.components.minutes < self.min_trajectory_length_minutes:
                continue

            flight_duration = flight.duration.components.minutes + 1
            num_prediction_samples = (flight_duration - self.min_trajectory_length_minutes) // self.data_period_minutes

            for sample_index in range(num_prediction_samples): # create a scene for each sample
                scene = Scene(traffic, start_time, self.input_time_minutes, self.horizon_time_minutes)
                scenes.append(scene)
                flight = flight.skip(minutes=self.data_period_minutes)
    
        return scenes
