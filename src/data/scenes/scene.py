from datetime import datetime, timedelta
import random

from traffic.core import Flight, Traffic

class Scene:
    """
    Represents a scene for trajectory prediction or analysis. 

    A scene contains:
    - All flights that are already present (i.e., have started) at the given input start time.
    - Only those flights whose total duration is at least (input_time_minutes + 1) minutes within the scene.
    - Each flight is split into two segments:
        * The "input" segment: the first `input_time_minutes` from the input start time.
        * The "horizon" segment: immediately follows the input segment and lasts `horizon_time_minutes`.
    """

    def __init__(self, traffic: Traffic, start_time: datetime, input_time_minutes: int, horizon_time_minutes: int):
        self.traffic = traffic
        self.input_time_minutes = input_time_minutes
        self.horizon_time_minutes = horizon_time_minutes

        self.input_start_time = start_time
        self.prediction_start_time = self.input_start_time + timedelta(minutes=input_time_minutes)
        self.prediction_end_time = self.prediction_start_time + timedelta(minutes=horizon_time_minutes)
        self.min_traffic_duration = self.input_time_minutes + 1  # Min length for all trajectories in the scene (input_time + 1 minute)

        self.input_flights, self.horizon_flights = self._get_flights()

    def __len__(self):
        """Number of flights in the scene."""
        return len(self.input_flights)

    def _get_flights(self) -> list[Flight]:
        """
        Gets all flights that are already present at the input start time (flight.start <= input_start_time)
        and that are at least min_traffic_duration_minutes long.
        """
        input_flights = []
        horizon_flights = []
        traffic_from_start_time = self.traffic.query(f"timestamp >= '{self.input_start_time}' and timestamp <= '{self.prediction_end_time}'")
        for flight in traffic_from_start_time:
            # Skip flights that are not yet present at the input start time or are too short
            if flight.start > self.input_start_time or flight.duration.components.minutes < self.min_traffic_duration:
                continue
            input_flight, horizon_flight = self._split_flight(flight)
            input_flights.append(input_flight)
            horizon_flights.append(horizon_flight)
        input_flights = self.permute_flights(input_flights)
        horizon_flights = self.permute_flights(horizon_flights)
        return input_flights, horizon_flights

    def _split_flight(self, flight: Flight) -> tuple[Flight, Flight]:
        """
        Splits the flight into input and horizon flights.
        """
        input_flight = flight.first(minutes=self.input_time_minutes)
        horizon_flight = flight.skip(minutes=self.input_time_minutes).first(minutes=self.horizon_time_minutes)
        return input_flight, horizon_flight

    def _permute_flights(self, flights: list[Flight]) -> list[Flight]:
        """
        Returns a permutation of the given flights using the input start time as the random seed.
        Same input start time yields the same permutation.
        Used to ensure that scenes with nearly the same input start time are ordered differently.
        """
        rng = random.Random(int(self.input_start_time.timestamp()))
        return rng.sample(flights, len(flights))