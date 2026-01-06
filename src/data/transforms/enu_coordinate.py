import torch
from pyproj import Transformer
import numpy as np


class ENUCoordinateTransform:
    def __init__(self, runway_lat: float, runway_lon: float, runway_alt: float):
        """
        Transform geodetic coordinates (latitude, longitude, altitude) to East, North, Up (ENU) coordinates.
        Expects input data to be in the following format: [lat, lon, alt, gs, track, vr]

        Args:
            runway_lat/lon/alt: The origin (0,0,0) of your local grid.
        Returns:
            A dictionary with the same keys as the input, but with the coordinates transformed to ENU coordinates.
            The first 3 columns of the input data are replaced with the ENU coordinates.
        """
        pipeline = (
            f"+proj=pipeline "
            f"+step +proj=cart +ellps=WGS84 "
            f"+step +proj=topocentric +lat_0={runway_lat} +lon_0={runway_lon} +h_0={runway_alt} "
        )
        self.transformer = Transformer.from_pipeline(pipeline)

    def __call__(self, sample: dict):
        for key in ["x", "y", "dec_in"]:
            data = sample[key] # [Seq, 6] -> [lat, lon, alt, gs, track, vr]
            
            # Extract Geodetic columns
            lats, lons, alts = data[:, 0].numpy(), data[:, 1].numpy(), data[:, 2].numpy()
            
            # Transform to East, North, Up (meters)
            e, n, u = self.transformer.transform(lats, lons, alts)
            
            # Replace the first 3 columns with ENU
            enu = torch.tensor(np.stack([e, n, u], axis=-1), dtype=torch.float32)
            sample[key] = torch.cat([enu, data[:, 3:]], dim=-1)
            
        return sample