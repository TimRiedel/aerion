from pyproj import Transformer
import torch
import numpy as np


class ENCoordinateTransform:
    def __init__(self, ref_lat: float, ref_lon: float):
        """
        Convert geodetic coordinates (lat, lon) to local East/North (meters),
        leaving all other features (including altitude) untouched.
        """
        self.transformer = Transformer.from_crs(
            crs_from="EPSG:4326",  # lat/lon only
            crs_to=f"+proj=aeqd +lat_0={ref_lat} +lon_0={ref_lon} +ellps=WGS84",
            always_xy=True
        )

    def __call__(self, sample: dict):
        for key in ["x", "y", "dec_in"]:
            data = sample[key]  # [Seq, 6] -> lat, lon, alt, gs, track, vr

            lons = data[:, 1].detach().cpu().numpy()
            lats = data[:, 0].detach().cpu().numpy()

            e, n = self.transformer.transform(lons, lats)

            # Replace the first two columns (lat/lon → E/N)
            en_tensor = torch.tensor(np.stack([e, n], axis=-1), dtype=data.dtype)
            sample[key] = torch.cat([en_tensor, data[:, 2:]], dim=-1)

        return sample