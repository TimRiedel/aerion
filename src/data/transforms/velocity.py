import torch

class VelocityTransform:
    def __init__(self, dt: float):
        """
        Compute velocity components (Speed_E, Speed_N, Vertical_Rate) in meters per second.

        Args:
            dt: Time step in seconds (used to compute vertical velocity from altitude differences)
        """
        self.dt = dt
        self.KNOTS_TO_MPS = 0.514444  # conversion factor

    def __call__(self, sample: dict):
        for key in ["x", "y", "dec_in"]:
            # Current columns: [E, N, altitude, gs, track, vertical_rate]
            data = sample[key]

            # --- 1. Horizontal velocity decomposition ---
            gs_mps = data[:, 3] * self.KNOTS_TO_MPS
            track_rad = torch.deg2rad(data[:, 4])

            speed_x = gs_mps * torch.sin(track_rad)  # East component
            speed_y = gs_mps * torch.cos(track_rad)  # North component

            # --- 2. Vertical velocity from altitude differences ---
            alt = data[:, 2]
            vertical_rate = torch.zeros_like(alt)
            if len(alt) > 1:
                delta_alt = alt[1:] - alt[:-1]
                v_z = delta_alt / self.dt
                vertical_rate[1:] = v_z
                vertical_rate[0] = v_z[0]  # backfill first element

            # --- 3. Stack into speed vector ---
            speed_vector = torch.stack([speed_x, speed_y, vertical_rate], dim=-1)
            sample[key] = torch.cat([data[:, :3], speed_vector], dim=-1)
        return sample