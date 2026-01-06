import torch

class ENUVelocityTransform:
    def __init__(self, dt: float):
        """
        Transform velocity data from knots to meters per second and computes velocity components in the East, North, and Up directions.
        Expects data to be in ENU coordinates.

        Args:
            dt: The resampling rate in seconds (e.g., 1.0 for 1Hz). 
                Crucial for turning difference into rate (m/s).
        Returns:
            A dictionary with the same keys as the input, but with the velocity components in the East, North, and Up directions added.
        """
        self.dt = dt

    def __call__(self, sample: dict):
        for key in ["x", "y", "dec_in"]:
            # Current columns: [E, N, U, gs, track, raw_vr]
            data = sample[key] 
            
            # 1. Horizontal Velocity Decomposition
            KNOTS_TO_MPS = 0.514444
            gs_mps = data[:, 3] * KNOTS_TO_MPS
            track_rad = torch.deg2rad(data[:, 4])
            
            v_e = gs_mps * torch.sin(track_rad)
            v_n = gs_mps * torch.cos(track_rad)
            
            # 2. Vertical Velocity from Altitude Difference
            u_coords = data[:, 2]
            u_diff = u_coords[1:] - u_coords[:-1]
            v_u = u_diff / self.dt
            
            # 3. Match Sequence Length
            # Since diff loses one element, we back-fill the first element 
            # to maintain the shape.
            v_u = torch.cat([v_u[0:1], v_u], dim=0)

            # 4. Final Feature Assembly
            vel_cartesian = torch.stack([v_e, v_n, v_u], dim=-1)
            sample[key] = torch.cat([data[:, :3], vel_cartesian], dim=-1)
            
        return sample