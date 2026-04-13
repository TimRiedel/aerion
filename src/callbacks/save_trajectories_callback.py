import os

import pandas as pd
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig


class SaveTrajectoriesCallback(pl.Callback):
    """
    Saves predicted test trajectories to a parquet file after the test epoch.

    The parquet file is written to the Hydra runtime output directory so that
    it sits alongside other Hydra outputs for the run.
    """

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save accumulated test trajectory predictions to a parquet file."""
        output_dir = HydraConfig.get().runtime.output_dir
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(
            pl_module._test_predictions,
            columns=["flight_id", "timestamp", "x_coord", "y_coord", "altitude"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

        output_path = os.path.join(output_dir, "test_predictions.parquet")
        df.to_parquet(output_path, index=False)
