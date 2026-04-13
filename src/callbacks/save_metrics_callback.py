import os

import pytorch_lightning as pl


class SaveMetricsCallback(pl.Callback):
    """
    Saves per-trajectory metrics to parquet files after each validation epoch.

    Mirrors the behaviour of ModelCheckpoint: always overwrites
    last_metrics.parquet, and keeps best_metrics.parquet updated whenever the
    monitored metric improves.

    The parquet files are written to the same directory as the model checkpoints
    so that the best parquet and the best checkpoint always correspond.

    Usage (registered automatically by Trainer when a checkpoint config exists):
        SaveMetricsCallback(dirpath=".outputs/checkpoints/...", monitor="val_loss", mode="min")
    """

    def __init__(self, dirpath: str, monitor: str = "val_loss", mode: str = "min"):
        """
        Args:
            dirpath: Directory where parquet files are saved.
            monitor: Metric name to track for best-epoch selection (must match a
                     value logged via self.log() in the LightningModule).
            mode: "min" if lower is better, "max" if higher is better.
        """
        super().__init__()
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.best_score: float | None = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save per-trajectory metrics and update best/last parquet files."""
        os.makedirs(self.dirpath, exist_ok=True)

        df = pl_module.val_metrics.to_dataframe()

        last_path = os.path.join(self.dirpath, "last_metrics.parquet")
        df.to_parquet(last_path, index=False)

        # Optionally update the best metrics based on the monitored score
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is not None:
            score = float(current_score)
            is_best = (
                self.best_score is None
                or (self.mode == "min" and score < self.best_score)
                or (self.mode == "max" and score > self.best_score)
            )
            if is_best:
                self.best_score = score
                best_path = os.path.join(self.dirpath, "best_metrics.parquet")
                df.to_parquet(best_path, index=False)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save per-trajectory test metrics to a parquet file."""
        os.makedirs(self.dirpath, exist_ok=True)
        df = pl_module.test_metrics.to_dataframe()
        test_path = os.path.join(self.dirpath, "test_metrics.parquet")
        df.to_parquet(test_path, index=False)
