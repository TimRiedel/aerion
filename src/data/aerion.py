import logging
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split
from hydra.utils import instantiate
from omegaconf import DictConfig

from data.approach_dataset import ApproachDataset

logger = logging.getLogger(__name__)


class AerionData(pl.LightningDataModule):
    def __init__(self, dataset_cfg: DictConfig, processing_cfg: DictConfig, dataloader_cfg: DictConfig, seed: int):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.processing_cfg = processing_cfg
        self.transform = None

        self.seed = seed

        if self.dataset_cfg.train_inputs_path == self.dataset_cfg.test_inputs_path or self.dataset_cfg.train_horizons_path == self.dataset_cfg.test_horizons_path:
            logger.warning("⚠️  Train and test inputs or horizons paths are the same. Make sure to test on a different dataset for correct evaluation.")


    def setup(self, stage: str):
        if stage == "test":
            self.test_ds = ApproachDataset(
                inputs_path=self.dataset_cfg.test_inputs_path,
                horizons_path=self.dataset_cfg.test_horizons_path,
                input_time_minutes=self.dataset_cfg.input_time_minutes,
                horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
                resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
                feature_cols=self.processing_cfg.feature_cols,
                transform=self.transform,
            )

        if stage == "fit":
            full_train_ds = ApproachDataset(
                inputs_path=self.dataset_cfg.train_inputs_path,
                horizons_path=self.dataset_cfg.train_horizons_path,
                input_time_minutes=self.dataset_cfg.input_time_minutes,
                horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
                resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
                feature_cols=self.processing_cfg.feature_cols,
                transform=self.transform,
            )

            val_pct = self.processing_cfg.validation_percentage
            self.train_ds, self.val_ds = random_split(
                full_train_ds, [1 - val_pct, val_pct], generator=torch.Generator().manual_seed(self.seed)
            )


    def train_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.train_ds)


    def val_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.val_ds)


    def test_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.test_ds)