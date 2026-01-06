import logging
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split
from torchvision import transforms as T
from traffic.data import airports
from hydra.utils import instantiate
from omegaconf import DictConfig

from data.approach_dataset import ApproachDataset
from data.transforms import *

logger = logging.getLogger(__name__)


class AerionData(pl.LightningDataModule):
    def __init__(self, dataset_cfg: DictConfig, processing_cfg: DictConfig, dataloader_cfg: DictConfig, seed: int, norm_mean: torch.Tensor = None, norm_std: torch.Tensor = None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.processing_cfg = processing_cfg

        # TODO: refactor this -> move ENU transforms to FlightFusion
        lat, lon = airports[self.processing_cfg.icao_code].latlon
        runway_alt = airports[self.processing_cfg.icao_code].altitude
        self.base_transform = T.Compose([
            ENUCoordinateTransform(
                runway_lat=lat,
                runway_lon=lon,
                runway_alt=runway_alt
            ),
            ENUVelocityTransform(
                dt=self.dataset_cfg.resampling_rate_seconds
            )
        ])

        self.seed = seed
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        if self.dataset_cfg.train_inputs_path == self.dataset_cfg.test_inputs_path or self.dataset_cfg.train_horizons_path == self.dataset_cfg.test_horizons_path:
            logger.warning("⚠️  Train and test inputs or horizons paths are the same. Make sure to test on a different dataset for correct evaluation.")


    def setup(self, stage: str):
        if stage == "fit":
            full_train_ds = ApproachDataset(
                inputs_path=self.dataset_cfg.train_inputs_path,
                horizons_path=self.dataset_cfg.train_horizons_path,
                input_time_minutes=self.dataset_cfg.input_time_minutes,
                horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
                resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
                feature_cols=self.processing_cfg.feature_cols,
            )

            # Split train and validation datasets
            val_pct = self.processing_cfg.validation_percentage
            self.train_ds, self.val_ds = random_split(
                full_train_ds, [1 - val_pct, val_pct], generator=torch.Generator().manual_seed(self.seed)
            )

            # Normalization transforms
            self.norm_mean, self.norm_std = self._compute_feature_stats(self.train_ds)
            self.train_ds.transform = self._get_transform(self.norm_mean, self.norm_std)
            self.val_ds.transform = self._get_transform(self.norm_mean, self.norm_std)

        if stage == "test":
            if self.norm_mean is None or self.norm_std is None:
                raise ValueError("Normalization mean and std must be provided for test dataset.")

            self.test_ds = ApproachDataset(
                inputs_path=self.dataset_cfg.test_inputs_path,
                horizons_path=self.dataset_cfg.test_horizons_path,
                input_time_minutes=self.dataset_cfg.input_time_minutes,
                horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
                resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
                feature_cols=self.processing_cfg.feature_cols,
                transform=self._get_transform(self.norm_mean, self.norm_std),
            )


    def train_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.train_ds)


    def val_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.val_ds, shuffle=False)


    def test_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.test_ds, shuffle=False)

    def _compute_feature_stats(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-feature mean and std for the trajectory data."""
        all_x = []
        for i in range(len(dataset)):
            sample = dataset[i]
            all_x.append(sample["x"])
        x_all = torch.cat(all_x, dim=0)
        mean = x_all.mean(dim=0)
        std = x_all.std(dim=0)
        return mean, std

    def _get_transform(self, mean: torch.Tensor, std: torch.Tensor):
        return T.Compose([self.base_transform, ZScoreNormalize(mean=mean, std=std)])