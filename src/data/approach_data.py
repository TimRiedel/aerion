import logging
from typing import Any, List

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import random_split
from torchvision import transforms as T

from data.collate import collate_samples
from data.datasets.approach_dataset import ApproachDataset
from data.features.feature_schema import FeatureSchema

logger = logging.getLogger(__name__)


class ApproachData(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cfg: DictConfig,
        processing_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        seed: int,
        feature_schema: FeatureSchema,
        num_trajectories_to_predict: int = None,
        num_waypoints_to_predict: int = None,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.data_processing_cfg = processing_cfg
        self.num_trajectories_to_predict = num_trajectories_to_predict
        self.num_waypoints_to_predict = num_waypoints_to_predict
        self.seed = seed
        self.feature_schema = feature_schema

        if self.dataset_cfg.train_inputs_path == self.dataset_cfg.test_inputs_path or self.dataset_cfg.train_horizons_path == self.dataset_cfg.test_horizons_path:
            logger.warning("⚠️  Train and test inputs or horizons paths are the same. Make sure to test on a different dataset for correct evaluation.")

    def setup(self, stage: str):
        if stage == "fit":
            full_train_ds = ApproachDataset(
                inputs_path=self.dataset_cfg.train_inputs_path,
                horizons_path=self.dataset_cfg.train_horizons_path,
                flightinfo_path=self.dataset_cfg.flightinfo_path,
                input_time_minutes=self.dataset_cfg.input_time_minutes,
                horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
                resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
                feature_schema=self.feature_schema,
                num_trajectories_to_predict=self.num_trajectories_to_predict,
                num_waypoints_to_predict=self.num_waypoints_to_predict,
            )

            val_pct = self.dataset_cfg.get("val_pct", 0.1)
            train_ds, val_ds = self._split_dataset(full_train_ds, val_pct)

            self.feature_schema.build_normalizers(train_ds)
            full_train_ds.transform = T.Compose(self._get_transforms()) # Important: set transform on full train dataset, not on subsets
            self.train_ds, self.val_ds = train_ds, val_ds

        if stage == "test":
            raise NotImplementedError("Test dataset and normalization are not implemented yet")

    def train_dataloader(self):
        return instantiate(
            self.dataloader_cfg,
            dataset=self.train_ds,
            collate_fn=collate_samples,
        )

    def val_dataloader(self):
        return instantiate(
            self.dataloader_cfg,
            dataset=self.val_ds,
            shuffle=False,
            collate_fn=collate_samples,
        )

    def test_dataloader(self):
        return instantiate(
            self.dataloader_cfg,
            dataset=self.test_ds,
            shuffle=False,
            collate_fn=collate_samples,
        )

    def _split_dataset(self, dataset: ApproachDataset, val_pct: float) -> tuple[ApproachDataset, ApproachDataset]:
        if self.num_trajectories_to_predict is None:
            generator = torch.Generator().manual_seed(self.seed)
            train_ds, val_ds = random_split(dataset, [1 - val_pct, val_pct], generator=generator)
        else:
            train_ds = val_ds = dataset
        return train_ds, val_ds

    def _get_transforms(self) -> List[Any]:
        transforms = []
        transforms.extend(self.feature_schema.build_transforms())
        return transforms
