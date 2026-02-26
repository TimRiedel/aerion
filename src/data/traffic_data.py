import logging
from functools import partial
from typing import Any, List

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import random_split
from torchvision import transforms as T

from data.collate import collate_samples
from data.datasets.traffic_dataset import TrafficDataset
from data.features.feature_schema import FeatureSchema
from data.scenes import SceneCreationStrategy

logger = logging.getLogger(__name__)


class TrafficData(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cfg: DictConfig,
        processing_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        scene_creation_strategy: SceneCreationStrategy,
        seed: int,
        feature_schema: FeatureSchema,
        max_n_agents: int = 15,
        num_trajectories_to_predict: int = None,
        num_waypoints_to_predict: int = None,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.data_processing_cfg = processing_cfg
        self.scene_creation_strategy = scene_creation_strategy
        self.max_n_agents = max_n_agents
        self.num_trajectories_to_predict = num_trajectories_to_predict
        self.num_waypoints_to_predict = num_waypoints_to_predict
        self.seed = seed
        self.feature_schema = feature_schema

        train_resampled = dataset_cfg.get("train_resampled_path", dataset_cfg.get("resampled_path"))
        test_resampled = dataset_cfg.get("test_resampled_path", train_resampled)
        if train_resampled == test_resampled:
            logger.warning("⚠️  Train and test resampled paths are the same. Make sure to test on a different dataset for correct evaluation.")

    def setup(self, stage: str):
        if stage == "fit":

            resampled_path = self.dataset_cfg.get("train_resampled_path", self.dataset_cfg.resampled_path)
            full_train_ds = TrafficDataset(
                resampled_path=resampled_path,
                flightinfo_path=self.dataset_cfg.flightinfo_path,
                input_time_minutes=self.dataset_cfg.input_time_minutes,
                horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
                resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
                feature_schema=self.feature_schema,
                scene_creation_strategy=self.scene_creation_strategy,
                num_trajectories_to_predict=self.num_trajectories_to_predict,
                num_waypoints_to_predict=self.num_waypoints_to_predict,
            )

            val_pct = self.dataset_cfg.get("val_pct", 0.1)
            train_ds, val_ds = self._split_dataset(full_train_ds, val_pct)

            self.feature_schema.build_normalizers(train_ds)
            full_train_ds.transform = T.Compose(self._get_transforms())
            self.train_ds, self.val_ds = train_ds, val_ds

        if stage == "test":
            raise NotImplementedError("Test dataset and normalization are not implemented yet")

    def train_dataloader(self):
        return instantiate(
            self.dataloader_cfg,
            dataset=self.train_ds,
            collate_fn=partial(collate_samples, max_n_agents=self.max_n_agents),
        )

    def val_dataloader(self):
        return instantiate(
            self.dataloader_cfg,
            dataset=self.val_ds,
            shuffle=False,
            collate_fn=partial(collate_samples, max_n_agents=self.max_n_agents),
        )

    def test_dataloader(self):
        return instantiate(
            self.dataloader_cfg,
            dataset=self.test_ds,
            shuffle=False,
            collate_fn=partial(collate_samples, max_n_agents=self.max_n_agents),
        )

    def _split_dataset(self, dataset: TrafficDataset, val_pct: float):
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
