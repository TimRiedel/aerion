from functools import partial
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch.utils.data import Subset, random_split
from torchvision import transforms as T

from data.datasets import ApproachDataset, TrafficDataset
from data.features import FeatureSchema
from data.utils import collate_samples

logger = logging.getLogger(__name__)


class BaseData(pl.LightningDataModule, ABC):
    def __init__(
        self,
        dataset_cfg: DictConfig,
        processing_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        seed: int,
        feature_schema: FeatureSchema,
        max_num_agents: int,
        num_trajectories_to_predict: int | None = None,
        num_waypoints_to_predict: int | None = None,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.data_processing_cfg = processing_cfg
        self.num_trajectories_to_predict = num_trajectories_to_predict
        self.num_waypoints_to_predict = num_waypoints_to_predict
        self.seed = seed
        self.feature_schema = feature_schema
        self.max_num_agents = max_num_agents
        self.k_folds = self.dataset_cfg.get("k_folds", 1)
        self.cross_val_fold_idx = self.dataset_cfg.get("cross_val_fold_idx", 0)

        if self.k_folds < 1:
            raise ValueError(f"k_folds must be at least 1, got {self.k_folds}.")
        if self.k_folds > 1 and not (0 <= self.cross_val_fold_idx < self.k_folds):
            raise ValueError(
                f"cross_val_fold_idx must be in [0, {self.k_folds - 1}] when k_folds={self.k_folds}, "
                f"got {self.cross_val_fold_idx}."
            )

    @abstractmethod
    def _build_full_train_dataset(self) -> Any:
        """Create the full training dataset instance for this data module."""

    @abstractmethod
    def _get_collate_fn(self):
        """Return the collate function used by this data module."""

    def setup(self, stage: str):
        if stage == "fit":
            full_train_ds = self._build_full_train_dataset()

            train_ds, val_ds = self._split_dataset(full_train_ds)

            self.feature_schema.build_normalizers(train_ds)
            full_train_ds.transform = T.Compose(self._get_transforms())
            self.train_ds, self.val_ds = train_ds, val_ds

        if stage == "test":
            raise NotImplementedError("Test dataset and normalization are not implemented yet")

    def train_dataloader(self):
        return instantiate(
            self.dataloader_cfg,
            dataset=self.train_ds,
            collate_fn=self._get_collate_fn(),
        )

    def val_dataloader(self):
        return instantiate(
            self.dataloader_cfg,
            dataset=self.val_ds,
            shuffle=False,
            collate_fn=self._get_collate_fn(),
        )

    def test_dataloader(self):
        return instantiate(
            self.dataloader_cfg,
            dataset=self.test_ds,
            shuffle=False,
            collate_fn=self._get_collate_fn(),
        )

    def _split_dataset_random(self, dataset) -> Tuple[Subset, Subset]:
        val_pct: float = self.data_processing_cfg.get("validation_percentage", 0.2)
        if not (0.0 < val_pct < 1.0):
            raise ValueError(f"validation_percentage must be between 0 and 1, got {val_pct}.")

        generator = torch.Generator().manual_seed(self.seed)
        train_ds, val_ds = random_split(
            dataset,
            [1.0 - val_pct, val_pct],
            generator=generator,
        )
        return train_ds, val_ds

    def _split_dataset(self, dataset) -> Tuple[Subset, Subset]:
        if self.num_trajectories_to_predict is not None:
            all_indices = list(range(len(dataset)))
            subset = Subset(dataset, all_indices)
            return subset, subset

        if self.k_folds == 1:
            return self._split_dataset_random(dataset)

        days_by_month = dataset.get_days_by_month(self.k_folds)

        day_to_fold: dict = {}
        random_generator = random.Random(self.seed)

        sorted_month_keys = sorted(days_by_month.keys())
        for month_key in sorted_month_keys:
            month_days = list(days_by_month[month_key])
            random_generator.shuffle(month_days)

            number_of_days_in_month = len(month_days)

            fold_sizes = [number_of_days_in_month // self.k_folds] * self.k_folds
            for fold_index in range(number_of_days_in_month % self.k_folds):
                fold_sizes[fold_index] += 1

            start_index = 0
            for fold_index in range(self.k_folds):
                end_index = start_index + fold_sizes[fold_index]
                for day in month_days[start_index:end_index]:
                    day_to_fold[day] = fold_index
                start_index = end_index

        training_indices: list[int] = []
        validation_indices: list[int] = []

        for index in range(len(dataset)):
            day_for_index = dataset.get_day_for_index(index)
            fold_index_for_day = day_to_fold[day_for_index]

            if fold_index_for_day == self.cross_val_fold_idx:
                validation_indices.append(index)
            else:
                training_indices.append(index)

        all_indices_set = set(range(len(dataset)))
        training_index_set = set(training_indices)
        validation_index_set = set(validation_indices)

        if not training_index_set.isdisjoint(validation_index_set):
            raise ValueError("Training and validation indices are not disjoint.")

        if training_index_set.union(validation_index_set) != all_indices_set:
            raise ValueError("Training and validation indices do not cover all dataset indices.")

        train_ds = Subset(dataset, training_indices)
        val_ds = Subset(dataset, validation_indices)
        return train_ds, val_ds

    def _get_transforms(self) -> List[Any]:
        transforms = []
        transforms.extend(self.feature_schema.build_transforms())
        return transforms


class TrafficData(BaseData):
    def _build_full_train_dataset(self) -> TrafficDataset:
        return TrafficDataset(
            trajectories_path=self.dataset_cfg.trajectories_path,
            scenes_path=self.dataset_cfg.scenes_path,
            flightinfo_path=self.dataset_cfg.flightinfo_path,
            input_time_minutes=self.dataset_cfg.input_time_minutes,
            horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
            resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
            max_num_agents=self.max_num_agents,
            feature_schema=self.feature_schema,
            num_trajectories_to_predict=self.num_trajectories_to_predict,
            num_waypoints_to_predict=self.num_waypoints_to_predict,
        )

    def _get_collate_fn(self):
        return partial(collate_samples, max_agents=self.max_num_agents)

class ApproachData(BaseData):
    def _build_full_train_dataset(self) -> ApproachDataset:
        return ApproachDataset(
            trajectories_path=self.dataset_cfg.trajectories_path,
            scenes_path=self.dataset_cfg.scenes_path,
            max_num_agents=self.max_num_agents,
            flightinfo_path=self.dataset_cfg.flightinfo_path,
            input_time_minutes=self.dataset_cfg.input_time_minutes,
            horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
            resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
            feature_schema=self.feature_schema,
            num_trajectories_to_predict=self.num_trajectories_to_predict,
            num_waypoints_to_predict=self.num_waypoints_to_predict,
        )

    def _get_collate_fn(self):
        return collate_samples