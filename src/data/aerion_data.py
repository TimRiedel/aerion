import logging
from typing import override
from data.transforms.normalize import FeatureSliceNormalizer
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split
from torchvision import transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig

from data.transforms import *
from data.approach_data import ApproachData
from data.datasets.aerion_dataset import AerionDataset

logger = logging.getLogger(__name__)


class AerionData(ApproachData):
    def __init__(self,
        dataset_cfg: DictConfig,
        processing_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        seed: int,
        pos_mean: torch.Tensor = None,
        pos_std: torch.Tensor = None,
        delta_mean: torch.Tensor = None,
        delta_std: torch.Tensor = None,
        dist_mean: torch.Tensor = None,
        dist_std: torch.Tensor = None,
        num_trajectories_to_predict: int = None,
        num_waypoints_to_predict: int = None,
        contexts_cfg: DictConfig = None,
    ):
        super().__init__(
            dataset_cfg=dataset_cfg,
            processing_cfg=processing_cfg,
            dataloader_cfg=dataloader_cfg,
            seed=seed,
            pos_mean=pos_mean,
            pos_std=pos_std,
            delta_mean=delta_mean,
            delta_std=delta_std,
            num_trajectories_to_predict=num_trajectories_to_predict,
            num_waypoints_to_predict=num_waypoints_to_predict,
        )
        self.contexts_cfg = contexts_cfg or {}
        self.dist_mean = dist_mean
        self.dist_std = dist_std

    @property
    def feature_groups(self):
        feature_groups = super().feature_groups
        feature_groups['dist'] = lambda sample: [sample['input_traj'][:, 6:14], sample['dec_in_traj'][:, 3:11]]
        return feature_groups

    def setup(self, stage: str):
        if stage == "fit":
            full_train_ds = AerionDataset(
                inputs_path=self.dataset_cfg.train_inputs_path,
                horizons_path=self.dataset_cfg.train_horizons_path,
                flightinfo_path=self.dataset_cfg.get("flightinfo_path"),
                input_time_minutes=self.dataset_cfg.input_time_minutes,
                horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
                resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
                num_trajectories_to_predict=self.num_trajectories_to_predict,
                num_waypoints_to_predict=self.num_waypoints_to_predict,
                contexts_cfg=self.contexts_cfg,
            )

            val_pct = self.dataset_cfg.get('val_pct', 0.1)
            train_ds, val_ds = self._split_dataset(full_train_ds, val_pct)

            self._compute_feature_stats(train_ds)
            full_train_ds.transform = T.Compose(self._get_transforms()) # Important: set transform on full train dataset, not on subsets
            self.train_ds, self.val_ds = train_ds, val_ds
            
        if stage == "test":
            if any(s is None for s in [self.pos_mean, self.pos_std, self.delta_mean, self.delta_std, self.dist_mean, self.dist_std]):
                raise ValueError("Normalization stats (pos_mean, pos_std, delta_mean, delta_std, dist_mean, dist_std) must be provided for test dataset.")

            self.test_ds = AerionDataset(
                inputs_path=self.dataset_cfg.test_inputs_path,
                horizons_path=self.dataset_cfg.test_horizons_path,
                flightinfo_path=self.dataset_cfg.get("flightinfo_path"),
                input_time_minutes=self.dataset_cfg.input_time_minutes,
                horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
                resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
                num_trajectories_to_predict=self.num_trajectories_to_predict,
                num_waypoints_to_predict=self.num_waypoints_to_predict,
                transform=T.Compose(self._get_transforms()),
                contexts_cfg=self.contexts_cfg,
            )


    def _get_transforms(self):
        transforms = super()._get_transforms()
        transforms.append(FeatureSliceNormalizer(name="input_traj", indices=(6, 14), mean=self.dist_mean, std=self.dist_std))
        transforms.append(FeatureSliceNormalizer(name="dec_in_traj", indices=(3, 11), mean=self.dist_mean, std=self.dist_std))
        return transforms