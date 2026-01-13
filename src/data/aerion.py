import logging
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split
from torchvision import transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig

from data.approach_dataset import ApproachDataset
from data.transforms import *

logger = logging.getLogger(__name__)


class AerionData(pl.LightningDataModule):
    def __init__(self,
        dataset_cfg: DictConfig,
        processing_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        seed: int,
        norm_mean: torch.Tensor = None,
        norm_std: torch.Tensor = None,
        num_trajectories_to_predict: int = None,
        num_waypoints_to_predict: int = None,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.processing_cfg = processing_cfg
        self.num_trajectories_to_predict = num_trajectories_to_predict
        self.num_waypoints_to_predict = num_waypoints_to_predict

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
                num_trajectories_to_predict=self.num_trajectories_to_predict,
                num_waypoints_to_predict=self.num_waypoints_to_predict,
            )

            # Split train and validation datasets, but only if num_trajectories_to_predict is not set for debugging
            if self.num_trajectories_to_predict is None:
                val_pct = self.processing_cfg.validation_percentage
                train_ds, val_ds = random_split(
                    full_train_ds, [1 - val_pct, val_pct], generator=torch.Generator().manual_seed(self.seed)
                )
            else:
                train_ds = val_ds = full_train_ds

            # Normalization transforms
            self.norm_mean, self.norm_std = self._compute_feature_stats(train_ds)
            full_train_ds.transform = self._get_transform(self.norm_mean, self.norm_std) # Important: set transform on full train dataset, not on subsets
            self.train_ds, self.val_ds = train_ds, val_ds

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
                num_trajectories_to_predict=self.num_trajectories_to_predict,
                num_waypoints_to_predict=self.num_waypoints_to_predict,
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
        all_feat = []
        for i in range(len(dataset)):
            sample = dataset[i]
            all_feat.append(sample["x"])
            all_feat.append(sample["y"])
        feat_all = torch.cat(all_feat, dim=0)
        mean = feat_all.mean(dim=0)
        std = feat_all.std(dim=0)
        # Avoid near-zero std -> huge normalized values
        std = std.clamp(min=1e-2)
        return mean, std

    def _get_transform(self, mean: torch.Tensor, std: torch.Tensor):
        return T.Compose([ZScoreNormalize(mean=mean, std=std)])