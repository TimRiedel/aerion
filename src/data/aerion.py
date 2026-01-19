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
        pos_mean: torch.Tensor = None,
        pos_std: torch.Tensor = None,
        delta_mean: torch.Tensor = None,
        delta_std: torch.Tensor = None,
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
        self.pos_mean = pos_mean
        self.pos_std = pos_std
        self.delta_mean = delta_mean
        self.delta_std = delta_std

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
                feature_cols=self.data_processing_cfg.feature_cols,
                num_trajectories_to_predict=self.num_trajectories_to_predict,
                num_waypoints_to_predict=self.num_waypoints_to_predict,
            )

            # Split train and validation datasets, but only if num_trajectories_to_predict is not set for debugging
            if self.num_trajectories_to_predict is None:
                val_pct = self.data_processing_cfg.validation_percentage
                train_ds, val_ds = random_split(
                    full_train_ds, [1 - val_pct, val_pct], generator=torch.Generator().manual_seed(self.seed)
                )
            else:
                train_ds = val_ds = full_train_ds

            self.pos_mean, self.pos_std, self.delta_mean, self.delta_std = self._compute_feature_stats(train_ds)
            full_train_ds.transform = self._get_transform() # Important: set transform on full train dataset, not on subsets
            self.train_ds, self.val_ds = train_ds, val_ds

        if stage == "test":
            if any(s is None for s in [self.pos_mean, self.pos_std, self.delta_mean, self.delta_std]):
                raise ValueError("Normalization stats (pos_mean, pos_std, delta_mean, delta_std) must be provided for test dataset.")

            self.test_ds = ApproachDataset(
                inputs_path=self.dataset_cfg.test_inputs_path,
                horizons_path=self.dataset_cfg.test_horizons_path,
                input_time_minutes=self.dataset_cfg.input_time_minutes,
                horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
                resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
                feature_cols=self.data_processing_cfg.feature_cols,
                num_trajectories_to_predict=self.num_trajectories_to_predict,
                num_waypoints_to_predict=self.num_waypoints_to_predict,
                transform=self._get_transform(),
            )


    def train_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.train_ds)


    def val_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.val_ds, shuffle=False)


    def test_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.test_ds, shuffle=False)

    def _compute_feature_stats(self, dataset: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute separate stats for positions and deltas.
        
        Returns:
            pos_mean: Mean of absolute positions [3]
            pos_std: Std of absolute positions [3]
            delta_mean: Mean of all deltas (input + output) [3]
            delta_std: Std of all deltas (input + output) [3]
        """
        all_pos = []    # For position stats (from x[:, :3])
        all_delta = []  # For delta stats (from x[:, 3:6] and y)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            x = sample["x"]  # [T_in, 6] - positions + deltas
            y = sample["y"]  # [H, 3] - target deltas
            
            all_pos.append(x[:, :3])      # Absolute positions from input
            all_delta.append(x[:, 3:6])   # Deltas from input sequence
            all_delta.append(y)           # Deltas from target sequence
        
        # Position stats
        pos_all = torch.cat(all_pos, dim=0)
        pos_mean = pos_all.mean(dim=0)
        pos_std = pos_all.std(dim=0).clamp(min=1e-2)
        
        # Delta stats (unified for all deltas)
        delta_all = torch.cat(all_delta, dim=0)
        delta_mean = delta_all.mean(dim=0)
        delta_std = delta_all.std(dim=0).clamp(min=1e-2)
        
        return pos_mean, pos_std, delta_mean, delta_std

    def _get_transform(self):
        transforms = []
        if self.data_processing_cfg.get("noise", None) is not None:
            noise_std_x = self.data_processing_cfg.noise.std_x
            noise_std_y = self.data_processing_cfg.noise.std_y
            noise_std_alt = self.data_processing_cfg.noise.std_alt
            transforms.append(DecoderInputNoise(noise_std=torch.tensor([noise_std_x, noise_std_y, noise_std_alt])))

        transforms.append(DeltaAwareNormalize(
            pos_mean=self.pos_mean, 
            pos_std=self.pos_std, 
            delta_mean=self.delta_mean, 
            delta_std=self.delta_std
        ))

        return T.Compose(transforms)