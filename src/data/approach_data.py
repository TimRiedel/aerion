import logging
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split
from torchvision import transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig

from data.datasets.approach_dataset import ApproachDataset
from data.transforms import *

logger = logging.getLogger(__name__)


class ApproachData(pl.LightningDataModule):
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

    @property
    def feature_groups(self):
        """Define feature groups to compute stats for.
        
        Override this method in subclasses to customize which features to track.
        
        Returns:
            dict: Mapping of feature group names to extraction functions.
                  Each function takes a sample and returns a list of tensors to concatenate.
                  Must contain 'pos' and 'delta' groups.
        """
        return {
            'pos': lambda sample: [sample['input_traj'][:, :3]],
            'delta': lambda sample: [
                sample['input_traj'][:, 3:6],
                sample['target_traj'],
            ]
        }


    def setup(self, stage: str):
        if stage == "fit":
            full_train_ds = ApproachDataset(
                inputs_path=self.dataset_cfg.train_inputs_path,
                horizons_path=self.dataset_cfg.train_horizons_path,
                flightinfo_path=self.dataset_cfg.flightinfo_path,
                input_time_minutes=self.dataset_cfg.input_time_minutes,
                horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
                resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
                num_trajectories_to_predict=self.num_trajectories_to_predict,
                num_waypoints_to_predict=self.num_waypoints_to_predict,
            )

            val_pct = self.dataset_cfg.get('val_pct', 0.1)
            train_ds, val_ds = self._split_dataset(full_train_ds, val_pct)

            self._compute_feature_stats(train_ds)
            full_train_ds.transform = T.Compose(self._get_transforms()) # Important: set transform on full train dataset, not on subsets
            self.train_ds, self.val_ds = train_ds, val_ds

        if stage == "test":
            if any(s is None for s in [self.pos_mean, self.pos_std, self.delta_mean, self.delta_std]):
                raise ValueError("Normalization stats (pos_mean, pos_std, delta_mean, delta_std) must be provided for test dataset.")

            self.test_ds = ApproachDataset(
                inputs_path=self.dataset_cfg.test_inputs_path,
                horizons_path=self.dataset_cfg.test_horizons_path,
                flightinfo_path=self.dataset_cfg.flightinfo_path,
                input_time_minutes=self.dataset_cfg.input_time_minutes,
                horizon_time_minutes=self.dataset_cfg.horizon_time_minutes,
                resampling_rate_seconds=self.dataset_cfg.resampling_rate_seconds,
                num_trajectories_to_predict=self.num_trajectories_to_predict,
                num_waypoints_to_predict=self.num_waypoints_to_predict,
                transform=self._get_transforms(),
            )
            self.test_ds.transform = T.Compose(self._get_transforms())

    def train_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.train_ds)


    def val_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.val_ds, shuffle=False)


    def test_dataloader(self):
        return instantiate(self.dataloader_cfg, dataset=self.test_ds, shuffle=False)


    def _split_dataset(self, dataset: ApproachDataset, val_pct: float) -> tuple[ApproachDataset, ApproachDataset]:
        if self.num_trajectories_to_predict is None:
            generator = torch.Generator().manual_seed(self.seed)
            train_ds, val_ds = random_split(dataset, [1 - val_pct, val_pct], generator=generator)
        else:
            train_ds = val_ds = dataset
        return train_ds, val_ds

    
    def _compute_feature_stats(self, dataset: torch.utils.data.Dataset):
        """Compute stats for all feature groups defined in feature_groups.
        
        Returns:
            Tuple of (mean, std) pairs for each feature group, flattened.
        """
        accumulators = {name: [] for name in self.feature_groups.keys()}
        
        for i in range(len(dataset)):
            sample = dataset[i]
            for name, extractor in self.feature_groups.items():
                tensors = extractor(sample)
                accumulators[name].extend(tensors)
        
        # Compute mean and std for each feature group
        stats = []
        for name in self.feature_groups.keys():
            all_features = torch.cat(accumulators[name], dim=0)
            mean = all_features.mean(dim=0)
            std = all_features.std(dim=0).clamp(min=1e-2)
            stats.extend([mean, std])
            setattr(self, f'{name}_mean', mean)
            setattr(self, f'{name}_std', std)

    def _get_transforms(self):
        transforms = []
        if self.data_processing_cfg.get("noise", None) is not None:
            noise_std_x = self.data_processing_cfg.noise.std_x
            noise_std_y = self.data_processing_cfg.noise.std_y
            noise_std_alt = self.data_processing_cfg.noise.std_alt
            transforms.append(DecoderInputNoise(noise_std=torch.tensor([noise_std_x, noise_std_y, noise_std_alt])))

        transforms.append(FeatureSliceNormalizer(name="input_traj", indices=(0, 3), mean=self.pos_mean, std=self.pos_std))
        transforms.append(FeatureSliceNormalizer(name="input_traj", indices=(3, 6), mean=self.delta_mean, std=self.delta_std))
        transforms.append(FeatureSliceNormalizer(name="dec_in_traj", indices=(0, 3), mean=self.delta_mean, std=self.delta_std))
        transforms.append(FeatureSliceNormalizer(name="target_traj", indices=(0, 3), mean=self.delta_mean, std=self.delta_std))
        return transforms