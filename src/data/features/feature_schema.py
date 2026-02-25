from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import Dataset

from data.features.feature_group import (
    FEATURE_REGISTRY,
    Altitude,
    DeltaXYZ,
    FeatureGroup,
    XYPosition,
)
from data.interface import RunwayData
from data.transforms.normalize import Denormalizer, FeatureSliceNormalizer, Normalizer


class FeatureSchema:
    def __init__(self, config: DictConfig) -> None:
        """
        Args:
            config: Feature config (e.g., from feature/aerion.yaml)
                    Must contain keys: encoder_inputs, decoder_inputs, outputs
                    May contain: params (shared params like centerline_distances)
        """
        params = dict(config.get("params", {}))
        encoder_input_names = list(config["encoder_inputs"])
        decoder_input_names = list(config["decoder_inputs"])

        self.encoder_groups: List[FeatureGroup] = []
        self.decoder_groups: List[FeatureGroup] = []
        self.output_groups: List[FeatureGroup] = [DeltaXYZ(start_idx=0)]

        self.encoder_by_name: Dict[str, FeatureGroup] = {}
        self.decoder_by_name: Dict[str, FeatureGroup] = {}

        start_idx = 0
        for group_name in encoder_input_names:
            cls = FEATURE_REGISTRY[group_name]
            group = cls(start_idx=start_idx, params=params)
            self.encoder_groups.append(group)
            self.encoder_by_name[group_name] = group
            start_idx = start_idx + group.width

        start_idx = 0
        for group_name in decoder_input_names:
            cls = FEATURE_REGISTRY[group_name]
            group = cls(start_idx=start_idx, params=params)
            self.decoder_groups.append(group)
            self.decoder_by_name[group_name] = group
            start_idx = start_idx + group.width


    @property
    def encoder_dim(self) -> int:
        return sum(group.width for group in self.encoder_groups)

    @property
    def decoder_dim(self) -> int:
        return sum(group.width for group in self.decoder_groups)

    @property
    def output_dim(self) -> int:
        return sum(group.width for group in self.output_groups)

    # ----- Dataset construction (called in Dataset.__getitem__) -----

    def _concat_features(self, groups: List[FeatureGroup], xyz_positions: torch.Tensor, xyz_deltas: torch.Tensor, runway: RunwayData) -> Tensor:
        parts = []
        for group in groups:
            parts.append(group.compute(xyz_positions, xyz_deltas, runway))
        return torch.cat(parts, dim=-1)

    def build_encoder_input(self, xyz_positions: torch.Tensor, xyz_deltas: torch.Tensor, runway: RunwayData) -> Tensor:
        return self._concat_features(self.encoder_groups, xyz_positions, xyz_deltas, runway)

    def build_decoder_input(self, xyz_positions: torch.Tensor, xyz_deltas: torch.Tensor, runway: RunwayData) -> Tensor:
        return self._concat_features(self.decoder_groups, xyz_positions, xyz_deltas, runway)

    def build_target(self, xyz_positions: torch.Tensor, xyz_deltas: torch.Tensor, runway: RunwayData) -> Tensor:
        return self._concat_features(self.output_groups, xyz_positions, xyz_deltas, runway)


    # ----- Data Normalization and Denormalization -----

    def build_normalizers(self, dataset: Dataset) -> None:
        """
        Build normalizers and denormalizers for all feature groups.
        """
        # 2. Iterate over all samples and collect values.
        all_pos: List[Tensor] = []
        all_delta: List[Tensor] = []
        all_group_values: Dict[str, List[Tensor]] = defaultdict(list)

        for i in range(len(dataset)):
            sample = dataset[i]

            valid_horizon = ~sample.target_padding_mask

            # 2.1 Stats for positions and deltas.
            all_pos.append(sample.xyz_positions.encoder_in)
            all_pos.append(sample.xyz_positions.target[valid_horizon])
            all_delta.append(sample.xyz_deltas.encoder_in)
            all_delta.append(sample.xyz_deltas.target[valid_horizon])

            # 2.2 Group-specific stats for groups other than xy positions, altitude and deltas.
            for group in self.encoder_groups:
                if not isinstance(group, (XYPosition, Altitude, DeltaXYZ)):
                    all_group_values[group.name].append(group.get_data(sample.trajectory.encoder_in))

            for group in self.decoder_groups:
                if not isinstance(group, (XYPosition, Altitude, DeltaXYZ)):
                    all_group_values[group.name].append(group.get_data(sample.trajectory.dec_in)[valid_horizon])

        # 3. Compute mean and std.
        pos_cat = torch.cat(all_pos, dim=0)
        pos_mean = pos_cat.mean(dim=0)
        pos_std = pos_cat.std(dim=0)

        delta_cat = torch.cat(all_delta, dim=0)
        delta_mean = delta_cat.mean(dim=0)
        delta_std = delta_cat.std(dim=0)

        stats = {}
        for group_name, group_values in all_group_values.items():
            group_cat = torch.cat(group_values, dim=0)
            group_mean = group_cat.mean(dim=0)
            group_std = group_cat.std(dim=0)
            stats[group_name] = (group_mean, group_std)


        # 4. Create normalizers
        self.normalize_positions = Normalizer(pos_mean, pos_std)
        self.normalize_deltas = Normalizer(delta_mean, delta_std)
        self.denormalize_positions = Denormalizer(pos_mean, pos_std)
        self.denormalize_deltas = Denormalizer(delta_mean, delta_std)

        for group in self.encoder_groups:
            if isinstance(group, (XYPosition, Altitude, DeltaXYZ)):
                group_mean, group_std = None, None
            else:
                group_mean, group_std = stats[group.name]
            self._create_normalizer_for_group(group, pos_mean, pos_std, delta_mean, delta_std, group_mean, group_std)

        for group in self.decoder_groups:
            if isinstance(group, (XYPosition, Altitude, DeltaXYZ)):
                group_mean, group_std = None, None
            else:
                group_mean, group_std = stats[group.name]
            self._create_normalizer_for_group(group, pos_mean, pos_std, delta_mean, delta_std, group_mean, group_std)


    def _create_normalizer_for_group(self, group: FeatureGroup, pos_mean: Tensor, pos_std: Tensor, delta_mean: Tensor, delta_std: Tensor, group_mean: Optional[Tensor], group_std: Optional[Tensor]) -> None:
        if isinstance(group, XYPosition):
            group.create_normalizer(pos_mean[:2], pos_std[:2])
        elif isinstance(group, Altitude):
            group.create_normalizer(pos_mean[2:3], pos_std[2:3])
        elif isinstance(group, DeltaXYZ):
            group.create_normalizer(delta_mean, delta_std)
        else:
            group.create_normalizer(group_mean, group_std)

    def build_transforms(self) -> List:
        transforms = []
        for group in self.encoder_groups:
            normalizer = FeatureSliceNormalizer(
                path=("trajectory", "encoder_in"),
                indices=(group.start_idx, group.end_idx),
                mean=group.mean,
                std=group.std
            )
            transforms.append(normalizer)

        for group in self.decoder_groups:
            normalizer = FeatureSliceNormalizer(
                path=("trajectory", "dec_in"),
                indices=(group.start_idx, group.end_idx),
                mean=group.mean,
                std=group.std
            )
            transforms.append(normalizer)

        normalizer = FeatureSliceNormalizer(
            path=("trajectory", "target"),
            indices=(0, self.output_dim),
            mean=self.normalize_deltas.mean,
            std=self.normalize_deltas.std
        )
        transforms.append(normalizer)
        return transforms

    def register_modules(self, module: nn.Module) -> None:
        """
        Register all normalizers and denormalizers as submodules of the given module.
        Ensures they are moved to the correct device with the model and included in checkpoints.
        Must be called after build_normalizers().
        """
        module.add_module("normalize_positions", self.normalize_positions)
        module.add_module("normalize_deltas", self.normalize_deltas)
        module.add_module("denormalize_positions", self.denormalize_positions)
        module.add_module("denormalize_deltas", self.denormalize_deltas)
        self.normalize_positions.to(module.device)
        self.normalize_deltas.to(module.device)
        self.denormalize_positions.to(module.device)
        self.denormalize_deltas.to(module.device)

        for i, group in enumerate(self.encoder_groups):
            module.add_module(f"normalize_encoder_{i}_{group.name}", group.normalizer)
            group.normalizer.to(module.device)
        for i, group in enumerate(self.decoder_groups):
            module.add_module(f"normalize_decoder_{i}_{group.name}", group.normalizer)
            group.normalizer.to(module.device)

    # ----- Model-side: autoregressive decoding -----

    def build_next_decoder_input(
        self,
        pred_deltas_norm: Tensor,
        current_position_abs: Tensor,
        runway: RunwayData,
    ) -> Tuple[Tensor, Tensor]:
        """
        Build the next normalized decoder input token during AR inference.

        Args:
            pred_deltas_norm: model output [B, 1, output_dim] (normalized)
            current_position_abs: [B, 3] absolute position before this step
            runway: batched RunwayData

        Returns:
            (next_decoder_token [B, 1, decoder_dim], updated_position_abs [B, 3])
        """
        pred_delta_abs = self.denormalize_deltas(pred_deltas_norm[:, 0, :])
        new_position_abs = current_position_abs + pred_delta_abs

        parts = []
        for group in self.decoder_groups:
            abs_features = group.build_next_decoder_input(
                new_position_abs, pred_delta_abs, runway
            )
            parts.append(group.normalize(abs_features))

        next_token = torch.cat(parts, dim=-1).unsqueeze(1)
        return next_token, new_position_abs

