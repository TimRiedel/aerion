from typing import Any, Dict

import torch

from data.compute.trajectory import compute_rtd, compute_trajectory_lengths, reconstruct_positions_from_deltas
from data.interface import PredictionSample
from models.base_module import BaseModule
from models.metrics import TrajectoryMetrics


class SingleAgentModule(BaseModule):
    def common_step(self,
        batch: PredictionSample,
        batch_idx: int,
        metrics: TrajectoryMetrics,
        prefix: str,
        num_trajectories_plotting: int = 6,
    ) -> torch.Tensor:
        input_traj = batch.trajectory.encoder_in
        input_pos_abs = batch.xyz_positions.encoder_in
        input_last_pos_abs = batch.last_input_pos_abs
        dec_in_traj = batch.trajectory.dec_in
        target_pos_abs = batch.xyz_positions.target
        target_pad_mask = batch.target_padding_mask
        target_rtd = batch.target_rtd
        runway = batch.runway
        flight_id = batch.flight_id

        pred_deltas_norm, _ = self.predict_autoregressively(input_traj, dec_in_traj, runway, initial_position_abs=input_last_pos_abs)

        pred_deltas_abs = self.feature_schema.denormalize_deltas(pred_deltas_norm)
        pred_pos_abs = reconstruct_positions_from_deltas(input_pos_abs, pred_deltas_abs)
        pred_pos_norm = self.feature_schema.normalize_positions(pred_pos_abs)
        target_pos_norm = self.feature_schema.normalize_positions(target_pos_abs)

        lengths = compute_trajectory_lengths(pred_deltas_abs, pred_pos_abs, runway.xyz[:, :2], target_pad_mask, self.delta_epsilon, self.terminal_area_m, self.min_consecutive_steps)

        # For loss: anchor to target_valid_len — stable gradients during early training
        pred_traj_distance_loss, _ = compute_rtd(pred_pos_abs, lengths.target_valid_len, runway.xyz, runway.bearing)
        # For metrics: use pred_valid_len — reflects where the model thinks the plane lands
        _, pred_rtd_metrics = compute_rtd(pred_pos_abs, lengths.pred_valid_len, runway.xyz, runway.bearing)

        loss, loss_info = self.loss(pred_pos_abs, target_pos_abs, pred_pos_norm, target_pos_norm, pred_deltas_abs, lengths, pred_traj_distance_loss, target_rtd, runway)
        self._log_loss(loss, loss_info, prefix=prefix, batch_size=len(input_traj))

        metrics.update(
            pred_pos_abs=pred_pos_abs,
            target_pos_abs=target_pos_abs,
            lengths=lengths,
            pred_rtd=pred_rtd_metrics,
            target_rtd=target_rtd,
            flight_id=list(flight_id),
        )
        self._plot_prediction_vs_target(
            input_pos_abs, target_pos_abs, pred_pos_abs, lengths, batch_idx, flight_id, target_rtd, pred_rtd_metrics,
            prefix=prefix, num_trajectories=num_trajectories_plotting
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, self.train_metrics, "train", num_trajectories_plotting=6)
    
    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, self.val_metrics, "val", num_trajectories_plotting=self.num_visualized_traj)
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Test step not implemented")