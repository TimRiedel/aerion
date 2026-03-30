from typing import Any, Dict, List

import torch

from data.compute.trajectory import compute_rtd, compute_trajectory_lengths, reconstruct_positions_from_deltas
from data.interface import RunwayData, PredictionSample
from models.base_module import BaseModule
from models.metrics import TrajectoryMetrics


class TrafficModule(BaseModule):
    def common_step(
        self,
        batch: PredictionSample,
        batch_idx: int,
        metrics: TrajectoryMetrics,
        prefix: str,
        num_trajectories_plotting: int = 6,
    ) -> torch.Tensor:
        if batch.agent_padding_mask is None:
            raise ValueError("Agent padding mask is required for traffic module")

        pred_deltas_norm, _ = self.predict_autoregressively(
            batch.trajectory.encoder_in,
            batch.trajectory.dec_in,
            batch.runway,
            initial_position_abs=batch.last_input_pos_abs,
            agent_padding_mask=batch.agent_padding_mask,
        )

        flattened_batch = self.flatten_batch(batch, pred_deltas_norm)
        pred_deltas_norm = flattened_batch["pred_deltas_norm"]
        input_pos_abs = flattened_batch["input_pos_abs"]
        target_pos_abs = flattened_batch["target_pos_abs"]
        target_pad_mask = flattened_batch["target_pad_mask"]
        target_rtd = flattened_batch["target_rtd"]
        runway = flattened_batch["runway"]
        flight_id = flattened_batch["flight_id"]
        valid_agent_count = flattened_batch["valid_agent_count"]

        pred_deltas_abs = self.feature_schema.denormalize_deltas(pred_deltas_norm)
        pred_pos_abs = reconstruct_positions_from_deltas(input_pos_abs, pred_deltas_abs)
        pred_pos_norm = self.feature_schema.normalize_positions(pred_pos_abs)
        target_pos_norm = self.feature_schema.normalize_positions(target_pos_abs)

        lengths = compute_trajectory_lengths(pred_deltas_abs, pred_pos_abs, runway.xyz[:, :2], target_pad_mask, self.pred_end_epsilon, self.pred_end_min_consecutive)

        # We use the raw cumulative trajectory distance for the loss, but for the metrics we add the distance to the threshold to get the RTD.
        pred_traj_distance, pred_rtd = compute_rtd(pred_pos_abs, lengths.pred_valid_len, runway.xyz, runway.bearing)

        loss, loss_info = self.loss(pred_pos_abs, target_pos_abs, pred_pos_norm, target_pos_norm, pred_deltas_abs, lengths, pred_traj_distance, target_rtd, runway)
        self._log_loss(loss, loss_info, prefix=prefix, batch_size=valid_agent_count)

        metrics.update(
            pred_pos_abs=pred_pos_abs,
            target_pos_abs=target_pos_abs,
            lengths=lengths,
            pred_rtd=pred_rtd,
            target_rtd=target_rtd,
            flight_id=flight_id,
        )
        self._plot_prediction_vs_target(
            input_pos_abs, target_pos_abs, pred_pos_abs, lengths, batch_idx, flight_id, target_rtd, pred_rtd,
            prefix=prefix, num_trajectories=num_trajectories_plotting
        )
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, self.train_metrics, "train", num_trajectories_plotting=6)

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, self.val_metrics, "val", num_trajectories_plotting=self.num_visualized_traj)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Test step not implemented")


    # --------------------------------------
    # Batch flattening
    # --------------------------------------

    def flatten_batch(
        self,
        batch: PredictionSample,
        pred_deltas_norm: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Flatten a batched multi-agent scene over the agent dimension.

        Converts [B, ..., N, ...] tensors into [B*N_valid, ...] by removing padded
        agent slots according to batch.agent_padding_mask.
        """
        B, _, N, _ = pred_deltas_norm.shape
        valid_agents = ~batch.agent_padding_mask.reshape(B * N)

        pred_deltas_norm_flat = self.flatten_btnf(pred_deltas_norm, valid_agents)
        input_pos_abs_flat = self.flatten_btnf(batch.xyz_positions.encoder_in, valid_agents)
        target_pos_abs_flat = self.flatten_btnf(batch.xyz_positions.target, valid_agents)
        target_pad_mask_flat = self.flatten_btn(batch.target_padding_mask, valid_agents)
        target_rtd_flat = batch.target_rtd.reshape(B * N)[valid_agents]
        runway_flat = self.flatten_runway(batch.runway, valid_agents)

        # Flatten flight IDs List[B][N] -> [B*N] and apply the same valid_agents mask
        flat_flight_ids: List[str] = [
            fid for scene_ids in batch.flight_id for fid in scene_ids
        ]
        valid_agents_list = valid_agents.cpu().tolist()
        flight_id_flat = [
            fid for fid, is_valid in zip(flat_flight_ids, valid_agents_list) if is_valid
        ]

        valid_agent_count = int(valid_agents.sum().item())

        return {
            "pred_deltas_norm": pred_deltas_norm_flat,
            "input_pos_abs": input_pos_abs_flat,
            "target_pos_abs": target_pos_abs_flat,
            "target_pad_mask": target_pad_mask_flat,
            "target_rtd": target_rtd_flat,
            "runway": runway_flat,
            "flight_id": flight_id_flat,
            "valid_agent_count": valid_agent_count,
        }

    def flatten_btnf(self, t: torch.Tensor, valid_agents: torch.Tensor) -> torch.Tensor:
        """Flatten [B, T, N, F] -> [B*N_valid, T, F] excluding padded agents."""
        B, T, N, F = t.shape
        return t.permute(0, 2, 1, 3).reshape(B * N, T, F)[valid_agents]

    def flatten_btn(self, t: torch.Tensor, valid_agents: torch.Tensor) -> torch.Tensor:
        """Flatten [B, T, N] -> [B*N_valid, T] excluding padded agents."""
        B, T, N = t.shape
        return t.permute(0, 2, 1).reshape(B * N, T)[valid_agents]

    def flatten_runway(
        self, runway: RunwayData, valid_agents: torch.Tensor
    ) -> RunwayData:
        """Reshape runway [B, N, F] -> [B*N_valid, F] excluding padded agents."""
        B, N, _ = runway.xyz.shape
        return RunwayData(
            xyz=runway.xyz.reshape(B * N, 3)[valid_agents],
            bearing=runway.bearing.reshape(B * N, 2)[valid_agents],
            length=runway.length.reshape(B * N)[valid_agents],
            centerline_points_xy=[
                cp.reshape(B * N, 2)[valid_agents] for cp in runway.centerline_points_xy
            ],
        )