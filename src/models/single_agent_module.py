from typing import Any, Dict, Optional

import torch
import torch.utils.checkpoint as checkpoint

from data.compute.trajectory import compute_rtd, reconstruct_positions_from_deltas
from data.interface import RunwayData, Sample
from models.base_module import BaseModule
from models.metrics import AccumulatedTrajectoryMetrics


class SingleAgentModule(BaseModule):
    def common_step(self, 
        batch: Sample,
        batch_idx: int,
        metrics: AccumulatedTrajectoryMetrics,
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
        pred_pos_abs = reconstruct_positions_from_deltas(input_pos_abs, pred_deltas_abs, target_pad_mask)
        pred_pos_norm = self.feature_schema.normalize_positions(pred_pos_abs)
        target_pos_norm = self.feature_schema.normalize_positions(target_pos_abs)

        # We use the raw cumulative trajectory distance for the loss, but for the metrics we add the distance to the threshold to get the RTD.
        pred_traj_distance, pred_rtd = compute_rtd(pred_pos_abs, target_pad_mask, runway.xyz, runway.bearing)
        
        loss, loss_info = self.loss(pred_pos_abs, target_pos_abs, pred_pos_norm, target_pos_norm, pred_deltas_abs, target_pad_mask, pred_traj_distance, target_rtd, runway)
        self._log_loss(loss, loss_info, prefix=prefix, batch_size=len(input_traj))
        
        metrics.update(
            pred_pos_abs=pred_pos_abs,
            target_pos_abs=target_pos_abs,
            target_pad_mask=target_pad_mask,
            pred_rtd=pred_rtd,
            target_rtd=target_rtd,
        )
        self._plot_prediction_vs_target(
            input_pos_abs, target_pos_abs, pred_pos_abs, target_pad_mask, batch_idx, flight_id, target_rtd, pred_rtd,
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
    # Prediction functions
    # --------------------------------------

    def predict_autoregressively(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        runway: RunwayData,
        initial_position_abs: torch.Tensor,
        num_steps: Optional[int] = None,
        memory: Optional[torch.Tensor] = None,
        continue_decoding = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict autoregressively.
        
        Args:
            input_traj: Normalized input trajectory [B, T_in, input_features]
            dec_in_traj: Normalized decoder input [B, H, dec_in_features] - used for initial token or as prefix
            runway: Runway dictionary containing "xyz" coordinates and "bearing"
            initial_position_abs: Starting position for autoregression in absolute (denormalized) coordinates [B, 3]
            num_steps: Number of steps to predict (default: horizon_seq_len)
            memory: Pre-computed encoder memory (optional, will encode if None)
            continue_decoding: Whether to take the full dec_in_traj as the initial decoder input (True) or only the first token (False). In the first case, recomputes the position from passed decoder deltas.
        Returns:
            Tuple of (predictions [B, num_steps, 3], memory)
        """
        num_steps = num_steps or self.horizon_seq_len
        
        # Encode if memory not provided
        if memory is None:
            memory = self.model.encode(input_traj)

        current_position_abs = initial_position_abs.clone()
        current_dec_in = dec_in_traj if continue_decoding else dec_in_traj[:, 0:1, :]

        # Autoregressive loop
        all_predictions_norm = []
        for i in range(num_steps):
            current_seq_len = current_dec_in.size(1)
            target_mask = self._generate_causal_mask(current_seq_len, input_traj.device)
            
            # Use gradient checkpointing during training to save memory
            if self.training:
                output = checkpoint.checkpoint(
                    lambda dec_in, mem, mask, pad_mask: self.model.decode(
                        dec_in, mem, causal_mask=mask, target_pad_mask=pad_mask
                    ),
                    current_dec_in,
                    memory,
                    target_mask,
                    None,
                    use_reentrant=False,
                )
            else:
                output = self.model.decode(
                    current_dec_in,
                    memory,
                    causal_mask=target_mask,
                    target_pad_mask=None,  # No padding mask during AR to avoid sequence length leakage
                )
            pred_deltas_norm = output[:, -1:, :]  # [B, 1, output_dim]
            all_predictions_norm.append(pred_deltas_norm)

            next_dec_in, current_position_abs = self.feature_schema.build_next_decoder_input(
                pred_deltas_norm, current_position_abs, runway
            )
            current_dec_in = torch.cat([current_dec_in, next_dec_in], dim=1)

        pred_deltas_norm = torch.cat(all_predictions_norm, dim=1)
        return pred_deltas_norm, memory