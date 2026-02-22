import torch
from typing import Any, Dict, Optional
from torch.utils import checkpoint

from data.transforms.normalize import Denormalizer, Normalizer
from models.base_module import BaseModule
from data.utils.trajectory import compute_rtd


class TransformerModule(BaseModule):
    def on_fit_start(self):
        dm = self.trainer.datamodule
        
        # Inputs are [Pos(3) + Delta(3)]
        input_mean = torch.cat([dm.pos_mean, dm.delta_mean], dim=0)
        input_std = torch.cat([dm.pos_std, dm.delta_std], dim=0)
        
        self.denormalize_inputs = Denormalizer(input_mean, input_std)
        self.denormalize_target_deltas = Denormalizer(dm.delta_mean, dm.delta_std)
        self.normalize_positions = Normalizer(dm.pos_mean, dm.pos_std)

        # Register as submodules so they're moved to the correct device automatically
        self.add_module("denormalize_inputs", self.denormalize_inputs)
        self.add_module("denormalize_target_deltas", self.denormalize_target_deltas)
        self.add_module("normalize_positions", self.normalize_positions)
        self.denormalize_inputs.to(self.device)
        self.denormalize_target_deltas.to(self.device)
        self.normalize_positions.to(self.device)

    def training_step(self, batch, batch_idx):
        input_traj = batch["input_traj"]
        target_traj = batch["target_traj"]
        dec_in_traj = batch["dec_in_traj"]
        target_pad_mask = batch["mask_traj"]
        target_rtd = batch["target_rtd"]
        flight_id = batch["flight_id"]
        runway = batch["runway"]
        
        pred_deltas_norm, _ = self._predict_autoregressively(input_traj, dec_in_traj)
        pred_deltas_abs = self.denormalize_target_deltas(pred_deltas_norm)
        input_pos_abs, target_pos_abs, pred_pos_abs = self._reconstruct_absolute_positions(input_traj, target_traj, pred_deltas_norm, target_pad_mask)
        pred_pos_norm = self.normalize_positions(pred_pos_abs)
        target_pos_norm = self.normalize_positions(target_pos_abs)

        # We use the raw cumulative trajectory distance for the loss, but for the metrics we add the distance to the threshold to get the RTD.
        pred_traj_distance, pred_rtd = compute_rtd(pred_pos_abs, target_pad_mask, runway["xyz"], runway["bearing"])

        loss, loss_info = self.loss(pred_pos_abs, target_pos_abs, pred_pos_norm, target_pos_norm, pred_deltas_abs, target_pad_mask, pred_traj_distance, target_rtd, runway)
        self._log_loss(loss, loss_info, prefix="train", batch_size=len(input_traj))

        self.train_metrics.update(
            pred_pos_abs=pred_pos_abs,
            target_pos_abs=target_pos_abs,
            target_pad_mask=target_pad_mask,
            pred_rtd=pred_rtd,
            target_rtd=target_rtd,
        )
        self._plot_prediction_vs_target(
            input_pos_abs, target_pos_abs, pred_pos_abs, target_pad_mask, batch_idx, flight_id, target_rtd, pred_rtd,
            prefix="train", num_trajectories=6
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_traj = batch["input_traj"]
        target_traj = batch["target_traj"]
        dec_in_traj = batch["dec_in_traj"]
        target_pad_mask = batch["mask_traj"]
        target_rtd = batch["target_rtd"]
        flight_id = batch["flight_id"]
        runway = batch["runway"]

        pred_deltas_norm, _ = self._predict_autoregressively(input_traj, dec_in_traj)
        pred_deltas_abs = self.denormalize_target_deltas(pred_deltas_norm)
        input_pos_abs, target_pos_abs, pred_pos_abs = self._reconstruct_absolute_positions(input_traj, target_traj, pred_deltas_norm, target_pad_mask)
        pred_pos_norm = self.normalize_positions(pred_pos_abs)
        target_pos_norm = self.normalize_positions(target_pos_abs)

        # We use the raw cumulative trajectory distance for the loss, but for the metrics we add the distance to the threshold to get the RTD.
        pred_traj_distance, pred_rtd = compute_rtd(pred_pos_abs, target_pad_mask, runway["xyz"], runway["bearing"])

        loss, _ = self.loss(pred_pos_abs, target_pos_abs, pred_pos_norm, target_pos_norm, pred_deltas_abs, target_pad_mask, pred_traj_distance, target_rtd, runway)
        self._log_loss(loss, prefix="val", batch_size=len(input_traj)) # do not log loss info for validation

        self.val_metrics.update(
            pred_pos_abs=pred_pos_abs,
            target_pos_abs=target_pos_abs,
            target_pad_mask=target_pad_mask,
            pred_rtd=pred_rtd,
            target_rtd=target_rtd,
        )
        self._plot_prediction_vs_target(
            input_pos_abs, target_pos_abs, pred_pos_abs, target_pad_mask, batch_idx, flight_id, target_rtd, pred_rtd,
            prefix="val", num_trajectories=self.num_visualized_traj
        )
        
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Test step not implemented")


    # --------------------------------------
    # Prediction functions
    # --------------------------------------

    def _predict_teacher_forcing(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        target_pad_mask: torch.Tensor,
        num_steps: Optional[int] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict using teacher forcing.
        
        Args:
            input_traj: Normalized input trajectory [B, T_in, input_features]
            dec_in_traj: Normalized decoder input (ground truth) [B, H, dec_in_features]
            target_pad_mask: Padding mask [B, H]
            num_steps: Number of steps to predict (default: horizon_seq_len)
            memory: Pre-computed encoder memory (optional, will encode if None)
            
        Returns:
            Tuple of (predictions [B, num_steps, 3], memory)
        """
        num_steps = num_steps or self.horizon_seq_len
        
        if memory is None:
            memory = self.model.encode(input_traj)
        
        # Slice inputs to num_steps
        dec_in = dec_in_traj[:, :num_steps, :]
        pad_mask = target_pad_mask[:, :num_steps] if target_pad_mask is not None else None
        
        causal_mask = self._generate_causal_mask(num_steps, input_traj.device)
        pred_deltas_norm = self.model.decode(
            dec_in,
            memory,
            causal_mask=causal_mask,
            target_pad_mask=pad_mask,
        )
        return pred_deltas_norm, memory

    
    def _predict_autoregressively(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        runway: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        memory: Optional[torch.Tensor] = None,
        continue_decoding = False,
        initial_position_abs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict autoregressively.
        
        Args:
            input_traj: Normalized input trajectory [B, T_in, input_features]
            dec_in_traj: Normalized decoder input [B, H, dec_in_features] - used for initial token or as prefix
            runway: Runway dictionary containing "xyz" coordinates and "bearing"
            num_steps: Number of steps to predict (default: horizon_seq_len)
            memory: Pre-computed encoder memory (optional, will encode if None)
            continue_decoding: Whether to take the full dec_in_traj as the initial decoder input (True) or only the first token (False). In the first case, recomputes the position from passed decoder deltas.
            initial_position_abs: Starting position for autoregression in absolute (denormalized) coordinates [B, 3] (optional, defaults to last decoder input position)
        Returns:
            Tuple of (predictions [B, num_steps, 3], memory)
        """
        num_steps = num_steps or self.horizon_seq_len
        
        # Encode if memory not provided
        if memory is None:
            memory = self.model.encode(input_traj)

        # Initialize position tracking and decoder input
        if continue_decoding:
            if initial_position_abs is None:
                raise ValueError("Initial_position must be provided if continue_decoding is True")
            current_position_abs = initial_position_abs.clone()
            current_dec_in = dec_in_traj
        else:
            input_abs = self.denormalize_inputs(input_traj)
            current_position_abs = input_abs[:, -1, :3].clone()  # [B, 3] # take last position of input trajectory as starting position
            current_dec_in = dec_in_traj[:, 0:1, :] # take first token of decoder input as decoder input

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
            pred_deltas_norm = output[:, -1:, :]  # [B, 1, 3]
            all_predictions_norm.append(pred_deltas_norm)
            
            current_dec_in = torch.cat([current_dec_in, pred_deltas_norm], dim=1)

        pred_deltas_norm = torch.cat(all_predictions_norm, dim=1)  # [B, num_steps, 3]
        return pred_deltas_norm, memory


    def _predict_scheduled_sampling(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        target_traj: torch.Tensor,
        runway: Dict[str, torch.Tensor],
        target_pad_mask: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Predict using scheduled sampling (curriculum learning).
        
        Uses teacher forcing for the first `tf_steps` timesteps, then switches to
        autoregressive prediction for the remaining timesteps.
        """
        tf_steps = self._compute_teacher_forcing_steps()
        ar_steps = self.horizon_seq_len - tf_steps
        
        # Log the current scheduled sampling state (once per epoch, on first batch)
        if batch_idx == 0:
            self.log("train/teacher_forcing_steps", float(tf_steps), on_step=False, on_epoch=True, batch_size=len(input_traj))
            self.log("train/autoregressive_steps", float(ar_steps), on_step=False, on_epoch=True, batch_size=len(input_traj))
        
        # Special case: Full teacher forcing
        if tf_steps >= self.horizon_seq_len:
            pred_traj, _ = self._predict_teacher_forcing(
                input_traj, dec_in_traj, target_pad_mask
            )
            return pred_traj
        
        # Special case: Full autoregressive
        if tf_steps <= 0:
            pred_traj, _ = self._predict_autoregressively(
                input_traj, dec_in_traj, runway
            )
            return pred_traj
        
        # Mixed mode: Teacher forcing prefix + autoregressive suffix
        # Teacher forcing phase
        tf_pred_deltas_norm, memory = self._predict_teacher_forcing(
            input_traj, dec_in_traj, target_pad_mask, num_steps=tf_steps
        )

        # Compute absolute position after TF phase
        input_abs = self.denormalize_inputs(input_traj)
        gt_deltas_abs = self.denormalize_target_deltas(target_traj[:, :tf_steps, :])
        position_before_last_tf_abs = input_abs[:, -1, :3] + gt_deltas_abs[:, :-1, :].sum(dim=1)  # [B, 3]

        # Build AR prefix: ground truth dec_in for TF steps + model's last prediction (full features)
        last_delta_norm = tf_pred_deltas_norm[:, -1:, :]
        last_tf_dec_in, position_after_last_tf_abs = self._build_next_decoder_input(
            last_delta_norm, position_before_last_tf_abs,
            runway["xyz"], runway["centerline_points_xy"]
        )
        ar_start_dec_in = torch.cat([dec_in_traj[:, :tf_steps, :], last_tf_dec_in], dim=1)
        
        # Autoregressive phase
        ar_pred_deltas_norm, _ = self._predict_autoregressively(
            input_traj, ar_start_dec_in, runway,
            num_steps=ar_steps,
            memory=memory,
            continue_decoding=True,
            initial_position_abs=position_after_last_tf_abs,
        )
        
        # Concatenate predictions
        pred_deltas_norm = torch.cat([tf_pred_deltas_norm, ar_pred_deltas_norm], dim=1)  # [B, H, 3]
        return pred_deltas_norm