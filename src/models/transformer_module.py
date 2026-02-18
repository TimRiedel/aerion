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

        if self.scheduled_sampling_enabled:
            pred_deltas_norm = self._predict_scheduled_sampling(input_traj, dec_in_traj, target_traj, target_pad_mask, batch_idx)
        else:
            pred_deltas_norm, _ = self._predict_teacher_forcing(input_traj, dec_in_traj, target_pad_mask)
        
        input_pos_abs, target_pos_abs, pred_pos_abs = self._reconstruct_absolute_positions(input_traj, target_traj, pred_deltas_norm, target_pad_mask)
        pred_pos_norm = self.normalize_positions(pred_pos_abs)
        target_pos_norm = self.normalize_positions(target_pos_abs)
        pred_rtd = compute_rtd(pred_pos_abs, target_pad_mask, runway["xyz"], runway["bearing"])

        loss, weighted_losses = self.loss(pred_pos_abs, target_pos_abs, pred_pos_norm, target_pos_norm, target_pad_mask, pred_rtd, target_rtd, runway)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))
        # for loss_name, loss_value in weighted_losses.items():
        #     self.log(f"train_{loss_name}_loss", loss_value, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))
        
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
        input_pos_abs, target_pos_abs, pred_pos_abs = self._reconstruct_absolute_positions(input_traj, target_traj, pred_deltas_norm, target_pad_mask)
        pred_pos_norm = self.normalize_positions(pred_pos_abs)
        target_pos_norm = self.normalize_positions(target_pos_abs)
        pred_rtd = compute_rtd(pred_pos_abs, target_pad_mask, runway["xyz"], runway["bearing"])

        loss, weighted_losses = self.loss(pred_pos_abs, target_pos_abs, pred_pos_norm, target_pos_norm, target_pad_mask, pred_rtd, target_rtd, runway)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))
        # for loss_name, loss_value in weighted_losses.items():
        #     self.log(f"val_{loss_name}_loss", loss_value, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))

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
        pred_traj = self.model.decode(
            dec_in,
            memory,
            causal_mask=causal_mask,
            target_pad_mask=pad_mask,
        )
        return pred_traj, memory

    
    def _predict_autoregressively(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        num_steps: Optional[int] = None,
        memory: Optional[torch.Tensor] = None,
        continued_decoding = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict autoregressively.
        
        Args:
            input_traj: Normalized input trajectory [B, T_in, input_features]
            dec_in_traj: Normalized decoder input [B, H, dec_in_features] - used for initial token or as prefix
            num_steps: Number of steps to predict (default: horizon_seq_len)
            memory: Pre-computed encoder memory (optional, will encode if None)
            continued_decoding: Whether to take the full dec_in_traj as the initial decoder input (True) or only the first token (False)
        Returns:
            Tuple of (predictions [B, num_steps, 3], memory)
        """
        num_steps = num_steps or self.horizon_seq_len
        
        # Encode if memory not provided
        if memory is None:
            memory = self.model.encode(input_traj)

        # Initialize decoder input either with the full dec_in_traj (for continued decoding) or only the first token
        current_dec_in = dec_in_traj if continued_decoding else dec_in_traj[:, 0:1, :]

        # Autoregressive loop
        all_predictions = []
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
            next_step_pred = output[:, -1:, :]  # [B, 1, output_features]
            all_predictions.append(next_step_pred)
            
            current_dec_in = torch.cat([current_dec_in, next_step_pred], dim=1)

        pred_traj = torch.cat(all_predictions, dim=1)  # [B, num_steps, output_features]
        return pred_traj, memory


    def _predict_scheduled_sampling(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        target_traj: torch.Tensor,
        target_pad_mask: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
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
                input_traj, dec_in_traj
            )
            return pred_traj
        
        # Mixed mode: Teacher forcing prefix + autoregressive suffix
        # Teacher forcing phase
        tf_pred, memory = self._predict_teacher_forcing(
            input_traj, dec_in_traj, target_pad_mask, num_steps=tf_steps
        )

        last_tf_pred = tf_pred[:, -1:, :]
        ar_start_dec_in = torch.cat([dec_in_traj[:, :tf_steps, :], last_tf_pred], dim=1)
        
        # Autoregressive phase
        ar_pred, _ = self._predict_autoregressively(
            input_traj, ar_start_dec_in,
            num_steps=ar_steps,
            memory=memory,
            continued_decoding=True,
        )
        
        # Concatenate predictions
        pred_traj = torch.cat([tf_pred, ar_pred], dim=1)  # [B, H, 3]
        return pred_traj