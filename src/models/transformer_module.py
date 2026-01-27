import torch
from typing import Any, Dict

from models.base_module import BaseModule
from data.utils.trajectory import compute_threshold_features


class TransformerModule(BaseModule):
    def training_step(self, batch, batch_idx):
        input_traj, target_traj, dec_in_traj, target_pad_mask = batch["input_traj"], batch["target_traj"], batch["dec_in_traj"], batch["mask_traj"]

        pred_traj = self._predict_teacher_forcing(input_traj, dec_in_traj, target_pad_mask)
        input_abs, target_abs, pred_abs = self._reconstruct_absolute_positions(input_traj, target_traj, pred_traj, target_pad_mask)

        loss = self.loss(pred_abs, target_abs, target_pad_mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))
        
        self.train_metrics.update(pred_abs, target_abs, target_pad_mask)
        self._visualize_prediction_vs_targets(
            input_abs, target_abs, pred_abs, target_pad_mask, batch_idx,
            prefix="train", num_trajectories=6
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_traj = batch["input_traj"]
        target_traj = batch["target_traj"]
        dec_in_traj = batch["dec_in_traj"]
        target_pad_mask = batch["mask_traj"]
        threshold_xy = batch["threshold_xy"]
        
        pred_traj = self._predict_autoregressively(input_traj, dec_in_traj, threshold_xy)
        input_abs, target_abs, pred_abs = self._reconstruct_absolute_positions(input_traj, target_traj, pred_traj, target_pad_mask)

        loss = self.loss(pred_abs, target_abs, target_pad_mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))

        self.val_metrics.update(pred_abs, target_abs, target_pad_mask)
        self._visualize_prediction_vs_targets(input_abs, target_abs, pred_abs, target_pad_mask, batch_idx, num_trajectories=self.num_visualized_traj)
        return loss
    

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Test step not implemented")


    def _predict_teacher_forcing(self, input_traj: torch.Tensor, dec_in_traj: torch.Tensor, target_pad_mask: torch.Tensor) -> torch.Tensor:
        causal_mask = self.model.transformer.generate_square_subsequent_mask(self.horizon_seq_len, dtype=torch.bool).to(input_traj.device)
        pred_traj = self.model(
            input_traj=input_traj,
            dec_in_traj=dec_in_traj,
            causal_mask=causal_mask,
            target_pad_mask=target_pad_mask 
        )
        return pred_traj

    
    def _predict_autoregressively(
        self, 
        input_traj: torch.Tensor, 
        dec_in_traj: torch.Tensor, 
        threshold_xy: torch.Tensor
    ) -> torch.Tensor:
        """Predict autoregressively (inference mode).
        
        Args:
            input_traj: Normalized input trajectory [B, T_in, 8]
            dec_in_traj: Normalized decoder input [B, H, 5] (only first token used)
            threshold_xy: Threshold coordinates [B, 2]
            
        Returns:
            Predicted deltas [B, H, 3]
        """
        # 1. Encode once
        memory = self.model.encode(input_traj)  # Shape: [Batch, Max_Input_Len, d_model]

        # 2. Get denormalized last input position for tracking current position
        input_abs = self.denormalize_inputs(input_traj)
        current_position = input_abs[:, -1, :3].clone()  # [B, 3] - last position (x, y, alt)

        # 3. Initialize: Start with the first token of dec_in_traj
        current_dec_in = dec_in_traj[:, 0:1, :]  # [B, 1, 5]
        all_predictions = []

        # 4. Autoregressive loop
        for i in range(self.horizon_seq_len):
            current_seq_len = current_dec_in.size(1)
            target_mask = self.model.transformer.generate_square_subsequent_mask(
                current_seq_len, dtype=torch.bool, device=input_traj.device
            )

            output = self.model.decode(
                current_dec_in, 
                memory, 
                causal_mask=target_mask,
                target_pad_mask=None,  # No padding mask during autoregressive validation to avoid sequence length leakage
            )
            next_step_pred = output[:, -1:, :]  # [B, 1, 3] - predicted delta (normalized)
            all_predictions.append(next_step_pred)
            
            # Update current position with denormalized predicted delta
            pred_delta_denorm = self.denormalize_target_deltas(next_step_pred)  # [B, 1, 3]
            current_position = current_position + pred_delta_denorm[:, 0, :]  # [B, 3]
            
            # Compute new threshold features for next decoder input
            new_thr_features = compute_threshold_features(
                current_position[:, :2],  # [B, 2] - x, y only
                threshold_xy              # [B, 2]
            )  # [B, 2]
            
            # Normalize threshold features using position stats
            pos_mean_xy = self.denormalize_inputs.mean[:2]  # [2]
            pos_std_xy = self.denormalize_inputs.std[:2]    # [2]
            new_thr_features_norm = (new_thr_features - pos_mean_xy) / pos_std_xy  # [B, 2]
            
            # Build next decoder input: [predicted_delta_norm, thr_features_norm]
            next_dec_in = torch.cat([
                next_step_pred,                        # [B, 1, 3]
                new_thr_features_norm.unsqueeze(1)     # [B, 1, 2]
            ], dim=-1)  # [B, 1, 5]
            
            current_dec_in = torch.cat([current_dec_in, next_dec_in], dim=1)

        pred_traj = torch.cat(all_predictions, dim=1)  # [B, H, 3]
        return pred_traj
