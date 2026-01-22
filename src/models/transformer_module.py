import torch
from typing import Any, Dict

from models.base_module import BaseModule


class TransformerModule(BaseModule):
    def training_step(self, batch, batch_idx):
        input_traj, target_traj, dec_in_traj, target_pad_mask = batch["input_traj"], batch["target_traj"], batch["dec_in_traj"], batch["mask_traj"]

        pred_traj = self._predict_teacher_forcing(input_traj, dec_in_traj, target_pad_mask)
            
        loss = self._compute_loss(pred_traj, target_traj, target_pad_mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_traj, target_traj, dec_in_traj, target_pad_mask = batch["input_traj"], batch["target_traj"], batch["dec_in_traj"], batch["mask_traj"]
        
        pred_traj = self._predict_autoregressively(input_traj, dec_in_traj, target_pad_mask)
        loss = self._compute_loss(pred_traj, target_traj, target_pad_mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))

        input_abs, target_abs, pred_abs = self._reconstruct_absolute_positions(input_traj, target_traj, pred_traj)
        self._evaluate_step(pred_abs, target_abs, target_pad_mask)
        self._visualize_prediction_vs_targets(input_abs, target_abs, pred_abs, target_pad_mask, batch_idx)
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

    
    def _predict_autoregressively(self, input_traj: torch.Tensor, dec_in_traj: torch.Tensor, target_pad_mask: torch.Tensor) -> torch.Tensor:
        # 1. Encode once
        memory = self.model.encode(input_traj) # Shape: [Batch, Max_Input_Len, d_model]

        # 2. Initialize: Start with the first token of dec_in_traj (current position)
        current_dec_in = dec_in_traj[:, 0:1, :] # Shape: [Batch, 1, Input_Dim]
        all_predictions = []

        # 3. Autoregressive loop
        for i in range(self.horizon_seq_len):
            # Generate mask for the current sequence length i+1
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
            next_step_pred = output[:, -1:, :]
            current_dec_in = torch.cat([current_dec_in, next_step_pred], dim=1)

            all_predictions.append(next_step_pred)

        pred_traj = torch.cat(all_predictions, dim=1)
        return pred_traj
