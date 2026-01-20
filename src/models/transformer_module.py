import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Any, Dict
from hydra.utils import instantiate

from models.base_module import BaseModule


class TransformerModule(BaseModule):
    def training_step(self, batch, batch_idx):
        x, y, dec_in, tgt_pad_mask = batch["x"], batch["y"], batch["dec_in"], batch["mask"]

        predictions = self._predict_teacher_forcing(x, dec_in, tgt_pad_mask)
            
        loss = self._compute_loss(predictions, y, tgt_pad_mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(x))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, dec_in, tgt_pad_mask = batch["x"], batch["y"], batch["dec_in"], batch["mask"]
        
        predictions = self._predict_autoregressively(x, dec_in, tgt_pad_mask)
        loss = self._compute_loss(predictions, y, tgt_pad_mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(x))

        input_abs, tgt_abs, pred_abs = self._reconstruct_absolute_positions(x, y, predictions, tgt_pad_mask)
        self._evaluate_step(pred_abs, tgt_abs, tgt_pad_mask)
        self._visualize_prediction_vs_targets(input_abs, tgt_abs, pred_abs, tgt_pad_mask, batch_idx)
        return loss
    

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Test step not implemented")


    def _predict_teacher_forcing(self, x: torch.Tensor, dec_in: torch.Tensor, tgt_pad_mask: torch.Tensor) -> torch.Tensor:
        causal_mask = self.model.transformer.generate_square_subsequent_mask(self.horizon_seq_len, dtype=torch.bool).to(x.device)
        predictions = self.model(
            src=x,
            tgt=dec_in,
            tgt_mask=causal_mask,
            tgt_pad_mask=tgt_pad_mask 
        )
        return predictions

    
    def _predict_autoregressively(self, x: torch.Tensor, dec_in: torch.Tensor, tgt_pad_mask: torch.Tensor) -> torch.Tensor:
        # 1. Encode once
        memory = self.model.encode(x) # Shape: [Batch, Max_Input_Len, d_model]

        # 2. Initialize: Start with the first token of dec_in
        current_input = dec_in[:, 0:1, :] # Shape: [Batch, 1, Input_Dim]
        all_predictions = []

        # 3. Autoregressive loop
        for i in range(self.horizon_seq_len):
            # Generate mask for the current sequence length i+1
            current_seq_len = current_input.size(1)
            tgt_mask = self.model.transformer.generate_square_subsequent_mask(
                current_seq_len, dtype=torch.bool, device=x.device
            )

            output = self.model.decode(current_input, memory, tgt_mask=tgt_mask)
            next_step_pred = output[:, -1:, :]
            current_input = torch.cat([current_input, next_step_pred], dim=1)

            all_predictions.append(next_step_pred)


        predictions = torch.cat(all_predictions, dim=1)
        return predictions


    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.model.parameters())
        
        if self.scheduler_cfg is None:
            return optimizer
        else:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            warmup_epochs = self.scheduler_cfg.get('warmup_epochs', 0)
            
            warmup_steps = steps_per_epoch * warmup_epochs
            total_steps = steps_per_epoch * self.trainer.max_epochs

            scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=1e-6, total_iters=warmup_steps),
                    CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
                ],
                milestones=[warmup_steps],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }