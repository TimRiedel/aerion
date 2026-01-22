import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from omegaconf import DictConfig

from models.base_module import BaseModule


class AerionModule(BaseModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        input_seq_len: int,
        horizon_seq_len: int,
        contexts_cfg: Optional[DictConfig] = None,
        scheduler_cfg: Optional[DictConfig] = None,
        num_visualized_traj: int = 10,
    ):
        self.contexts_cfg = contexts_cfg or {}
        
        # Inject contexts_cfg into model params before parent __init__ calls instantiate
        model_cfg["params"]["contexts_cfg"] = self.contexts_cfg
        
        super().__init__(
            model_cfg=model_cfg,
            optimizer_cfg=optimizer_cfg,
            input_seq_len=input_seq_len,
            horizon_seq_len=horizon_seq_len,
            scheduler_cfg=scheduler_cfg,
            num_visualized_traj=num_visualized_traj,
        )
    
    
    def training_step(self, batch, batch_idx):
        input_traj = batch["input_traj"]
        target_traj = batch["target_traj"]
        dec_in_traj = batch["dec_in_traj"]
        target_pad_mask = batch["mask_traj"]
        contexts = self._extract_contexts(batch)

        pred_traj = self._predict_teacher_forcing(input_traj, dec_in_traj, target_pad_mask, contexts)
            
        loss = self._compute_loss(pred_traj, target_traj, target_pad_mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_traj = batch["input_traj"]
        target_traj = batch["target_traj"]
        dec_in_traj = batch["dec_in_traj"]
        target_pad_mask = batch["mask_traj"]
        contexts = self._extract_contexts(batch)
        
        pred_traj = self._predict_autoregressively(input_traj, dec_in_traj, target_pad_mask, contexts)
        loss = self._compute_loss(pred_traj, target_traj, target_pad_mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))

        input_abs, target_abs, pred_abs = self._reconstruct_absolute_positions(input_traj, target_traj, pred_traj)
        self._evaluate_step(pred_abs, target_abs, target_pad_mask)
        self._visualize_prediction_vs_targets(input_abs, target_abs, pred_abs, target_pad_mask, batch_idx)
        return loss
    

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Test step not implemented")


    def _predict_teacher_forcing(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        target_pad_mask: torch.Tensor,
        contexts: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Predict using teacher forcing (training mode)."""
        causal_mask = self._generate_causal_mask(self.horizon_seq_len, input_traj.device)
        pred_traj = self.model(
            input_traj=input_traj,
            dec_in_traj=dec_in_traj,
            causal_mask=causal_mask,
            target_pad_mask=target_pad_mask,
            contexts=contexts,
        )
        return pred_traj

    
    def _predict_autoregressively(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        target_pad_mask: torch.Tensor,
        contexts: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Predict autoregressively (inference mode)."""
        # 1. Encode once (with context)
        memory = self.model.encode(input_traj, contexts=contexts)

        # 2. Initialize: Start with the first token of dec_in_traj
        current_dec_in = dec_in_traj[:, 0:1, :]
        all_predictions = []

        # 3. Autoregressive loop
        for i in range(self.horizon_seq_len):
            current_seq_len = current_dec_in.size(1)
            target_mask = self._generate_causal_mask(current_seq_len, input_traj.device)
            
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


    # --------------------------------------
    # Context-related helper functions
    # --------------------------------------

    def _is_context_enabled(self, name: str) -> bool:
        return self.contexts_cfg.get(name, {}).get("enabled", False)
    
    def _extract_contexts(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        contexts = {}
        if self._is_context_enabled("flightinfo") and "flightinfo" in batch:
            contexts["flightinfo"] = batch["flightinfo"]
        return contexts
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(seq_len, dtype=torch.bool, device=device)