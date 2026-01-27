import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Any, Dict, Optional
from omegaconf import DictConfig

from models.base_module import BaseModule
from data.utils.trajectory import compute_threshold_features


class AerionModule(BaseModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        input_seq_len: int,
        horizon_seq_len: int,
        contexts_cfg: Optional[DictConfig] = None,
        scheduler_cfg: Optional[DictConfig] = None,
        loss_cfg: Optional[DictConfig] = None,
        scheduled_sampling_cfg: Optional[DictConfig] = None,
        num_visualized_traj: int = 10,
    ):
        
        # Inject contexts_cfg into model params before parent __init__ calls instantiate
        self.contexts_cfg = contexts_cfg or {}
        model_cfg["params"]["contexts_cfg"] = self.contexts_cfg
        
        super().__init__(
            model_cfg=model_cfg,
            optimizer_cfg=optimizer_cfg,
            input_seq_len=input_seq_len,
            horizon_seq_len=horizon_seq_len,
            scheduler_cfg=scheduler_cfg,
            loss_cfg=loss_cfg,
            num_visualized_traj=num_visualized_traj,
        )

        scheduled_sampling_cfg = scheduled_sampling_cfg or {}
        self.scheduled_sampling_enabled = scheduled_sampling_cfg.get("enabled", False)
        self.teacher_forcing_epochs = scheduled_sampling_cfg.get("teacher_forcing_epochs", 10)
        self.transition_epochs = scheduled_sampling_cfg.get("transition_epochs", 20)
    
    
    def training_step(self, batch, batch_idx):
        input_traj = batch["input_traj"]
        target_traj = batch["target_traj"]
        dec_in_traj = batch["dec_in_traj"]
        target_pad_mask = batch["mask_traj"]
        threshold_xy = batch["threshold_xy"]
        contexts = self._extract_contexts(batch)

        if self.scheduled_sampling_enabled:
            pred_traj = self._predict_scheduled_sampling(
                input_traj, dec_in_traj, target_traj, threshold_xy, target_pad_mask, contexts
            )
        else:
            pred_traj, _, _ = self._predict_teacher_forcing(input_traj, dec_in_traj, target_pad_mask, contexts)
        
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
        contexts = self._extract_contexts(batch)
        
        pred_traj, _, _, _ = self._predict_autoregressively(input_traj, dec_in_traj, threshold_xy, contexts)
        input_abs, target_abs, pred_abs = self._reconstruct_absolute_positions(input_traj, target_traj, pred_traj, target_pad_mask)

        loss = self.loss(pred_abs, target_abs, target_pad_mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))

        self.val_metrics.update(pred_abs, target_abs, target_pad_mask)
        self._visualize_prediction_vs_targets(
            input_abs, target_abs, pred_abs, target_pad_mask, batch_idx, 
            prefix="val", num_trajectories=self.num_visualized_traj
        )

        return loss
    

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Test step not implemented")


    def _predict_teacher_forcing(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        target_pad_mask: torch.Tensor,
        contexts: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        memory: Optional[torch.Tensor] = None,
        flightinfo_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict using teacher forcing.
        
        Args:
            input_traj: Normalized input trajectory [B, T_in, 8]
            dec_in_traj: Normalized decoder input (ground truth) [B, H, 5]
            target_pad_mask: Padding mask [B, H]
            contexts: Dictionary of context tensors
            num_steps: Number of steps to predict (default: horizon_seq_len)
            memory: Pre-computed encoder memory (optional, will encode if None)
            flightinfo_emb: Pre-computed flightinfo embedding (optional)
            
        Returns:
            Tuple of (predictions [B, num_steps, 3], memory, flightinfo_emb)
        """
        num_steps = num_steps or self.horizon_seq_len
        
        # Encode if memory not provided
        if memory is None:
            memory = self.model.encode(input_traj, contexts=contexts)
        
        # Encode flightinfo if not provided and enabled
        if flightinfo_emb is None and self.model.use_flightinfo and "flightinfo" in contexts:
            flightinfo_emb = self.model.flightinfo_encoder(contexts["flightinfo"])
        
        # Slice inputs to num_steps
        dec_in = dec_in_traj[:, :num_steps, :]
        pad_mask = target_pad_mask[:, :num_steps] if target_pad_mask is not None else None
        
        causal_mask = self._generate_causal_mask(num_steps, input_traj.device)
        pred_traj = self.model.decode(
            dec_in,
            memory,
            causal_mask=causal_mask,
            target_pad_mask=pad_mask,
            flightinfo_emb=flightinfo_emb,
        )
        return pred_traj, memory, flightinfo_emb

    
    def _predict_autoregressively(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        threshold_xy: torch.Tensor,
        contexts: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        memory: Optional[torch.Tensor] = None,
        flightinfo_emb: Optional[torch.Tensor] = None,
        initial_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict autoregressively.
        
        Args:
            input_traj: Normalized input trajectory [B, T_in, 8]
            dec_in_traj: Normalized decoder input [B, H, 5] - used for initial token or as prefix
            threshold_xy: Threshold coordinates [B, 2]
            contexts: Dictionary of context tensors
            num_steps: Number of steps to predict (default: horizon_seq_len)
            memory: Pre-computed encoder memory (optional, will encode if None)
            flightinfo_emb: Pre-computed flightinfo embedding (optional)
            initial_position: Starting position [B, 3] (optional, defaults to last input position)
            
        Returns:
            Tuple of (predictions [B, num_steps, 3], memory, flightinfo_emb, final_position [B, 3])
        """
        num_steps = num_steps or self.horizon_seq_len
        
        # Encode if memory not provided
        if memory is None:
            memory = self.model.encode(input_traj, contexts=contexts)

        # Encode flightinfo if not provided and enabled
        if flightinfo_emb is None and self.model.use_flightinfo and "flightinfo" in contexts:
            flightinfo_emb = self.model.flightinfo_encoder(contexts["flightinfo"])

        # Initialize position tracking
        if initial_position is None:
            input_abs = self.denormalize_inputs(input_traj)
            current_position = input_abs[:, -1, :3].clone()  # [B, 3]
        else:
            current_position = initial_position.clone()

        # Initialize decoder input with first token
        current_dec_in = dec_in_traj[:, 0:1, :]  # [B, 1, 5]
        all_predictions = []

        # Autoregressive loop
        for i in range(num_steps):
            current_seq_len = current_dec_in.size(1)
            target_mask = self._generate_causal_mask(current_seq_len, input_traj.device)
            
            # Use gradient checkpointing during training to save memory
            if self.training:
                output = checkpoint.checkpoint(
                    lambda dec_in, mem, mask, pad_mask, emb: self.model.decode(
                        dec_in, mem, causal_mask=mask, target_pad_mask=pad_mask, flightinfo_emb=emb
                    ),
                    current_dec_in,
                    memory,
                    target_mask,
                    None,
                    flightinfo_emb,
                    use_reentrant=False,
                )
            else:
                output = self.model.decode(
                    current_dec_in,
                    memory,
                    causal_mask=target_mask,
                    target_pad_mask=None,  # No padding mask during AR to avoid sequence length leakage
                    flightinfo_emb=flightinfo_emb,
                )
            next_step_pred = output[:, -1:, :]  # [B, 1, 3]
            all_predictions.append(next_step_pred)
            
            # Update current position with denormalized predicted delta
            pred_delta_denorm = self.denormalize_target_deltas(next_step_pred)  # [B, 1, 3]
            current_position = current_position + pred_delta_denorm[:, 0, :]  # [B, 3]
            
            # Compute new threshold features for next decoder input
            new_thr_features = compute_threshold_features(
                current_position[:, :2],  # [B, 2]
                threshold_xy              # [B, 2]
            )  # [B, 2]
            
            # Normalize threshold features
            pos_mean_xy = self.denormalize_inputs.mean[:2]
            pos_std_xy = self.denormalize_inputs.std[:2]
            new_thr_features_norm = (new_thr_features - pos_mean_xy) / pos_std_xy  # [B, 2]
            
            # Build next decoder input
            next_dec_in = torch.cat([
                next_step_pred,                        # [B, 1, 3]
                new_thr_features_norm.unsqueeze(1)     # [B, 1, 2]
            ], dim=-1)  # [B, 1, 5]
            
            current_dec_in = torch.cat([current_dec_in, next_dec_in], dim=1)

        pred_traj = torch.cat(all_predictions, dim=1)  # [B, num_steps, 3]
        return pred_traj, memory, flightinfo_emb, current_position


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

    # --------------------------------------
    # Scheduled Sampling
    # --------------------------------------

    def _compute_teacher_forcing_steps(self) -> int:
        """Compute the number of teacher forcing steps based on current epoch."""
        current_epoch = self.current_epoch
        
        # Phase 1: Full teacher forcing
        if current_epoch < self.teacher_forcing_epochs:
            return self.horizon_seq_len
        
        # Phase 2: Linear transition
        transition_start = self.teacher_forcing_epochs
        transition_end = self.teacher_forcing_epochs + self.transition_epochs
        
        if current_epoch < transition_end:
            # Linear interpolation: horizon_seq_len -> 0
            progress = (current_epoch - transition_start) / self.transition_epochs
            tf_steps = int(self.horizon_seq_len * (1 - progress))
            return tf_steps
        
        # Phase 3: Full autoregressive
        return 0

    def _predict_scheduled_sampling(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        target_traj: torch.Tensor,
        threshold_xy: torch.Tensor,
        target_pad_mask: torch.Tensor,
        contexts: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Predict using scheduled sampling (curriculum learning).
        
        Uses teacher forcing for the first `tf_steps` timesteps, then switches to
        autoregressive prediction for the remaining timesteps.
        """
        tf_steps = self._compute_teacher_forcing_steps()
        ar_steps = self.horizon_seq_len - tf_steps
        
        # Log the current scheduled sampling state (once per epoch, on first batch)
        if self.trainer.global_step % len(self.trainer.datamodule.train_dataloader()) == 0:
            self.log("train/teacher_forcing_steps", float(tf_steps), on_step=False, on_epoch=True)
            self.log("train/autoregressive_steps", float(ar_steps), on_step=False, on_epoch=True)
        
        # Special case: Full teacher forcing
        if tf_steps >= self.horizon_seq_len:
            pred_traj, _, _ = self._predict_teacher_forcing(
                input_traj, dec_in_traj, target_pad_mask, contexts
            )
            return pred_traj
        
        # Special case: Full autoregressive
        if tf_steps <= 0:
            pred_traj, _, _, _ = self._predict_autoregressively(
                input_traj, dec_in_traj, threshold_xy, contexts
            )
            return pred_traj
        
        # Mixed mode: Teacher forcing prefix + autoregressive suffix
        # 1. Teacher forcing phase
        tf_pred, memory, flightinfo_emb = self._predict_teacher_forcing(
            input_traj, dec_in_traj, target_pad_mask, contexts, num_steps=tf_steps
        )
        
        # 2. Compute position after teacher forcing (using ground truth deltas)
        input_abs = self.denormalize_inputs(input_traj)
        initial_position = input_abs[:, -1, :3].clone()
        gt_deltas_denorm = self.denormalize_target_deltas(target_traj[:, :tf_steps, :])
        position_after_tf = initial_position + gt_deltas_denorm.sum(dim=1)  # [B, 3]
        
        # 3. Build decoder input for AR phase: start with last TF prediction
        last_tf_pred = tf_pred[:, -1:, :]  # [B, 1, 3]
        
        # Compute threshold features at position after TF
        thr_features = compute_threshold_features(position_after_tf[:, :2], threshold_xy)
        pos_mean_xy = self.denormalize_inputs.mean[:2]
        pos_std_xy = self.denormalize_inputs.std[:2]
        thr_features_norm = (thr_features - pos_mean_xy) / pos_std_xy
        
        ar_start_token = torch.cat([
            last_tf_pred,
            thr_features_norm.unsqueeze(1)
        ], dim=-1)  # [B, 1, 5]
        
        # 4. Autoregressive phase
        ar_pred, _, _, _ = self._predict_autoregressively(
            input_traj, ar_start_token, threshold_xy, contexts,
            num_steps=ar_steps,
            memory=memory,
            flightinfo_emb=flightinfo_emb,
            initial_position=position_after_tf,
        )
        
        # 5. Concatenate predictions
        pred_traj = torch.cat([tf_pred, ar_pred], dim=1)  # [B, H, 3]
        return pred_traj