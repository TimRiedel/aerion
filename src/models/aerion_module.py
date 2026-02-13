import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Any, Dict, Optional
from omegaconf import DictConfig

from models.base_module import BaseModule
from data.transforms.normalize import Denormalizer, Normalizer
from data.utils.runway import get_distances_to_centerline


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
            scheduled_sampling_cfg=scheduled_sampling_cfg,
            num_visualized_traj=num_visualized_traj,
        )

    def on_fit_start(self):
        dm = self.trainer.datamodule
        
        # Inputs are [Pos(3) + Delta(3)]
        input_mean = torch.cat([dm.pos_mean, dm.delta_mean, dm.dist_mean], dim=0)
        input_std = torch.cat([dm.pos_std, dm.delta_std, dm.dist_std], dim=0)
        
        self.denormalize_inputs = Denormalizer(input_mean, input_std)
        self.denormalize_target_deltas = Denormalizer(dm.delta_mean, dm.delta_std)
        self.denormalize_distances = Denormalizer(dm.dist_mean, dm.dist_std)

        self.normalize_abs_positions = Normalizer(dm.pos_mean, dm.pos_std)
        self.normalize_distances = Normalizer(dm.dist_mean, dm.dist_std)

        self.add_module("denormalize_inputs", self.denormalize_inputs)
        self.add_module("denormalize_target_deltas", self.denormalize_target_deltas)
        self.add_module("denormalize_distances", self.denormalize_distances)
        self.add_module("normalize_abs_positions", self.normalize_abs_positions)
        self.add_module("normalize_distances", self.normalize_distances)
        self.denormalize_inputs.to(self.device)
        self.denormalize_target_deltas.to(self.device)
        self.denormalize_distances.to(self.device)
        self.normalize_abs_positions.to(self.device)
        self.normalize_distances.to(self.device)
    
    
    def training_step(self, batch, batch_idx):
        input_traj = batch["input_traj"]
        target_traj = batch["target_traj"]
        dec_in_traj = batch["dec_in_traj"]
        target_pad_mask = batch["mask_traj"]
        runway = batch["runway"]
        contexts = self._extract_contexts(batch)

        if self.scheduled_sampling_enabled:
            pred_deltas_norm = self._predict_scheduled_sampling(
                input_traj, dec_in_traj, target_traj, runway, target_pad_mask, contexts, batch_idx
            )
        else:
            pred_deltas_norm, _, _ = self._predict_teacher_forcing(input_traj, dec_in_traj, target_pad_mask, contexts)
        
        input_pos_abs, target_pos_abs, pred_pos_abs = self._reconstruct_absolute_positions(input_traj, target_traj, pred_deltas_norm, target_pad_mask)
        pred_pos_norm = self.normalize_abs_positions(pred_pos_abs)
        target_pos_norm = self.normalize_abs_positions(target_pos_abs)
        
        loss = self.loss(pred_pos_abs, target_pos_abs, pred_pos_norm, target_pos_norm, target_pad_mask, runway)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))
        
        self.train_metrics.update(pred_pos_abs, target_pos_abs, target_pad_mask)
        self._visualize_prediction_vs_targets(
            input_pos_abs, target_pos_abs, pred_pos_abs, target_pad_mask, batch_idx,
            prefix="train", num_trajectories=6
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_traj = batch["input_traj"]
        target_traj = batch["target_traj"]
        dec_in_traj = batch["dec_in_traj"]
        target_pad_mask = batch["mask_traj"]
        runway = batch["runway"]
        contexts = self._extract_contexts(batch)
        
        pred_deltas_norm, _, _ = self._predict_autoregressively(input_traj, dec_in_traj, runway, contexts)
        input_pos_abs, target_pos_abs, pred_pos_abs = self._reconstruct_absolute_positions(input_traj, target_traj, pred_deltas_norm, target_pad_mask)
        pred_pos_norm = self.normalize_abs_positions(pred_pos_abs)
        target_pos_norm = self.normalize_abs_positions(target_pos_abs)

        loss = self.loss(pred_pos_abs, target_pos_abs, pred_pos_norm, target_pos_norm, target_pad_mask, runway)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(input_traj))

        self.val_metrics.update(pred_pos_abs, target_pos_abs, target_pad_mask)
        self._visualize_prediction_vs_targets(
            input_pos_abs, target_pos_abs, pred_pos_abs, target_pad_mask, batch_idx, 
            prefix="val", num_trajectories=self.num_visualized_traj
        )

        return loss
    

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Test step not implemented")


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


    # --------------------------------------
    # Prediction functions
    # --------------------------------------

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
            input_traj: Normalized input trajectory [B, T_in, input_features]
            dec_in_traj: Normalized decoder input (ground truth) [B, H, dec_in_features]
            target_pad_mask: Padding mask [B, H]
            contexts: Dictionary of context tensors
            num_steps: Number of steps to predict (default: horizon_seq_len)
            memory: Pre-computed encoder memory (optional, will encode if None)
            flightinfo_emb: Pre-computed flightinfo embedding (optional)
            
        Returns:
            Tuple of (predictions [B, num_steps, 3], memory, flightinfo_emb)
        """
        num_steps = num_steps or self.horizon_seq_len
        
        if memory is None:
            memory = self.model.encode(input_traj, contexts=contexts)
        
        if flightinfo_emb is None and self.model.use_flightinfo and "flightinfo" in contexts:
            flightinfo_emb = self.model.flightinfo_encoder(contexts["flightinfo"])
        
        # Slice inputs to num_steps
        dec_in = dec_in_traj[:, :num_steps, :]
        pad_mask = target_pad_mask[:, :num_steps] if target_pad_mask is not None else None
        
        causal_mask = self._generate_causal_mask(num_steps, input_traj.device)
        pred_deltas_norm = self.model.decode(
            dec_in,
            memory,
            causal_mask=causal_mask,
            target_pad_mask=pad_mask,
            flightinfo_emb=flightinfo_emb,
        )
        return pred_deltas_norm, memory, flightinfo_emb

    
    def _predict_autoregressively(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        runway: Dict[str, torch.Tensor],
        contexts: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        memory: Optional[torch.Tensor] = None,
        continue_decoding = False,
        flightinfo_emb: Optional[torch.Tensor] = None,
        initial_position_abs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict autoregressively.
        
        Args:
            input_traj: Normalized input trajectory [B, T_in, input_features]
            dec_in_traj: Normalized decoder input [B, H, dec_in_features] - used for initial token or as prefix
            runway: Runway dictionary containing "xyz" coordinates and "bearing"
            contexts: Dictionary of context tensors
            num_steps: Number of steps to predict (default: horizon_seq_len)
            memory: Pre-computed encoder memory (optional, will encode if None)
            continue_decoding: Whether to take the full dec_in_traj as the initial decoder input (True) or only the first token (False). In the first case, recomputes the position from passed decoder deltas.
            flightinfo_emb: Pre-computed flightinfo embedding (optional)
            initial_position_abs: Starting position for autoregression in absolute (denormalized) coordinates [B, 3] (optional, defaults to last decoder input position)
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
            pred_deltas_norm = output[:, -1:, :]  # [B, 1, 3]
            all_predictions_norm.append(pred_deltas_norm)
            
            next_dec_in, current_position_abs = self._build_next_decoder_input(pred_deltas_norm, current_position_abs, runway["xyz"], runway["centerline_points_xy"])
            current_dec_in = torch.cat([current_dec_in, next_dec_in], dim=1)

        pred_deltas_norm = torch.cat(all_predictions_norm, dim=1)  # [B, num_steps, 3]
        return pred_deltas_norm, memory, flightinfo_emb

    def _build_next_decoder_input(self, pred_deltas_norm: torch.Tensor, current_position_abs: torch.Tensor, threshold_xyz: torch.Tensor, centerline_points_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Update current position with denormalized predicted delta
        pred_delta_abs = self.denormalize_target_deltas(pred_deltas_norm)  # [B, 1, 3]
        current_position_abs = current_position_abs + pred_delta_abs[:, 0, :]  # [B, 3]
        current_position_norm = self.normalize_abs_positions(current_position_abs).unsqueeze(1) # [B, 1, 3]
        
        # Compute distances to threshold and centerline points and normalize
        threshold_xy = threshold_xyz[:, :2]  # [B, 2]
        dist_threshold_abs = get_distances_to_centerline(current_position_abs[:, :2], [threshold_xy])
        dist_centerline_points_abs = get_distances_to_centerline(current_position_abs[:, :2], centerline_points_xy)
        dist_abs = torch.cat([dist_threshold_abs, dist_centerline_points_abs], dim=-1)
        dist_norm = self.normalize_distances(dist_abs).unsqueeze(1) # [B, 1, N]

        next_dec_in = torch.cat([pred_deltas_norm, dist_norm], dim=-1)  # [B, 1, dec_in_features]
        return next_dec_in, current_position_abs


    def _predict_scheduled_sampling(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        target_traj: torch.Tensor,
        runway: Dict[str, torch.Tensor],
        target_pad_mask: torch.Tensor,
        contexts: Dict[str, torch.Tensor],
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
            pred_traj, _, _ = self._predict_teacher_forcing(
                input_traj, dec_in_traj, target_pad_mask, contexts
            )
            return pred_traj
        
        # Special case: Full autoregressive
        if tf_steps <= 0:
            pred_traj, _, _ = self._predict_autoregressively(
                input_traj, dec_in_traj, runway, contexts
            )
            return pred_traj
        
        # Mixed mode: Teacher forcing prefix + autoregressive suffix
        # Teacher forcing phase
        tf_pred_deltas_norm, memory, flightinfo_emb = self._predict_teacher_forcing(
            input_traj, dec_in_traj, target_pad_mask, contexts, num_steps=tf_steps
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
        ar_pred_deltas_norm, _, _ = self._predict_autoregressively(
            input_traj, ar_start_dec_in, runway, contexts,
            num_steps=ar_steps,
            memory=memory,
            flightinfo_emb=flightinfo_emb,
            continue_decoding=True,
            initial_position_abs=position_after_last_tf_abs,
        )
        
        # Concatenate predictions
        pred_deltas_norm = torch.cat([tf_pred_deltas_norm, ar_pred_deltas_norm], dim=1)  # [B, H, 3]
        return pred_deltas_norm