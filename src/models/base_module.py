import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch import nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.transforms.normalize import Denormalizer
from models.losses import ADELoss, CompositeApproachLoss
from data.utils.trajectory import reconstruct_absolute_from_deltas
from visualization.predictions_targets import plot_predictions_targets



class BaseModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        input_seq_len: int,
        horizon_seq_len: int,
        scheduler_cfg: DictConfig = None,
        loss_cfg: DictConfig = None,
        num_visualized_traj: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule"])
        self.model = instantiate(model_cfg["params"])
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.input_seq_len = input_seq_len
        self.horizon_seq_len = horizon_seq_len

        # Evaluation horizons
        predefined_horizons = [1, 3, 6, 15, 30, 45, 60, 75, 90, 105, 120]
        self.evaluation_horizons = [h for h in predefined_horizons if h <= self.horizon_seq_len]
        self.num_visualized_traj = num_visualized_traj

        if loss_cfg is not None:
            self.loss = instantiate(loss_cfg)
        else:
            self.loss = ADELoss()
        

    # --------------------------------------
    # Optimizer and Scheduler
    # --------------------------------------

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


    # --------------------------------------
    # Normalization storage
    # --------------------------------------

    def on_fit_start(self):
        dm = self.trainer.datamodule
        
        # Inputs are [Pos(3) + Delta(3) + ThresholdFeatures(2)]
        input_mean = torch.cat([dm.pos_mean, dm.delta_mean, dm.pos_mean[:2]], dim=0)
        input_std = torch.cat([dm.pos_std, dm.delta_std, dm.pos_std[:2]], dim=0)
        
        self.denormalize_inputs = Denormalizer(input_mean, input_std)
        self.denormalize_target_deltas = Denormalizer(dm.delta_mean, dm.delta_std)

        # Register as submodules so they're moved to the correct device automatically
        self.add_module("denormalize_inputs", self.denormalize_inputs)
        self.add_module("denormalize_target_deltas", self.denormalize_target_deltas)
        self.denormalize_inputs.to(self.device)
        self.denormalize_target_deltas.to(self.device)
        

    # --------------------------------------
    # Training and Validation
    # --------------------------------------

    def _reconstruct_absolute_positions(
        self,
        input_traj: torch.Tensor,
        target_traj: torch.Tensor,
        pred_traj: torch.Tensor,
        target_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct absolute positions from normalized trajectories.
        
        This is a convenience method that uses the utility function for trajectory reconstruction.
        
        Args:
            input_traj: Normalized input trajectory [batch_size, input_seq_len, 8]
            target_traj: Normalized target deltas [batch_size, horizon_seq_len, 3]
            pred_traj: Normalized prediction deltas [batch_size, horizon_seq_len, 3]
            target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
            
        Returns:
            input_abs: Denormalized input absolute positions [batch_size, input_seq_len, 8]
            target_abs: Reconstructed target absolute positions [batch_size, horizon_seq_len, 3]
            pred_abs: Reconstructed prediction absolute positions [batch_size, horizon_seq_len, 3]
        """
        return reconstruct_absolute_from_deltas(
            input_traj=input_traj,
            pred_deltas=pred_traj,
            target_deltas=target_traj,
            denormalize_inputs=self.denormalize_inputs,
            denormalize_target_deltas=self.denormalize_target_deltas,
            target_pad_mask=target_pad_mask,
        )

    # --------------------------------------
    # Validation metrics
    # --------------------------------------

    def on_validation_epoch_start(self):
        H = self.horizon_seq_len          # All horizons
        F = self.model.output_dim         # Number of output features (predicted deltas)

        device = self.device

        # Per-horizon accumulators
        self.val_sum_abs_error = torch.zeros(H, F, device=device)
        self.val_sum_sq_error = torch.zeros(H, F, device=device)
        self.val_sum_dist_2d = torch.zeros(H, device=device)
        self.val_sum_dist_3d = torch.zeros(H, device=device)
        self.val_sum_fde_2d = torch.tensor(0.0, device=device)
        self.val_sum_fde_3d = torch.tensor(0.0, device=device)

        # Aggregate MDE accumulators (Max Displacement Error)
        self.val_sum_max_dist_2d = torch.tensor(0.0, device=device)
        self.val_sum_max_dist_3d = torch.tensor(0.0, device=device)

        # Count of valid waypoints and trajectories
        self.val_count_valid_waypoints = torch.zeros(H, device=device) # Shape: [H]
        self.val_count_traj = torch.tensor(0.0, device=device) # Shape: [1]

    def _evaluate_step(
        self, 
        pred_abs: torch.Tensor, 
        target_abs: torch.Tensor, 
        target_pad_mask: torch.Tensor,
    ):
        """
        Store metric accumulators for validation step.
        
        Reconstructs absolute positions from deltas and computes metrics.
        
        Args:
            pred_abs: Model predictions (absolute positions) [batch_size, horizon_seq_len, 3]
            target_abs: Target values (absolute positions) [batch_size, horizon_seq_len, 3]
            target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
        """
        valid_mask = ~target_pad_mask # [B, H]
        last_valid_index = (valid_mask.sum(dim=1) - 1).clamp(min=0) # [B]
        batch_indices = torch.arange(pred_abs.size(0), device=pred_abs.device) # [B]
        
        # 1. Feature-wise Errors on absolute positions [B, H, F]
        diff = pred_abs - target_abs
        abs_err = diff.abs()
        sq_err = diff ** 2

        # 2. Euclidean Distances [B, H]
        dist_2d = torch.norm(diff[:, :, :2], dim=2)  # Position distances (X, Y)
        dist_3d = torch.norm(diff[:, :, :3], dim=2)  # Position + altitude distances

        # 3. Masking
        # Zero out invalid entries for summation
        # Expand mask for features: [B, H] -> [B, H, 1]
        valid_mask_unsqueezed = valid_mask.unsqueeze(-1)
        abs_err_masked = abs_err * valid_mask_unsqueezed
        sq_err_masked = sq_err * valid_mask_unsqueezed
        dist_2d_masked = dist_2d * valid_mask
        dist_3d_masked = dist_3d * valid_mask
        fde_2d = dist_2d_masked[batch_indices, last_valid_index]
        fde_3d = dist_3d_masked[batch_indices, last_valid_index]

        # 4. Accumulate per horizon (Sum over batch dimension)
        self.val_sum_abs_error += abs_err_masked.sum(dim=0)
        self.val_sum_sq_error += sq_err_masked.sum(dim=0)
        self.val_sum_dist_2d += dist_2d_masked.sum(dim=0)
        self.val_sum_dist_3d += dist_3d_masked.sum(dim=0)
        self.val_sum_fde_2d += fde_2d.sum()
        self.val_sum_fde_3d += fde_3d.sum()

        # 5. MDE (Max Displacement Error) calculation
        # Use masked max: set padded positions to -inf so they're ignored in max()
        dist_2d_masked[~valid_mask] = -torch.inf
        dist_3d_masked[~valid_mask] = -torch.inf
        traj_max_dist_2d = dist_2d_masked.max(dim=1).values # [B]
        traj_max_dist_3d = dist_3d_masked.max(dim=1).values # [B]
        has_valid_points = valid_mask.any(dim=1)            # [B]
        self.val_sum_max_dist_2d += traj_max_dist_2d[has_valid_points].sum()
        self.val_sum_max_dist_3d += traj_max_dist_3d[has_valid_points].sum()

        self.val_count_valid_waypoints += valid_mask.sum(dim=0)
        self.val_count_traj += valid_mask.any(dim=1).sum()

    def on_validation_epoch_end(self):
        for tensor in [  # Needed for DDP
            self.val_sum_abs_error,
            self.val_sum_sq_error,
            self.val_sum_dist_2d,
            self.val_sum_dist_3d,
            self.val_sum_fde_2d,
            self.val_sum_fde_3d,
            self.val_sum_max_dist_2d,
            self.val_sum_max_dist_3d,
            self.val_count_valid_waypoints,
            self.val_count_traj,
        ]:
            self.trainer.strategy.reduce(tensor, reduce_op="sum")

        valid_points_per_horizon = self.val_count_valid_waypoints.clamp(min=1.0) # Shape: [H]
        total_valid_points = self.val_count_valid_waypoints.sum().clamp(min=1.0) # Scalar
        total_trajectories = self.val_count_traj.clamp(min=1.0)                  # Scalar

        # 1. ADE (Average of Euclidean Distance per valid waypoint)
        ade2d_seq = self.val_sum_dist_2d / valid_points_per_horizon     # Shape: [H]
        ade3d_seq = self.val_sum_dist_3d / valid_points_per_horizon     # Shape: [H]
        ade2d_scalar = self.val_sum_dist_2d.sum() / total_valid_points  # Scalar
        ade3d_scalar = self.val_sum_dist_3d.sum() / total_valid_points  # Scalar
        
        # 2. MAE (Mean Absolute Error) and RMSE (Root Mean Square Error)
        valid_points_per_horizon = valid_points_per_horizon.unsqueeze(1)    # Shape: [H, 1]
        mae = self.val_sum_abs_error / valid_points_per_horizon             # Shape: [H, F]
        rmse = torch.sqrt(self.val_sum_sq_error / valid_points_per_horizon) # Shape: [H, F]

        # 3. MDE (Average Max Displacement Error)
        mde2d_scalar = self.val_sum_max_dist_2d / total_trajectories # Scalar
        mde3d_scalar = self.val_sum_max_dist_3d / total_trajectories # Scalar

        # 4. FDE (Average Final Displacement Error)
        fde2d_scalar = self.val_sum_fde_2d / total_trajectories # Scalar
        fde3d_scalar = self.val_sum_fde_3d / total_trajectories # Scalar

        # --- Logging Line Plots ---
        self._horizon_line_plot(ade2d_seq, "ADE2D")
        self._horizon_line_plot(ade3d_seq, "ADE3D")
        for feature in ["X", "Y", "Altitude"]:
            self._horizon_line_plot(mae, "MAE", feature)
            self._horizon_line_plot(rmse, "RMSE", feature)
        
        # --- Logging Aggregates (Scalars) ---
        self.log_dict({
            "val/ADE2D": ade2d_scalar,
            "val/ADE3D": ade3d_scalar,
            "val/FDE2D": fde2d_scalar,
            "val/FDE3D": fde3d_scalar,
            "val/MDE2D": mde2d_scalar,
            "val/MDE3D": mde3d_scalar,
        })

 
    # --------------------------------------
    # Logging and Visualization
    # --------------------------------------

    def _horizon_line_plot(self, metric: torch.Tensor, metric_name: str, feature_name: str = None):
        horizons = np.arange(1, self.horizon_seq_len + 1).tolist()
        if feature_name is not None:
            feat_idx = ["X", "Y", "Altitude"].index(feature_name)
            metric = metric[:, feat_idx]
        metric_values = metric.detach().cpu().tolist()
        data = [[x, y] for (x, y) in zip(horizons, metric_values)]

        y_column = "Error" if feature_name is None else f"{feature_name} Error"
        x_column = "Horizon"
        table = wandb.Table(data=data, columns = [x_column, y_column])

        if feature_name is None:
            category = f"val-ade-horizons"
        else:
            category = f"val-{metric_name.lower()}-horizons"
            metric_name = f"{metric_name} {feature_name}"

        plot = wandb.plot.line(table, x_column, y_column, title=f"{metric_name} vs Horizon")
        self.logger.experiment.log({f"{category}/{metric_name}" : plot})


    def _visualize_prediction_vs_targets(self, input_abs: torch.Tensor, target_abs: torch.Tensor, pred_abs: torch.Tensor, target_pad_mask: torch.Tensor, batch_idx: int):
        """
        Visualize the prediction vs targets for the first num_visualized_traj trajectories in batch index 0.

        Args:
            input_abs: Input absolute positions [batch_size, horizon_seq_len, 3]
            target_abs: Target absolute positions [batch_size, horizon_seq_len, 3]
            pred_abs: Predicted absolute positions [batch_size, horizon_seq_len, 3]
            target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
            batch_idx: Batch index
        """
        if batch_idx != 0:
            return

        for i in range(min(self.num_visualized_traj, input_abs.shape[0])):
            input_abs_i = input_abs[i].detach().cpu().float().numpy()
            target_abs_i = target_abs[i].detach().cpu().float().numpy()
            pred_abs_i = pred_abs[i].detach().cpu().float().numpy()
            target_pad_mask_i = target_pad_mask[i].detach().cpu().numpy()

            fig, ax = plot_predictions_targets(input_abs_i, target_abs_i, pred_abs_i, target_pad_mask_i, "EDDB") # TODO: add icao
            self.logger.experiment.log({
                f"val-predictions-targets/batch_{batch_idx}_traj_{i}": wandb.Image(fig)
            })
            plt.close(fig)





