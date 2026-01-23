import wandb
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch import nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.transforms.normalize import ZScoreDenormalize
from visualization.predictions_targets import plot_predictions_targets



class BaseModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        input_seq_len: int,
        horizon_seq_len: int,
        scheduler_cfg: DictConfig = None,
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
        
        # Loss function
        self.criterion = nn.MSELoss(reduction="none")

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

        # Store stats for denormalization
        self.register_buffer("pos_mean", dm.pos_mean)
        self.register_buffer("pos_std", dm.pos_std)
        self.register_buffer("delta_mean", dm.delta_mean)
        self.register_buffer("delta_std", dm.delta_std)

        input_mean = torch.cat([dm.pos_mean, dm.delta_mean], dim=0)
        input_std = torch.cat([dm.pos_std, dm.delta_std], dim=0)
        self.denormalize_inputs = ZScoreDenormalize(input_mean, input_std) # Inputs are positions and deltas [6]
        self.denormalize_targets = ZScoreDenormalize(dm.delta_mean, dm.delta_std) # Targets and decoder input are pure deltas [3]

    # --------------------------------------
    # Training and Validation
    # --------------------------------------
    
    def _compute_loss(
        self,
        pred_traj: torch.Tensor,
        target_traj: torch.Tensor,
        target_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.
        
        Args:
            pred_traj: Model predictions [batch_size, horizon_seq_len, num_features]
            target_traj: Target values [batch_size, horizon_seq_len, num_features]
            target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
            
        Returns:
            Scalar loss value
        """
        active_loss = ~target_pad_mask 
        
        loss = self.criterion(pred_traj[active_loss], target_traj[active_loss]).mean()
        return loss


    # --------------------------------------
    # Validation metrics
    # --------------------------------------

    def on_validation_epoch_start(self):
        H = self.horizon_seq_len          # All horizons
        F = self.model.decoder_input_dim  # Number features

        device = self.device

        # Per-horizon accumulators
        self.val_sum_abs_error = torch.zeros(H, F, device=device)
        self.val_sum_sq_error = torch.zeros(H, F, device=device)
        self.val_sum_dist_2d = torch.zeros(H, device=device)
        self.val_sum_dist_3d = torch.zeros(H, device=device)
        self.val_count_valid_waypoints = torch.zeros(H, device=device) # Shape: [H]

        # Aggregate MDE accumulators (Max Displacement Error)
        self.val_sum_of_max_dist_2d = torch.tensor(0.0, device=device)
        self.val_sum_of_max_dist_3d = torch.tensor(0.0, device=device)
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

        # 4. Accumulate per horizon (Sum over batch dimension)
        self.val_sum_abs_error += abs_err_masked.sum(dim=0)
        self.val_sum_sq_error += sq_err_masked.sum(dim=0)
        self.val_sum_dist_2d += dist_2d_masked.sum(dim=0)
        self.val_sum_dist_3d += dist_3d_masked.sum(dim=0)
        self.val_count_valid_waypoints += valid_mask.sum(dim=0)

        # 5. MDE (Max Displacement Error) calculation
        traj_max_dist_2d = dist_2d_masked.max(dim=1).values # [B]
        traj_max_dist_3d = dist_3d_masked.max(dim=1).values # [B]
        has_valid_points = valid_mask.any(dim=1)         # [B]
        self.val_sum_of_max_dist_2d += traj_max_dist_2d[has_valid_points].sum()
        self.val_sum_of_max_dist_3d += traj_max_dist_3d[has_valid_points].sum()
        self.val_count_traj += has_valid_points.sum()


    def on_validation_epoch_end(self):
        for tensor in [  # Needed for DDP
            self.val_sum_abs_error,
            self.val_sum_sq_error,
            self.val_sum_dist_2d,
            self.val_sum_dist_3d,
            self.val_sum_of_max_dist_2d,
            self.val_sum_of_max_dist_3d,
            self.val_count_valid_waypoints,
            self.val_count_traj,
        ]:
            self.trainer.strategy.reduce(tensor, reduce_op="sum")

        # count.shape: [H, 1]
        count = self.val_count_valid_waypoints.clamp(min=1.0).unsqueeze(1)
        
        # mae.shape = rmse.shape: [H, F]
        mae = self.val_sum_abs_error / count
        rmse = torch.sqrt(self.val_sum_sq_error / count)

        # ade.shape: [H] -> Remove feature dim for 1D tensors
        count_1d = count.squeeze(1)
        ade2d_seq = self.val_sum_dist_2d / count_1d
        ade3d_seq = self.val_sum_dist_3d / count_1d

        # --- Logging Plots (filtered by evaluation_horizons) ---
        eval_indices = [h - 1 for h in self.evaluation_horizons]
        
        # Extract specific horizons for plotting
        mae_plot = mae[eval_indices]
        rmse_plot = rmse[eval_indices]
        ade2d_plot = ade2d_seq[eval_indices].unsqueeze(1)
        ade3d_plot = ade3d_seq[eval_indices].unsqueeze(1)

        for feature in ["X", "Y", "Altitude"]:
            self._log_error_vs_horizon(mae_plot, "MAE", feature)
            self._log_error_vs_horizon(rmse_plot, "RMSE", feature)

        self._log_error_vs_horizon(ade2d_plot, "ADE2D")
        self._log_error_vs_horizon(ade3d_plot, "ADE3D")

        # --- Logging Aggregates (Scalars) ---
        total_valid_points = self.val_count_valid_waypoints.sum() # Number of valid waypoints over all horizons
        total_trajectories = self.val_count_traj.clamp(min=1.0)

        # ADE (Average of Euclidean Distance per valid waypoint)
        ade2d_scalar = self.val_sum_dist_2d.sum() / total_valid_points
        ade3d_scalar = self.val_sum_dist_3d.sum() / total_valid_points

        # MDE (Average of Max Displacement per trajectory)
        mde2d_scalar = self.val_sum_of_max_dist_2d / total_trajectories
        mde3d_scalar = self.val_sum_of_max_dist_3d / total_trajectories

        self.log_dict({
            "val/ADE2D": ade2d_scalar,
            "val/ADE3D": ade3d_scalar,
            # "val/FDE": ade3d_seq[-1], # TODO: must be FDE at last horizon (without padding -> separately aggregated)
            "val/MDE2D": mde2d_scalar,
            "val/MDE3D": mde3d_scalar,
        })

    def _reconstruct_absolute_positions(self, input_traj: torch.Tensor, target_traj: torch.Tensor, pred_traj: torch.Tensor) -> torch.Tensor:
        input_abs = self.denormalize_inputs(input_traj)
        target_deltas = self.denormalize_targets(target_traj)
        pred_deltas = self.denormalize_targets(pred_traj)

        last_position_abs = input_abs[:,-1,:3].unsqueeze(1)
        target_abs = last_position_abs + target_deltas.cumsum(dim=1)
        pred_abs = last_position_abs + pred_deltas.cumsum(dim=1)

        return input_abs, target_abs, pred_abs

        
    # --------------------------------------
    # Logging and Visualization
    # --------------------------------------

    def _log_error_vs_horizon(self, metric: torch.Tensor, metric_name: str, feature_name: str = None):
        if feature_name is not None:
            features = ["X", "Y", "Altitude"]
            feat_idx = features.index(feature_name)
            metric = metric[:, feat_idx].detach().cpu().tolist()
            name = f"{metric_name} {feature_name}"
        else:
            # For 1D metrics like ADE2D, ADE3D -> Input metric shape: [H, 1]
            metric = metric.squeeze(1).detach().cpu().tolist()
            name = metric_name

        table = wandb.Table(columns=[name, "Horizon"])
        for i, val in enumerate(metric):
            h = self.evaluation_horizons[i]
            table.add_data(val, h)

        fields = {"value": name, "label": "Horizon"}
        custom_chart = wandb.plot_table(
            vega_spec_name="timriedel/error_by_horizon",
            data_table=table,
            fields=fields,
        )

        self.logger.experiment.log({
            f"val-{metric_name.lower()}/{name}": custom_chart
        })


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





