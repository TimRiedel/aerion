import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.transforms.normalize import Denormalizer
from models.metrics import ADELoss, AccumulatedTrajectoryMetrics
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
        self.num_visualized_traj = num_visualized_traj

        if loss_cfg is not None:
            self.loss = instantiate(loss_cfg)
        else:
            self.loss = ADELoss()
        
        # Initialize metric accumulators (will be properly initialized in epoch start hooks)
        self.val_metrics = None
        self.train_metrics = None
        

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
    # Metrics accumulation and logging
    # --------------------------------------

    def on_train_epoch_start(self):
        self.train_metrics = AccumulatedTrajectoryMetrics(self.horizon_seq_len, self.device)

    def on_validation_epoch_start(self):
        self.val_metrics = AccumulatedTrajectoryMetrics(self.horizon_seq_len, self.device)

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute(
            reduce_op="sum",
            strategy=self.trainer.strategy,
        )
        
        self._log_metrics(metrics, "train")
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute(
            reduce_op="sum",
            strategy=self.trainer.strategy if hasattr(self.trainer, 'strategy') else None,
        )
        
        self._log_metrics(metrics, "val")
        self.val_metrics.reset()

    # --------------------------------------
    # Reconstruction of absolute positions
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
            target_deltas=target_traj,
            pred_deltas=pred_traj,
            denormalize_inputs=self.denormalize_inputs,
            denormalize_target_deltas=self.denormalize_target_deltas,
            target_pad_mask=target_pad_mask,
        )

 
    # --------------------------------------
    # Logging and Visualization
    # --------------------------------------

    def _log_metrics(self, metrics: dict, prefix: str):
        self.log_dict({
            f"{prefix}/ADE2D": metrics["ade_2d_scalar"],
            f"{prefix}/ADE3D": metrics["ade_3d_scalar"],
            f"{prefix}/FDE2D": metrics["fde_2d_scalar"],
            f"{prefix}/FDE3D": metrics["fde_3d_scalar"],
            f"{prefix}/MDE2D": metrics["mde_2d_scalar"],
            f"{prefix}/MDE3D": metrics["mde_3d_scalar"],
        })

        self._horizon_line_plot(metrics["ade_2d_per_horizon"], "ADE2D", prefix)
        self._horizon_line_plot(metrics["ade_3d_per_horizon"], "ADE3D", prefix)
        for feature in ["X", "Y", "Altitude"]:
            self._horizon_line_plot(metrics["mae_per_horizon"], "MAE", prefix, feature)
            self._horizon_line_plot(metrics["rmse_per_horizon"], "RMSE", prefix, feature)

    def _horizon_line_plot(self, metric: torch.Tensor, metric_name: str, prefix: str, feature_name: str = None):
        if feature_name is not None:
            feat_idx = ["X", "Y", "Altitude"].index(feature_name)
            metric = metric[:, feat_idx]
        metric_values = metric.detach().cpu().tolist()

        # Find the first index in metric_values that is zero, limit horizons to that length
        zero_idx = next((i for i, value in enumerate(metric_values) if value == 0), len(metric_values))
        horizons = np.arange(1, zero_idx + 1).tolist() if zero_idx > 0 else []
        metric_values = metric_values[:zero_idx] if zero_idx > 0 else []

        data = [[x, y] for (x, y) in zip(horizons, metric_values)]

        y_column = "Error" if feature_name is None else f"{feature_name} Error"
        x_column = "Horizon"
        table = wandb.Table(data=data, columns=[x_column, y_column])

        if feature_name is None:
            full_metric_name = metric_name
            category = f"{prefix}-ade-horizons"
        else:
            full_metric_name = f"{metric_name} {feature_name}"
            category = f"{prefix}-{metric_name.lower()}-horizons"

        plot = wandb.plot.line(table, x_column, y_column, title=f"{full_metric_name} vs Horizon")
        self.logger.experiment.log({f"{category}/{full_metric_name}": plot})

    def _visualize_prediction_vs_targets(
        self,
        input_abs: torch.Tensor,
        target_abs: torch.Tensor,
        pred_abs: torch.Tensor,
        target_pad_mask: torch.Tensor,
        batch_idx: int,
        prefix: str = "val",
        num_trajectories: int = 6,
    ):
        """
        Visualize the prediction vs targets for trajectories in batch index 0.

        Args:
            input_abs: Input absolute positions [batch_size, input_seq_len, 3]
            target_abs: Target absolute positions [batch_size, horizon_seq_len, 3]
            pred_abs: Predicted absolute positions [batch_size, horizon_seq_len, 3]
            target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
            batch_idx: Batch index
            prefix: Prefix for visualization names (e.g., "train", "val")
            num_trajectories: Number of trajectories to visualize (default: 6)
        """
        if batch_idx != 0:
            return

        for i in range(min(num_trajectories, input_abs.shape[0])):
            input_abs_i = input_abs[i].detach().cpu().float().numpy()
            target_abs_i = target_abs[i].detach().cpu().float().numpy()
            pred_abs_i = pred_abs[i].detach().cpu().float().numpy()
            target_pad_mask_i = target_pad_mask[i].detach().cpu().numpy()

            fig, ax = plot_predictions_targets(input_abs_i, target_abs_i, pred_abs_i, target_pad_mask_i, "EDDB") # TODO: add icao
            self.logger.experiment.log({
                f"{prefix}-predictions-targets/batch_{batch_idx}_traj_{i}": wandb.Image(fig)
            })
            plt.close(fig)





