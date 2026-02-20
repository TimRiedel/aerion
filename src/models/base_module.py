import io
from PIL import Image
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch import nn

from models.metrics import AccumulatedTrajectoryMetrics, CompositeApproachLoss
from data.utils.trajectory import reconstruct_absolute_from_deltas
from visualization import *


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        loss_cfg: DictConfig,
        input_seq_len: int,
        horizon_seq_len: int,
        scheduler_cfg: DictConfig = None,
        scheduled_sampling_cfg: DictConfig = None,
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

        self.loss = CompositeApproachLoss(loss_configs=loss_cfg)

        scheduled_sampling_cfg = scheduled_sampling_cfg or {}
        self.scheduled_sampling_enabled = scheduled_sampling_cfg.get("enabled", False)
        self.teacher_forcing_epochs = scheduled_sampling_cfg.get("teacher_forcing_epochs", 1)
        self.transition_epochs = scheduled_sampling_cfg.get("transition_epochs", 15)

        # Initialize metric accumulators (will be properly initialized in epoch start hooks)
        self.val_metrics = None
        self.train_metrics = None

    # --------------------------------------
    # Optimizer and Scheduler
    # --------------------------------------

    def configure_optimizers(self):
        # Note: use self.parameters() instead of self.model.parameters() to include loss weight parameters
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        
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

    def _log_loss(self, loss: torch.Tensor, loss_info: dict = None, prefix: str = "val", batch_size: int = None):
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        if loss_info is None:
            return

        loss_weights = loss_info.get("loss_weights", {})
        for name, weight in loss_weights.items():
            self.log(f"{prefix}_loss_weights/{name}", weight, on_step=False, on_epoch=True, batch_size=batch_size)

        loss_sigmas = loss_info.get("loss_sigmas", {})
        for name, sigma in loss_sigmas.items():
            self.log(f"{prefix}_loss_sigmas/{name}", sigma, on_step=False, on_epoch=True, batch_size=batch_size)

    def _log_metrics(self, metrics: dict, prefix: str):
        self.log_dict({
            f"{prefix}/ADE2D": metrics["ade_2d_scalar"],
            f"{prefix}/ADE3D": metrics["ade_3d_scalar"],
            f"{prefix}/FDE2D": metrics["fde_2d_scalar"],
            f"{prefix}/FDE3D": metrics["fde_3d_scalar"],
            f"{prefix}/MDE2D": metrics["mde_2d_scalar"],
            f"{prefix}/MDE3D": metrics["mde_3d_scalar"],
            f"{prefix}/RTDE": metrics["rtde_scalar"],
            f"{prefix}/RelativeRTDE": metrics["rtde_relative_scalar"],
        })

        self._horizon_line_plot(metrics["ade_2d_per_horizon"], "ADE2D", prefix)
        self._horizon_line_plot(metrics["ade_3d_per_horizon"], "ADE3D", prefix)
        for feature in ["X", "Y", "Altitude"]:
            self._horizon_line_plot(metrics["mae_per_horizon"], "MAE", prefix, feature)
            self._horizon_line_plot(metrics["rmse_per_horizon"], "RMSE", prefix, feature)
        
        self._log_histogram(metrics["traj_ade_2d_values"], "ADE2D", prefix)
        self._log_histogram(metrics["traj_ade_3d_values"], "ADE3D", prefix)
        self._log_histogram(metrics["traj_fde_2d_values"], "FDE2D", prefix)
        self._log_histogram(metrics["traj_fde_3d_values"], "FDE3D", prefix)
        self._log_histogram(metrics["traj_rtde_relative_values"], "Relative-RTDE", prefix, is_rtd=True)
        
        self._plot_rtde_violins(metrics["traj_rtd_target_values"], metrics["traj_rtde_values"], metrics["traj_rtde_relative_values"], prefix=prefix)
        self._plot_rtd_scatter(metrics["traj_rtd_target_values"], metrics["traj_rtd_pred_values"], metrics["traj_rtde_relative_values"], prefix=prefix)

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
    
    def _log_histogram(self, values: torch.Tensor, metric_name: str, prefix: str, is_rtd: bool = False):
        """
        Log histogram of per-trajectory metric values.
        
        Args:
            values: Tensor of per-trajectory metric values [N]
            metric_name: Name of the metric (e.g., "ADE2D", "FDE3D")
            prefix: Prefix for logging (e.g., "train", "val")
        """
        if values.numel() == 0:
            return
        
        if is_rtd:
            category = f"{prefix}-rtd"
            vega_spec_name = "timriedel/histogram-binned-percentages"
        else:
            category = f"{prefix}-histograms"
            vega_spec_name = "timriedel/histogram-binned"

        values_np = values.detach().cpu().numpy()
        data = [[i, float(value)] for i, value in enumerate(values_np)]
        table = wandb.Table(
            data=data,
            columns=["step", metric_name],
        )
        fields = {"value": metric_name, "title": f"Histogram of {metric_name}"}
        histogram = wandb.plot_table(
            vega_spec_name=vega_spec_name,
            data_table=table,
            fields=fields,
        )
        self.logger.experiment.log({
            f"{category}/hist-{metric_name}": histogram
        })

    def _plot_prediction_vs_target(
        self,
        input_abs: torch.Tensor,
        target_abs: torch.Tensor,
        pred_abs: torch.Tensor,
        target_pad_mask: torch.Tensor,
        batch_idx: int,
        flight_id: str = None,
        target_rtd: float = None,
        pred_rtd: float = None,
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
            flight_id: Flight ID [batch_size]
            target_rtd: Target RTD in meters [batch_size]
            pred_rtd: Target RTD in meters [batch_size]
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
            target_rtd_i = target_rtd[i].detach().cpu().float().numpy()
            pred_rtd_i = pred_rtd[i].detach().cpu().float().numpy()
            flight_id_i = flight_id[i]

            fig, _ = plot_predictions_targets(input_abs_i, target_abs_i, pred_abs_i, target_pad_mask_i, "EDDB", flight_id_i, target_rtd_i, pred_rtd_i) # TODO: add icao
            self.logger.experiment.log({
                f"{prefix}-predictions-targets/batch_{batch_idx}_traj_{i}": self.fig_to_wandb_image(fig)
            })
            plt.close(fig)

    def _plot_rtde_violins(
        self,
        rtd_target: torch.Tensor,
        rtde: torch.Tensor,
        rtde_relative: torch.Tensor,
        prefix: str = "val",
    ):
        rtd_target_km = (rtd_target / 1000.0).detach().cpu().float().numpy()
        rtde_km = (rtde / 1000.0).detach().cpu().float().numpy()
        rtde_relative = rtde_relative.detach().cpu().float().numpy()

        fig, _ = plot_rtde_violins(rtd_target_km, rtde_km, is_relative_rtde=False)
        self.logger.experiment.log({f"{prefix}-rtd/RTDE-Violins": self.fig_to_wandb_image(fig)})
        plt.close(fig)

        fig, _ = plot_rtde_violins(rtd_target_km, rtde_relative, is_relative_rtde=True)
        self.logger.experiment.log({f"{prefix}-rtd/Relative-RTDE-Violins": self.fig_to_wandb_image(fig)})
        plt.close(fig)

    def _plot_rtd_scatter(self, 
        rtd_target: torch.Tensor,
        rtde: torch.Tensor,
        rtde_relative: torch.Tensor,
        prefix: str = "val",
    ):
        rtd_target_km, rtde_km = rtd_target / 1000.0, rtde / 1000.0

        rtd_target_km = rtd_target_km.detach().cpu().float().numpy()
        rtde_km = rtde_km.detach().cpu().float().numpy()
        rtde_relative = rtde_relative.detach().cpu().float().numpy()

        fig, _ = plot_rtd_scatter(rtd_target_km, rtde_km, rtde_relative)
        self.logger.experiment.log({
            f"{prefix}-rtd/RTD-Scatter": self.fig_to_wandb_image(fig)
        })
        plt.close(fig)

    # --------------------------------------
    # Utility
    # --------------------------------------

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(seq_len, dtype=torch.bool, device=device)

    def fig_to_wandb_image(self, fig: plt.Figure) -> wandb.Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        return wandb.Image(Image.open(buf))





