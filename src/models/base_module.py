import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.checkpoint as checkpoint_util
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.features import FeatureSchema
from data.interface import RunwayData
from models.losses import CompositeApproachLoss
from models.metrics import TrajectoryMetrics, TrajectoryMetricsResult
from visualization import plot_predictions_targets, plot_rtd_error_line, plot_rtd_scatter, plot_rtde_violins


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        loss_cfg: DictConfig,
        input_seq_len: int,
        horizon_seq_len: int,
        feature_schema: FeatureSchema,
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
        self.feature_schema = feature_schema
        self.num_visualized_traj = num_visualized_traj

        self.loss = CompositeApproachLoss(loss_configs=loss_cfg)

        # Initialize metric accumulators (will be properly initialized in epoch start hooks)
        self.val_metrics = None
        self.train_metrics = None

    def on_fit_start(self):
        self.feature_schema.register_modules(self)

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
    # Metrics accumulation and logging
    # --------------------------------------

    def on_train_epoch_start(self):
        self.train_metrics = TrajectoryMetrics(self.horizon_seq_len, self.device)

    def on_validation_epoch_start(self):
        self.val_metrics = TrajectoryMetrics(self.horizon_seq_len, self.device)

    def on_train_epoch_end(self):
        result = self.train_metrics.compute()
        self._log_metrics(result, "train")

    def on_validation_epoch_end(self):
        result = self.val_metrics.compute()
        self._log_metrics(result, "val")

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

    def _log_metrics(self, result: TrajectoryMetricsResult, prefix: str):
        displacement = result.displacement
        position = result.position
        rtd = result.rtd
        horizon = result.horizon

        self.log_dict({
            f"{prefix}/ADE2D": displacement.ade_mean,
            f"{prefix}/FDE2D": displacement.fde_mean,
            f"{prefix}/MDE2D": displacement.mde_mean,
            f"{prefix}/RTD_ME": rtd.mean_error,
            f"{prefix}/RTD_MAE": rtd.mean_abs_error,
            f"{prefix}/RTD_MPE": rtd.mean_pct_error,
            f"{prefix}/RTD_MAPE": rtd.mean_abs_pct_error,
            f"{prefix}/RTD_ME_StdDev": rtd.std_error,
            f"{prefix}/RTD_MAE_StdDev": rtd.std_abs_error,
            f"{prefix}/RTD_MPE_StdDev": rtd.std_pct_error,
            f"{prefix}/RTD_MAPE_StdDev": rtd.std_abs_pct_error,
            f"{prefix}/Altitude_MAE": position.altitude_mae_mean,
        })

        self._horizon_line_plot(horizon.ade, "ADE2D", prefix)
        for feature in ["X", "Y", "Altitude"]:
            self._horizon_line_plot(horizon.mae, "MAE", prefix, feature)
            self._horizon_line_plot(horizon.rmse, "RMSE", prefix, feature)

        self._log_histogram(displacement.ade_trajectories, "ADE2D", prefix)
        self._log_histogram(displacement.fde_trajectories, "FDE2D", prefix)
        self._log_histogram(rtd.rtdpe_trajectories, "RTD PE", prefix, is_rtd=True)

        self._plot_rtde_violins(rtd.rtd_target_trajectories, rtd.rtde_trajectories, rtd.rtdpe_trajectories, prefix=prefix)
        self._plot_rtd_error_line(rtd.rtd_target_trajectories, rtd.rtde_trajectories, rtd.rtdpe_trajectories, prefix=prefix)
        self._plot_rtd_scatter(rtd.rtd_target_trajectories, rtd.rtd_pred_trajectories, rtd.rtdpe_trajectories, prefix=prefix)

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
        input_pos_abs: torch.Tensor,
        target_pos_abs: torch.Tensor,
        pred_pos_abs: torch.Tensor,
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
            input_pos_abs: Input absolute positions [batch_size, input_seq_len, 3]
            target_pos_abs: Target absolute positions [batch_size, horizon_seq_len, 3]
            pred_pos_abs: Predicted absolute positions [batch_size, horizon_seq_len, 3]
            target_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
            batch_idx: Batch index
            flight_id: Flight ID [batch_size]
            target_rtd: Target RTD in meters [batch_size]
            pred_rtd: Predicted RTD in meters [batch_size]
            prefix: Prefix for visualization names (e.g., "train", "val")
            num_trajectories: Number of trajectories to visualize (default: 6)
        """
        if batch_idx != 0:
            return

        num_plotted_trajectories = 0
        for i in range(input_pos_abs.shape[0]):
            if target_rtd[i] <= 80000: # skip trajectories that are close to the runway
                continue
            input_abs_i = input_pos_abs[i].detach().cpu().float().numpy()
            target_abs_i = target_pos_abs[i].detach().cpu().float().numpy()
            pred_abs_i = pred_pos_abs[i].detach().cpu().float().numpy()
            target_pad_mask_i = target_pad_mask[i].detach().cpu().numpy()
            target_rtd_i = target_rtd[i].detach().cpu().float().numpy()
            pred_rtd_i = pred_rtd[i].detach().cpu().float().numpy()
            flight_id_i = flight_id[i]

            fig, _ = plot_predictions_targets(input_abs_i, target_abs_i, pred_abs_i, target_pad_mask_i, "EDDB", flight_id_i, target_rtd_i, pred_rtd_i)
            self.logger.experiment.log({
                f"{prefix}-predictions-targets/batch_{batch_idx}_traj_{num_plotted_trajectories}": self.fig_to_wandb_image(fig)
            })
            plt.close(fig)

            if num_plotted_trajectories >= num_trajectories:
                break
            num_plotted_trajectories += 1

    def _plot_rtde_violins(
        self,
        rtd_target: torch.Tensor,
        rtde: torch.Tensor,
        rtdpe: torch.Tensor,
        prefix: str = "val",
    ):
        rtd_target_km = (rtd_target / 1000.0).detach().cpu().float().numpy()
        rtde_km = (rtde / 1000.0).detach().cpu().float().numpy()
        rtdpe = rtdpe.detach().cpu().float().numpy()

        fig, _ = plot_rtde_violins(rtd_target_km, rtde_km, is_relative_rtde=False)
        self.logger.experiment.log({f"{prefix}-rtd/RTDE-Violins": self.fig_to_wandb_image(fig)})
        plt.close(fig)

        fig, _ = plot_rtde_violins(rtd_target_km, rtdpe, is_relative_rtde=True)
        self.logger.experiment.log({f"{prefix}-rtd/RTD-PE-Violins": self.fig_to_wandb_image(fig)})
        plt.close(fig)

    def _plot_rtd_error_line(
        self,
        rtd_target: torch.Tensor,
        rtde: torch.Tensor,
        rtdpe: torch.Tensor,
        prefix: str = "val",
    ):
        rtd_target_km = (rtd_target / 1000.0).detach().cpu().float().numpy()
        rtde_km = (rtde / 1000.0).detach().cpu().float().numpy()
        rtdpe = rtdpe.detach().cpu().float().numpy()

        fig, _ = plot_rtd_error_line(rtd_target_km, rtde_km, rtdpe)
        self.logger.experiment.log({f"{prefix}-rtd/RTD-MAE-MAPE-Line": self.fig_to_wandb_image(fig)})
        plt.close(fig)

    def _plot_rtd_scatter(self,
        rtd_target: torch.Tensor,
        rtde: torch.Tensor,
        rtdpe: torch.Tensor,
        prefix: str = "val",
    ):
        rtd_target_km, rtde_km = rtd_target / 1000.0, rtde / 1000.0

        rtd_target_km = rtd_target_km.detach().cpu().float().numpy()
        rtde_km = rtde_km.detach().cpu().float().numpy()
        rtdpe = rtdpe.detach().cpu().float().numpy()

        fig, _ = plot_rtd_scatter(rtd_target_km, rtde_km, rtdpe)
        self.logger.experiment.log({
            f"{prefix}-rtd/RTD-Scatter": self.fig_to_wandb_image(fig)
        })
        plt.close(fig)

    # --------------------------------------
    # Utility
    # --------------------------------------

    def _generate_causal_mask(self, seq_len: int, device: torch.device, num_agents: int = 1) -> torch.Tensor:
        """Generate a causal mask. For multi-agent (num_agents > 1), generates a
        block-causal mask [T*N, T*N] where all agents at the same or earlier
        time steps are visible."""
        if num_agents <= 1:
            return nn.Transformer.generate_square_subsequent_mask(seq_len, dtype=torch.bool, device=device)
        total = seq_len * num_agents
        time_idx = torch.arange(total, device=device) // num_agents
        return time_idx.unsqueeze(0) > time_idx.unsqueeze(1)

    # --------------------------------------
    # Autoregressive prediction
    # --------------------------------------

    def predict_autoregressively(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        runway: RunwayData,
        initial_position_abs: torch.Tensor,
        num_steps: Optional[int] = None,
        memory: Optional[torch.Tensor] = None,
        continue_decoding: bool = False,
        agent_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict autoregressively. Works for both single-agent and multi-agent tensors.

        Args:
            input_traj: Normalized input trajectory [B, T_in, F] or [B, T_in, N, F].
            dec_in_traj: Normalized decoder input [B, H, F] or [B, H, N, F].
            runway: Batched RunwayData.
            initial_position_abs: Starting position [B, 3] or [B, N, 3].
            num_steps: Number of steps to predict (default: horizon_seq_len).
            memory: Pre-computed encoder memory (optional, will encode if None).
            continue_decoding: If True, use full dec_in_traj as initial decoder input.
            agent_padding_mask: Optional [B, N] — True = padded agent slot.

        Returns:
            Tuple of (predicted deltas [B, num_steps, F] or [B, num_steps, N, F], memory).
        """
        num_steps = num_steps or self.horizon_seq_len
        has_agent_dim = input_traj.ndim == 4
        N = input_traj.size(2) if has_agent_dim else 1

        enc_padding = None
        if agent_padding_mask is not None:
            enc_padding = agent_padding_mask.unsqueeze(1)  # [B, 1, N]

        if memory is None:
            memory = self.model.encode(input_traj, target_padding_mask=enc_padding)

        current_position_abs = initial_position_abs.clone()
        current_dec_in = dec_in_traj if continue_decoding else dec_in_traj[:, 0:1]

        all_predictions_norm = []
        for i in range(num_steps):
            current_seq_len = current_dec_in.size(1)
            target_mask = self._generate_causal_mask(current_seq_len, input_traj.device, num_agents=N)

            dec_padding = None # Default is no padding mask during AR to avoid sequence length leakage
            if agent_padding_mask is not None: # only padded agents are masked, but not the sequence length
                dec_padding = agent_padding_mask.unsqueeze(1).expand(-1, current_seq_len, -1)

            if self.training:
                output = checkpoint_util.checkpoint(
                    lambda dec_in, mem, mask, pad_mask: self.model.decode(
                        dec_in, mem, causal_mask=mask, target_padding_mask=pad_mask
                    ),
                    current_dec_in,
                    memory,
                    target_mask,
                    dec_padding,
                    use_reentrant=False,
                )
            else:
                output = self.model.decode(
                    current_dec_in,
                    memory,
                    causal_mask=target_mask,
                    target_padding_mask=dec_padding,
                )

            pred_deltas_norm = output[:, -1:]
            all_predictions_norm.append(pred_deltas_norm)

            next_dec_in, current_position_abs = self.feature_schema.build_next_decoder_input(
                pred_deltas_norm, current_position_abs, runway
            )
            current_dec_in = torch.cat([current_dec_in, next_dec_in], dim=1)

        pred_deltas_norm = torch.cat(all_predictions_norm, dim=1)
        return pred_deltas_norm, memory

    def fig_to_wandb_image(self, fig: plt.Figure) -> wandb.Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        return wandb.Image(Image.open(buf))





