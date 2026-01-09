import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Dict, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate

from data.transforms.normalize import ZScoreDenormalize


class TransformerModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        input_seq_len: int,
        horizon_seq_len: int,
        num_waypoints_to_predict: int = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule"])
        self.model = instantiate(model_cfg["params"])
        self.optimizer_cfg = optimizer_cfg
        self.input_seq_len = input_seq_len
        self.horizon_seq_len = horizon_seq_len
        self.num_waypoints_to_predict = num_waypoints_to_predict if num_waypoints_to_predict is not None else horizon_seq_len

        # Evaluation horizons
        predefined_horizons = [1, 10, 20, 40, 60, 80, 100, 120]
        self.evaluation_horizons = [h for h in predefined_horizons if h <= self.num_waypoints_to_predict]
        
        # Loss function
        self.criterion = nn.MSELoss(reduction="none")
        

    def on_fit_start(self):
        dm = self.trainer.datamodule

        self.register_buffer("norm_mean", dm.norm_mean)
        self.register_buffer("norm_std", dm.norm_std)

        self.denormalize = ZScoreDenormalize(dm.norm_mean, dm.norm_std)


    def on_validation_epoch_start(self):
        H = len(self.evaluation_horizons) # Number horizons
        F = self.model.input_dim          # Number features

        device = self.device

        self.val_sum_abs_error = torch.zeros(H, F, device=device)
        self.val_sum_sq_error = torch.zeros(H, F, device=device)
        self.val_sum_distance_error = torch.zeros(H, device=device)
        self.val_count = torch.zeros(H, device=device)


    def on_validation_epoch_end(self):
        for tensor in [  # Needed for DDP
            self.val_sum_abs_error,
            self.val_sum_sq_error,
            self.val_sum_distance_error,
            self.val_count,
        ]:
            self.trainer.strategy.reduce(tensor, reduce_op="sum")

        # count.shape: [num horizons, 1]
        count = self.val_count.clamp(min=1.0).unsqueeze(1)

        # mae.shape = rmse.shape = mde.shape: [num horizons, num features]
        mae = self.val_sum_abs_error / count
        rmse = torch.sqrt(self.val_sum_sq_error / count)

        # Reshape from (H,) to (H, 1) to match log_table_metric expectations
        # mde.shape: [num horizons, 1]
        val_sum_distance_error = self.val_sum_distance_error.unsqueeze(1)
        mde = (val_sum_distance_error / count)

        for feature in ["E", "N", "Altitude", "Speed_E", "Speed_N", "Vertical_Rate"]:
            self._log_error_vs_horizon(mae, "MAE", feature)
            self._log_error_vs_horizon(rmse, "RMSE", feature)

        self._log_error_vs_horizon(mde, "MDE")


    def training_step(self, batch, batch_idx):
        x, y, dec_in, tgt_pad_mask = batch["x"], batch["y"], batch["dec_in"], batch["mask"]
        x, y, dec_in, tgt_pad_mask = self._limit_data_to_effective_horizon(x, y, dec_in, tgt_pad_mask)

        causal_mask = self.model.transformer.generate_square_subsequent_mask(self.num_waypoints_to_predict, dtype=torch.bool).to(x.device)
        outputs = self.model(
            src=x,
            tgt=dec_in,
            tgt_mask=causal_mask,
            tgt_pad_mask=tgt_pad_mask 
        )

        loss = self._compute_loss(outputs, y, tgt_pad_mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(x))
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        x, y, dec_in, tgt_pad_mask = batch["x"], batch["y"], batch["dec_in"], batch["mask"]
        x, y, dec_in, tgt_pad_mask = self._limit_data_to_effective_horizon(x, y, dec_in, tgt_pad_mask)

        causal_mask = self.model.transformer.generate_square_subsequent_mask(self.num_waypoints_to_predict, dtype=torch.bool).to(x.device)
        outputs = self.model(
            src=x,
            tgt=dec_in,
            tgt_mask=causal_mask,
            tgt_pad_mask=tgt_pad_mask 
        )

        loss = self._compute_loss(outputs, y, tgt_pad_mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(x))

        denorm_predictions = self.denormalize(outputs, batched=True)
        denorm_targets = self.denormalize(y, batched=True)

        for h_idx, h in enumerate(self.evaluation_horizons):
            t = h - 1  # horizon index

            # Valid trajectories at time/horizon t (only these trajectories which are not padded at t)
            valid_traj_at_t = ~tgt_pad_mask[:, t] 
            if valid_traj_at_t.sum() == 0:
                continue

            # p.shape = gt.shape: [num valid trajectories at t, num features]
            p = denorm_predictions[valid_traj_at_t, t]
            gt = denorm_targets[valid_traj_at_t, t]

            # abs_err.shape = sq_err.shape: [num valid trajectories at t, num features]
            abs_err = torch.abs(p - gt)
            sq_err = (p - gt) ** 2

            # Feature-wise
            # val_sum_abs_error.shape = val_sum_sq_error.shape: [num horizons, num features]
            self.val_sum_abs_error[h_idx] += abs_err.sum(dim=0)
            self.val_sum_sq_error[h_idx] += sq_err.sum(dim=0)

            # Distance error (EN + altitude in first 3 dims)
            # val_sum_distance_error.shape: [num horizons]
            dist_err = torch.norm(p[:, :3] - gt[:, :3], dim=1)
            self.val_sum_distance_error[h_idx] += dist_err.sum()

            self.val_count[h_idx] += valid_traj_at_t.sum()
        
        return loss
    

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Test step not implemented")


    def _limit_data_to_effective_horizon(self, x: torch.Tensor, y: torch.Tensor, dec_in: torch.Tensor, tgt_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        effective_horizon = self.num_waypoints_to_predict if self.num_waypoints_to_predict is not None else self.horizon_seq_len
        y = y[:, :effective_horizon]
        dec_in = dec_in[:, :effective_horizon]
        tgt_pad_mask = tgt_pad_mask[:, :effective_horizon]
        return x, y, dec_in, tgt_pad_mask
    

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        tgt_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.
        
        Args:
            predictions: Model predictions [batch_size, horizon_seq_len, num_features]
            targets: Target values [batch_size, horizon_seq_len, num_features]
            tgt_pad_mask: Padding mask [batch_size, horizon_seq_len] (True for padded positions)
            
        Returns:
            Scalar loss value
        """
        active_loss = ~tgt_pad_mask 
        
        loss = self.criterion(predictions[active_loss], targets[active_loss]).mean()
        return loss


    def _log_error_vs_horizon(self, metric: torch.Tensor, metric_name: str, feature_name: str = None):
        if feature_name is not None:
            features = ["E", "N", "Altitude", "Speed_E", "Speed_N", "Vertical_Rate"]
            feat_idx = features.index(feature_name)
            metric = metric[:, feat_idx].detach().cpu().tolist()
            name = f"{metric_name} {feature_name}"
        else:
            feature_name = "Distance"
            name = metric_name

        table = wandb.Table(columns=[name, "Horizon"])
        for h_idx, h in enumerate(self.evaluation_horizons):
            v = metric[h_idx]
            table.add_data(v, h) # categorical x-axis

        fields = {"value": name, "label": "Horizon"}
        custom_chart = wandb.plot_table(
            vega_spec_name="timriedel/error_by_horizon",
            data_table=table,
            fields=fields,
        )

        self.logger.experiment.log({
            f"val-{metric_name.lower()}/{name}": custom_chart
        })
    

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.model.parameters())
        
        scheduler_cfg = self.optimizer_cfg.get('scheduler', None)
        if scheduler_cfg is None:
            return optimizer
        else:
            scheduler = instantiate(scheduler_cfg, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }