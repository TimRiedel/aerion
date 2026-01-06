from hydra.utils import instantiate
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf


class TransformerModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        input_seq_len: int,
        horizon_seq_len: int,
        norm_mean: Optional[torch.Tensor] = None,
        norm_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule"])
        self.model = instantiate(model_cfg["params"])
        nn.Transformer()
        self.optimizer_cfg = optimizer_cfg
        self.input_seq_len = input_seq_len
        self.horizon_seq_len = horizon_seq_len

        # self.example_input_array = torch.Tensor(1, input_seq_len, 6) # Debug only
        
        # Loss function
        self.criterion = nn.MSELoss(reduction="none")
        
        # Register normalization buffers
        if norm_mean is not None and norm_std is not None:
            self.register_buffer("norm_mean", norm_mean)
            self.register_buffer("norm_std", norm_std)
        
    
    def training_step(self, batch, batch_idx):
        x, y, dec_in, tgt_pad_mask = batch["x"], batch["y"], batch["dec_in"], batch["mask"]

        causal_mask = self.model.transformer.generate_square_subsequent_mask(self.horizon_seq_len, dtype=torch.bool).to(x.device)
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

        causal_mask = self.model.transformer.generate_square_subsequent_mask(self.horizon_seq_len, dtype=torch.bool).to(x.device)
        outputs = self.model(
            src=x,
            tgt=dec_in,
            tgt_mask=causal_mask,
            tgt_pad_mask=tgt_pad_mask 
        )

        loss = self._compute_loss(outputs, y, tgt_pad_mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(x))
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Test step not implemented")
    
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