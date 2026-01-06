from datetime import datetime
import warnings
from typing import List
from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

warnings.filterwarnings("ignore", message=".*srun.*")
warnings.filterwarnings("ignore", message=".*Lightning model registry.*")


class Trainer(pl.Trainer):
    def __init__(self, trainer_cfg: DictConfig, callbacks_cfg: DictConfig, logger_cfg: DictConfig):
        super().__init__(**trainer_cfg)
        self.callbacks = self._setup_callbacks(callbacks_cfg)
        self.logger = self._setup_logger(logger_cfg)
        
    def _setup_callbacks(self, callbacks_cfg: DictConfig) -> List[pl.Callback]:
        callbacks = []

        if callbacks_cfg.get("early_stopping", None) is not None:
            early_stopping_cfg = callbacks_cfg["early_stopping"]
            early_stopping = EarlyStopping(
                monitor=early_stopping_cfg.get("monitor", "val_loss"),
                mode=early_stopping_cfg.get("mode", "min"),
                patience=early_stopping_cfg.get("patience", 10),
                min_delta=early_stopping_cfg.get("min_delta", 0.0)
            )
            callbacks.append(early_stopping)

        if callbacks_cfg.get("learning_rate_monitor", None) is not None:
            learning_rate_monitor_cfg = callbacks_cfg["learning_rate_monitor"]
            learning_rate_monitor = LearningRateMonitor(
                logging_interval=learning_rate_monitor_cfg.get("logging_interval", "epoch")
            )
            callbacks.append(learning_rate_monitor)

        if callbacks_cfg.get("checkpoint", None) is not None:
            checkpoint_cfg = callbacks_cfg["checkpoint"]
            dirpath = checkpoint_cfg.get("dirpath", ".outputs/checkpoints")
            dirpath = dirpath + f"/{datetime.now().strftime('%Y-%m-%d')}/{datetime.now().strftime('%H-%M-%S')}"
            checkpoint = ModelCheckpoint(
                dirpath=dirpath,
                filename=checkpoint_cfg.get("filename", "{epoch}-{val_loss:.2f}"),
                monitor=checkpoint_cfg.get("monitor", "val_loss"),
                mode=checkpoint_cfg.get("mode", "min"),
                save_top_k=checkpoint_cfg.get("save_top_k", 1),
                save_last=checkpoint_cfg.get("save_last", True)
            )
            callbacks.append(checkpoint)

        return callbacks

    def _setup_logger(self, logger_cfg: DictConfig) -> WandbLogger:
        return instantiate(logger_cfg)