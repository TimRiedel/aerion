import warnings
from typing import List
from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
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
        for callback_cfg in callbacks_cfg.values():
            callbacks.append(instantiate(callback_cfg))
        return callbacks

    def _setup_logger(self, logger_cfg: DictConfig) -> WandbLogger:
        return instantiate(logger_cfg)