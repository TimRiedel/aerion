import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from trainer import Trainer
from data import *

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="execute_aerion")
def main(cfg: DictConfig) -> None:
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.get('seed', None) is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    trainer = Trainer(cfg["trainer"], cfg["callbacks"], cfg["wandb"])
    trainer.logger.config = cfg

    data = AerionData(cfg["dataset"], cfg["data_processing"], cfg["dataloader"], cfg.seed)

if __name__ == "__main__":
    main()