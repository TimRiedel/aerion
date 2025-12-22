import hydra
from omegaconf import DictConfig
import torch

from trainer import Trainer


@hydra.main(version_base=None, config_path="../configs", config_name="execute_aerion")
def main(cfg: DictConfig) -> None:
    if cfg.get('seed', None) is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    trainer = Trainer(cfg["trainer"], cfg["callbacks"], cfg["wandb"])
    trainer.logger.config = cfg

if __name__ == "__main__":
    main()