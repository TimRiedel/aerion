import logging
from omegaconf import DictConfig, OmegaConf

from utils.trainer import Trainer

logger = logging.getLogger(__name__)


def log_important_parameters(cfg: DictConfig, input_seq_len: int, horizon_seq_len: int) -> None:
    num_trajectories_to_predict = cfg.get("debug", {}).get("num_trajectories_to_predict", None)

    formatted = f"""\
    ----------------------------------------
    Parameters for {cfg.experiment_name}:
    ----------------------------------------
    Model name:                 {cfg.model.name}
    Batch size:                 {cfg.dataloader.batch_size}
    Max Epochs:                 {cfg.trainer.max_epochs}
    Learning rate:              {cfg.optimizer.lr}
    Weight Decay:               {cfg.optimizer.weight_decay}
    Dropout:                    {cfg.model.params.dropout}

    Dataset:                    {cfg.dataset.name}
    Input Length:               {input_seq_len}
    Horizon Length:             {horizon_seq_len}
    Num traject. to predict:    {num_trajectories_to_predict}
    """
    logger.info("\n%s", formatted)

def log_hydra_config_to_wandb(cfg: DictConfig, trainer: Trainer) -> None:
    trainer.logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True),
        allow_val_change=True
    )

def add_wandb_tags(cfg: DictConfig) -> None:
    cfg["wandb"]["tags"].append(cfg["model"]["name"])
    cfg["wandb"]["tags"].append(cfg["dataset"]["name"])
    return cfg

def calculate_seq_len(time_minutes: int, resampling_rate_seconds: int) -> int:
    return time_minutes * 60 // resampling_rate_seconds