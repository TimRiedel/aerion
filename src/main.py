import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from trainer import Trainer
from data import *
from models import *


# Enable Tensor Core optimization for better performance on CUDA devices with Tensor Cores
torch.set_float32_matmul_precision('high')
os.environ.setdefault("WANDB_DIR", ".wandb")
logger = logging.getLogger(__name__)


def train(cfg: DictConfig, input_seq_len: int, horizon_seq_len: int) -> None:
    trainer = Trainer(cfg["trainer"], cfg["callbacks"], cfg["wandb"])
    log_hydra_config_to_wandb(cfg, trainer)

    num_trajectories_to_predict = cfg.get("execution", {}).get("num_trajectories_to_predict", None)
    num_visualized_traj = cfg.get("execution", {}).get("num_visualized_traj", 10)
    contexts_cfg = cfg.get("contexts", {})

    data = AerionData(
        cfg["dataset"],
        cfg["data_processing"],
        cfg["dataloader"],
        cfg.seed,
        num_trajectories_to_predict=num_trajectories_to_predict,
        num_waypoints_to_predict=horizon_seq_len,
        contexts_cfg=contexts_cfg,
    )
    
    # Instantiate correct module based on model name
    model_cfg = OmegaConf.to_container(cfg["model"], resolve=True) # Convert to regular dict to allow modifications
    optimizer_cfg = OmegaConf.to_container(cfg["optimizer"], resolve=True)

    if cfg["model"]["name"] == "aerion":
        model = AerionModule(
            model_cfg,
            optimizer_cfg,
            input_seq_len,
            horizon_seq_len,
            contexts_cfg=contexts_cfg,
            scheduler_cfg=cfg.get("scheduler", None),
            loss_cfg=cfg.get("loss", None),
            num_visualized_traj=num_visualized_traj,
        )
    else:
        model = TransformerModule(
            model_cfg,
            optimizer_cfg,
            input_seq_len,
            horizon_seq_len,
            scheduler_cfg=cfg.get("scheduler", None),
            loss_cfg=cfg.get("loss", None),
            num_visualized_traj=num_visualized_traj,
        )
    
    log_important_parameters(cfg, trainer, input_seq_len, horizon_seq_len)
    trainer.fit(model, data)


def test(cfg: DictConfig, input_seq_len: int, horizon_seq_len: int) -> None:
    # TODO: When loading model from checkpoint, obtain norm_mean and norm_std from the checkpoint
    # and pass it to the datamodule for correct normalization data loading transforms
    raise NotImplementedError("Test stage not implemented. Loading model checkpoints is not supported yet.")


def calculate_seq_len(time_minutes: int, resampling_rate_seconds: int) -> int:
    return time_minutes * 60 // resampling_rate_seconds

def log_hydra_config_to_wandb(cfg: DictConfig, trainer: Trainer) -> None:
    trainer.logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True),
        allow_val_change=True
    )

def log_important_parameters(cfg: DictConfig, trainer: Trainer, input_seq_len: int, horizon_seq_len: int) -> None:
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

def add_wandb_tags(cfg: DictConfig) -> None:
    # Add model and dataset tags as default to wandb tags
    cfg["wandb"]["tags"].append(cfg["model"]["name"])
    cfg["wandb"]["tags"].append(cfg["dataset"]["name"])
    
    # Add tags for all enabled contexts
    contexts_cfg = cfg.get("contexts", {})
    for context_name, context_config in contexts_cfg.items():
        if context_config.get("enabled", False):
            cfg["wandb"]["tags"].append(context_name)
    return cfg


@hydra.main(version_base=None, config_path="../configs", config_name="execute_aerion")
def main(cfg: DictConfig) -> None:
    if cfg.get('seed', None) is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    input_seq_len = calculate_seq_len(cfg["dataset"]["input_time_minutes"], cfg["dataset"]["resampling_rate_seconds"])
    horizon_seq_len = calculate_seq_len(cfg["dataset"]["horizon_time_minutes"], cfg["dataset"]["resampling_rate_seconds"])
    
    # Apply num_waypoints_to_predict limit if specified
    num_waypoints_to_predict = cfg.get("execution", {}).get("num_waypoints_to_predict", None)
    if num_waypoints_to_predict is not None:
        horizon_seq_len = min(horizon_seq_len, num_waypoints_to_predict)

    cfg = add_wandb_tags(cfg)

    if cfg.stage == "train" or cfg.stage == "fit":
        train(cfg, input_seq_len, horizon_seq_len)
    elif cfg.stage == "test":
        test(cfg, input_seq_len, horizon_seq_len)


if __name__ == "__main__":
    main()