import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from trainer import Trainer
from data import *
from models import *

# Enable Tensor Core optimization for better performance on CUDA devices with Tensor Cores
torch.set_float32_matmul_precision('high')
logger = logging.getLogger(__name__)

def train(cfg: DictConfig, input_seq_len: int, horizon_seq_len: int) -> None:
    trainer = Trainer(cfg["trainer"], cfg["callbacks"], cfg["wandb"])
    trainer.logger.config = cfg

    data = AerionData(cfg["dataset"], cfg["data_processing"], cfg["dataloader"], cfg.seed)

    num_waypoints_to_predict = cfg.get("debug", {}).get("num_waypoints_to_predict", None)
    model = TransformerModule(
        cfg["model"],
        cfg["optimizer"],
        input_seq_len,
        horizon_seq_len,
        num_waypoints_to_predict=num_waypoints_to_predict,
    )
    trainer.fit(model, data)


def test(cfg: DictConfig, input_seq_len: int, horizon_seq_len: int) -> None:
    # TODO: When loading model from checkpoint, obtain norm_mean and norm_std from the checkpoint
    # and pass it to the datamodule for correct normalization data loading transforms
    raise NotImplementedError("Test stage not implemented. Loading model checkpoints is not supported yet.")


def calculate_seq_len(time_minutes: int, resampling_rate_seconds: int) -> int:
    return time_minutes * 60 // resampling_rate_seconds


@hydra.main(version_base=None, config_path="../configs", config_name="execute_aerion")
def main(cfg: DictConfig) -> None:
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.get('seed', None) is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    input_seq_len = calculate_seq_len(cfg["dataset"]["input_time_minutes"], cfg["dataset"]["resampling_rate_seconds"])
    horizon_seq_len = calculate_seq_len(cfg["dataset"]["horizon_time_minutes"], cfg["dataset"]["resampling_rate_seconds"])

    if cfg.stage == "train" or cfg.stage == "fit":
        train(cfg, input_seq_len, horizon_seq_len)
    elif cfg.stage == "test":
        test(cfg, input_seq_len, horizon_seq_len)


if __name__ == "__main__":
    main()