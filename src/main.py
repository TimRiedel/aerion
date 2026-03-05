import logging
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from data import ApproachData, FeatureSchema, TrafficData
from models import SingleAgentModule, TrafficModule
from utils import *


# Enable Tensor Core optimization for better performance on CUDA devices with Tensor Cores
torch.set_float32_matmul_precision('high')
os.environ.setdefault("WANDB_DIR", ".wandb")
logger = logging.getLogger(__name__)


def train(cfg: DictConfig, input_seq_len: int, horizon_seq_len: int) -> None:
    trainer = Trainer(cfg["trainer"], cfg["callbacks"], cfg["wandb"])
    log_hydra_config_to_wandb(cfg, trainer)

    execution_cfg = cfg.get("execution", {})
    num_trajectories_to_predict = execution_cfg.get("num_trajectories_to_predict", None)
    num_visualized_traj = execution_cfg.get("num_visualized_traj", 10)
    multi_agent_prediction = execution_cfg.get("multi_agent_prediction", False)
    dataset_max_num_agents = execution_cfg.get("max_num_agents", None)

    # Convert OmegaConf to regular dict to allow modifications
    model_cfg = OmegaConf.to_container(cfg["model"], resolve=True)
    optimizer_cfg = OmegaConf.to_container(cfg["optimizer"], resolve=True)
    loss_cfg = OmegaConf.to_container(cfg["loss"], resolve=True)

    feature_schema = FeatureSchema(cfg["features"])
    model_cfg["params"]["encoder_input_dim"] = feature_schema.encoder_dim
    model_cfg["params"]["decoder_input_dim"] = feature_schema.decoder_dim
    model_cfg["params"]["output_dim"] = feature_schema.output_dim

    if dataset_max_num_agents is None:
        max_num_agents = calculate_max_num_agents(cfg["dataset"]["scenes_path"])
    else:
        max_num_agents = dataset_max_num_agents

    if multi_agent_prediction:
        model_cfg["params"]["max_num_agents"] = max_num_agents

        data = TrafficData(
            cfg["dataset"],
            cfg["data_processing"],
            cfg["dataloader"],
            cfg["seed"],
            feature_schema=feature_schema,
            max_num_agents=max_num_agents,
        )
        model = TrafficModule(
            model_cfg,
            optimizer_cfg,
            loss_cfg,
            input_seq_len,
            horizon_seq_len,
            feature_schema=feature_schema,
            scheduler_cfg=cfg.get("scheduler", None),
            num_visualized_traj=num_visualized_traj,
        )
    else:
        data = ApproachData(
            cfg["dataset"],
            cfg["data_processing"],
            cfg["dataloader"],
            cfg["seed"],
            feature_schema=feature_schema,
            max_num_agents=max_num_agents,
            num_trajectories_to_predict=num_trajectories_to_predict,
            num_waypoints_to_predict=horizon_seq_len,
        )
        model = SingleAgentModule(
            model_cfg,
            optimizer_cfg,
            loss_cfg,
            input_seq_len,
            horizon_seq_len,
            feature_schema=feature_schema,
            scheduler_cfg=cfg.get("scheduler", None),
            num_visualized_traj=num_visualized_traj,
        )
    
    log_important_parameters(cfg, input_seq_len, horizon_seq_len, max_num_agents)
    trainer.fit(model, data)


def test(cfg: DictConfig, input_seq_len: int, horizon_seq_len: int) -> None:
    raise NotImplementedError("Test stage not implemented. Loading model checkpoints is not supported yet.")


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