# Configuration Guide

This guide explains how to use the nested Hydra configuration system.

## Configuration Structure

The framework uses Hydra's config groups to organize configurations into logical groups:

- `model/` - Model-specific configurations
- `dataset/` - Dataset-specific configurations  
- `trainer/` - Training settings (epochs, devices, precision, etc.)

Additionally, execution configs (e.g., `execute_aerion.yaml`, `execute_flightbert.yaml`) provide complete configurations for specific models by composing base configs and model-specific settings.

## Base Config

The `configs/base_config.yaml` file contains common settings shared across all models. You usually do not adjust this file.

```yaml
defaults:
  - trainer: default
  - _self_

# --------------------------------------
# General settings
# --------------------------------------
experiment_name: "unconfigured_experiment"
stage: "train"
seed: 42
load_checkpoint_path: null # Only used for prediction and evaluation

# ...
```

## Execution Configs

Execution configs provide complete configurations for specific models. They extend `base_config` and specify model, dataset, optimizer, and other settings:

```yaml
defaults:
  - base_config
  - model: aerion
  - dataset: single_day
  - optimizer: adam
  - _self_

# --------------------------------------
# Overrides
# --------------------------------------
trainer:
  max_epochs: 100

optimizer:
  _target_: torch.optim.Adam
  learning_rate: 1e-3
  weight_decay: 0.0

dataset:
  inputs_path: "path/to/inputs"
  horizons_path: "path/to/horizons"
  dataloader:
    batch_size: 32
```

You can use the `_target_` parameter to instantiate a specific class.

## Command Overrides
The following command overrids attributes of configuration files from the command line, without the need to add a new configuration file:
- Uses `execute_aerion` as the base execution config
- Uses `aerion` model from `configs/model/aerion.yaml`
- Uses `january` dataset from `configs/dataset/january.yaml`
- Uses `adamw` optimizer from `configs/optimizer/adamw.yaml`
- Overrides specific values for epochs, learning rate, batch size, and wandb settings
```bash
python src/train.py --config-name=execute_aerion \
    model=aerion \
    dataset=january \
    optimizer=adamw \
    trainer.max_epochs=100 \
    optimizer.learning_rate=0.0001 \
    dataset.dataloader.batch_size=1024 \
    wandb.project=my_project \
    wandb.name=transformer_experiment_001
```

## Creating New Configurations
Creating new configurations is useful, if you want to persistently store a given set of parameters.

### Adding a New Model Config

1. Create `configs/model/your_model.yaml`:

```yaml
# @package _global_
name: "your_model"
input_dim: 128
hidden_dim: 256
output_dim: 64
# ...
```

2. Use it in an execution config or override:

```bash
python src/train.py --config-name=base_config model=your_model
```

### Adding a New Dataset Config

1. Create `configs/dataset/your_dataset.yaml`:

```yaml
# @package _global_
inputs_path: "path/to/your/inputs"
horizons_path: "path/to/your/horizons"
dataloader:
  batch_size: 64
  num_workers: 8
```

2. Use it in an execution config or override:

```bash
python src/train.py --config-name=execute_aerion dataset=your_dataset
```

### Adding a Scheduler

Schedulers are configured at the top level of execution configs. If a scheduler config exists, it will be used:

```yaml
scheduler:
  name: "cosine"  # Options: cosine, step, reduce_on_plateau
  T_max: 100      # For cosine annealing (defaults to trainer.max_epochs if not specified)
```

If no scheduler config is present, training uses a constant learning rate.

## Configuration Precedence

Configuration values are merged in this precedence:

1. Command-line overrides
2. Values in execution config (after `_self_`)
3. Default configs from `defaults` list (model, dataset, optimizer, etc.)
4. Base config defaults

## Tips

1. **Use `@package _global_`** in config group files to merge them into the root config
2. **Use execution configs** for complete experiment setups - they compose base configs with model-specific settings
3. **Override at command line** for quick experiments without editing files
4. **Use config composition** to mix and match different components
5. **Use bash execution scripts** for experiment presets - copy and modify execution scripts for different configurations
6. **Scheduler configuration**: Add a top-level `scheduler` section to your execution config to enable learning rate scheduling. If omitted, no scheduler is used.

