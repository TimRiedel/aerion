# Configuration Guide

This guide explains how to use the nested Hydra configuration system.

## Configuration Structure

The framework uses Hydra's config groups to organize configurations into logical groups:

- `model/` - Model-specific configurations
- `dataset/` - Dataset-specific configurations  
- `trainer/` - Training settings (epochs, devices, precision, etc.)
- `optimizer/` - Optimizer configurations

Additionally, execution configs (e.g., `execute_aerion.yaml`, `execute_flightbert.yaml`) provide complete configurations for specific models by composing base configs and model-specific settings.

## Base Config

The `configs/base_config.yaml` file contains common settings shared across all models:

```yaml
defaults:
  - trainer: default
  - _self_

seed: 42
experiment_name: "unconfigured_experiment"

# Checkpoint, early stopping, wandb, etc.
```

## Execution Configs

Execution configs provide complete configurations for specific models. They extend `base_config` and specify model, dataset, optimizer, and other settings:

```yaml
defaults:
  - base_config
  - model: aerion
  - dataset: base_dataset
  - optimizer: adam
  - _self_

optimizer:
  name: "adam"
  learning_rate: 1e-3

scheduler:
  name: "cosine"
  T_max: 100

trainer:
  max_epochs: 100

dataset:
  inputs_path: "path/to/inputs"
  horizons_path: "path/to/horizons"
```

## Creating New Configurations

### Adding a New Model Config

1. Create `configs/model/your_model.yaml`:

```yaml
# @package _global_
name: "your_model"
input_dim: 128
hidden_dim: 256
output_dim: 64
```

2. Use it in an execution config or override:

```bash
python src/train.py --config-name=execute_aerion model=your_model
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

### Adding a New Optimizer Config

1. Create `configs/optimizer/your_optimizer.yaml`:

```yaml
# @package _global_
name: "adamw"
learning_rate: 0.001
weight_decay: 0.01
```

2. Use it in an execution config or override:

```bash
python src/train.py --config-name=execute_aerion optimizer=your_optimizer
```

### Adding a Scheduler

Schedulers are configured at the top level of execution configs. If a scheduler config exists, it will be used:

```yaml
scheduler:
  name: "cosine"  # Options: cosine, step, reduce_on_plateau
  T_max: 100      # For cosine annealing (defaults to trainer.max_epochs if not specified)
```

For step scheduler:
```yaml
scheduler:
  name: "step"
  step_size: 20
  gamma: 0.7
```

For reduce on plateau:
```yaml
scheduler:
  name: "reduce_on_plateau"
  mode: "min"
  factor: 0.5
  patience: 5
  monitor: "val_loss"
```

If no scheduler config is present, training uses a constant learning rate.

## Overriding Configuration Values

You can override any configuration value from the command line:

```bash
# Override nested values using dot notation
python src/train.py --config-name=execute_aerion optimizer.learning_rate=0.0001

# Override multiple values
python src/train.py --config-name=execute_aerion \
    trainer.max_epochs=50 \
    dataset.dataloader.batch_size=64

# Combine config selection and overrides
python src/train.py --config-name=execute_aerion \
    model=your_model \
    optimizer.learning_rate=0.001
```

## Configuration Precedence

Configuration values are merged in this precedence:

1. Command-line overrides
2. Values in execution config (after `_self_`)
3. Default configs from `defaults` list (model, dataset, optimizer, etc.)
4. Base config defaults

## Accessing Config in Code

In your Python code, access configurations like this:

```python
@hydra.main(version_base=None, config_path="../configs", config_name="execute_your_model")
def train(cfg: DictConfig):
    # Access model config
    model_name = cfg.model.name
    model_hidden_dim = cfg.model.hidden_dim
    
    # Access dataset config
    inputs_path = cfg.dataset.inputs_path
    horizons_path = cfg.dataset.horizons_path
    batch_size = cfg.dataset.dataloader.batch_size
    
    # Access trainer config
    max_epochs = cfg.trainer.max_epochs
    
    # Access optimizer config
    learning_rate = cfg.optimizer.learning_rate
    
    # Access scheduler config (may be None)
    scheduler_config = cfg.get('scheduler')
    if scheduler_config:
        scheduler_name = scheduler_config.name
```

## Tips

1. **Use `@package _global_`** in config group files to merge them into the root config
2. **Use execution configs** for complete experiment setups - they compose base configs with model-specific settings
3. **Override at command line** for quick experiments without editing files
4. **Use config composition** to mix and match different components
5. **Use bash execution scripts** for experiment presets - copy and modify execution scripts for different configurations
6. **Scheduler configuration**: Add a top-level `scheduler` section to your execution config to enable learning rate scheduling. If omitted, no scheduler is used.

## Example: Complete Training Command

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

This command:
- Uses `execute_aerion` as the base execution config
- Uses `aerion` model from `configs/model/aerion.yaml`
- Uses `january` dataset from `configs/dataset/january.yaml`
- Uses `adamw` optimizer from `configs/optimizer/adamw.yaml`
- Overrides specific values for epochs, learning rate, batch size, and wandb settings
