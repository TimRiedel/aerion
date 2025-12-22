# Aerion
AERION - Approach Estimation and Representation of Intent for Operational Navigation is a deep learning trajectory prediction model that supports pilot descent planning and Continuous Descent Operations (CDO) in terminal airspace.

The model is currently under development as part of the master's thesis "Learning Multi-Modal Representations for Aircraft Approach Trajectory Prediction" at the Hasso Plattner Institut (HPI) in Potsdam, in collaboration with the Institute of Flight Guidance at the German Aerospace Center (DLR) in Braunschweig.

## Features

- **Configuration Management**: Hydra + OmegaConf for flexible experiment configuration. See CONFIG_GUIDE.md.
- **Deep Learning**: PyTorch + PyTorch Lightning for model training
- **Experiment Tracking**: Weights and Biases (wandb) for logging metrics, GPU/RAM usage, and visualizations
- **Distributed Training**: Multi-GPU support on single nodes if `trainer.accelerator: gpu` and `trainer.devices: auto`
- **Standardized Evaluation**: Consistent evaluation pipeline across models

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd aerion
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yaml
conda activate aerion
```
