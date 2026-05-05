# Hydra Configuration Guide

This project uses [Hydra](https://hydra.cc/) for configuration management, replacing the previous argparse-based approach.

## Configuration Structure

The configuration is organized in the `configs/` directory:

```
configs/
├── train.yaml           # Main training configuration
├── data/
│   └── default.yaml     # Data-related settings
├── model/
│   └── default.yaml     # Model architecture settings
├── diffusion/
│   └── default.yaml     # Diffusion process settings
└── optimizer/
    ├── adam.yaml        # Adam optimizer settings
    └── sgd.yaml         # SGD optimizer settings
```

## Running Training

### Basic Usage

Run training with default configuration:
```bash
uv run python src/uaag2/train.py
```

### Override Configuration

Override specific parameters from the command line:
```bash
# Change learning rate and batch size
uv run python src/uaag2/train.py optimizer.lr=0.001 data.batch_size=64

# Change number of epochs and GPUs
uv run python src/uaag2/train.py num_epochs=100 gpus=2

# Use SGD optimizer instead of Adam
uv run python src/uaag2/train.py optimizer=sgd

# Change multiple model parameters
uv run python src/uaag2/train.py model.num_layers=10 model.sdim=512
```

### Load from Checkpoint

```bash
uv run python src/uaag2/train.py load_ckpt=/path/to/checkpoint.ckpt
```

### Multi-run Experiments

Run experiments with different hyperparameters:
```bash
# Sweep over learning rates
uv run python src/uaag2/train.py -m optimizer.lr=0.0001,0.0005,0.001

# Sweep over multiple parameters
uv run python src/uaag2/train.py -m \
    optimizer.lr=0.0001,0.001 \
    data.batch_size=32,64 \
    model.num_layers=5,7,10
```

## Configuration Groups

### Optimizer

Switch between optimizers:
```bash
# Use Adam (default)
uv run python src/uaag2/train.py optimizer=adam

# Use SGD
uv run python src/uaag2/train.py optimizer=sgd
```

### Logger

Switch between logging backends:
```bash
# Use Weights & Biases (default)
uv run python src/uaag2/train.py logger_type=wandb

# Use TensorBoard
uv run python src/uaag2/train.py logger_type=tensorboard
```

## Creating Custom Configurations

### Create a New Experiment Config

Create `configs/experiment/my_experiment.yaml`:
```yaml
# @package _global_

# Override default values
defaults:
  - override /optimizer: sgd

num_epochs: 500
seed: 123

optimizer:
  lr: 0.001

model:
  num_layers: 10
  sdim: 512

data:
  batch_size: 64
```

Run it:
```bash
uv run python src/uaag2/train.py +experiment=my_experiment
```

### Create Custom Model Configuration

Create `configs/model/large.yaml`:
```yaml
# Large model variant
sdim: 512
vdim: 128
edim: 64
num_layers: 12
```

Use it:
```bash
uv run python src/uaag2/train.py model=large
```

## Output Directory

Hydra automatically manages output directories. By default, outputs are saved to:
- `models/run<timestamp>/` - Training outputs and checkpoints
- `outputs/<date>/<time>/` - Hydra logs and config snapshots

The resolved configuration is automatically saved to `config.yaml` in the output directory.

## Key Features

### 1. Automatic Config Validation
Hydra validates your configuration at runtime and provides clear error messages for typos or missing values.

### 2. Config Composition
Easily compose configurations from multiple files:
```yaml
defaults:
  - model: default
  - data: custom_dataset
  - optimizer: adam
  - diffusion: default
```

### 3. Command-line Override
Override any configuration value without modifying files:
```bash
uv run python src/uaag2/train.py model.sdim=512 data.batch_size=128
```

### 4. Reproducibility
Each run automatically saves the complete resolved configuration, making experiments fully reproducible.

## Migration from Argparse

Previous argparse arguments map to Hydra config as follows:

| Argparse | Hydra Config |
|----------|-------------|
| `--lr 0.001` | `optimizer.lr=0.001` |
| `--batch-size 64` | `data.batch_size=64` |
| `--num-layers 10` | `model.num_layers=10` |
| `--timesteps 1000` | `diffusion.timesteps=1000` |
| `--gpus 2` | `gpus=2` |
| `--seed 42` | `seed=42` |

## Tips

1. **View resolved config**: Add `--cfg job` to see the complete configuration:
   ```bash
   uv run python src/uaag2/train.py --cfg job
   ```

2. **Validate without running**: Use `--help` to see all available options:
   ```bash
   uv run python src/uaag2/train.py --help
   ```

3. **Tab completion**: Enable shell completion for easier parameter discovery:
   ```bash
   eval "$(python src/uaag2/train.py --shell-completion=bash)"
   ```

## Further Reading

- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Hydra Patterns and Best Practices](https://hydra.cc/docs/patterns/overview/)
