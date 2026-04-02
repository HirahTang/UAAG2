# Quick Start with Hydra Configuration

## TL;DR

**Before:**
```bash
python src/uaag2/train.py --lr 0.001 --batch-size 64 --num-layers 10
```

**After:**
```bash
uv run python src/uaag2/train.py optimizer.lr=0.001 data.batch_size=64 model.num_layers=10
```

## Common Commands

### Run with Defaults
```bash
uv run python src/uaag2/train.py
```

### Quick Test (5 epochs, small model)
```bash
uv run python src/uaag2/train.py +experiment=quick_test
```

### Production Training
```bash
uv run python src/uaag2/train.py +experiment=production
```

### Custom Hyperparameters
```bash
uv run python src/uaag2/train.py \
  optimizer.lr=0.001 \
  data.batch_size=128 \
  model.num_layers=12 \
  gpus=4
```

### Load from Checkpoint
```bash
uv run python src/uaag2/train.py \
  load_ckpt=/path/to/checkpoint.ckpt \
  optimizer.lr=0.0001
```

### Hyperparameter Sweep
```bash
uv run python src/uaag2/train.py -m \
  optimizer.lr=0.0001,0.001,0.01 \
  data.batch_size=32,64,128
```

## Config Groups

### Model Configs
- Located in `configs/model/`
- Default: `default.yaml`

**Key parameters:**
- `model.sdim` - Scalar dimension
- `model.vdim` - Vector dimension
- `model.num_layers` - Number of layers
- `model.dropout_prob` - Dropout probability

### Data Configs
- Located in `configs/data/`
- Default: `default.yaml`

**Key parameters:**
- `data.batch_size` - Training batch size
- `data.mask_rate` - Masking rate (0-1)
- `data.train_size` - Training set proportion
- `data.pocket_noise` - Enable pocket noise

### Optimizer Configs
- Located in `configs/optimizer/`
- Options: `adam.yaml`, `sgd.yaml`

**Key parameters:**
- `optimizer.lr` - Learning rate
- `optimizer.lr_scheduler` - Scheduler type
- `optimizer.grad_clip_val` - Gradient clipping

### Diffusion Configs
- Located in `configs/diffusion/`
- Default: `default.yaml`

**Key parameters:**
- `diffusion.timesteps` - Number of timesteps
- `diffusion.noise_scheduler` - Scheduler type
- `diffusion.loss_weighting` - Loss weighting strategy

## View Configuration

### See all parameters
```bash
uv run python src/uaag2/train.py --cfg job
```

### See help
```bash
uv run python src/uaag2/train.py --help
```

## Create Custom Experiment

1. Create `configs/experiment/my_exp.yaml`:
```yaml
# @package _global_

defaults:
  - override /optimizer: adam

num_epochs: 100
gpus: 2

model:
  num_layers: 8
  sdim: 384

data:
  batch_size: 64

optimizer:
  lr: 0.0005
```

2. Run it:
```bash
uv run python src/uaag2/train.py +experiment=my_exp
```

## Parameter Reference

### Common Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpus` | int | 1 | Number of GPUs |
| `num_epochs` | int | 300 | Training epochs |
| `seed` | int | 42 | Random seed |
| `logger_type` | str | wandb | Logger type (wandb/tensorboard) |

### Model Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model.sdim` | int | 256 | Scalar dimension |
| `model.vdim` | int | 64 | Vector dimension |
| `model.num_layers` | int | 7 | Number of layers |
| `model.dropout_prob` | float | 0.3 | Dropout probability |

### Data Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data.batch_size` | int | 32 | Batch size |
| `data.mask_rate` | float | 0.5 | Mask rate (0-1) |
| `data.train_size` | float | 0.99 | Training proportion |
| `data.pocket_noise` | bool | false | Enable pocket noise |

### Optimizer Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer.lr` | float | 5e-4 | Learning rate |
| `optimizer.grad_clip_val` | float | 10.0 | Gradient clipping |
| `optimizer.lr_scheduler` | str | reduce_on_plateau | LR scheduler |

### Diffusion Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diffusion.timesteps` | int | 500 | Diffusion timesteps |
| `diffusion.noise_scheduler` | str | cosine | Noise scheduler |
| `diffusion.loss_weighting` | str | snr_t | Loss weighting |

## Tips

1. **Tab completion**: Hydra supports shell completion (see docs)
2. **Config validation**: Hydra validates types automatically
3. **Reproducibility**: Config is automatically saved with each run
4. **Multi-run**: Use `-m` flag for parameter sweeps
5. **Override experiments**: Combine experiment with overrides

## Need More Help?

- See `docs/HYDRA_CONFIG.md` for comprehensive guide
- See `MIGRATION_HYDRA.md` for migration details
- Check `configs/experiment/` for example configurations
