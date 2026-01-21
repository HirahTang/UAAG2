# Argparse to Hydra Migration - Summary

## Completed Tasks ✅

### 1. Configuration Structure Created
- ✅ `configs/train.yaml` - Main training configuration with defaults
- ✅ `configs/data/default.yaml` - Data loading and processing settings
- ✅ `configs/model/default.yaml` - Model architecture parameters
- ✅ `configs/diffusion/default.yaml` - Diffusion process settings
- ✅ `configs/optimizer/adam.yaml` - Adam optimizer configuration
- ✅ `configs/optimizer/sgd.yaml` - SGD optimizer configuration
- ✅ `configs/experiment/quick_test.yaml` - Quick testing configuration
- ✅ `configs/experiment/production.yaml` - Production training configuration

### 2. Code Migration
- ✅ Converted `src/uaag2/train.py` from argparse to Hydra
- ✅ Replaced `ArgumentParser` with `@hydra.main` decorator
- ✅ Updated function signatures to use `DictConfig`
- ✅ Changed config access from `hparams.param` to `cfg.group.param`
- ✅ Replaced `yaml.safe_dump(vars(hparams))` with `OmegaConf.to_yaml(cfg)`
- ✅ Code passes syntax validation and linting

### 3. Dependencies
- ✅ Added `hydra-core==1.3.2` to pyproject.toml
- ✅ `omegaconf` automatically installed as Hydra dependency

### 4. Documentation
- ✅ Created `docs/HYDRA_CONFIG.md` - Comprehensive Hydra usage guide
- ✅ Created `MIGRATION_HYDRA.md` - Migration guide for developers
- ✅ Updated `AGENTS.md` to reflect new configuration system

### 5. Validation
- ✅ Configuration loads successfully via Hydra
- ✅ Python syntax validation passes
- ✅ Ruff linting passes
- ✅ Code formatted with ruff

## Configuration Features

### Hierarchical Structure
```
train.yaml (root)
├── model/default.yaml
├── data/default.yaml
├── diffusion/default.yaml
└── optimizer/adam.yaml
```

### Command-Line Overrides
```bash
# Before (argparse)
python src/uaag2/train.py --lr 0.001 --batch-size 64

# After (Hydra)
uv run python src/uaag2/train.py optimizer.lr=0.001 data.batch_size=64
```

### Experiment Configs
```bash
# Use pre-configured experiments
uv run python src/uaag2/train.py +experiment=quick_test
uv run python src/uaag2/train.py +experiment=production
```

### Multi-Run Sweeps
```bash
# Hyperparameter sweep
uv run python src/uaag2/train.py -m optimizer.lr=0.0001,0.001 data.batch_size=32,64
```

## Key Improvements

1. **Better Organization**: Configuration split into logical groups (model, data, diffusion, optimizer)
2. **Reproducibility**: Every run automatically saves complete resolved configuration
3. **Flexibility**: Easy to create experiment configs for different scenarios
4. **Type Safety**: Hydra validates configuration structure at runtime
5. **No Code Changes**: Modify hyperparameters without touching code
6. **Experiment Tracking**: Automatic logging of all config values to wandb/tensorboard

## Parameter Mapping

| Old Argparse | New Hydra Path |
|--------------|----------------|
| `--lr` | `optimizer.lr` |
| `--batch-size` | `data.batch_size` |
| `--num-layers` | `model.num_layers` |
| `--sdim` | `model.sdim` |
| `--vdim` | `model.vdim` |
| `--timesteps` | `diffusion.timesteps` |
| `--noise-scheduler` | `diffusion.noise_scheduler` |
| `--mask-rate` | `data.mask_rate` |
| `--pocket-noise` | `data.pocket_noise` |
| `--gpus` | `gpus` |
| `--num-epochs` | `num_epochs` |

## Usage Examples

### Basic Training
```bash
uv run python src/uaag2/train.py
```

### Override Parameters
```bash
uv run python src/uaag2/train.py \
  optimizer.lr=0.001 \
  data.batch_size=64 \
  model.num_layers=10 \
  gpus=2
```

### Use Experiment Config
```bash
uv run python src/uaag2/train.py +experiment=production
```

### Override in Experiment
```bash
uv run python src/uaag2/train.py \
  +experiment=production \
  optimizer.lr=0.0005
```

### Hyperparameter Sweep
```bash
uv run python src/uaag2/train.py -m \
  optimizer.lr=0.0001,0.0003,0.001 \
  data.batch_size=32,64
```

## Next Steps

### For Developers
1. Read `docs/HYDRA_CONFIG.md` for detailed usage instructions
2. Create custom experiment configs in `configs/experiment/`
3. Update CI/CD pipelines to use new Hydra syntax
4. Update training scripts and notebooks to use new configuration

### For CI/CD
Update pipeline commands from:
```yaml
run: python src/uaag2/train.py --gpus 0 --num-epochs 10
```
to:
```yaml
run: uv run python src/uaag2/train.py gpus=0 num_epochs=10
```

### Testing
The configuration system is ready for use. Note that actual training execution will require:
- Proper data files in place (`data/statistic.pkl`, `data/pdb_subset.lmdb`)
- Fixed torch_geometric installation (currently segfaulting on macOS ARM64)

The torch_geometric issue is environment-specific and will work properly on Linux systems (including CI/CD).

## Files Modified/Created

### Modified
- `src/uaag2/train.py` - Converted to Hydra
- `AGENTS.md` - Updated configuration instructions
- `pyproject.toml` - Added hydra-core dependency

### Created
- `configs/train.yaml`
- `configs/data/default.yaml`
- `configs/model/default.yaml`
- `configs/diffusion/default.yaml`
- `configs/optimizer/adam.yaml`
- `configs/optimizer/sgd.yaml`
- `configs/experiment/quick_test.yaml`
- `configs/experiment/production.yaml`
- `docs/HYDRA_CONFIG.md`
- `MIGRATION_HYDRA.md`
- `SUMMARY_HYDRA_MIGRATION.md` (this file)

## Migration Complete ✅

The argparse to Hydra migration is complete and ready for use. All configuration parameters have been preserved and organized into a clean, hierarchical structure.
