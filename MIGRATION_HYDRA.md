# Migration to Hydra Configuration Management

## Summary

The project has been migrated from argparse-based configuration to Hydra, providing better configuration management, reproducibility, and experiment tracking.

## What Changed

### Before (Argparse)
```bash
python src/uaag2/train.py \
  --lr 0.001 \
  --batch-size 64 \
  --num-layers 10 \
  --timesteps 1000 \
  --gpus 2
```

### After (Hydra)
```bash
uv run python src/uaag2/train.py \
  optimizer.lr=0.001 \
  data.batch_size=64 \
  model.num_layers=10 \
  diffusion.timesteps=1000 \
  gpus=2
```

## Key Benefits

1. **Organized Configuration**: Settings are now grouped logically (model, data, diffusion, optimizer)
2. **Reproducibility**: Every run automatically saves complete configuration
3. **Easy Experiments**: Create reusable experiment configs instead of long command lines
4. **Type Safety**: Hydra validates configuration at runtime
5. **Multi-run**: Easy parameter sweeps for hyperparameter tuning
6. **No Code Changes**: Change hyperparameters without modifying code

## Files Created

### Configuration Files
- `configs/train.yaml` - Main training configuration
- `configs/data/default.yaml` - Data-related settings
- `configs/model/default.yaml` - Model architecture settings
- `configs/diffusion/default.yaml` - Diffusion process settings
- `configs/optimizer/adam.yaml` - Adam optimizer settings
- `configs/optimizer/sgd.yaml` - SGD optimizer settings
- `configs/experiment/quick_test.yaml` - Quick testing experiment
- `configs/experiment/production.yaml` - Production training experiment

### Documentation
- `docs/HYDRA_CONFIG.md` - Comprehensive Hydra usage guide
- `MIGRATION_HYDRA.md` - This migration guide

### Code Changes
- `src/uaag2/train.py` - Converted from argparse to Hydra

## Migration Guide

### For Developers

If you have custom training scripts or commands, update them as follows:

| Old Argparse Flag | New Hydra Config Path |
|-------------------|----------------------|
| `--lr` | `optimizer.lr` |
| `--batch-size` | `data.batch_size` |
| `--num-layers` | `model.num_layers` |
| `--sdim` | `model.sdim` |
| `--vdim` | `model.vdim` |
| `--timesteps` | `diffusion.timesteps` |
| `--noise-scheduler` | `diffusion.noise_scheduler` |
| `--gpus` | `gpus` |
| `--num-epochs` | `num_epochs` |
| `--seed` | `seed` |
| `--mask-rate` | `data.mask_rate` |
| `--pocket-noise` | `data.pocket_noise` |
| `--use-metadata-sampler` | `data.use_metadata_sampler` |

### For CI/CD

Update your CI/CD pipelines to use the new command structure:

**Before:**
```yaml
- name: Train model
  run: python src/uaag2/train.py --gpus 0 --num-epochs 10 --batch-size 32
```

**After:**
```yaml
- name: Train model
  run: uv run python src/uaag2/train.py gpus=0 num_epochs=10 data.batch_size=32
```

### For Experiments

Instead of maintaining shell scripts with long command lines, create experiment configs:

**Before (shell script):**
```bash
#!/bin/bash
python src/uaag2/train.py \
  --lr 0.001 \
  --batch-size 64 \
  --num-layers 10 \
  --sdim 512 \
  --timesteps 1000 \
  --noise-scheduler cosine \
  --gpus 2 \
  --num-epochs 300
```

**After (config file `configs/experiment/my_exp.yaml`):**
```yaml
# @package _global_
defaults:
  - override /optimizer: adam

num_epochs: 300
gpus: 2

model:
  num_layers: 10
  sdim: 512

data:
  batch_size: 64

diffusion:
  timesteps: 1000
  noise_scheduler: cosine

optimizer:
  lr: 0.001
```

**Run with:**
```bash
uv run python src/uaag2/train.py +experiment=my_exp
```

## Examples

### Quick Test Run
```bash
uv run python src/uaag2/train.py +experiment=quick_test
```

### Production Training
```bash
uv run python src/uaag2/train.py +experiment=production
```

### Custom Override
```bash
uv run python src/uaag2/train.py \
  +experiment=production \
  optimizer.lr=0.0005 \
  data.batch_size=128
```

### Hyperparameter Sweep
```bash
uv run python src/uaag2/train.py -m \
  optimizer.lr=0.0001,0.0003,0.001 \
  data.batch_size=32,64,128
```

## Troubleshooting

### Configuration not found
**Error:** `Could not find config`
**Solution:** Make sure you're running from the repository root or adjust the `config_path` in the Hydra decorator.

### Invalid configuration
**Error:** `ConfigAttributeError`
**Solution:** Check your YAML syntax and ensure all referenced files exist in the configs directory.

### Old argparse command not working
**Solution:** Update your command to use Hydra syntax (see migration table above).

## Rollback (if needed)

The argparse code is preserved in git history. To rollback:
```bash
git show HEAD~1:src/uaag2/train.py > src/uaag2/train.py
```

However, we recommend adapting to Hydra as it provides significant benefits for ML experimentation.

## Additional Resources

- [Hydra Configuration Guide](docs/HYDRA_CONFIG.md)
- [Hydra Official Documentation](https://hydra.cc/)
- [Example Experiments](configs/experiment/)

## Support

For questions or issues with the new configuration system, please:
1. Check `docs/HYDRA_CONFIG.md` for detailed usage
2. Review example experiments in `configs/experiment/`
3. Open an issue on the repository
