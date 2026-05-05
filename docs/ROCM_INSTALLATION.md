# ROCm Installation Guide for AMD GPUs

This guide explains how to install and use UAAG2 with ROCm support on AMD GPUs, such as those available on the LUMI supercomputer.

## Quick Start

### Option 1: Using the Installation Script (Recommended)

```bash
# On LUMI or any system with AMD GPUs
bash install_rocm.sh
```

This script will:
- Install PyTorch 2.7.0 with ROCm 6.2 support
- Install PyTorch Geometric extensions compiled for ROCm
- Install all other project dependencies

### Option 2: Manual Installation

```bash
# Install PyTorch with ROCm
uv pip install --upgrade \
    torch==2.7.0+rocm6.2 \
    --index-url https://download.pytorch.org/whl/rocm6.2

# Install other dependencies
uv sync

# Build PyG extensions from source (no prebuilt ROCm wheels available)
# Note: torch-spline-conv is not used in this codebase
uv pip install --no-build-isolation --upgrade \
    torch-cluster==1.6.3 \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18
```

**Important Notes:**
- PyTorch Geometric extensions (torch-cluster, torch-scatter, torch-sparse) don't have prebuilt ROCm wheels
- They will be **built from source** automatically, which requires:
  - C++ compiler (gcc/g++)
  - On LUMI: `module load buildtools/23.09`
- Build time: ~5-10 minutes depending on system
- torch-spline-conv is not used in this codebase and is excluded

## Verification

After installation, verify that ROCm is properly detected:

```bash
uv run python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output on systems with AMD GPUs:
```
ROCm available: True
Device: AMD Instinct MI250X  # or your specific GPU model
```

## Running Training on AMD GPUs

Once installed, you can run training exactly as you would with CUDA:

```bash
# Using single GPU
uv run python src/uaag2/train.py gpus=1

# Using multiple GPUs
uv run python src/uaag2/train.py gpus=4

# Using experiment config
uv run python src/uaag2/train.py +experiment=production gpus=8
```

## LUMI-Specific Instructions

### Environment Setup on LUMI

On the LUMI supercomputer, load the appropriate modules before installation:

```bash
# Load required modules
module load LUMI/23.09
module load rocm/6.2.0
module load cray-python/3.11.7

# Clone and setup
git clone https://github.com/HirahTang/UAAG2.git
cd UAAG2

# Install uv if not available
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Install with ROCm support
bash install_rocm.sh
```

### SLURM Job Script for LUMI

Example SLURM script for running training on LUMI:

```bash
#!/bin/bash
#SBATCH --job-name=uaag2-train
#SBATCH --account=project_xxxxx
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --mem=480G

# Load modules
module load LUMI/23.09
module load rocm/6.2.0
module load cray-python/3.11.7

# Activate environment
cd $SCRATCH/UAAG2
export PATH="$HOME/.cargo/bin:$PATH"

# Run training
uv run python src/uaag2/train.py \
    +experiment=production \
    gpus=8 \
    data.batch_size=64 \
    num_epochs=100
```

## Platform Support Matrix

| Platform | GPU Type | Backend | Installation Method |
|----------|----------|---------|---------------------|
| Linux x86_64 (default) | NVIDIA | CUDA 12.8 | `uv sync` |
| Linux x86_64 (LUMI/AMD) | AMD Instinct | ROCm 6.2 | `bash install_rocm.sh` |
| macOS (Apple Silicon) | Apple GPU | MPS | `uv sync` |
| Linux ARM64 | CPU only | - | `uv sync` |

## Troubleshooting

### ROCm Not Detected

If `torch.cuda.is_available()` returns `False`:

1. **Check ROCm installation:**
   ```bash
   rocm-smi
   ```

2. **Verify ROCm version:**
   ```bash
   cat /opt/rocm/.info/version
   ```
   Should show version 6.2 or compatible.

3. **Check environment variables:**
   ```bash
   echo $ROCM_PATH
   echo $HIP_VISIBLE_DEVICES
   ```

### Build Failures for PyG Extensions

PyG extensions (torch-cluster, torch-scatter, torch-sparse) **must be built from source** for ROCm as there are no prebuilt wheels available.

**Common issues:**

1. **Missing compiler:**
   ```bash
   # On LUMI
   module load buildtools/23.09

   # On Ubuntu/Debian
   sudo apt-get install build-essential

   # On RHEL/CentOS
   sudo yum groupinstall "Development Tools"
   ```

2. **ROCm headers not found:**
   ```bash
   # Ensure ROCm is in your path
   export ROCM_PATH=/opt/rocm
   export HIP_PATH=/opt/rocm/hip
   ```

3. **Try installing with verbose output:**
   ```bash
   uv pip install -v --no-build-isolation torch-cluster==1.6.3
   ```

4. **Build one extension at a time** to identify which one fails:
   ```bash
   uv pip install --no-build-isolation torch-scatter==2.1.2
   uv pip install --no-build-isolation torch-cluster==1.6.3
   uv pip install --no-build-isolation torch-sparse==0.6.18
   ```

### Version Compatibility

- **ROCm 6.2** is required for PyTorch 2.7.0
- For older ROCm versions (5.7), use PyTorch 2.4.0
- Check [PyTorch ROCm Support Matrix](https://pytorch.org/get-started/locally/) for compatibility

## Performance Optimization

### ROCm-Specific Tuning

```python
# In your training script or config
import torch

# Enable TF32 for AMD GPUs (if supported)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Optimize for AMD architecture
torch.backends.cudnn.benchmark = True
```

### Memory Management

AMD GPUs may have different memory characteristics:

```bash
# Monitor GPU memory on AMD
watch -n 1 rocm-smi

# Set memory fraction if needed
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
```

## Additional Resources

- [PyTorch ROCm Documentation](https://pytorch.org/get-started/locally/#rocm)
- [LUMI Documentation](https://docs.lumi-supercomputer.eu/)
- [ROCm GitHub](https://github.com/ROCm/ROCm)
- [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

## Support

For ROCm-specific issues:
- Check the [PyTorch ROCm forum](https://discuss.pytorch.org/)
- For LUMI-specific issues, contact [LUMI support](https://lumi-supercomputer.eu/user-support/)
- For project-specific issues, open an issue on GitHub
