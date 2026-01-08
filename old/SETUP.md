# UAAG2 Environment Setup Guide

**Uncanonical Amino Acid Generative Model v2** - A diffusion-based model for generating uncanonical amino acids given proteomic information.

This guide provides instructions for setting up the UAAG2 environment using **uv** (ultra-fast Python package installer) on both **NVIDIA GPUs** and **AMD GPUs** (including HPC systems like LUMI).

---

## Table of Contents
- [System Requirements](#system-requirements)
- [Installing uv](#installing-uv)
- [Option 1: NVIDIA GPU Setup](#option-1-nvidia-gpu-setup)
- [Option 2: AMD GPU Setup (ROCm)](#option-2-amd-gpu-setup-rocm)
- [Installation Verification](#installation-verification)
- [Running the Code](#running-the-code)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Hardware
- **GPU Memory**: 16GB+ (24GB+ recommended for training)
- **RAM**: 32GB+
- **Storage**: 50GB+ free space

### Software Prerequisites
- Python 3.8-3.10 (uv will manage this)
- CUDA 11.8+ (for NVIDIA) or ROCm 5.4+ (for AMD)
- uv package manager (instructions below)

---

## Installing uv

Install uv using the official installer:

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip if you prefer
pip install uv

# Verify installation
uv --version
```

For Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Option 1: NVIDIA GPU Setup

### Step 1: Create Python Environment with uv

```bash
# Navigate to UAAG2 directory
cd /path/to/UAAG2

# Create virtual environment with uv
uv venv --python 3.10 .venv

# Activate the environment
source .venv/bin/activate  # Linux/macOS
# or on Windows: .venv\Scripts\activate
```

### Step 2: Install PyTorch with CUDA

```bash
# Install PyTorch 2.1.0 with CUDA 11.8 using uv
uv pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Install PyTorch Geometric

```bash
# Install PyG and extensions using uv
uv pip install torch-geometric
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### Step 4: Install Core Dependencies

```bash
# Install PyTorch Lightning and other ML libraries
uv pip install pytorch-lightning==2.0.0
uv pip install wandb tensorboard
uv pip install torch-ema  # Exponential Moving Average for model weights

# Install chemistry libraries
# Note: RDKit requires conda or system packages, install via conda or use conda-forge builds
uv pip install rdkit  # or: conda install -c conda-forge rdkit -y
uv pip install biopython
uv pip install openbabel-wheel

# Install scientific computing libraries
uv pip install numpy scipy scikit-learn
uv pip install pandas
uv pip install networkx

# Install utilities
uv pip install tqdm pyyaml omegaconf
uv pip install lmdb
uv pip install ipython  # For interactive debugging in some scripts
```

**Note on RDKit:** If `uv pip install rdkit` fails, you can either:
1. Install RDKit via conda: `conda install -c conda-forge rdkit -y`
2. Use system packages: `sudo apt-get install python3-rdkit` (Ubuntu/Debian)

### Step 5: Install Optional Tools

```bash
# For molecule evaluation
uv pip install posebusters

# For visualization
uv pip install matplotlib seaborn py3Dmol
```

### Step 6: Set Library Path (if needed)

```bash
# Add to ~/.bashrc or run before each session
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib
```

---

## Option 2: AMD GPU Setup (ROCm)

### Prerequisites for AMD GPUs

**Supported AMD GPUs:**
- MI250X, MI250, MI210, MI100 (data center)
- RX 7900 XTX, RX 7900 XT, RX 6900 XT, RX 6800 XT (consumer, limited support)

**For LUMI users:** ROCm is pre-installed. Load the appropriate modules.

### Step 1: Setup ROCm Environment

#### On LUMI or HPC with ROCm Modules:

```bash
# Load ROCm modules (adjust versions as needed)
module load LUMI/22.08
module load rocm/5.4.3

# Create virtual environment with uv
cd /path/to/UAAG2
uv venv --python 3.10 .venv
source .venv/bin/activate
```

#### On Standalone System with ROCm:

```bash
# Verify ROCm installation
rocm-smi

# Create virtual environment with uv
cd /path/to/UAAG2
uv venv --python 3.10 .venv
source .venv/bin/activate
```

### Step 2: Install PyTorch with ROCm

```bash
# Install PyTorch built for ROCm using uv
uv pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/rocm5.6

# Verify PyTorch + ROCm
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Note:** `torch.cuda.is_available()` returns `True` on AMD GPUs with ROCm - this is expected behavior!

### Step 3: Install PyTorch Geometric for ROCm

```bash
# Install PyG
uv pip install torch-geometric

# Build PyG extensions from source for ROCm
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv --no-binary :all:
```

**Alternative (pre-built ROCm wheels):**
```bash
# Try ROCm-specific wheels if available
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+rocm5.6.html
```

### Step 4: Install Core Dependencies

```bash
# Install PyTorch Lightning
uv pip install pytorch-lightning==2.0.0

# Logging and experiment tracking
uv pip install wandb tensorboard
uv pip install torch-ema  # Exponential Moving Average for model weights

# Chemistry libraries
# Note: For RDKit, you may need conda or system packages
uv pip install rdkit  # or: conda install -c conda-forge rdkit -y
uv pip install biopython
uv pip install openbabel-wheel

# Scientific computing libraries
uv pip install numpy scipy scikit-learn
uv pip install pandas
uv pip install networkx

# Utilities
uv pip install tqdm pyyaml omegaconf
uv pip install lmdb
uv pip install ipython  # For interactive debugging in some scripts
```

### Step 5: Install Optional Tools

```bash
# Molecule evaluation
uv pip install posebusters

# Visualization
uv pip install matplotlib seaborn py3Dmol
```

### Step 6: Set Environment Variables for ROCm

```bash
# Add to ~/.bashrc or create a setup script
export HSA_OVERRIDE_GFX_VERSION=9.0.8  # Adjust for your GPU (e.g., 9.0.8 for MI250X)
export ROCM_HOME=/opt/rocm  # Adjust to your ROCm installation path
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# For virtual environment
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH
```

**For LUMI specifically:**
```bash
# LUMI-specific optimizations
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# Create cache directory
mkdir -p $MIOPEN_USER_DB_PATH
```

---

## Installation Verification

### Test PyTorch and GPU

```bash
python << EOF
import torch
import torch_geometric

print("="*60)
print("UAAG2 Environment Verification")
print("="*60)

# PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test tensor operations
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print(f"GPU tensor test: PASSED")

# PyTorch Geometric
print(f"PyG version: {torch_geometric.__version__}")

# RDKit
try:
    from rdkit import Chem
    print(f"RDKit: Available")
except:
    print(f"RDKit: NOT FOUND")

# PyTorch Lightning
try:
    import pytorch_lightning as pl
    print(f"PyTorch Lightning: {pl.__version__}")
except:
    print(f"PyTorch Lightning: NOT FOUND")

print("="*60)
EOF
```

### Test UAAG2 Model Loading

```bash
cd /path/to/UAAG2
python << EOF
import sys
sys.path.append('.')
from uaag.e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork

print("Testing UAAG2 model import...")
model = DenoisingEdgeNetwork(
    num_atom_features=64,
    num_bond_types=5,
    hn_dim=(256, 64),
    edge_dim=32,
    num_layers=5
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print("UAAG2 model import: SUCCESS")
EOF
```

---

## Running the Code

### Training

```bash
# Activate environment
source .venv/bin/activate  # Linux/macOS
# or on Windows: .venv\Scripts\activate

# Set library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib

# Run training
bash run_train.sh

# Or with specific GPU
CUDA_VISIBLE_DEVICES=0 python scripts/run_train.py \
    --logger-type wandb \
    --batch-size 8 \
    --gpus 1 \
    --num-epochs 1000 \
    --id my_experiment
```

### Sampling/Generation

```bash
# Generate ligands
bash run_sampling_sanity_check_test_scale.sh

# Or manually
python scripts/generate_ligand.py \
    --load-ckpt /path/to/checkpoint.ckpt \
    --benchmark-path /path/to/benchmark.pt \
    --save-dir ./outputs \
    --num-samples 100
```

### Molecule Evaluation

```bash
# Evaluate generated molecules
python scripts/evaluate_mol_samples.py \
    /path/to/samples_directory \
    -o evaluation_results.csv
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA/ROCm Not Detected

**NVIDIA:**
```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA
nvcc --version

# Reinstall PyTorch with correct CUDA version
uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**AMD:**
```bash
# Check ROCm
rocm-smi

# Verify ROCm version
cat /opt/rocm/.info/version

# Set GPU architecture
export HSA_OVERRIDE_GFX_VERSION=9.0.8  # Adjust for your GPU
```

#### 2. PyTorch Geometric Installation Fails

```bash
# Install dependencies first
uv pip install ninja

# Build from source
uv pip install torch-scatter torch-sparse --no-binary :all:

# Or use specific wheel URL
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cu118.html
```

#### 3. Out of Memory Errors

```bash
# Reduce batch size in training script
--batch-size 4  # or smaller

# Use gradient accumulation
--accumulate-grad-batches 4

# For AMD/ROCm, clear cache
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
```

#### 4. Library Path Issues

```bash
# Add to shell profile (~/.bashrc or ~/.zshrc)
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH

# Or add to activation script
echo 'export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH' >> .venv/bin/activate

# Then reactivate
deactivate && source .venv/bin/activate
```

#### 5. ROCm-Specific Issues

**MIOpen Cache Issues:**
```bash
# Clear MIOpen cache
rm -rf ~/.cache/miopen
mkdir -p /tmp/${USER}-miopen-cache
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
```

**Compatibility Issues:**
```bash
# For older AMD GPUs, try different GFX version
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # RX 6000 series
export HSA_OVERRIDE_GFX_VERSION=9.0.0   # MI100
```

#### 6. Slow Performance on AMD

```bash
# Enable TunableOp for better performance
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1

# Use hipBLAS for better GEMM performance
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library
```

---

## Performance Optimization

### NVIDIA GPUs

```bash
# Enable TF32 for faster training (Ampere and newer)
export NVIDIA_TF32_OVERRIDE=1

# Use torch.compile (PyTorch 2.0+)
# Add to training script: model = torch.compile(model)

# Enable cuDNN benchmarking
# Add to training script: torch.backends.cudnn.benchmark = True
```

### AMD GPUs (ROCm)

```bash
# Enable all optimizations
export HSA_OVERRIDE_GFX_VERSION=9.0.8  # Your GPU
export PYTORCH_TUNABLEOP_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
export ROCM_PATH=/opt/rocm

# For multi-GPU
export NCCL_DEBUG=INFO
export RCCL_KERNEL_COLL_TRACE_ENABLE=1
```

---

## Additional Resources

### Documentation
- [UAAG2 Pipeline Usage](PIPELINE_USAGE.md)
- [PyTorch ROCm Documentation](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [LUMI Documentation](https://docs.lumi-supercomputer.eu/)

### Support
- Check existing issues in the repository
- LUMI users: [LUMI User Support](https://www.lumi-supercomputer.eu/user-support/)
- ROCm users: [ROCm GitHub Issues](https://github.com/RadeonOpenCompute/ROCm/issues)

---

## Quick Start Summary

### NVIDIA GPU (One-liner)
```bash
conda create -n uaag2 python=3.10 -y && \
conda activate uaag2 && \
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html && \
conda install -c conda-forge rdkit openbabel -y && \
pip install pytorch-lightning wandb tensorboard torch-ema biopython tqdm pyyaml omegaconf lmdb && \
pip install numpy scipy scikit-learn pandas networkx ipython && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
### AMD GPU/ROCm (One-liner)
```bash
conda create -n uaag2_rocm python=3.10 -y && \
conda activate uaag2_rocm && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6 && \
pip install torch-geometric torch-scatter torch-sparse --no-binary :all: && \
conda install -c conda-forge rdkit openbabel -y && \
pip install pytorch-lightning wandb tensorboard torch-ema biopython tqdm pyyaml omegaconf lmdb && \
pip install numpy scipy scikit-learn pandas networkx ipython && \
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export HSA_OVERRIDE_GFX_VERSION=9.0.8
```ort LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export HSA_OVERRIDE_GFX_VERSION=9.0.8
```

---

**Last Updated:** January 2026  
**Tested On:** NVIDIA A100, AMD MI250X (LUMI), NVIDIA RTX 4090
