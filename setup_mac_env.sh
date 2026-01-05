#!/bin/bash
################################################################################
# UAAG2 Mac Environment Setup Script
# 
# This script sets up a conda environment for UAAG2 on macOS for local
# development and testing before running experiments on the cluster.
#
# Usage:
#   bash setup_mac_env.sh
#
# Author: UAAG2 Team
# Date: January 2026
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "UAAG2 Mac Environment Setup"
echo "================================================================================"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed!${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓${NC} Conda found: $(conda --version)"
echo ""

# Environment name
ENV_NAME="uaag2_mac"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Warning: Environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Exiting. Please use a different environment name or remove manually."
        exit 1
    fi
fi

echo "================================================================================"
echo "Step 1: Creating conda environment with Python 3.10"
echo "================================================================================"
conda create -n ${ENV_NAME} python=3.10 -y

echo ""
echo "================================================================================"
echo "Step 2: Installing PyTorch (CPU version for Mac)"
echo "================================================================================"
# For Mac, use CPU version of PyTorch
conda run -n ${ENV_NAME} pip install torch torchvision torchaudio

echo ""
echo "================================================================================"
echo "Step 3: Installing PyTorch Geometric"
echo "================================================================================"
conda run -n ${ENV_NAME} pip install torch-geometric
conda run -n ${ENV_NAME} pip install torch-scatter torch-sparse torch-cluster torch-spline-conv

echo ""
echo "================================================================================"
echo "Step 4: Installing Chemistry Libraries"
echo "================================================================================"
# Install RDKit and OpenBabel via conda-forge (more reliable on Mac)
conda install -n ${ENV_NAME} -c conda-forge rdkit openbabel -y
conda run -n ${ENV_NAME} pip install biopython

echo ""
echo "================================================================================"
echo "Step 5: Installing Scientific Computing Libraries"
echo "================================================================================"
conda run -n ${ENV_NAME} pip install numpy scipy scikit-learn pandas networkx

echo ""
echo "================================================================================"
echo "Step 6: Installing ML/DL Libraries"
echo "================================================================================"
conda run -n ${ENV_NAME} pip install pytorch-lightning==2.0.0
conda run -n ${ENV_NAME} pip install torch-ema
conda run -n ${ENV_NAME} pip install wandb tensorboard

echo ""
echo "================================================================================"
echo "Step 7: Installing Utilities"
echo "================================================================================"
conda run -n ${ENV_NAME} pip install tqdm pyyaml omegaconf lmdb ipython

echo ""
echo "================================================================================"
echo "Step 8: Installing Optional Evaluation Tools"
echo "================================================================================"
echo "Installing PoseBusters (may take a while)..."
conda run -n ${ENV_NAME} pip install posebusters || echo -e "${YELLOW}Warning: PoseBusters installation failed. You can install it later if needed.${NC}"

echo ""
echo "================================================================================"
echo "Step 9: Verifying Installation"
echo "================================================================================"

# Create a verification script
VERIFY_SCRIPT=$(mktemp)
cat > ${VERIFY_SCRIPT} << 'EOF'
import sys
print("="*80)
print("UAAG2 Environment Verification")
print("="*80)

# Test core packages
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import torch_geometric
    print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")
except ImportError as e:
    print(f"✗ PyTorch Geometric: {e}")
    sys.exit(1)

try:
    from rdkit import Chem
    print(f"✓ RDKit: Available")
except ImportError as e:
    print(f"✗ RDKit: {e}")
    sys.exit(1)

try:
    import openbabel
    print(f"✓ OpenBabel: Available")
except ImportError as e:
    print(f"✗ OpenBabel: {e}")
    sys.exit(1)

try:
    import networkx
    print(f"✓ NetworkX: {networkx.__version__}")
except ImportError as e:
    print(f"✗ NetworkX: {e}")

try:
    import pytorch_lightning as pl
    print(f"✓ PyTorch Lightning: {pl.__version__}")
except ImportError as e:
    print(f"✗ PyTorch Lightning: {e}")
    sys.exit(1)

try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import pandas
    print(f"✓ Pandas: {pandas.__version__}")
except ImportError as e:
    print(f"✗ Pandas: {e}")

try:
    import scipy
    print(f"✓ SciPy: {scipy.__version__}")
except ImportError as e:
    print(f"✗ SciPy: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn: {e}")

try:
    from posebusters import PoseBusters
    print(f"✓ PoseBusters: Available")
except ImportError:
    print(f"⚠ PoseBusters: Not installed (optional)")

print("="*80)
print("All core packages installed successfully!")
print("="*80)
EOF

conda run -n ${ENV_NAME} python ${VERIFY_SCRIPT}
rm ${VERIFY_SCRIPT}

echo ""
echo "================================================================================"
echo "Setup Complete!"
echo "================================================================================"
echo ""
echo -e "${GREEN}Environment '${ENV_NAME}' has been successfully created!${NC}"
echo ""
echo "To activate the environment, run:"
echo -e "  ${YELLOW}conda activate ${ENV_NAME}${NC}"
echo ""
echo "To test UAAG2 locally:"
echo "  1. Activate the environment: conda activate ${ENV_NAME}"
echo "  2. Navigate to UAAG2 directory"
echo "  3. Run a small test with limited data"
echo ""
echo "For cluster deployment:"
echo "  - See SETUP.md for GPU-specific instructions"
echo "  - Use the NVIDIA GPU setup for cluster nodes"
echo ""
echo "================================================================================"
