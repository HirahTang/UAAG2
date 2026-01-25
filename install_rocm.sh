#!/bin/bash
# Installation script for ROCm-enabled PyTorch on AMD GPUs (e.g., LUMI supercomputer)
# Usage: bash install_rocm.sh

set -e

echo "Installing PyTorch with ROCm 6.2 support for AMD GPUs..."

# Set environment to use ROCm backend
export UV_INDEX_URL=https://download.pytorch.org/whl/rocm6.2

# Install PyTorch with ROCm
uv pip install --upgrade \
    torch==2.7.0+rocm6.2 \
    --index-url https://download.pytorch.org/whl/rocm6.2

# Install PyG extensions for ROCm
uv pip install --upgrade \
    torch-cluster==1.6.3+pt27rocm6.2 \
    torch-scatter==2.1.2+pt27rocm6.2 \
    torch-sparse==0.6.18+pt27rocm6.2 \
    torch-spline-conv==1.2.2+pt27rocm6.2 \
    --find-links https://data.pyg.org/whl/torch-2.7.0+rocm6.2.html

# Install other dependencies
uv sync

echo "✅ ROCm installation complete!"
echo ""
echo "To verify ROCm installation, run:"
echo "  uv run python -c 'import torch; print(f\"ROCm available: {torch.cuda.is_available()}\"); print(f\"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}\")'"
