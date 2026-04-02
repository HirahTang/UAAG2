#!/bin/bash
# Installation script for ROCm-enabled PyTorch on AMD GPUs (e.g., LUMI supercomputer)
# Usage: bash install_rocm.sh

set -e

echo "Installing PyTorch with ROCm 6.2 support for AMD GPUs..."

# Install PyTorch with ROCm
echo "→ Installing PyTorch 2.7.0+rocm6.2..."
uv pip install --upgrade \
    torch==2.7.0+rocm6.2 \
    --index-url https://download.pytorch.org/whl/rocm6.2

# Install other dependencies (PyTorch Geometric, etc.)
echo "→ Installing PyTorch Geometric and other dependencies..."
uv sync

# Build PyG extensions from source (no prebuilt ROCm wheels available)
# Note: torch-spline-conv is not used in the codebase, so we skip it
echo "→ Building PyG extensions from source (this may take several minutes)..."
echo "  Required: torch-cluster, torch-scatter, torch-sparse"

# Ensure build tools are available
if ! command -v gcc &> /dev/null; then
    echo "⚠️  Warning: gcc not found. PyG extension builds may fail."
    echo "   On LUMI, run: module load buildtools/23.09"
fi

# Build from source - these will use ROCm automatically if PyTorch was built with ROCm
uv pip install --no-build-isolation --upgrade \
    torch-cluster==1.6.3 \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18

echo "✅ ROCm installation complete!"
echo ""
echo "To verify ROCm installation, run:"
echo "  uv run python -c 'import torch; print(f\"ROCm available: {torch.cuda.is_available()}\"); print(f\"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}\")'"
echo ""
echo "To verify PyG extensions, run:"
echo "  uv run python -c 'import torch_cluster, torch_scatter, torch_sparse; print(\"✓ All PyG extensions imported successfully\")'"
