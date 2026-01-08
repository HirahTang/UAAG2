#!/bin/bash
################################################################################
# UAAG2 Mac Environment Activation Helper
#
# Quick activation script for UAAG2 development on Mac
#
# Usage:
#   source activate_mac.sh
#
################################################################################

# Source conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Error: Conda not found. Please install Miniconda first."
    return 1
fi

# Set OpenMP workaround
export KMP_DUPLICATE_LIB_OK=TRUE

# Activate environment
conda activate uaag2_mac

# Confirm activation
if [ $? -eq 0 ]; then
    echo "✓ UAAG2 Mac environment activated!"
    echo "  Python: $(python --version)"
    echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
    echo ""
    echo "You can now run UAAG2 scripts with --gpus 0 for CPU testing."
else
    echo "✗ Failed to activate environment. Please run setup_mac_env.sh first."
    return 1
fi
