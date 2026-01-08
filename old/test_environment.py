#!/usr/bin/env python
"""
UAAG2 Environment Test Script

Tests all critical imports required for UAAG2 development.
Run this after setting up the Mac environment to verify everything works.

Usage:
    conda activate uaag2_mac
    python test_environment.py
"""

import sys
import os

def test_imports():
    """Test all critical package imports."""
    print("="*80)
    print("UAAG2 Environment Test")
    print("="*80)
    print()
    
    failures = []
    
    # Core ML/DL packages
    tests = [
        ("PyTorch", "import torch; print(f'  Version: {torch.__version__}')"),
        ("PyTorch Geometric", "import torch_geometric; print(f'  Version: {torch_geometric.__version__}')"),
        ("PyTorch Lightning", "import pytorch_lightning as pl; print(f'  Version: {pl.__version__}')"),
        ("NumPy", "import numpy as np; print(f'  Version: {np.__version__}')"),
        ("SciPy", "import scipy; print(f'  Version: {scipy.__version__}')"),
        ("Pandas", "import pandas as pd; print(f'  Version: {pd.__version__}')"),
        ("NetworkX", "import networkx as nx; print(f'  Version: {nx.__version__}')"),
        ("Scikit-learn", "import sklearn; print(f'  Version: {sklearn.__version__}')"),
        
        # Chemistry packages
        ("RDKit", "from rdkit import Chem; from rdkit import __version__; print(f'  Version: {__version__}')"),
        ("OpenBabel", "import openbabel; print('  Available')"),
        ("BioPython", "import Bio; print(f'  Version: {Bio.__version__}')"),
        
        # Utilities
        ("LMDB", "import lmdb; print(f'  Version: {lmdb.__version__}')"),
        ("YAML", "import yaml; print('  Available')"),
        ("OmegaConf", "import omegaconf; print(f'  Version: {omegaconf.__version__}')"),
        ("TQDM", "import tqdm; print(f'  Version: {tqdm.__version__}')"),
        
        # Logging/tracking
        ("Weights & Biases", "import wandb; print(f'  Version: {wandb.__version__}')"),
        ("TensorBoard", "import tensorboard; print(f'  Version: {tensorboard.__version__}')"),
        
        # Optional
        ("PoseBusters (optional)", "from posebusters import PoseBusters; print('  Available')"),
    ]
    
    for name, import_str in tests:
        try:
            print(f"Testing {name}...", end=" ")
            exec(import_str)
            print("✓")
        except ImportError as e:
            print(f"✗")
            failures.append((name, str(e)))
        except Exception as e:
            print(f"⚠ (imported but error: {e})")
    
    print()
    print("="*80)
    
    if failures:
        print("FAILED IMPORTS:")
        for name, error in failures:
            print(f"  ✗ {name}: {error}")
        print()
        print("Some packages failed to import. Please check the installation.")
        return False
    else:
        print("✓ All packages imported successfully!")
        print()
        print("Your UAAG2 Mac environment is ready for development!")
        return True

def test_uaag_imports():
    """Test UAAG2 package imports."""
    print()
    print("="*80)
    print("Testing UAAG2 Package Imports")
    print("="*80)
    print()
    
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    uaag_tests = [
        ("UAAG Data Module", "from uaag.data.uaag_dataset import UAAG2DataModule"),
        ("UAAG Trainer", "from uaag.equivariant_diffusion import Trainer"),
        ("UAAG Utils", "from uaag.utils import load_data, load_model"),
        ("UAAG Callbacks", "from uaag.callbacks.ema import ExponentialMovingAverage"),
        ("E3 Mol Diffusion", "from uaag.e3moldiffusion import coordsatomsbonds"),
    ]
    
    failures = []
    for name, import_str in uaag_tests:
        try:
            print(f"Testing {name}...", end=" ")
            exec(import_str)
            print("✓")
        except ImportError as e:
            print(f"✗ ({e})")
            failures.append((name, str(e)))
        except Exception as e:
            print(f"⚠ (error: {e})")
    
    print()
    if failures:
        print("Some UAAG2 imports failed (this is OK if files are missing):")
        for name, error in failures:
            print(f"  ⚠ {name}")
    else:
        print("✓ All UAAG2 imports successful!")
    
    return True

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("UAAG2 Mac Environment Verification")
    print("="*80 + "\n")
    
    # Test basic imports
    success = test_imports()
    
    # Test UAAG imports
    try:
        test_uaag_imports()
    except Exception as e:
        print(f"\nNote: UAAG imports test skipped or failed: {e}")
    
    print("\n" + "="*80)
    if success:
        print("✓ Environment setup complete and verified!")
        print("\nNext steps:")
        print("  1. Try running a small test script")
        print("  2. See MAC_DEVELOPMENT.md for local development workflow")
        print("  3. Use --gpus 0 flag for CPU-only training on Mac")
    else:
        print("✗ Some tests failed. Please review errors above.")
        sys.exit(1)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
