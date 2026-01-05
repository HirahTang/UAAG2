# UAAG2 Mac Environment Setup - Complete!

## ✓ Installation Summary

Your UAAG2 Mac development environment has been successfully set up!

### What Was Installed

1. **Miniconda** (already present, initialized for zsh)
2. **Conda Environment**: `uaag2_mac` with Python 3.10
3. **Core ML/DL Packages**:
   - PyTorch 2.9.1 (CPU version for Apple Silicon)
   - PyTorch Geometric 2.7.0
   - PyTorch Lightning 2.0.0
   - torch-scatter, torch-sparse, torch-cluster, torch-spline-conv

4. **Chemistry Libraries**:
   - RDKit 2025.09.4
   - OpenBabel 3.1.1
   - BioPython 1.86

5. **Scientific Computing**:
   - NumPy 2.2.6
   - SciPy 1.15.3
   - Pandas 2.3.3
   - NetworkX 3.4.2
   - Scikit-learn 1.7.2

6. **Utilities**:
   - LMDB 1.7.5
   - OmegaConf 2.3.0
   - YAML, TQDM, IPython

7. **Logging/Tracking**:
   - Weights & Biases 0.23.1
   - TensorBoard 2.20.0

8. **Optional**:
   - PoseBusters 0.6.3

### Fixed Issues

1. **OpenMP Conflict**: Added `export KMP_DUPLICATE_LIB_OK=TRUE` to `~/.zshrc`
2. **Import Errors**: Fixed deprecated imports in `uaag/utils.py`:
   - Removed `from torch_geometric.utils.sort_edge_index import sort_edge_index`
   - Removed `from torch_geometric.utils.subgraph import subgraph`
   - Consolidated to `from torch_geometric.utils import ...`

3. **PyTorch Geometric Extensions**: Installed torch-scatter, torch-sparse, etc. with `--no-build-isolation` flag

---

## 🚀 Quick Start

### Activate Environment

```bash
# Option 1: Use the helper script
source activate_mac.sh

# Option 2: Manual activation
conda activate uaag2_mac
export KMP_DUPLICATE_LIB_OK=TRUE  # Already in your .zshrc
```

### Test Installation

```bash
python test_environment.py
```

### Run a Test Script

```bash
# Example: Test with small data (when you have data)
conda activate uaag2_mac
python scripts/run_train.py \
    --data-path ./data/test_small.pt \
    --batch-size 2 \
    --num-epochs 2 \
    --gpus 0 \
    --id local_test \
    --save-dir ./test_runs
```

**Important**: Always use `--gpus 0` on Mac for CPU-only training!

---

## 📁 New Files Created

1. **activate_mac.sh** - Quick environment activation script
2. **test_environment.py** - Comprehensive environment verification script
3. **SETUP_SUMMARY.md** - This file

---

## ⚙️ Environment Configuration

### Shell Configuration (~/.zshrc)

Added the following to fix OpenMP conflicts:

```bash
# UAAG2 Mac Development - Fix OpenMP conflict
export KMP_DUPLICATE_LIB_OK=TRUE
```

This takes effect in new terminal sessions. For the current session, the environment variables are already set.

---

## 🔧 Development Workflow

### 1. Local Development Cycle

```bash
# 1. Activate environment
conda activate uaag2_mac

# 2. Make code changes
vim scripts/my_experiment.py

# 3. Test with small data (CPU only)
python scripts/my_experiment.py --gpus 0 --batch-size 2

# 4. Debug if needed
ipython  # Interactive Python with all packages available
```

### 2. Prepare for Cluster

Once local tests pass:

```bash
# On your Mac - commit changes
git add .
git commit -m "Tested feature locally"
git push

# On cluster - pull and run with GPU
ssh your_cluster
cd UAAG2
git pull
conda activate uaag2  # GPU environment
sbatch run_train.sh
```

---

## 📝 Key Differences: Mac vs Cluster

| Aspect | Mac (Local) | Cluster (GPU) |
|--------|-------------|---------------|
| **Environment** | `uaag2_mac` | `uaag2` or `uaag2_rocm` |
| **PyTorch** | CPU (2.9.1) | CUDA 11.8+ or ROCm |
| **Batch Size** | 1-4 (small) | 8-32+ (large) |
| **Training** | Slow (testing only) | Fast (production) |
| **GPU Flag** | `--gpus 0` | `--gpus 1` or more |
| **Data** | Small subset | Full dataset |

---

## 🐛 Troubleshooting

### Issue: OpenMP Error
**Solution**: Already fixed! The environment variable is in your `.zshrc`. If you still see it:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Issue: Import Errors
**Solution**: Already fixed! The problematic imports in `uaag/utils.py` have been corrected.

### Issue: Out of Memory
**Solution**: Reduce batch size
```bash
python scripts/run_train.py --batch-size 1 --accumulate-grad-batches 8
```

### Issue: Missing Data Files
**Expected**: Some imports may show warnings about missing data files (like `data/aa_graph.json`). This is normal if you haven't prepared your data yet.

### Issue: Package Conflicts
**Solution**: Recreate environment
```bash
conda deactivate
conda env remove -n uaag2_mac -y
bash setup_mac_env.sh
```

---

## 📚 Next Steps

1. **✓ Environment Setup** - Complete!
2. **Prepare Test Data** - Create a small subset for local testing
3. **Run Local Tests** - Test scripts with `--gpus 0`
4. **Develop Features** - Make changes and test locally
5. **Deploy to Cluster** - Run full experiments on GPU

### Recommended Commands to Try

```bash
# 1. Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 2. Check RDKit
python -c "from rdkit import Chem; print('RDKit works!')"

# 3. Check UAAG imports
python -c "from uaag.utils import load_data; print('UAAG utils works!')"

# 4. Run IPython for interactive testing
ipython
```

---

## 📖 Documentation

- **Local Development**: See [MAC_DEVELOPMENT.md](MAC_DEVELOPMENT.md)
- **Cluster Setup**: See [SETUP.md](SETUP.md)
- **Pipeline Usage**: See [PIPELINE_USAGE.md](PIPELINE_USAGE.md)

---

## ✅ Verification Checklist

- [x] Conda installed and initialized
- [x] Python 3.10 environment created
- [x] PyTorch 2.9.1 (CPU) installed
- [x] PyTorch Geometric 2.7.0 installed
- [x] torch-scatter and extensions installed
- [x] RDKit and OpenBabel installed
- [x] All scientific packages installed
- [x] OpenMP conflict fixed
- [x] Import errors in uaag/utils.py fixed
- [x] Environment verified with test script
- [x] Helper scripts created

---

## 🎉 Success!

Your UAAG2 Mac development environment is fully set up and ready to use!

**Quick activation**: `source activate_mac.sh`

**Test everything**: `python test_environment.py`

Happy coding! 🚀
