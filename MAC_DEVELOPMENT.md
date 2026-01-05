# UAAG2 Local Development Guide (Mac)

This guide helps you set up UAAG2 on your local Mac for development and small-scale testing before running large experiments on the cluster.

---

## Quick Start

### 1. Run the Automated Setup Script

```bash
cd /path/to/UAAG2
bash setup_mac_env.sh
```

This will:
- Create a conda environment named `uaag2_mac`
- Install all required packages (CPU versions for Mac)
- Verify the installation

### 2. Activate the Environment

```bash
conda activate uaag2_mac
```

### 3. Verify Installation

```bash
python -c "import torch; import torch_geometric; import rdkit; print('Ready!')"
```

---

## Manual Setup (Alternative)

If you prefer manual installation or the script fails:

### Step 1: Create Environment

```bash
conda create -n uaag2_mac python=3.10 -y
conda activate uaag2_mac
```

### Step 2: Install PyTorch (CPU)

```bash
pip install torch torchvision torchaudio
```

### Step 3: Install PyTorch Geometric

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv
```

### Step 4: Install Chemistry Tools

```bash
conda install -c conda-forge rdkit openbabel -y
pip install biopython
```

### Step 5: Install Dependencies

```bash
# ML/DL frameworks
pip install pytorch-lightning==2.0.0
pip install torch-ema wandb tensorboard

# Scientific computing
pip install numpy scipy scikit-learn pandas networkx

# Utilities
pip install tqdm pyyaml omegaconf lmdb ipython

# Optional: Evaluation tools
pip install posebusters
```

---

## Local Development Workflow

### 1. Testing with Small Datasets

Create a test dataset with a few samples:

```bash
# Use a subset of your data
python scripts/construct_aa_eqgat.py \
    --input-dir ./data/test_small \
    --output-file ./data/test_small.pt
```

### 2. Quick Training Test

Run a quick training sanity check:

```bash
python scripts/run_train.py \
    --data-path ./data/test_small.pt \
    --batch-size 2 \
    --num-epochs 2 \
    --gpus 0 \
    --id local_test \
    --save-dir ./test_runs
```

**Note:** On Mac without GPU, set `--gpus 0` to use CPU.

### 3. Testing Generation

Test molecule generation:

```bash
python scripts/generate_ligand.py \
    --load-ckpt ./test_runs/checkpoint.ckpt \
    --num-samples 5 \
    --save-dir ./test_outputs
```

### 4. Evaluate Results

```bash
python scripts/evaluate_mol_samples.py \
    ./test_outputs \
    -o test_evaluation.csv
```

---

## Mac vs Cluster Differences

| Aspect | Mac (Local) | Cluster (GPU) |
|--------|-------------|---------------|
| **PyTorch** | CPU version | CUDA 11.8+ version |
| **Batch Size** | 1-4 (small) | 8-32+ (large) |
| **Training Speed** | Slow (CPU) | Fast (GPU) |
| **Data Size** | Small subset | Full dataset |
| **Purpose** | Testing, debugging | Production runs |

---

## Transitioning to Cluster

### 1. Test Locally First

Always test your code on Mac with small data before submitting cluster jobs:

```bash
# Local test (Mac)
conda activate uaag2_mac
python scripts/run_train.py --batch-size 2 --num-epochs 1 --gpus 0
```

### 2. Prepare for Cluster

Once local tests pass, prepare cluster job:

```bash
# On cluster, use GPU environment
conda activate uaag2  # or uaag2_rocm for AMD GPUs
```

### 3. Adjust Parameters for GPU

```bash
# Cluster run (GPU)
python scripts/run_train.py \
    --batch-size 32 \
    --num-epochs 1000 \
    --gpus 1 \
    --logger-type wandb
```

---

## Development Tips

### Working with Small Datasets

Create a development dataset:

```bash
# Take first 100 samples
head -n 100 data/full_dataset.csv > data/dev_dataset.csv
```

### Fast Iteration

Use these flags for quick testing:

```bash
python scripts/run_train.py \
    --batch-size 2 \
    --num-epochs 2 \
    --fast-dev-run  # PyTorch Lightning flag for quick test
```

### Debugging

Enable detailed logging:

```bash
python scripts/run_train.py \
    --log-every-n-steps 1 \
    --verbose
```

### Memory Management on Mac

If you run out of memory:

```bash
# Reduce batch size
--batch-size 1

# Limit number of workers
--num-workers 0

# Use gradient accumulation
--accumulate-grad-batches 4
```

---

## Common Issues & Solutions

### Issue 1: Out of Memory

**Solution:** Reduce batch size and enable gradient accumulation

```bash
python scripts/run_train.py --batch-size 1 --accumulate-grad-batches 8
```

### Issue 2: Slow Training

**Expected:** CPU training is naturally slower. Use for testing only, not full training.

### Issue 3: Package Conflicts

**Solution:** Recreate environment

```bash
conda deactivate
conda env remove -n uaag2_mac -y
bash setup_mac_env.sh
```

### Issue 4: RDKit/OpenBabel Issues

**Solution:** Install via conda-forge

```bash
conda install -c conda-forge rdkit openbabel -y
```

---

## Project Structure for Development

```
UAAG2/
├── scripts/               # Main scripts
│   ├── run_train.py      # Training
│   ├── generate_ligand.py # Generation
│   └── evaluate_mol_samples.py # Evaluation
├── uaag/                  # Core package
│   ├── data/             # Data loading
│   ├── diffusion/        # Diffusion models
│   └── e3moldiffusion/   # E3 equivariant models
├── data/                  # Data directory
│   ├── dev_small/        # Small dev dataset
│   └── full/             # Full dataset (for cluster)
├── configs/               # Configuration files
├── test_runs/            # Local test outputs
└── setup_mac_env.sh      # This setup script
```

---

## Recommended Development Cycle

1. **Write/Modify Code Locally** (on Mac)
   ```bash
   # Edit files in your IDE
   vim scripts/my_experiment.py
   ```

2. **Test Locally with Small Data**
   ```bash
   conda activate uaag2_mac
   python scripts/my_experiment.py --data small_test --gpus 0
   ```

3. **Debug Issues**
   ```bash
   # Use IPython for debugging
   ipython
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add new feature"
   git push
   ```

5. **Run on Cluster**
   ```bash
   # SSH to cluster
   ssh your_cluster
   
   # Pull latest code
   cd UAAG2
   git pull
   
   # Activate GPU environment
   conda activate uaag2
   
   # Submit job
   sbatch run_train.sh
   ```

---

## Environment Variables

Add to your `~/.zshrc` or `~/.bash_profile`:

```bash
# UAAG2 development
export UAAG2_DEV=1
export UAAG2_DATA_DIR=~/UAAG2/data
export UAAG2_OUTPUT_DIR=~/UAAG2/outputs

# For local testing
alias uaag2-activate='conda activate uaag2_mac'
alias uaag2-test='python scripts/run_train.py --gpus 0 --batch-size 2'
```

---

## Next Steps

1. ✅ Set up environment: `bash setup_mac_env.sh`
2. ✅ Test installation: Run verification script
3. 📝 Prepare small test dataset
4. 🧪 Run local training test
5. 🚀 Move to cluster for full experiments

For cluster-specific setup, see [SETUP.md](SETUP.md)

---

## Support

- **Local Issues:** Check this guide
- **Cluster Issues:** See SETUP.md
- **Code Questions:** Check repository documentation
- **Package Issues:** Recreate environment with setup script
