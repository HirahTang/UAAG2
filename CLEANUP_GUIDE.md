# Cleanup and Archival Script for .mol Files

## Overview

The `cleanup_mol_files.sh` script automates the process of:
1. **Evaluating** all generated molecules with PoseBusters
2. **Compressing** .mol files into tar.gz archives
3. **Deleting** .mol structure files while preserving important results

This is automatically run after post-processing in the `run_assay_postprocess.sh` script.

## Usage

### Automatic (via post-processing)
```bash
./submit_assay_jobs.sh A0A247D711_LISMN
# Post-processing will automatically run cleanup after all iterations complete
```

### Manual
```bash
sbatch cleanup_mol_files.sh <protein_id> <model_name> <num_samples>
```

Example:
```bash
sbatch cleanup_mol_files.sh A0A247D711_LISMN UAAG_model 1000
```

## What Gets Processed

For each iteration (0-4), the script processes:
```
/scratch/project_465002574/ProteinGymSampling/
└── runUAAG_model/
    ├── A0A247D711_LISMN_UAAG_model_1000_iter0/Samples/
    ├── A0A247D711_LISMN_UAAG_model_1000_iter1/Samples/
    ├── A0A247D711_LISMN_UAAG_model_1000_iter2/Samples/
    ├── A0A247D711_LISMN_UAAG_model_1000_iter3/Samples/
    └── A0A247D711_LISMN_UAAG_model_1000_iter4/Samples/
```

## Step-by-Step Process

### Step 1: PoseBusters Evaluation

Evaluates all .mol files using the `evaluate_mol_samples.py` script:

```bash
python scripts/evaluate_mol_samples.py <samples_dir> -o posebusters_evaluation.csv
```

Creates:
- `posebusters_evaluation.csv` - Detailed results for each molecule
- `posebusters_evaluation_summary.txt` - Overall statistics

### Step 2: Compression

Creates compressed archives of all molecule files:

```bash
find . -name "*.mol" -o -name "all_molecules.sdf" | tar -czf archive.tar.gz
```

Archives are saved to:
```
/scratch/project_465002574/UNAAGI_archives/UAAG_model/
├── A0A247D711_LISMN_UAAG_model_1000_iter0_mol_files.tar.gz
├── A0A247D711_LISMN_UAAG_model_1000_iter1_mol_files.tar.gz
├── ...
└── A0A247D711_LISMN_UAAG_model_1000_iter4_mol_files.tar.gz
```

### Step 3: Deletion

Deletes only molecule structure files:

**Files DELETED:**
- `*.mol` - Individual molecule files
- `all_molecules.sdf` - Consolidated molecule files
- `batch_*/` - Batch directories
- `iter_*/` - Iteration directories
- `final/` - Final subdirectories

**Files PRESERVED:**
- `*_aa_table.csv` - Amino acid identity tables
- `results.json` - Validity/connectivity metrics
- `aa_distribution.csv` - Amino acid distribution
- `posebusters_evaluation.csv` - PoseBusters results
- `posebusters_evaluation_summary.txt` - Summary statistics

### Step 4: Verification

Verifies cleanup success:
- Counts remaining .mol files (should be 0)
- Checks for preserved CSV/JSON files
- Reports disk usage after cleanup

## Example Directory Structure

### Before Cleanup
```
Samples/
├── ILE_10/
│   ├── iter_0/
│   │   ├── batch_0/
│   │   │   └── final/
│   │   │       └── ligand.mol          ← Will be deleted
│   │   └── ...
│   ├── ILE_10_aa_table.csv             ← Preserved
│   └── results.json                    ← Preserved
├── aa_distribution.csv                 ← Preserved
└── all_molecules.sdf                   ← Will be deleted
```

### After Cleanup
```
Samples/
├── ILE_10/
│   ├── ILE_10_aa_table.csv             ← Preserved
│   └── results.json                    ← Preserved
├── aa_distribution.csv                 ← Preserved
├── posebusters_evaluation.csv          ← New
└── posebusters_evaluation_summary.txt  ← New
```

### Archive Contents
```
A0A247D711_LISMN_UAAG_model_1000_iter0_mol_files.tar.gz contains:
├── ILE_10/iter_0/batch_0/final/ligand.mol
├── ILE_10/iter_0/batch_1/final/ligand.mol
├── ...
└── all_molecules.sdf (if using consolidated output)
```

## Disk Space Savings

Typical savings per iteration:
- **Before**: ~2-5 GB (thousands of .mol files)
- **After**: ~50-200 MB (CSV/JSON files only)
- **Archive**: ~200-500 MB (compressed)

**Overall**: ~80-90% disk space reduction while preserving all analysis results.

## Recovering .mol Files

To extract molecules from archive:

```bash
# Extract all
tar -xzf A0A247D711_LISMN_UAAG_model_1000_iter0_mol_files.tar.gz

# Extract specific file
tar -xzf archive.tar.gz path/to/specific/ligand.mol

# List contents
tar -tzf archive.tar.gz | less
```

## Integration with Workflow

The cleanup is automatically integrated into the post-processing workflow:

1. **Sampling** (50 array tasks) → Creates .mol files
2. **Post-processing** → Runs `post_analysis.py`
3. **Cleanup** (automatic) → Evaluates, compresses, deletes

To disable automatic cleanup, comment out the cleanup section in `run_assay_postprocess.sh`.

## Monitoring

Check cleanup logs:
```bash
tail -f /scratch/project_465002574/UAAG_logs/cleanup_<JOB_ID>.log
```

Check archives:
```bash
ls -lh /scratch/project_465002574/UNAAGI_archives/UAAG_model/
```

## Troubleshooting

### PoseBusters evaluation fails
- Check if PoseBusters is installed: `pip install posebusters`
- Script continues even if evaluation fails

### Archives not created
- Check disk space: `df -h /scratch/project_465002574/`
- Check permissions on archive directory

### Important files deleted
Archives contain all molecule files - extract to recover.
