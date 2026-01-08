# UAAG2 Full Pipeline Usage Guide

## Overview
The pipeline has been updated to use 10 data partitions (splits 0-9) for better parallelization.

## Configuration Files

### 1. `slurm_config/slurm_config.txt`
- **Structure**: 251 lines (header + 250 data rows)
- **Format**: `ArrayID | ID | Baseline | Split`
- **Content**: 25 proteins × 10 splits = 250 array jobs
- **Split range**: 0-9 (10 partitions per protein)

Example entries:
```
ArrayID id              baseline                        split
0       A0A247D711_LISMN A0A247D711_LISMN_Stadelmann_2021.csv  0
3       DN7A_SACS2       DN7A_SACS2_baselines.csv              0
28      DN7A_SACS2       DN7A_SACS2_baselines.csv              1
53      DN7A_SACS2       DN7A_SACS2_baselines.csv              2
...
228     DN7A_SACS2       DN7A_SACS2_baselines.csv              9
249     ERBB2_HUMAN      ERBB2_HUMAN_Elazar_2016.csv           9
```

### 2. `generate_ligand.py` Partition Logic
Updated to cleanly handle 10 partitions:
- Uses `NUM_PARTITIONS = 10` constant
- Dynamic partition creation via loop
- Validates split_index range (0-9)
- Last partition gets any remaining residues

## Pipeline Execution

### Run Command
```bash
bash run_full_pipeline.sh
```

### What Happens
For each of 5 iterations (_1 to _5):
1. **Submits 250 sampling jobs** (array 0-249)
   - Each job reads protein ID, baseline, and split from config
   - Generates samples for 1/10th of the protein's residues
   - Run ID format: `{PROTEIN}_{MODEL}_variational_sampling_{NUM}_iter{1-5}_split{0-9}`

2. **Submits 250 analysis jobs** (array 0-249)
   - Depends on corresponding sampling job completion
   - Analyzes generated samples against baselines
   - Saves results to iteration-specific directories

### Total Job Count
- **Per iteration**: 500 jobs (250 sampling + 250 analysis)
- **All 5 iterations**: 2,500 jobs total

## Configuration Parameters

Edit these in `run_full_pipeline.sh`:
```bash
MODEL=Full_mask_5_virtual_node_mask_token_atomic_only_mask_diffusion_0917
CKPT_PATH=/path/to/checkpoint/last.ckpt
CONFIG_FILE=/home/qcx679/hantang/UAAG2/slurm_config/slurm_config.txt
NUM_SAMPLES=1000           # Samples per job
BATCH_SIZE=8               # Batch size for generation
VIRTUAL_NODE_SIZE=15       # Virtual node parameter
TOTAL_NUM=100              # For analysis normalization
```

## Output Structure
```
/datasets/biochem/unaagi/ProteinGymSampling/
└── run{PROTEIN}_{MODEL}_variational_sampling_{NUM}_iter{1-5}_split{0-9}/
    └── Samples/
        └── aa_distribution.csv

/home/qcx679/hantang/UAAG2/results/
└── {MODEL}_variational_sampling_{NUM}/
    └── {PROTEIN}_{MODEL}_variational_sampling_{NUM}_iter{1-5}_split{0-9}/
        ├── full_table.csv
        ├── results.csv
        └── *.png (plots)
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# Check logs
ls logs/sampling_iter_*.log
ls logs/analysis_iter_*.log

# Monitor specific iteration
squeue -u $USER | grep "UAAG_samp_1"  # Iteration 1 sampling
squeue -u $USER | grep "UAAG_anal_3"  # Iteration 3 analysis
```

## Key Improvements

1. **Better Parallelization**: 10 splits vs previous 5 = more efficient GPU usage
2. **Config-driven**: Single source of truth for protein/baseline mappings
3. **Cleaner Code**: No hardcoded if-elif chains in partition logic
4. **Scalable**: Easy to add more proteins or change split count
5. **Traceable**: Each job has unique run ID with protein and split info
