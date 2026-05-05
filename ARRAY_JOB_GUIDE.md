# Array Job Submission Guide

## Quick Start

### Submit a Single Assay
```bash
./submit_assay_jobs.sh ENVZ_ECOLI
```
This submits 50 array tasks (5 iterations × 10 splits) for ENVZ_ECOLI.

### Submit All Assays
```bash
./submit_assay_jobs.sh all
```
This submits 1,250 array tasks total (25 assays × 50 tasks each).

### Direct Submission (Advanced)
```bash
sbatch run_assay_array.sh slurm_config/assays/DN7A_SACS2.txt
```

## File Structure

### Configuration Files
```
slurm_config/assays/
├── A0A247D711_LISMN.txt    (50 rows: 5 iterations × 10 splits)
├── ENVZ_ECOLI.txt           (50 rows: 5 iterations × 10 splits)
├── DN7A_SACS2.txt           (50 rows: 5 iterations × 10 splits)
└── ...                      (22 more assays)
```

Each config file contains:
```
ArrayID  id          baseline                     split  iteration
0        DN7A_SACS2  DN7A_SACS2_baselines.csv    0      0
1        DN7A_SACS2  DN7A_SACS2_baselines.csv    1      0
...
49       DN7A_SACS2  DN7A_SACS2_baselines.csv    9      4
```

### Job Scripts
- `run_assay_array.sh` - Main SLURM array job script
- `submit_assay_jobs.sh` - Convenient wrapper for submission

## Array Job Details

### SLURM Configuration
```bash
#SBATCH --array=0-49          # 50 tasks per assay
#SBATCH --gpus-per-node=1     # 1 GPU per task
#SBATCH --mem=60G             # 60GB RAM per task
#SBATCH --time=2-00:00:00     # 2 days max per task
```

### Task Assignment
- **Array Task ID 0-9**: Iteration 0, Splits 0-9
- **Array Task ID 10-19**: Iteration 1, Splits 0-9
- **Array Task ID 20-29**: Iteration 2, Splits 0-9
- **Array Task ID 30-39**: Iteration 3, Splits 0-9
- **Array Task ID 40-49**: Iteration 4, Splits 0-9

### Output Structure
```
/scratch/project_465002574/ProteinGymSampling/
└── run<MODEL>/<PROTEIN>_<MODEL>_<NUM_SAMPLES>_iter<I>_split<S>/
    └── Samples/
        ├── all_molecules.sdf     # Consolidated output (if using --consolidate-to-sdf)
        ├── aa_distribution.csv   # Post-processing result
        └── results.json          # Validity/connectivity metrics
```

### Log Files
```
/scratch/project_465002574/UAAG_logs/
└── array_<JOB_ID>_<TASK_ID>.log
```

## Monitoring Jobs

### Check Queue Status
```bash
squeue -u $USER
```

### Check Specific Job
```bash
squeue -j <JOB_ID>
```

### Count Running/Pending Tasks
```bash
squeue -u $USER -h | wc -l
```

### Monitor Array Job Progress
```bash
squeue -j <JOB_ID> -t RUNNING,PENDING
```

## Examples

### Submit One Assay
```bash
./submit_assay_jobs.sh ENVZ_ECOLI
# Output:
# Submitting array job for: ENVZ_ECOLI
#   Config: slurm_config/assays/ENVZ_ECOLI.txt
#   Tasks: 0-49 (5 iterations × 10 splits)
#   ✓ Submitted: Job 12345 (50 array tasks)
```

### Submit Three Specific Assays
```bash
./submit_assay_jobs.sh ENVZ_ECOLI
./submit_assay_jobs.sh DN7A_SACS2
./submit_assay_jobs.sh CCDB_ECOLI
```

### List Available Assays
```bash
./submit_assay_jobs.sh
# Shows list of all 25 available assays
```

## Resource Usage

### Per Assay (50 tasks)
- **GPU hours**: 50 tasks × 2 days max = 100 GPU-days
- **Storage**: ~1-5 GB per task (with SDF consolidation)

### All Assays (1,250 tasks)
- **GPU hours**: 1,250 tasks × 2 days max = 2,500 GPU-days
- **Storage**: ~1.25-6.25 TB total

## Troubleshooting

### Check Failed Tasks
```bash
sacct -j <JOB_ID> --format=JobID,State,ExitCode
```

### Resubmit Failed Tasks
```bash
sbatch --array=5,12,23 run_assay_array.sh slurm_config/assays/ENVZ_ECOLI.txt
```

### Cancel All Tasks
```bash
scancel <JOB_ID>
```

### Cancel Specific Array Tasks
```bash
scancel <JOB_ID>_5,<JOB_ID>_12
```

## Advanced Usage

### Custom Parameters
Edit `run_assay_array.sh` to modify:
- `NUM_SAMPLES` (default: 1000)
- `BATCH_SIZE` (default: 8)
- `VIRTUAL_NODE_SIZE` (default: 15)
- Time limits, memory, etc.

### Job Dependencies
Submit analysis jobs after sampling completes:
```bash
SAMPLE_JOB=$(sbatch --parsable run_assay_array.sh slurm_config/assays/ENVZ_ECOLI.txt)
sbatch --dependency=afterok:${SAMPLE_JOB} analysis_job.sh
```
