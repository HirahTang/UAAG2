# SLURM Config Files - Assay Index

This directory contains 25 separate configuration files, one for each assay.

## File Structure

Each file contains 50 rows with the following columns:
- **ArrayID**: 0-49 (unique identifier for each job)
- **id**: Protein/assay identifier
- **baseline**: Baseline CSV file for comparison
- **split**: Data split number (0-9)
- **iteration**: Iteration number (0-4)

## Organization

- **5 iterations** (0-4)
- **10 splits per iteration** (0-9)
- **Total: 50 array tasks per assay**

## Assay Files

1. A0A247D711_LISMN.txt
2. AICDA_HUMAN.txt
3. ARGR_ECOLI.txt
4. B2L11_HUMAN.txt
5. CCDB_ECOLI.txt
6. DLG4_RAT.txt
7. DN7A_SACS2.txt
8. ENVZ_ECOLI.txt
9. ENV_HV1B9.txt
10. ERBB2_HUMAN.txt
11. FKBP3_HUMAN.txt
12. HCP_LAMBD.txt
13. IF1_ECOLI.txt
14. ILF3_HUMAN.txt
15. OTU7A_HUMAN.txt
16. PKN1_HUMAN.txt
17. RS15_GEOSE.txt
18. SBI_STAAM.txt
19. SCIN_STAAR.txt
20. SOX30_HUMAN.txt
21. SQSTM_MOUSE.txt
22. SUMO1_HUMAN.txt
23. TAT_HV1BR.txt
24. VG08_BPP22.txt
25. VRPI_BPT7.txt

## Usage

To use these in SLURM scripts, reference the specific assay file:

```bash
CONFIG_FILE=slurm_config/assays/DN7A_SACS2.txt
ID=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG_FILE})
SPLIT_INDEX=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG_FILE})
ITERATION=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $5}' ${CONFIG_FILE})
```

## Migration from Original Config

The original `slurm_config.txt` has been split into these individual files with the addition of an `iteration` column to better organize the experimental runs.
