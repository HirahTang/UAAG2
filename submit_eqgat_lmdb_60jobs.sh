#!/bin/bash
set -euo pipefail

# Submit 60 shard jobs, then submit merge job with afterok dependency.
# Customize through exported env vars before running this script.

ARRAY_JOB_ID=$(sbatch run_build_eqgat_lmdb_array_sbatch.sh | awk '{print $4}')
echo "Submitted shard array job: ${ARRAY_JOB_ID}"

MERGE_JOB_ID=$(sbatch --dependency=afterok:${ARRAY_JOB_ID} run_merge_eqgat_lmdb_sbatch.sh | awk '{print $4}')
echo "Submitted merge job: ${MERGE_JOB_ID} (afterok:${ARRAY_JOB_ID})"
