#!/bin/bash
#SBATCH --job-name=eqgat_lmdb_shard
#SBATCH --account=project_465002574
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-119%60
#SBATCH -o /scratch/project_465002574/UAAG_logs/eqgat_lmdb_shard_%A_%a.log
#SBATCH -e /scratch/project_465002574/UAAG_logs/eqgat_lmdb_shard_%A_%a.log

set -euo pipefail

echo "============================================================================"
echo "UAAG2 EQGAT LMDB Shard Build"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "============================================================================"

# Optional module setup for LUMI; safe to keep if modules are available.
module load LUMI || true
module load CrayEnv || true

PYTHON_BIN=${PYTHON_BIN:-/flash/project_465002574/unaagi_env/bin/python}
REPO_DIR=${REPO_DIR:-/flash/project_465002574/UAAG2_main}

PDB_DIR=${PDB_DIR:-/scratch/project_465002574/PDB/PDB_cleaned}
LATENT_ROOT_128=${LATENT_ROOT_128:-/scratch/project_465002574/PDB/PDB_128}
LATENT_ROOT_20=${LATENT_ROOT_20:-/scratch/project_465002574/PDB/PDB_20}

OUTPUT_DIR=${OUTPUT_DIR:-/scratch/project_465002574/PDB/uaag2_eqgat_lmdb_shards}
OUTPUT_PREFIX_BASE=${OUTPUT_PREFIX_BASE:-uaag2_eqgat_duallat_shard}
NUM_SHARDS=${NUM_SHARDS:-120}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID}

POCKET_RADIUS=${POCKET_RADIUS:-10.0}
EDGE_RADIUS=${EDGE_RADIUS:-8.0}

mkdir -p "$OUTPUT_DIR"

cd "$REPO_DIR"

echo "Python: $PYTHON_BIN"
echo "Repo: $REPO_DIR"
echo "PDB_DIR: $PDB_DIR"
echo "LATENT_ROOT_128: $LATENT_ROOT_128"
echo "LATENT_ROOT_20: $LATENT_ROOT_20"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "NUM_SHARDS: $NUM_SHARDS"
echo "SHARD_INDEX: $SHARD_INDEX"

time "$PYTHON_BIN" scripts/build_eqgat_lmdb_from_pdb.py \
  --pdb_dir "$PDB_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --output_prefix "${OUTPUT_PREFIX_BASE}_${SHARD_INDEX}" \
  --pocket_radius "$POCKET_RADIUS" \
  --edge_radius "$EDGE_RADIUS" \
  --latent_root_128 "$LATENT_ROOT_128" \
  --latent_root_20 "$LATENT_ROOT_20" \
  --num_shards "$NUM_SHARDS" \
  --shard_index "$SHARD_INDEX"

echo "End time: $(date)"
echo "Finished shard ${SHARD_INDEX}/${NUM_SHARDS}"
