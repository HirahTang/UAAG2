#!/bin/bash
set -euo pipefail

# Demo submit: sample a small random subset of PDBs, then run EQGAT LMDB build+merge.
#
# Usage:
#   ./submit_eqgat_lmdb_demo_subset.sh [SAMPLE_SIZE] [SOURCE_PDB_DIR] [OUTPUT_LMDB_NAME]
#
# Examples:
#   ./submit_eqgat_lmdb_demo_subset.sh
#   ./submit_eqgat_lmdb_demo_subset.sh 200 /scratch/project_465002574/PDB/PDB_cleaned uaag2_eqgat_demo_200
#
# Optional env vars:
#   NUM_SHARDS=10               # number of array shards for demo
#   MAX_CONCURRENT_SHARDS=5     # array concurrency cap
#   OUTPUT_DIR=/scratch/project_465002574/PDB/uaag2_eqgat_lmdb_shards
#   DEMO_ROOT=/scratch/project_465002574/PDB/demo_subsets
#   FILE_GLOB="*.pdb"          # file pattern inside source dir
#   PYTHON_SUBSET_BIN=python3    # python used on login node for random subset prep
#   PYTHON_BIN=/flash/project_465002574/unaagi_env/bin/python
#   REPO_DIR=/flash/project_465002574/UAAG2_main
#   LATENT_ROOT_128=/scratch/project_465002574/PDB/PDB_128
#   LATENT_ROOT_20=/scratch/project_465002574/PDB/PDB_20

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,40p' "$0"
  exit 0
fi

SAMPLE_SIZE="${1:-200}"
SOURCE_PDB_DIR="${2:-/scratch/project_465002574/PDB/PDB_cleaned}"
OUTPUT_LMDB_NAME_RAW="${3:-uaag2_eqgat_demo_${SAMPLE_SIZE}}"

if ! [[ "$SAMPLE_SIZE" =~ ^[0-9]+$ ]] || [[ "$SAMPLE_SIZE" -lt 1 ]]; then
  echo "Error: SAMPLE_SIZE must be a positive integer, got: $SAMPLE_SIZE" >&2
  exit 1
fi

OUTPUT_LMDB_NAME="${OUTPUT_LMDB_NAME_RAW%.lmdb}"
if [[ -z "$OUTPUT_LMDB_NAME" ]]; then
  echo "Error: OUTPUT_LMDB_NAME cannot be empty." >&2
  exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-10}"
MAX_CONCURRENT_SHARDS="${MAX_CONCURRENT_SHARDS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_465002574/PDB/uaag2_eqgat_lmdb_shards}"
DEMO_ROOT="${DEMO_ROOT:-/scratch/project_465002574/PDB/demo_subsets}"
FILE_GLOB="${FILE_GLOB:-*.pdb}"
PYTHON_SUBSET_BIN="${PYTHON_SUBSET_BIN:-python3}"

# Runtime env consumed by run_build_eqgat_lmdb_array_sbatch.sh
PYTHON_BIN="${PYTHON_BIN:-/flash/project_465002574/unaagi_env/bin/python}"
REPO_DIR="${REPO_DIR:-/flash/project_465002574/UAAG2_main}"
LATENT_ROOT_128="${LATENT_ROOT_128:-/scratch/project_465002574/PDB/PDB_128}"
LATENT_ROOT_20="${LATENT_ROOT_20:-/scratch/project_465002574/PDB/PDB_20}"

if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "Error: NUM_SHARDS must be a positive integer, got: $NUM_SHARDS" >&2
  exit 1
fi

if ! [[ "$MAX_CONCURRENT_SHARDS" =~ ^[0-9]+$ ]] || [[ "$MAX_CONCURRENT_SHARDS" -lt 1 ]]; then
  echo "Error: MAX_CONCURRENT_SHARDS must be a positive integer, got: $MAX_CONCURRENT_SHARDS" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_PDB_DIR" ]]; then
  echo "Error: source folder not found: $SOURCE_PDB_DIR" >&2
  exit 1
fi
if ! command -v "$PYTHON_SUBSET_BIN" >/dev/null 2>&1; then
  echo "Error: subset prep python not found: $PYTHON_SUBSET_BIN" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEMO_INPUT_DIR="${DEMO_ROOT}/pdb_subset_${SAMPLE_SIZE}_${TIMESTAMP}"
OUTPUT_PREFIX_BASE="${OUTPUT_LMDB_NAME}_shard"
ARRAY_SPEC="0-$((NUM_SHARDS - 1))%${MAX_CONCURRENT_SHARDS}"

mkdir -p "$DEMO_INPUT_DIR"

echo "Preparing random subset..."
echo "  SOURCE_PDB_DIR: $SOURCE_PDB_DIR"
echo "  DEMO_INPUT_DIR: $DEMO_INPUT_DIR"
echo "  SAMPLE_SIZE: $SAMPLE_SIZE"
echo "  FILE_GLOB: $FILE_GLOB"
echo "  PYTHON_BIN (job runtime): $PYTHON_BIN"
echo "  REPO_DIR (job runtime): $REPO_DIR"
echo "  LATENT_ROOT_128: $LATENT_ROOT_128"
echo "  LATENT_ROOT_20: $LATENT_ROOT_20"

"$PYTHON_SUBSET_BIN" - "$SOURCE_PDB_DIR" "$DEMO_INPUT_DIR" "$SAMPLE_SIZE" "$FILE_GLOB" <<'PY'
import random
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
sample_size = int(sys.argv[3])
file_glob = sys.argv[4]

candidates = sorted([p for p in source.glob(file_glob) if p.is_file()])
if not candidates:
    candidates = sorted([p for p in source.iterdir() if p.is_file() and not p.name.startswith('.')])

if not candidates:
    raise SystemExit(f"No input files found under {source}")

k = min(sample_size, len(candidates))
selected = random.sample(candidates, k)

for src in selected:
    dst = target / src.name
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)

print(f"Selected {k} files out of {len(candidates)}")
print(f"Subset directory: {target}")
PY

echo "Submitting demo shard array..."
ARRAY_JOB_ID=$(sbatch --parsable \
  --array="$ARRAY_SPEC" \
  --export=ALL,PDB_DIR="$DEMO_INPUT_DIR",OUTPUT_DIR="$OUTPUT_DIR",OUTPUT_PREFIX_BASE="$OUTPUT_PREFIX_BASE",FINAL_PREFIX="$OUTPUT_LMDB_NAME",NUM_SHARDS="$NUM_SHARDS",PYTHON_BIN="$PYTHON_BIN",REPO_DIR="$REPO_DIR",LATENT_ROOT_128="$LATENT_ROOT_128",LATENT_ROOT_20="$LATENT_ROOT_20" \
  run_build_eqgat_lmdb_array_sbatch.sh)
if [[ -z "${ARRAY_JOB_ID}" || ! "${ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "Error: failed to submit shard array job. sbatch returned: ${ARRAY_JOB_ID}" >&2
  exit 1
fi

echo "Submitted shard array job: ${ARRAY_JOB_ID} (array=${ARRAY_SPEC})"

echo "Submitting merge job..."
MERGE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:${ARRAY_JOB_ID} \
  --export=ALL,PDB_DIR="$DEMO_INPUT_DIR",OUTPUT_DIR="$OUTPUT_DIR",OUTPUT_PREFIX_BASE="$OUTPUT_PREFIX_BASE",FINAL_PREFIX="$OUTPUT_LMDB_NAME",NUM_SHARDS="$NUM_SHARDS",PYTHON_BIN="$PYTHON_BIN",REPO_DIR="$REPO_DIR",LATENT_ROOT_128="$LATENT_ROOT_128",LATENT_ROOT_20="$LATENT_ROOT_20" \
  run_merge_eqgat_lmdb_sbatch.sh)
if [[ -z "${MERGE_JOB_ID}" || ! "${MERGE_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "Error: failed to submit merge job. sbatch returned: ${MERGE_JOB_ID}" >&2
  exit 1
fi

echo "Submitted merge job: ${MERGE_JOB_ID} (afterok:${ARRAY_JOB_ID})"
echo "Final LMDB target: ${OUTPUT_DIR}/${OUTPUT_LMDB_NAME}.lmdb"
