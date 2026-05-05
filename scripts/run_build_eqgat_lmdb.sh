#!/usr/bin/env bash
set -euo pipefail

PDB_DIR=""
OUTPUT_DIR=""
OUTPUT_PREFIX="uaag_eqgat"
POCKET_RADIUS="10.0"
EDGE_RADIUS="8.0"
LATENT_ROOT_128="/scratch/project_465002574/PDB/PDB_128"
LATENT_ROOT_20="/scratch/project_465002574/PDB/PDB_20"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdb_dir)
      PDB_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output_prefix)
      OUTPUT_PREFIX="$2"
      shift 2
      ;;
    --pocket_radius)
      POCKET_RADIUS="$2"
      shift 2
      ;;
    --edge_radius)
      EDGE_RADIUS="$2"
      shift 2
      ;;
    --latent_root_128)
      LATENT_ROOT_128="$2"
      shift 2
      ;;
    --latent_root_20)
      LATENT_ROOT_20="$2"
      shift 2
      ;;
    --python_bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$PDB_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --pdb_dir <pdb_folder> --output_dir <output_folder> [options]"
  exit 1
fi

"$PYTHON_BIN" scripts/build_eqgat_lmdb_from_pdb.py \
  --pdb_dir "$PDB_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --output_prefix "$OUTPUT_PREFIX" \
  --pocket_radius "$POCKET_RADIUS" \
  --edge_radius "$EDGE_RADIUS" \
  --latent_root_128 "$LATENT_ROOT_128" \
  --latent_root_20 "$LATENT_ROOT_20"
