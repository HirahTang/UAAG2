#!/bin/bash

set -euo pipefail

# Remove water molecules (HOH/WAT/H2O) from all PDBs.
#
# Defaults target your LUMI scratch paths, but can be overridden with args:
#   ./extract_pdb_water.sh [INPUT_DIR] [OUTPUT_DIR]

INPUT_DIR=${1:-/scratch/project_465002574/PDB/PDB_cleaned}
OUTPUT_DIR=${2:-/scratch/project_465002574/PDB/PDB_water}

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: input directory does not exist: $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Input dir:    $INPUT_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo "Start time:   $(date)"

shopt -s nullglob
mapfile -t pdb_files < <(find "$INPUT_DIR" -type f \( -name "*.pdb" -o -name "*.PDB" \) | sort)

if [[ ${#pdb_files[@]} -eq 0 ]]; then
  echo "No PDB files found in: $INPUT_DIR"
  exit 0
fi

processed=0
files_with_removed_water=0
total_removed_water_atoms=0

for pdb in "${pdb_files[@]}"; do
  rel_path=${pdb#"$INPUT_DIR"/}

  out_pdb="$OUTPUT_DIR/$rel_path"

  mkdir -p "$(dirname "$out_pdb")"

  awk -v out_pdb="$out_pdb" '
    function trim(s) {
      gsub(/^ +| +$/, "", s)
      return s
    }

    {
      rec = substr($0, 1, 6)
      is_coord = (rec == "ATOM  " || rec == "HETATM")

      if (is_coord) {
        resn = trim(substr($0, 18, 3))
        atom = trim(substr($0, 13, 4))

        is_water = (resn == "HOH" || resn == "WAT" || resn == "H2O")

        if (is_water) {
          removed_water_atoms++
          next
        }
      }

      print $0 >> out_pdb
    }

    END {
      print removed_water_atoms + 0
    }
  ' "$pdb" > /tmp/water_count.$$ 

  water_count=$(cat /tmp/water_count.$$)
  rm -f /tmp/water_count.$$

  processed=$((processed + 1))
  total_removed_water_atoms=$((total_removed_water_atoms + water_count))

  if [[ "$water_count" -gt 0 ]]; then
    files_with_removed_water=$((files_with_removed_water + 1))
  fi

done

echo "Processed files:        $processed"
echo "Files with water found: $files_with_removed_water"
echo "Removed water atoms:    $total_removed_water_atoms"
echo "End time:               $(date)"
