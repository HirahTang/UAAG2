#!/usr/bin/env python3
"""
Split slurm_config.txt into 25 separate files, one per assay.
Each file contains: ArrayID, id, baseline, split, iteration
- iterations: 0-4
- splits: 0-9 for each iteration
- Total: 50 rows per file (5 iterations × 10 splits)
"""

import os
from pathlib import Path

def main():
    # Read the original config file
    config_file = Path("slurm_config/slurm_config.txt")
    output_dir = Path("slurm_config/assays")
    output_dir.mkdir(exist_ok=True)
    
    # Parse the original file to get unique assays
    assays = {}  # {id: baseline}
    
    with open(config_file, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip()
        
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    array_id, protein_id, baseline, split = parts
                    if protein_id not in assays:
                        assays[protein_id] = baseline
    
    print(f"Found {len(assays)} unique assays")
    
    # Create a file for each assay
    for idx, (protein_id, baseline) in enumerate(sorted(assays.items())):
        output_file = output_dir / f"{protein_id}.txt"
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("ArrayID\tid\tbaseline\tsplit\titeration\n")
            
            # Write data: 5 iterations × 10 splits = 50 rows
            array_id = 0
            for iteration in range(5):
                for split in range(10):
                    f.write(f"{array_id}\t{protein_id}\t{baseline}\t{split}\t{iteration}\n")
                    array_id += 1
        
        print(f"Created: {output_file} ({array_id} rows)")
    
    print(f"\nSummary:")
    print(f"  Total assays: {len(assays)}")
    print(f"  Rows per file: 50 (5 iterations × 10 splits)")
    print(f"  Output directory: {output_dir}")

if __name__ == "__main__":
    main()
