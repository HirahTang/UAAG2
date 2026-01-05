#!/usr/bin/env python3
"""
Tool script to convert .mol files to .sdf format.

Usage:
    python mol_2_sdf.py input.mol output.sdf
    python mol_2_sdf.py input_dir/ output_dir/  # Convert all .mol files in directory
    python mol_2_sdf.py input.mol               # Output to input.sdf
"""

import os
import sys
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import SDWriter


def convert_mol_to_sdf(mol_file, sdf_file=None, verbose=True):
    """
    Convert a single .mol file to .sdf format.
    
    Args:
        mol_file (str): Path to input .mol file
        sdf_file (str, optional): Path to output .sdf file. If None, uses same name with .sdf extension
        verbose (bool): Print status messages
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    mol_path = Path(mol_file)
    
    if not mol_path.exists():
        print(f"Error: Input file '{mol_file}' not found")
        return False
    
    # Determine output file path
    if sdf_file is None:
        sdf_path = mol_path.with_suffix('.sdf')
    else:
        sdf_path = Path(sdf_file)
    
    try:
        # Read molecule from .mol file
        mol = Chem.MolFromMolFile(str(mol_path), removeHs=False, strictParsing=False)
        
        if mol is None:
            print(f"Error: Failed to parse molecule from '{mol_file}'")
            return False
        
        # Write molecule to .sdf file
        writer = SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        
        if verbose:
            print(f"✓ Converted: {mol_file} → {sdf_path}")
        
        return True
        
    except Exception as e:
        print(f"Error converting '{mol_file}': {str(e)}")
        return False


def convert_directory(input_dir, output_dir=None, verbose=True):
    """
    Convert all .mol files in a directory to .sdf format.
    
    Args:
        input_dir (str): Path to input directory
        output_dir (str, optional): Path to output directory. If None, uses input_dir
        verbose (bool): Print status messages
        
    Returns:
        tuple: (num_success, num_failed)
    """
    input_path = Path(input_dir)
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return (0, 0)
    
    # Determine output directory
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .mol files
    mol_files = list(input_path.glob('*.mol'))
    
    if not mol_files:
        print(f"No .mol files found in '{input_dir}'")
        return (0, 0)
    
    if verbose:
        print(f"Found {len(mol_files)} .mol file(s) in '{input_dir}'")
        print(f"Output directory: '{output_path}'")
        print("-" * 60)
    
    num_success = 0
    num_failed = 0
    
    for mol_file in mol_files:
        sdf_file = output_path / mol_file.with_suffix('.sdf').name
        
        if convert_mol_to_sdf(mol_file, sdf_file, verbose=verbose):
            num_success += 1
        else:
            num_failed += 1
    
    if verbose:
        print("-" * 60)
        print(f"Conversion complete: {num_success} succeeded, {num_failed} failed")
    
    return (num_success, num_failed)


def main():
    parser = argparse.ArgumentParser(
        description='Convert .mol files to .sdf format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python mol_2_sdf.py molecule.mol
  python mol_2_sdf.py molecule.mol output.sdf
  
  # Convert all files in directory
  python mol_2_sdf.py input_dir/
  python mol_2_sdf.py input_dir/ output_dir/
  
  # Quiet mode
  python mol_2_sdf.py molecule.mol -q
        """
    )
    
    parser.add_argument('input', help='Input .mol file or directory containing .mol files')
    parser.add_argument('output', nargs='?', default=None, 
                       help='Output .sdf file or directory (optional)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress status messages')
    parser.add_argument('--remove-hs', action='store_true',
                       help='Remove hydrogens from molecules')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    verbose = not args.quiet
    
    # Check if input is a file or directory
    if input_path.is_file():
        # Single file conversion
        success = convert_mol_to_sdf(args.input, args.output, verbose=verbose)
        sys.exit(0 if success else 1)
        
    elif input_path.is_dir():
        # Directory conversion
        num_success, num_failed = convert_directory(args.input, args.output, verbose=verbose)
        sys.exit(0 if num_failed == 0 else 1)
        
    else:
        print(f"Error: '{args.input}' is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
