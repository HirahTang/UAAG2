#!/usr/bin/env python3
"""
Evaluate .mol samples in nested directories using PoseBusters.

This script:
1. Finds all .mol files in nested directories
2. Converts them to .sdf format in a temporary directory
3. Evaluates each .sdf file using PoseBusters
4. Records results (success rate, bust metrics) to CSV
5. Cleans up temporary .sdf files

Usage:
    python evaluate_mol_samples.py <input_dir> [options]
    
Example:
    python evaluate_mol_samples.py /datasets/biochem/unaagi/ProteinGymSampling/runA0A247D711_LISMN_Full_mask_5_virtual_node_mask_token_atomic_only_mask_diffusion_0915/Samples/ALA_58 -o results.csv
"""

import os
import sys
import argparse
import csv
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import SDWriter

# Try to import posebusters
try:
    from posebusters import PoseBusters
    POSEBUSTERS_AVAILABLE = True
except ImportError:
    POSEBUSTERS_AVAILABLE = False
    print("Warning: PoseBusters not available. Install with: pip install posebusters")


def find_mol_files(root_dir, verbose=True):
    """
    Recursively find all .mol files in directory.
    
    Args:
        root_dir (str): Root directory to search
        verbose (bool): Print status messages
        
    Returns:
        list: List of Path objects for .mol files
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    mol_files = list(root_path.rglob('*.mol'))
    
    if verbose:
        print(f"Found {len(mol_files)} .mol files in {root_dir}")
    
    return mol_files


def convert_mol_to_sdf(mol_file, output_dir, input_path):
    """
    Convert a single .mol file to .sdf format.
    
    Args:
        mol_file (Path): Path to input .mol file
        output_dir (Path): Directory for output .sdf file
        input_path (Path): Root input path for relative path calculation
        
    Returns:
        tuple: (success: bool, sdf_path: Path or None, error_msg: str or None)
    """
    try:
        # Create output path maintaining relative structure
        sdf_filename = mol_file.relative_to(input_path).with_suffix('.sdf')
        # replace / in sdf_filename with _
        sdf_filename = str(sdf_filename).replace(os.sep, '_')
        sdf_path = output_dir / sdf_filename
        
        # Read molecule from .mol file
        mol = Chem.MolFromMolFile(str(mol_file), removeHs=False, strictParsing=False)
        
        if mol is None:
            return (False, None, "Failed to parse molecule")
        
        # Write molecule to .sdf file
        writer = SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        
        return (True, sdf_path, None)
        
    except Exception as e:
        return (False, None, str(e))


def evaluate_with_bust(sdf_file, mol_file=None):
    """
    Evaluate a molecule using PoseBusters.
    
    Args:
        sdf_file (Path): Path to .sdf file
        mol_file (Path, optional): Original .mol file path for reference
        
    Returns:
        dict: Evaluation results including individual checks and overall pass/fail
    """
    if not POSEBUSTERS_AVAILABLE:
        return {
            'bust_available': False,
            'error': 'PoseBusters not installed'
        }
    
    try:
        # Initialize PoseBusters
        buster = PoseBusters(config='mol')
        
        # Run evaluation
        results = buster.bust([str(sdf_file)])
        
        # Extract results (bust returns a dataframe)
        if results is not None and len(results) > 0:
            result_dict = results.iloc[0].to_dict()
            result_dict['bust_available'] = True
            
            # Check all criteria - if ANY check fails, molecule fails
            # PoseBusters returns True for passed checks, False for failed
            all_checks_passed = True
            for key, value in result_dict.items():
                # Skip non-check columns
                if key in ['file_name', 'molecule', 'mol_pred_loaded', 'bust_available']:
                    continue
                # If any check is False, the molecule fails
                if isinstance(value, bool) and value == False:
                    all_checks_passed = False
                    break
            
            # Store both passing criteria
            result_dict['pass_all_criteria'] = all_checks_passed
            result_dict['pass_mol_loaded'] = result_dict.get('mol_pred_loaded', False)
            return result_dict
        else:
            return {
                'bust_available': True,
                'error': 'No results returned',
                'pass_all_criteria': False,
                'pass_mol_loaded': False
            }
            
    except Exception as e:
        return {
            'bust_available': True,
            'error': str(e),
            'pass_all_criteria': False,
            'pass_mol_loaded': False
        }


def process_samples(input_dir, output_csv, temp_dir=None, keep_sdf=False, verbose=True):
    """
    Process all .mol samples: convert, evaluate, and record results.
    
    Args:
        input_dir (str): Root directory containing .mol files
        output_csv (str): Output CSV file path
        temp_dir (str, optional): Temporary directory for .sdf files
        keep_sdf (bool): Keep .sdf files after evaluation
        verbose (bool): Print detailed status messages
        
    Returns:
        dict: Summary statistics
    """
    # Setup directories
    input_path = Path(input_dir)
    
    if temp_dir is None:
        temp_path = Path('temp_sdf_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    else:
        temp_path = Path(temp_dir)
    
    temp_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\nProcessing .mol samples from: {input_dir}")
        print(f"Temporary .sdf directory: {temp_path}")
        print(f"Output CSV: {output_csv}")
        print("=" * 80)
    
    # Find all .mol files
    mol_files = find_mol_files(input_dir, verbose=verbose)
    
    if not mol_files:
        print("No .mol files found!")
        return {'total': 0, 'conversion_success': 0, 'conversion_failed': 0, 
                'pass_all_criteria': 0, 'pass_mol_loaded': 0, 'bust_failed': 0}
    
    # Statistics
    stats = {
        'total': len(mol_files),
        'conversion_success': 0,
        'conversion_failed': 0,
        'pass_all_criteria': 0,
        'pass_mol_loaded': 0,
        'bust_failed': 0,
        'bust_error': 0,
        'criteria_counts': {}  # Track individual criteria pass rates
    }
    
    # Prepare CSV file
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all results first
    results = []
    
    if verbose:
        print("\n1. Converting .mol files to .sdf format...")
        iterator = tqdm(mol_files, desc="Converting", unit="file")
    else:
        iterator = mol_files
    
    for mol_file in iterator:
        # Get relative path for better tracking
        try:
            rel_path = mol_file.relative_to(input_path)
        except ValueError:
            rel_path = mol_file
        
        # replace / in rel_path with _
        rel_path_str = str(rel_path).replace(os.sep, '_')
        
        result = {
            'mol_file': str(rel_path_str),
            'mol_full_path': str(mol_file),
            'conversion_success': False,
            'sdf_file': None,
            'conversion_error': None,
            'bust_evaluated': False,
            'pass_all_criteria': False,
            'pass_mol_loaded': False
        }
        
        # Convert to SDF
        success, sdf_path, error_msg = convert_mol_to_sdf(mol_file, temp_path, input_path)
        
        result['conversion_success'] = success
        result['sdf_file'] = str(sdf_path) if sdf_path else None
        result['conversion_error'] = error_msg
        
        if success:
            stats['conversion_success'] += 1
            result['sdf_created'] = True
        else:
            stats['conversion_failed'] += 1
            result['sdf_created'] = False
        
        results.append(result)
    
    # Evaluate with PoseBusters
    if verbose:
        print("\n2. Evaluating molecules with PoseBusters...")
        iterator = tqdm(results, desc="Evaluating", unit="molecule")
    else:
        iterator = results
    
    for result in iterator:
        if result['conversion_success'] and result['sdf_file']:
            sdf_path = Path(result['sdf_file'])
            
            if sdf_path.exists():
                bust_results = evaluate_with_bust(sdf_path, Path(result['mol_full_path']))
                result['bust_evaluated'] = True
                
                # Merge bust results into main result
                result.update(bust_results)
                
                # Track statistics for both passing criteria
                if bust_results.get('pass_all_criteria', False):
                    stats['pass_all_criteria'] += 1
                
                if bust_results.get('pass_mol_loaded', False):
                    stats['pass_mol_loaded'] += 1
                
                # Update final status based on strict criteria
                if bust_results.get('pass_all_criteria', False):
                    result['final_status'] = 'PASS_ALL'
                elif bust_results.get('pass_mol_loaded', False):
                    result['final_status'] = 'PASS_LOADED'
                else:
                    stats['bust_failed'] += 1
                    result['final_status'] = 'FAIL'
                
                # Track individual criteria success rates
                for key, value in bust_results.items():
                    if isinstance(value, bool) and key not in ['bust_available', 'pass_all_criteria', 'pass_mol_loaded']:
                        if key not in stats['criteria_counts']:
                            stats['criteria_counts'][key] = {'passed': 0, 'failed': 0}
                        if value:
                            stats['criteria_counts'][key]['passed'] += 1
                        else:
                            stats['criteria_counts'][key]['failed'] += 1
                    
                if 'error' in bust_results:
                    stats['bust_error'] += 1
            else:
                result['bust_evaluated'] = False
                result['final_status'] = 'CONVERSION_FAILED'
        else:
            result['bust_evaluated'] = False
            result['final_status'] = 'CONVERSION_FAILED'
    
    # Write results to CSV
    if verbose:
        print("\n3. Writing results to CSV...")
    
    # Determine all column names from results
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    # Order important columns first
    important_cols = ['mol_file', 'final_status', 'conversion_success', 'bust_evaluated', 
                      'pass_all_criteria', 'pass_mol_loaded', 'conversion_error', 'error']
    other_cols = sorted([k for k in all_keys if k not in important_cols and k != 'mol_full_path'])
    column_order = important_cols + other_cols
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_order, extrasaction='ignore')
        writer.writeheader()
        
        for result in results:
            writer.writerow(result)
    
    # Write summary statistics
    summary_path = csv_path.parent / (csv_path.stem + '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MOLECULE EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Input Directory: {input_dir}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CONVERSION STATISTICS:\n")
        f.write(f"  Total .mol files:        {stats['total']}\n")
        f.write(f"  Successfully converted:  {stats['conversion_success']} ({stats['conversion_success']/stats['total']*100:.2f}%)\n")
        f.write(f"  Conversion failed:       {stats['conversion_failed']} ({stats['conversion_failed']/stats['total']*100:.2f}%)\n\n")
        
        if stats['conversion_success'] > 0:
            f.write("POSEBUSTERS EVALUATION:\n")
            f.write(f"  Evaluated molecules:     {stats['conversion_success']}\n\n")
            
            f.write("PASSING RATES:\n")
            f.write(f"  Pass (all criteria):     {stats['pass_all_criteria']} ({stats['pass_all_criteria']/stats['conversion_success']*100:.2f}%)\n")
            f.write(f"  Pass (mol loaded):       {stats['pass_mol_loaded']} ({stats['pass_mol_loaded']/stats['conversion_success']*100:.2f}%)\n")
            f.write(f"  Failed:                  {stats['bust_failed']} ({stats['bust_failed']/stats['conversion_success']*100:.2f}%)\n")
            if stats['bust_error'] > 0:
                f.write(f"  Evaluation errors:       {stats['bust_error']}\n")
            
            # Write individual criteria success rates
            if stats['criteria_counts']:
                f.write("\nINDIVIDUAL CRITERIA SUCCESS RATES:\n")
                for criterion, counts in sorted(stats['criteria_counts'].items()):
                    total_checked = counts['passed'] + counts['failed']
                    pass_rate = counts['passed'] / total_checked * 100 if total_checked > 0 else 0
                    f.write(f"  {criterion:30s}: {counts['passed']:4d}/{total_checked:4d} ({pass_rate:6.2f}%)\n")
        
        f.write("\n")
        f.write("OVERALL SUCCESS RATES:\n")
        f.write(f"  Strict (all criteria):   {stats['pass_all_criteria']}/{stats['total']} ({stats['pass_all_criteria']/stats['total']*100:.2f}%)\n")
        f.write(f"  Lenient (mol loaded):    {stats['pass_mol_loaded']}/{stats['total']} ({stats['pass_mol_loaded']/stats['total']*100:.2f}%)\n")
        f.write("="*80 + "\n")
    
    # Clean up temporary .sdf files
    if not keep_sdf:
        if verbose:
            print("\n4. Cleaning up temporary .sdf files...")
        try:
            shutil.rmtree(temp_path)
            if verbose:
                print(f"   Removed: {temp_path}")
        except Exception as e:
            print(f"Warning: Failed to remove temporary directory: {e}")
    else:
        if verbose:
            print(f"\n4. Keeping .sdf files in: {temp_path}")
    
    # Print summary
    if verbose:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total .mol files:          {stats['total']}")
        print(f"Successfully converted:    {stats['conversion_success']} ({stats['conversion_success']/stats['total']*100:.2f}%)")
        print(f"Conversion failed:         {stats['conversion_failed']} ({stats['conversion_failed']/stats['total']*100:.2f}%)")
        
        if stats['conversion_success'] > 0:
            print(f"\nPoseBusters evaluation:")
            print(f"  Pass (all criteria):     {stats['pass_all_criteria']} ({stats['pass_all_criteria']/stats['conversion_success']*100:.2f}%)")
            print(f"  Pass (mol loaded):       {stats['pass_mol_loaded']} ({stats['pass_mol_loaded']/stats['conversion_success']*100:.2f}%)")
            print(f"  Failed:                  {stats['bust_failed']} ({stats['bust_failed']/stats['conversion_success']*100:.2f}%)")
        
        print(f"\nOverall success rates:")
        print(f"  Strict (all criteria):   {stats['pass_all_criteria']}/{stats['total']} ({stats['pass_all_criteria']/stats['total']*100:.2f}%)")
        print(f"  Lenient (mol loaded):    {stats['pass_mol_loaded']}/{stats['total']} ({stats['pass_mol_loaded']/stats['total']*100:.2f}%)")
        print(f"\nResults saved to:          {csv_path}")
        print(f"Summary saved to:          {summary_path}")
        print("="*80)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate .mol samples using PoseBusters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python evaluate_mol_samples.py /path/to/samples -o results.csv
  python evaluate_mol_samples.py /path/to/samples -o results.csv --keep-sdf
  python evaluate_mol_samples.py /path/to/samples -o results.csv --temp-dir ./temp_sdf
        """
    )
    
    parser.add_argument('input_dir', 
                       help='Root directory containing .mol files (will search recursively)')
    parser.add_argument('-o', '--output', default='evaluation_results.csv',
                       help='Output CSV file path (default: evaluation_results.csv)')
    parser.add_argument('--temp-dir', default=None,
                       help='Temporary directory for .sdf files (default: auto-generated)')
    parser.add_argument('--keep-sdf', action='store_true',
                       help='Keep .sdf files after evaluation (default: remove)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Check if PoseBusters is available
    if not POSEBUSTERS_AVAILABLE:
        print("\nError: PoseBusters is not installed!")
        print("Install with: pip install posebusters")
        print("Or: conda install -c conda-forge posebusters")
        sys.exit(1)
    
    verbose = not args.quiet
    
    try:
        stats = process_samples(
            input_dir=args.input_dir,
            output_csv=args.output,
            temp_dir=args.temp_dir,
            keep_sdf=args.keep_sdf,
            verbose=verbose
        )
        
        # Exit with error code if no samples passed (using strict criteria)
        if stats['pass_all_criteria'] == 0 and stats['total'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
