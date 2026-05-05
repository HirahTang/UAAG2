import shutil
import os
import sys
import argparse
from tqdm import tqdm
import subprocess
from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
import warnings

# Suppress BioPython warnings
warnings.filterwarnings('ignore')

class HydrogenSelect(Select):
    """Select all atoms except hydrogens"""
    def accept_atom(self, atom):
        # Remove hydrogen atoms
        if atom.element == 'H':
            return False
        return True

def contains_amino_acids(structure):
    """Check if structure contains any amino acid residues (protein/peptide)"""
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if residue is a standard amino acid
                if is_aa(residue, standard=True):
                    return True
    return False

def main(args):
    
    if not os.path.exists(args.cleaned_dir):
        os.makedirs(args.cleaned_dir)
    
    # Get list of .cif files
    cif_file_list = [f for f in os.listdir(args.pdb_dir) if f.endswith('.cif')]
    
    parser = MMCIFParser(QUIET=True)
    io = PDBIO()
    
    skipped_files = []
    processed_files = []
    
    for cif_file in tqdm(cif_file_list, desc="Processing CIF files"):
        file_id = os.path.splitext(cif_file)[0]
        cif_path = os.path.join(args.pdb_dir, cif_file)
        
        print(f'Processing {cif_file}')
        
        try:
            # Parse CIF file
            structure = parser.get_structure(file_id, cif_path)
            
            # Check if structure contains amino acids
            if not contains_amino_acids(structure):
                print(f'  Skipping {cif_file}: No amino acids found (likely DNA/RNA)')
                skipped_files.append((cif_file, 'no_amino_acids'))
                continue
            
            # Get number of models
            num_models = len(structure)
            
            # Process each model
            for model_idx, model in enumerate(structure):
                if num_models > 1:
                    # Multi-model: name as {id}_0.pdb, {id}_1.pdb, etc.
                    output_filename = f'{file_id}_{model_idx}.pdb'
                else:
                    # Single model: name as {id}.pdb
                    output_filename = f'{file_id}.pdb'
                
                output_path = os.path.join(args.cleaned_dir, output_filename)
                
                # Set structure to current model
                io.set_structure(model)
                
                # Save to PDB format, removing hydrogens but keeping water and other heteroatoms
                io.save(output_path, select=HydrogenSelect())
                
                print(f'  Saved: {output_filename}')
            
            processed_files.append((cif_file, num_models))
            
        except Exception as e:
            print(f'  Error processing {cif_file}: {str(e)}')
            skipped_files.append((cif_file, f'error: {str(e)}'))
            continue
    
    # Print summary
    print('\n' + '='*60)
    print('PROCESSING SUMMARY')
    print('='*60)
    print(f'Total files processed: {len(processed_files)}')
    print(f'Total files skipped: {len(skipped_files)}')
    
    if skipped_files:
        print('\nSkipped files:')
        for fname, reason in skipped_files:
            print(f'  - {fname}: {reason}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean CIF files: remove hydrogens, keep water, filter DNA/RNA, handle multi-model structures')
    parser.add_argument('--pdb_dir', type=str, default='data/pdb', help='Directory containing CIF files')
    parser.add_argument('--cleaned_dir', type=str, default='data/pdb_processed', help='Directory to store cleaned PDB files')
    args = parser.parse_args()
    main(args)