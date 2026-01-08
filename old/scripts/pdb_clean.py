import shutil
import os
import sys
import argparse
from tqdm import tqdm
import subprocess

def main(args):
    

    pdb_file_list = os.listdir(args.pdb_dir)
    
    if not os.path.exists(args.cleaned_dir):
        os.mkdir(args.cleaned_dir)
    
    test = 100
    
    for pdb in tqdm(pdb_file_list):
        pdb_cleaned = pdb.split('.')[0] + '_tidy.pdb'
        pdb_orig_path = os.path.join(args.pdb_dir, pdb)
        pdb_process_path = os.path.join(args.cleaned_dir, pdb_cleaned)
        print(f'running {pdb}')
        try:
            subprocess.run(f'pdb_delhetatm {pdb_orig_path} | pdb_delelem -H | pdb_tidy > {pdb_process_path}', check=True, text=True, shell=True)
        except:
            continue
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean PDB files')
    parser.add_argument('--pdb_dir', type=str, default='data/pdb', help='Directory containing protein pdb files')
    parser.add_argument('--cleaned_dir', type=str, default='data/pdb_processed', help='Directory to store cleaned protein pdb files')
    args = parser.parse_args()
    main(args)