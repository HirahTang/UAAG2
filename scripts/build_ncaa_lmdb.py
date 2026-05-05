#!/usr/bin/env python3
"""
Convert NCAA .pt files to NCAA.lmdb.

Each .pt file is a list of PyG Data objects representing NCAA structures.
The stored format is pre-_post_process (raw), matching PDBBind.lmdb, so
PDBBindDataset can load it directly.

Transformations applied:
  - Remove is_in_ring
  - Add edge_ligand = all-ones (no pocket, all edges are ligand-ligand)
  - Convert bool tensors (is_aromatic, is_in_ring) to int64
  - Keep charges as raw formal charges (encoded by _post_process_graph)
  - Keep compound_id field (used by _post_process_graph to set id=compound_id)

Usage:
    python build_ncaa_lmdb.py \
        --in-dir  /scratch/project_465002574/PDB/naa \
        --out-dir /scratch/project_465002574/PDB/NCAA
"""
import argparse
import os
import pickle
from pathlib import Path

import lmdb
import torch
from torch_geometric.data import Data


def _convert_item(item: Data, source_name: str) -> Data:
    """Convert one NCAA Data item to the pre-_post_process LMDB format."""
    n_edges = item.edge_index.size(1)

    # Convert bool tensors to int64 for consistent downstream handling
    is_aromatic = item.is_aromatic
    if is_aromatic.dtype == torch.bool:
        is_aromatic = is_aromatic.long()

    is_backbone = item.is_backbone
    if is_backbone.dtype == torch.bool:
        is_backbone = is_backbone.long()

    is_ligand = item.is_ligand
    if is_ligand.dtype == torch.bool:
        is_ligand = is_ligand.float()

    return Data(
        x=item.x,                                          # int64, ATOM_ENCODER values
        pos=item.pos,                                      # float32, 3D coordinates
        edge_index=item.edge_index,                        # int64 [2, E]
        edge_attr=item.edge_attr,                          # int64, bond types (BOND_ENCODER)
        edge_ligand=torch.ones(n_edges, dtype=torch.float),# all ligand edges (no pocket)
        charges=item.charges,                              # int64, raw formal charges
        degree=item.degree,                                # int64
        is_aromatic=is_aromatic,                           # int64
        hybridization=item.hybridization,                  # int64
        is_backbone=is_backbone,                           # int64/float
        is_ligand=is_ligand,                               # float32, all 1s
        compound_id=item.compound_id,                      # str — used by _post_process_graph
        source_name=source_name,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  required=True,
                    help="Directory containing .pt files (e.g. /scratch/.../naa)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory; NCAA.lmdb will be written here")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "NCAA.lmdb")

    pt_files = sorted(Path(args.in_dir).glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {args.in_dir}")
        return

    print(f"Found {len(pt_files)} .pt file(s): {[f.name for f in pt_files]}")

    env = lmdb.open(out_path, map_size=5 * 1024 ** 3, subdir=False)

    keys_written: list[bytes] = []
    stats: dict[str, int] = {}
    global_idx = 0

    for pt_file in pt_files:
        source_name = pt_file.stem
        print(f"\nLoading {pt_file.name} …", flush=True)
        raw_list = torch.load(str(pt_file), map_location="cpu", weights_only=False)
        print(f"  {len(raw_list)} items", flush=True)

        ok = skip = 0
        for local_idx, item in enumerate(raw_list):
            try:
                converted = _convert_item(item, source_name)
            except Exception as exc:
                print(f"  SKIP [{source_name}:{local_idx}] {exc}")
                skip += 1
                continue

            # Key: file-stem + global index → unique across both files
            key = f"{source_name}_{global_idx:07d}".encode()
            payload = pickle.dumps(converted, protocol=pickle.HIGHEST_PROTOCOL)
            with env.begin(write=True) as txn:
                txn.put(key, payload)
            keys_written.append(key)
            ok += 1
            global_idx += 1

        stats[source_name] = ok
        print(f"  Written: {ok}  Skipped: {skip}", flush=True)

    # Store key index
    with env.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys_written, protocol=pickle.HIGHEST_PROTOCOL))

    env.close()

    print(f"\n{'='*50}")
    print(f"NCAA.lmdb written to: {out_path}")
    for src, n in stats.items():
        print(f"  {src:<35} {n:6d} items")
    print(f"  {'TOTAL':<35} {len(keys_written):6d} items")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
