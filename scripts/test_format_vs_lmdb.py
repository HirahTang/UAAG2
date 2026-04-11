#!/usr/bin/env python3
"""
Format comparison test: verify UAAG2DatasetPDB output matches the Data format
of graphs stored in /scratch/project_465002574/unaagi_whole_v1.lmdb

Checks:
  1. Same set of top-level field names (excluding is_in_ring which is removed)
  2. Same dtypes for each field
  3. Same tensor ranks (1D vs 2D, etc.)
  4. Edge_index has shape [2, E], edge_attr shape [E], edge_ligand shape [E]
  5. x and pos consistent with num_nodes
"""
import sys, os
sys.path.insert(0, "/flash/project_465002574/UAAG2_main/src")

import lmdb, pickle, random, time
from pathlib import Path
import torch
from torch_geometric.data import Data

from uaag2.datasets.pdb_dataset import (
    _parse_pdb_cached,
    _build_graph,
)

LMDB_PATH = "/scratch/project_465002574/pdb_demo_100.lmdb"
PDB_DIR   = "/scratch/project_465002574/PDB/PDB_processed"
EDGE_RADIUS   = 5.0
POCKET_RADIUS = 10.0

# ---- Load a few LMDB graphs ----
print("Loading LMDB samples...")
env = lmdb.open(LMDB_PATH, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, map_size=1 << 30)
lmdb_samples = []
with env.begin() as txn:
    cursor = txn.cursor()
    for i, (_, v) in enumerate(cursor):
        if i >= 5:
            break
        lmdb_samples.append(pickle.loads(v))
env.close()
print(f"  Loaded {len(lmdb_samples)} LMDB graphs")

# Inspect field names and dtypes of first LMDB graph
lmdb_ref = lmdb_samples[0]
# Normalise: remove is_in_ring from LMDB reference (it's removed from codebase)
lmdb_fields = {k: getattr(lmdb_ref, k) for k in lmdb_ref.keys()
               if k != "is_in_ring" and not k.startswith("protein_mpnn")}
print("\nLMDB reference fields:")
for name, val in sorted(lmdb_fields.items()):
    if isinstance(val, torch.Tensor):
        print(f"  {name:35s}  shape={tuple(val.shape)}  dtype={val.dtype}")
    else:
        print(f"  {name:35s}  type={type(val).__name__}  val={val}")

# ---- Build a PDB graph ----
print("\nBuilding PDB graph...")
all_pdbs = list(Path(PDB_DIR).glob("*.pdb"))
random.seed(0)
random.shuffle(all_pdbs)

pdb_graph = None
for pdb in all_pdbs[:50]:
    try:
        af, bf, residues = _parse_pdb_cached(str(pdb))
    except Exception:
        continue
    aa_residues = [r for r in residues if r.is_amino_acid]
    for idx, res in enumerate(aa_residues):
        g = _build_graph(
            center_residue=res,
            center_idx=idx,
            all_residues=residues,
            atom_features=af,
            bond_features=bf,
            pocket_radius=POCKET_RADIUS,
            edge_radius=EDGE_RADIUS,
            latent_128=None,
            latent_20=None,
            compound_id=0,
        )
        if g is not None:
            pdb_graph = g
            break
    if pdb_graph is not None:
        break

assert pdb_graph is not None, "Could not build any PDB graph"
pdb_fields = {k: getattr(pdb_graph, k) for k in pdb_graph.keys()
              if not k.startswith("protein_mpnn")}
print(f"  Built graph from {pdb.name}")
print("\nPDB graph fields:")
for name, val in sorted(pdb_fields.items()):
    if isinstance(val, torch.Tensor):
        print(f"  {name:35s}  shape={tuple(val.shape)}  dtype={val.dtype}")
    else:
        print(f"  {name:35s}  type={type(val).__name__}  val={val}")

# ---- Compare ----
print("\n--- Comparison ---")
failures = []

# 1. Field presence (PDB should have everything LMDB has, except is_in_ring)
for name in lmdb_fields:
    if name not in pdb_fields:
        failures.append(f"MISSING field in PDB graph: {name}")

for name in pdb_fields:
    if name not in lmdb_fields:
        print(f"  NOTE: PDB has extra field '{name}' not in LMDB (OK if optional)")

# 2. dtype and rank match
for name in lmdb_fields:
    if name not in pdb_fields:
        continue
    lv = lmdb_fields[name]
    pv = pdb_fields[name]
    if isinstance(lv, torch.Tensor) and isinstance(pv, torch.Tensor):
        if lv.dtype != pv.dtype:
            failures.append(f"  DTYPE mismatch '{name}': LMDB={lv.dtype} PDB={pv.dtype}")
        if lv.dim() != pv.dim():
            failures.append(f"  RANK mismatch '{name}': LMDB={lv.dim()}D PDB={pv.dim()}D")

# 3. is_in_ring absent from PDB graph
if "is_in_ring" in [k for k in pdb_graph.keys()]:
    failures.append("FAIL: is_in_ring present in PDB graph!")

# 4. Structural sanity
n = pdb_graph.x.shape[0]
assert pdb_graph.pos.shape == (n, 3), f"pos shape mismatch: {pdb_graph.pos.shape}"
assert pdb_graph.edge_index.shape[0] == 2, "edge_index not [2, E]"
E = pdb_graph.edge_index.shape[1]
if pdb_graph.edge_attr.shape[0] != E:
    failures.append(f"edge_attr length {pdb_graph.edge_attr.shape[0]} != edges {E}")
if pdb_graph.edge_ligand.shape[0] != E:
    failures.append(f"edge_ligand length {pdb_graph.edge_ligand.shape[0]} != edges {E}")
assert n <= 400, f"node count {n} exceeds 400"

if failures:
    print("\nFAILURES:")
    for f in failures:
        print(" ", f)
    sys.exit(1)
else:
    print("\nALL CHECKS PASSED")
    print(f"  nodes={n}  edges={E}  is_in_ring absent=True")
