#!/usr/bin/env python3
"""
Direct demo: call _build_graph on a handful of PDBs directly (bypasses index
build entirely). Tests:
  - Graphs are built without errors
  - is_in_ring field is ABSENT from the Data object
  - node count <= 400
  - required fields are present
"""
import sys, time, random
sys.path.insert(0, "/flash/project_465002574/UAAG2_main/src")

from pathlib import Path
import torch

# Import internal helpers directly
from uaag2.datasets.pdb_dataset import (
    _parse_pdb_cached,
    _build_graph,
)

PDB_DIR = "/scratch/project_465002574/PDB/PDB_processed"

REQUIRED_FIELDS = ["x", "pos", "edge_index", "edge_attr", "edge_ligand",
                   "charges", "degree", "is_aromatic", "hybridization",
                   "is_backbone", "is_ligand"]
FORBIDDEN_FIELDS = ["is_in_ring"]

# Pick 20 PDB files at random
all_pdbs = list(Path(PDB_DIR).glob("*.pdb"))
random.seed(42)
sample = random.sample(all_pdbs, min(20, len(all_pdbs)))

EDGE_RADIUS   = 5.0
POCKET_RADIUS = 10.0

ok = skip = err = 0
for pdb in sample:
    t0 = time.time()
    try:
        atom_features, bond_features, residues = _parse_pdb_cached(str(pdb))
    except Exception as e:
        print(f"  PARSE_ERR {pdb.name}: {e}")
        err += 1
        continue

    # Try every residue as ligand until we get a graph
    graph = None
    aa_residues = [r for r in residues if r.is_amino_acid]
    for idx, res in enumerate(aa_residues):
        graph = _build_graph(
            center_residue=res,
            center_idx=idx,
            all_residues=residues,
            atom_features=atom_features,
            bond_features=bond_features,
            pocket_radius=POCKET_RADIUS,
            edge_radius=EDGE_RADIUS,
            latent_128=None,
            latent_20=None,
            compound_id=0,
        )
        if graph is not None:
            break

    elapsed = time.time() - t0

    if graph is None:
        print(f"  SKIP {pdb.name} — no valid graph (all residues skipped)")
        skip += 1
        continue

    # Check forbidden fields absent
    for f in FORBIDDEN_FIELDS:
        assert not hasattr(graph, f) or getattr(graph, f) is None, \
            f"FAIL: forbidden field '{f}' present in graph for {pdb.name}"

    # Check required fields present
    for f in REQUIRED_FIELDS:
        assert hasattr(graph, f) and getattr(graph, f) is not None, \
            f"FAIL: required field '{f}' missing in graph for {pdb.name}"

    n = graph.x.shape[0]
    assert n <= 400, f"FAIL: {n} nodes > 400 for {pdb.name}"

    print(f"  OK  {pdb.name:30s}  nodes={n:3d}  edges={graph.edge_index.shape[1]:5d}"
          f"  time={elapsed:.3f}s")
    ok += 1

print(f"\n--- {ok} OK / {skip} skipped / {err} errors out of {len(sample)} ---")
print("Fields:", [k for k in graph.keys()])
assert "is_in_ring" not in [k for k in graph.keys()], "is_in_ring still in graph!"
print("PASS: is_in_ring absent from all graphs")
