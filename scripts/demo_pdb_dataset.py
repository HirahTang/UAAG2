#!/usr/bin/env python3
"""Quick smoke test for UAAG2DatasetPDB."""
import sys
sys.path.insert(0, "/flash/project_465002574/UAAG2_main/src")

import time
import torch

from uaag2.datasets.pdb_dataset import UAAG2DatasetPDB

PDB_DIR = "/scratch/project_465002574/PDB/PDB_processed"

class Params:
    virtual_node = False
    max_virtual_nodes = 5
    pocket_radius = 10.0
    edge_radius = 5.0

params = Params()

print("Building dataset index (first run builds cache)...")
t0 = time.time()
ds = UAAG2DatasetPDB(
    pdb_dir=PDB_DIR,
    latent_root_128=None,
    latent_root_20=None,
    mask_rate=0.0,
    params=params,
    max_retries=10,
)
print(f"  Index built in {time.time()-t0:.1f}s, {len(ds)} items")

print("\nFetching 10 random samples...")
import random
random.seed(42)
indices = random.sample(range(len(ds)), min(10, len(ds)))

ok = 0
for idx in indices:
    t1 = time.time()
    data = ds[idx]
    elapsed = time.time() - t1
    if data is None:
        print(f"  [{idx}] -> None (skipped)")
        continue
    n = data.x.shape[0]
    e = data.edge_index.shape[1]
    lig = int(data.is_ligand.sum().item())
    ring_sum = data.is_in_ring.sum().item()
    print(f"  [{idx}] nodes={n:3d}  edges={e:5d}  ligand_atoms={lig}  "
          f"is_in_ring_sum={ring_sum:.0f}  time={elapsed:.3f}s")
    assert ring_sum == 0.0, f"is_in_ring not zeroed at idx {idx}!"
    assert n <= 400, f"node count {n} exceeds 400!"
    ok += 1

print(f"\n{ok}/10 samples OK")
print("Fields:", [k for k in data.keys()])
print("x dtype:", data.x.dtype, " pos dtype:", data.pos.dtype)
