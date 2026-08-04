"""Static molecular-topology condition for conditional UAAG2.

The condition is a frozen GROVER-base embedding of the *whole residue* topology
(the ``is_ligand == 1`` subgraph: backbone N/CA/C/O + sidechain), precomputed
offline into a lookup table. Nothing runs GROVER at training time.

Layout of the table produced by ``grover/_work/build_condition_table.py``:

    emb          (M, D) float32   mean || max || n_atoms  (D = 3201 for GROVER-base)
    mean, std    (D,)   float32   per-dimension normalisation stats
    smiles       (M,)   object    canonical SMILES, the identity of each row
    record_ids   (R,)   object    "<source>:<compound_id>"
    record_rows  (R,)   int32     row index into emb for each record

Lookup is by ``compound_id``, which both the LMDB builder and the on-the-fly PDB
dataset attach to every graph. A miss returns zeros, which the model treats
exactly as a dropped condition -- so uncovered residues train unconditionally
instead of being dropped from the data.
"""
from typing import List, Optional, Sequence

import numpy as np
import torch


class ConditionTable:
    def __init__(self, path: str, normalize: bool = True):
        self.path = path
        z = np.load(path, allow_pickle=True)
        self.emb = torch.from_numpy(np.asarray(z["emb"], dtype=np.float32))
        self.dim = int(self.emb.shape[1])
        self.normalize = normalize
        if normalize:
            mu = torch.from_numpy(np.asarray(z["mean"], dtype=np.float32))
            sd = torch.from_numpy(np.asarray(z["std"], dtype=np.float32)).clamp_min(1e-6)
            self.emb = (self.emb - mu) / sd
        self.smiles = [str(s) for s in z["smiles"]]

        # compound_id -> row. Record ids are stored source-prefixed; key on the bare
        # compound_id so lookups work regardless of which source a batch came from.
        rec_ids = [str(s) for s in z["record_ids"]]
        rec_rows = np.asarray(z["record_rows"], dtype=np.int64)
        self.by_id = {}
        for rid, row in zip(rec_ids, rec_rows):
            bare = rid.split(":", 1)[1] if ":" in rid else rid
            self.by_id.setdefault(bare, int(row))
        self.by_smiles = {s: i for i, s in enumerate(self.smiles)}

        # zero vector shares storage; never written to
        self._zero = torch.zeros(self.dim, dtype=torch.float32)

    def __len__(self):
        return self.emb.shape[0]

    def lookup(self, compound_ids: Sequence[str]) -> (torch.Tensor, torch.Tensor):
        """Return (G, D) embeddings and a (G,) bool mask of which ids were found."""
        rows, found = [], torch.zeros(len(compound_ids), dtype=torch.bool)
        for i, cid in enumerate(compound_ids):
            r = self.by_id.get(str(cid))
            if r is None:
                rows.append(self._zero)
            else:
                rows.append(self.emb[r])
                found[i] = True
        return torch.stack(rows) if rows else torch.zeros(0, self.dim), found

    def lookup_smiles(self, smiles: Sequence[str]) -> (torch.Tensor, torch.Tensor):
        """Sampling-time path: condition on a chosen topology by canonical SMILES."""
        rows, found = [], torch.zeros(len(smiles), dtype=torch.bool)
        for i, s in enumerate(smiles):
            r = self.by_smiles.get(str(s))
            if r is None:
                rows.append(self._zero)
            else:
                rows.append(self.emb[r])
                found[i] = True
        return torch.stack(rows) if rows else torch.zeros(0, self.dim), found

    def coverage(self, compound_ids: Sequence[str]) -> float:
        if not len(compound_ids):
            return 0.0
        _, found = self.lookup(compound_ids)
        return float(found.float().mean())
