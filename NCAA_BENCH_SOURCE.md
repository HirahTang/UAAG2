# NCAA_BENCH — what it is, why it exists, how to use it

**As of 2026-09-21.** Companion to `NCAA_PDB_CCD_REPAIR.md`. Written for hand-off.
Everything below was read from the LMDB directly unless marked OPEN.

---

## TL;DR

NCAA_BENCH is a small, purpose-built training source: **RDKit conformers of the 20
benchmark non-canonical residues, generated from their correct SMILES.** It exists
because *none of the other sources contains those molecules correctly*, so without it the
model is asked at evaluation time for residues it has never seen.

It is **not** mined from the PDB, so it never passes through `MolFromPDBFile` and carries
none of the all-SINGLE-bond corruption described in `NCAA_PDB_CCD_REPAIR.md`.

---

## Contents (verified by reading the LMDB)

```
/scratch/project_465002574/PDB/NCAA_BENCH/NCAA.lmdb      12.8 MB   (LUMI)
~/hantang/lite_data/PDB/NCAA_BENCH/NCAA.lmdb              1.8 MB   (Hendrix, lite subset)

1,390 conformers | 20 distinct residues | capped at 75 conformers each
```

| | | | | |
|---|---|---|---|---|
| 2NP 75 | 2TH 75 | 3TH 75 | ABU 75 | AHP 68 |
| AIB 75 | AOC 71 | BZT 75 | CHA 59 | CPA 50 |
| HSM 48 | MEA 75 | MEB 75 | MEF 75 | MEG 75 |
| NLE 44 | NVA 75 | TBU 75 | TME 75 | YME 75 |

Counts below 75 (NLE 44, HSM 48, CPA 50, CHA 59, AHP 68, AOC 71) are where RDKit
embedding produced fewer distinct conformers, not a curation decision.

### Record format

LMDB keys are sequential `ncaa_benchmark_conformers_0000000…`, plus a `__keys__` entry
holding the pickled key list. Values are `torch_geometric.data.Data` with:

```
edge_index  edge_attr  pos  edge_ligand  charges  degree
is_aromatic is_in_ring hybridization  is_backbone  is_ligand
compound_id = "ncaa_bench_<CODE>_conf_<N>"
source_name = "ncaa_benchmark_conformers"
```

Same schema as the other sources, so it drops into `PDBBindDataset` unchanged.

**`is_backbone = 0` throughout**, like NCAA_indep — every atom including N/CA/C/O is a
generation target. This matters: it is the same convention `--all-atom-generation`
imposes on CATH, and the reason those two sources are task-compatible.

---

## Why it exists — the gap it fills

Three separate reasons the benchmark NCAAs were unreachable without it:

1. **NCAA_PDB's codes collide.** Its `2NP`, `2TH`, `3TH`, `AOC`, `BZT`, `CPA`, `MEA`,
   `MEB`, `MEF`, `MEG`, `TBU`, `TME`, `YME` are *different chemical components* from the
   benchmark's — the three-letter codes clash with real PDB components. 13 of the 20.
   (AIB, ABU, AHP, CHA, HSM, NLE, NVA do genuinely appear in NCAA_PDB.)
2. **NCAA_PDB's bond orders were corrupt** for every non-standard residue until the CCD
   repair — see the companion doc.
3. **CATH contains only the 20 standard residues.**

So for 13 of the 20 benchmark NCAAs, NCAA_BENCH is the **only** source of the correct
molecule, and no amount of repairing NCAA_PDB changes that.

### Measured consequence of omitting it

Seven of nine model families had never seen a single benchmark NCAA. Adding NCAA_BENCH
was one of three changes that took **AIB from 0% to 66%** in the UNAAGI-lite work. (The
other two: `--all-atom-generation`, and the `convert_edge_to_bond` fix. The three were
applied together and never ablated individually — OPEN.)

---

## How to use it

```
--use-ncaa-bench \
--ncaa-bench-lmdb <path>/NCAA_BENCH/NCAA.lmdb \
--ncaa-bench-weight 10
```

Handled in `scripts/run_train.py` (~line 119) via `PDBBindDataset`, same class as the
other sources.

**Weighting matters more than for the other sources.** Sampling share is
`weight x dataset size`, and NCAA_BENCH is tiny (1,390 records against NCAA_PDB's 38,863
and CATH-1k). At weight 10 in the lite run it reached roughly 34% of sampling mass. At
weight 1 it would be a rounding error, and the 13 collision-only residues would be
effectively absent again.

**This is the lever for those 13 residues.** If they underperform, raise this weight
before changing anything else.

---

## Caveats

- **Conformers, not crystal environments.** Every record is a free residue with no
  pocket. It teaches the model what the molecule *is*, not how it sits in a protein. For
  a pocket-conditioned model, NCAA_PDB and CATH supply the environment; NCAA_BENCH
  supplies the chemistry for residues nothing else covers.
- **75 conformers per residue is a small, possibly redundant sample.** Whether the model
  benefits from more diverse conformers, or is effectively memorising 20 molecules, is
  untested (OPEN).
- **Same 20 residues as the evaluation benchmark.** This is deliberate — the alternative
  was evaluating on molecules absent from training — but it means the benchmark is
  *not* held out. Any claim about generalisation to unseen NCAAs needs a different split.
- **CHA is in here (59 conformers) but is unmeasurable by graph-isomorphism scoring**, as
  it is heavy-atom-graph-identical to PHE. Use exact/canonical-SMILES scoring for CHA.

---

## Provenance (OPEN)

Built 2026-08-18. The generating script was not located in this session — `check_benchmarks.py`
in `UAAG2_conditional/scripts/` references NCAA_BENCH but is a checker, not a builder.
If conformers ever need regenerating, source SMILES are in
`/datasets/biochem/unaagi/condition_embeddings/cp2_experiment_conditions.npz`
(`codes`, `smiles`), which is the authoritative list of the 40 benchmark residues and is
verified clean (norms 36-78, no double-z-scoring).

---

## Related

- `NCAA_PDB_CCD_REPAIR.md` — the bond-order corruption in the companion source, the CCD
  fix, and its status.
- Evaluation: a sample counts as correct when it **sanitizes and its heavy-atom graph is
  isomorphic to the requested residue**, with exact canonical SMILES reported alongside.
  Do not use `aa_check` against `AA_GRAPH_DICT`.
- Sampling: pin node count with `--exact-node-count`, use **NFE >= 250** (the default
  `--every-k-step 25` gives 20 NFE off a 500-step chain).
