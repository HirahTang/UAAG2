# NCAA_PDB bond-order repair — status, evidence, and how to finish it

**As of 2026-09-21.** Written for hand-off to another session. Everything below is
verified unless explicitly marked OPEN.

---

## TL;DR

`NCAA_PDB.lmdb` was built through `Chem.MolFromPDBFile`, which has bond-order templates
for the **20 standard residues only**. Every non-standard residue in it was therefore
stored with **all-SINGLE bonds** — so AIB, whose real structure is `CC(C)(N)C=O`, is
stored as `CC(C)(N)CO`: the backbone carbonyl became a hydroxyl.

The fix is the wwPDB Chemical Component Dictionary, applied by `ccd_bonds.py`. It is
**ported and smoke-tested on LUMI**. What remains is patching the builder and rebuilding
the LMDB.

---

## Why it matters

Training on the uncorrected LMDB is worse than merely missing data, because the
**condition and the target disagree**:

- the condition vector is looked up by residue code and encodes the *correct* molecule
  (`cp2_experiment_conditions.npz` holds correct SMILES)
- the reconstruction target has a hydroxyl where the carbonyl belongs

Bond order is an explicitly supervised CTMC output (`sample_ctmc_tauleaping`, mapped out
by `convert_edge_to_bond`), so the gradient teaches *"condition says aldehyde → generate
alcohol"*.

This is very likely connected to the dominant generation failure measured in the
UNAAGI-lite grid: `NC(CO)…` instead of `NC(C=O)…` accounts for **91% of NLE's misses and
98% of 2NP's**. NOTE (OPEN): the lite models never trained on NCAA_PDB and still showed
this error, so NCAA_PDB is not the sole source. Worth checking whether `NCAA.lmdb`
(NCAA_indep) shares the same `MolFromPDBFile` path.

---

## The fix

`ccd_bonds.py` re-assigns intra-residue bond orders from the CCD, keyed on
`(residue code, unordered atom-name pair)`.

Key properties (read from the source, not assumed):

- **Standard residues are skipped** (`STANDARD_AA` frozenset) → CATH behaviour is
  bit-identical to before the change.
- **Formal charges are applied before bond orders.** A repaired DOUBLE onto e.g. a
  quaternary N fails RDKit's valence check unless the CCD charge goes on too.
- **Unmatched atom-name pairs are left untouched and counted**, never forced. A
  three-letter-code collision therefore degrades to a **no-op, not a corruption**.
- `repair_and_sanitize(mol, table, max_reverts=25)` sanitizes after repair; on a valence
  failure it identifies the offending residue, blocks just that one, rebuilds from
  pristine and retries. Last resort returns the unrepaired molecule. It never drops a
  structure.
- Returns `(mol, stats)` with `residues / no_template / bonds_seen / bonds_changed /
  bonds_unmatched / reverted_residues / sanitized` so you can verify it fired.

### Provenance of the table

Built by `UNAAGI/scripts/build_ccd_bond_table.py` from the wwPDB CCD:

```
source: /datasets/biochem/unaagi/protenix_data/common/components.cif   (Hendrix)
loop:   _chem_comp_bond.comp_id .atom_id_1 .atom_id_2 .value_order .pdbx_aromatic_flag
output: {comp_id: {(name1, name2): (order, aromatic_bool)}}, atom-name pair sorted
```

48,832 components, 3,138 with charges. The `pdbx_aromatic_flag` is preserved, which is
what restores HIS imidazole / TRP indole aromaticity. Snapshot came bundled with the
Protenix data distribution rather than downloaded fresh; pickle built 2026-08-31.

---

## Verification (run on LUMI, reproducible)

Synthetic all-single AIB through `repair_and_sanitize`:

```
AIB bonds before:  SINGLE SINGLE SINGLE SINGLE SINGLE
AIB bonds after :  SINGLE SINGLE DOUBLE SINGLE SINGLE
stats: residues 1, bonds_seen 5, bonds_changed 1, bonds_unmatched 0, sanitized True
SMILES after: CC(C)(N)C=O          <-- was CC(C)(N)CO
```

---

## Coverage: nothing needs filtering

Sampled 5,552 of 38,863 NCAA_PDB entries (direct key fetch, stride 7), parsing the
residue code from `compound_id` (format `PDBID_CHAIN+POS_CODE`, code is the **last**
field):

```
501 distinct residue codes
standard residues            0     0.0%
non-std WITH CCD entry   5,552   100.0%   <- repairable
non-std NO CCD entry         0     0.0%   <- nothing to drop
```

100% coverage follows from construction: these residues came from real PDB depositions,
so their codes are in the CCD by definition. **The filtering step originally planned is
unnecessary.**

Composition (top codes): MLY 403, SEP 292, CME 281, TPO 205, CSO 200, LLP 178, PTR 164,
FME 152, **AIB 149**, DLE 133, DLY 121, DAL 120, CSX 85, CSD 82, ORN 77, MLE 75.

---

## CAVEAT: benchmark three-letter codes collide with the CCD

Checked all 40 benchmark residues (heavy-atom count of `CCD[code]` vs the benchmark
SMILES). **27 match, 13 are a different chemical component entirely:**

```
collision: 2NP 2TH 3TH AOC BZT CPA MEA MEB MEF MEG TBU TME YME

  2NP  bench 15 HA -> CCD 13      CPA  bench 10 -> CCD 37
  2TH  bench 10    -> CCD 22      MEF  bench 12 -> CCD 33
  AOC  bench 10    -> CCD 22      TME  bench  8 -> CCD  3
  BZT  bench 14    -> CCD 38      MEB  bench  7 -> CCD  4
```

Consequences:

- Harmless for the repair itself (atom names don't match → no-op).
- But NCAA_PDB's `2NP`/`CPA`/`MEA`/… are **not** the benchmark's molecules. AIB *is*.
- So those 13 benchmark NCAAs reach the model **only via NCAA_BENCH**, regardless of
  NCAA_PDB. NCAA_BENCH weight remains the lever for them.

Same class of bug as `AA_GRAPH_DICT` holding the wrong isomer for Ahp/Bzt/Tme — a code
collision, not a parsing failure.

---

## File locations

| What | Where |
|---|---|
| `ccd_bonds.py` (ported) | LUMI `/flash/project_465002574/UAAG2_conditional/src/uaag2/datasets/ccd_bonds.py` |
| `ccd_bonds.py` (origin) | Hendrix `~/hantang/UNAAGI/vendored_uaag2/uaag2/datasets/ccd_bonds.py` |
| CCD table (57 MB) | LUMI `/scratch/project_465003401/ccd/ccd_bond_orders.pkl` |
| CCD table (origin) | Hendrix `~/hantang/UNAAGI/data/ccd_bond_orders.pkl` |
| Table builder | Hendrix `~/hantang/UNAAGI/scripts/build_ccd_bond_table.py` |
| NCAA_PDB source (18 GB) | LUMI `/scratch/project_465002574/PDB/NCAA_PDB` |
| NCAA_PDB LMDB (11 GB, CONTAMINATED) | LUMI `/scratch/project_465002574/PDB/NCAA_PDB_lmdb/NCAA_PDB.lmdb` |
| NCAA_indep | LUMI `/scratch/project_465002574/PDB/NCAA` (378 MB) |
| NCAA_BENCH | LUMI `/scratch/project_465002574/PDB/NCAA_BENCH` (13 MB) |

Activate with `UAAG2_CCD_BONDS=/scratch/project_465003401/ccd/ccd_bond_orders.pkl`.
If the table is absent, `ccd_bonds` is a **no-op** — it never hard-fails a loader.

LUMI rdkit env for ad-hoc checks: `/project/project_465003401/rdkit_env`
(`module load cray-python/3.11.7` first). Note `import uaag2` pulls in `wandb`; load
`ccd_bonds.py` directly via `importlib` to avoid that.

---

## THE REBUILD — DONE 2026-09-22, and what it actually required

Four defects, each of which produces a *wrong dataset rather than an error*. All four
must be active or the output is silently wrong.

```
UAAG2_SITES_CSV=/scratch/project_465003401/sites.csv        one graph per NCAA SITE
UAAG2_CCD_BONDS=/scratch/project_465003401/ccd/ccd_bond_orders.pkl   bond orders
UAAG2_LATENTS_OPTIONAL=1                                    latent-free, as the original
```

### 1. The input pipeline was unrecorded

`build_eqgat_lmdb_from_pdb.py` wants `.pdb`; the source is 14,937 `.cif`. The missing
stage is `scripts/process_cif.py` (SLURM array: strips H/waters, splits multi-model).
The intermediate PDBs were deleted after the original build, which is why neither cluster
had any. Stage 1 took <1 min for 9,729 structures on 32 tasks.

### 2. The subset is resolution <= 2.0 A, not `fine/`

Derived by matching site counts against the original LMDB's 38,863 entries:

```
res<=1.5 (= fine/)  12,513    res<=2.2   49,564
res<=2.0            38,737 <- matches    res<=2.5   61,937
```

**No `full_atom_ok` filter** was applied (requiring it gives 33,963). 9,729 structures ->
10,101 PDBs after multi-model splits.

### 3. Latents are REQUIRED by the stock builder

`PDB_128`/`PDB_20` do not exist on LUMI, and a missing latent makes the builder `continue`
past the residue -> an empty LMDB. Patched at both the guard and the `Data` construction.

### 4. The `is_aa` gate silently dropped 23% of sites  <-- the subtle one

`if not residue.is_amino_acid: continue` ran BEFORE the site filter. `Bio.PDB.is_aa()`
matches a fixed table that omits the rarer NCAA/PTM codes and every glycosylation site.
Measured on 826 listed sites: 764 matched the filter key (92.5%) but **194 (23.5%) were
dropped before reaching it**. Fixed so a site listed in `sites.csv` is authoritative.
This alone took the build from ~19,300 graphs to 28,946.

Also: compound_id. The builder's default is `{stem}_{res_name}_{resseq}_{chain}_{res_id}`;
the original LMDB uses `10JU_A71_SEP`. Matched when the site filter is active.

### Yield (be honest about this)

```
sites expected (<=2.0 A)     38,737
candidate sites found        33,974   88%
  - unsupported atoms        -5,024   15% of candidates (element vocabulary; left as-is)
graphs written               28,946   74% of the original 38,863
structures yielding >=1      7,337 / 10,101
```

Known losses: ~301 structures whose 5-character CCD codes (2023+ depositions) cannot be
written to legacy PDB at all -- `resName` has 3 columns, so the code spills into chainID
and resSeq. ~3,600 sites remain unaccounted; possibly another systematic filter, NOT yet
diagnosed (OPEN).

### Verification

```
bond types over 7,105 graphs: {0: 684826, 1: 97268, 2: 22084, 3: 17408, 4: 12}
                                             doubles + triples were ZERO before the repair
composition matches sites.csv ranking: HYP 593, MLY 361, SEP 277, AIB 258, ...
AIB: 1OB4_A2_AIB, 50 atoms total, 6 ligand atoms (AIB has exactly 6 heavy atoms)
AIB total ~1,032 vs sites.csv prediction 1,026
```

### Output

```
/scratch/project_465003401/ncaa_pdb_rebuild/final/NCAA_PDB.lmdb     <- USE THIS
/scratch/project_465002574/PDB/NCAA_PDB_lmdb/NCAA_PDB.lmdb          <- old, CONTAMINATED, keep for diffing
```

Patched files (all with `.bak_*` siblings, all gated behind env vars so default behaviour
is unchanged):
`scripts/build_eqgat_lmdb_from_pdb.py`, `src/uaag2/datasets/pdb_dataset.py`.

---

## REMAINING WORK

(Steps 1-3 below are DONE -- see above. Kept for reference.)

1. **Patch `read_structure_and_mol()`** in
   `scripts/build_eqgat_lmdb_from_pdb.py` (LUMI line 203/207 — identical on Hendrix).
   Replace:

   ```python
   Chem.SanitizeMol(rdkit_mol)
   Chem.AssignStereochemistry(rdkit_mol)
   ```

   with a `repair_and_sanitize(rdkit_mol, _CCD_TABLE)` call (loads the table once at
   module level), keeping `AssignStereochemistry` after it.

2. **Patch `_read_mol_and_structure()`** in `src/uaag2/datasets/pdb_dataset.py:157` the
   same way, so the on-the-fly loader and the LMDB builder agree.

3. **Time one shard**, then rebuild all 38,863 entries from the 18 GB source. The
   original build was sharded (`NCAA_PDB_meta_shard0..13`). Write the new LMDB to
   `project_465003401` storage (90,000 TB-h untouched; `project_465002574` is at 73%).

4. **Then train.** Intended config: CATH-1k + NCAA_indep + NCAA_BENCH + repaired
   NCAA_PDB, additive injection, two arms (GROVER 3201-d / MorganFP 2049-d),
   `--all-atom-generation`, `--virtual-node 0`, cond-dropout 0.2, pocket-dropout 0.2,
   8 GPUs, 47:59:00 on LUMI `standard-g` (8 GPUs = exactly one node; max walltime is
   `2-00:00:00`). Account `project_465003401` (17,995 GPU-h free).

---

## Related context another session will need

- **Sampling must pin the node count** (`--exact-node-count`) and use **NFE ≥ 250**.
  The default `--every-k-step 25` gives 20 NFE off a 500-step chain and costs ~24 points
  of end-to-end.
- **`data/aa_graph.json` is repaired on Hendrix only, uncommitted** (original at
  `.bak_pre_labelfix`). Without it, AHP/BZT/TME score 0% because the reference was a
  different constitutional isomer. It must be staged wherever evaluation runs.
- **Evaluation metric in use**: a sample counts as correct when it *sanitizes and its
  heavy-atom graph is isomorphic to the requested residue* (bond orders ignored), with
  exact canonical-SMILES reported alongside. Do **not** use `aa_check` against
  `AA_GRAPH_DICT` — it matches first-wins against a vocabulary (CHA resolves to PHE) and
  runs on unsanitized molecules.
- **LUMI repo is on branch `adaln-conditioning` with 35 uncommitted files.** The CCD port
  adds a 36th. Hendrix tree is not a git repo at all.
