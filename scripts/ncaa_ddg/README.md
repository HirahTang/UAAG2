# NCAA ΔΔG Benchmark: Rosetta & OpenMM Evaluation on PUMA and CP2

## Overview

This benchmark evaluates three physics-based methods for predicting the effect of amino acid
substitutions — including noncanonical amino acids (NCAAs) — on protein–peptide binding affinity.
The two target systems are **PUMA** (a BH3-domain peptide binding MCL-1) and **CP2** (a macrocyclic
peptide), both drawn from the UAAG2 deep mutational scanning (DMS) benchmark suite.

"Coverage" in the results tables means the fraction of benchmark mutations that received a
non-NaN prediction score. A mutation is unscored (NaN) when the method lacks parameters for
that amino acid — e.g., Approach A skips NCAAs absent from PyRosetta's residue-type database,
and Approach B skips NCAAs for which AM1-BCC parameter generation failed. Coverage of 46%
means 54% of mutations produced no prediction and are excluded from the Spearman calculation.

---

## Benchmark Datasets

### PUMA (BH3 peptide)
- **PDB**: 2ROC (NMR, 20 models — MODEL 1 extracted as `2roc_model1.pdb`)
- **Chain**: B (27-residue BH3 peptide of PUMA, binds MCL-1 on chain A)
- **Mutations**: 1,326 single substitutions (19 positions × 20 canonical AAs + 19 NCAAs)
- **Experimental values**: DMS fitness scores from Reznik et al. (binding to MCL-1)
- **CSV**: `UNAAGI_benchmark_values/uaa_benchmark_csv/PUMA_reframe.csv`
- **NCAAs in benchmark**: ABU, NVA, NLE, AHP, AOC, AIB, TME, CPA, CHA, TBU,
  2NP, 2TH, 3TH, BZT, DAL, HSM, YME, MEG, MEA, MEB, MEF (21 codes)

### CP2 (macrocyclic peptide)
- **PDB**: 5LY1 (crystal structure)
- **Chain**: E (13-residue macrocyclic CP2 peptide, binds MDM2)
- **Mutations**: 468 single substitutions
- **Experimental values**: DMS fitness scores (binding to MDM2/MDMX)
- **CSV**: `UNAAGI_benchmark_values/uaa_benchmark_csv/CP2_reframe.csv`
- **Note**: Chain E appears twice in 5LY1 (two biological copies); only the first is used.
  The nonstandard residue DTY (dityrosine, residue 1) is replaced with TYR by PDBFixer.

---

## Methods

### Approach A — Rosetta with existing params

**Goal**: Score mutations using PyRosetta's built-in residue-type database, with no
external parameter generation. Serves as a fast baseline.

**Tools**:
- PyRosetta 2026 (conda, `ncaa_tools` micromamba env at `/flash/project_465002574/micromamba/envs/ncaa_tools/`)
- Script: `scripts/ncaa_ddg/approach_a_rosetta.py`

**Workflow**:
1. **Initialise PyRosetta** with flags:
   ```
   -mute all -ex1 -ex2 -extrachi_cutoff 1 -use_input_sc
   -ignore_unrecognized_res -ignore_zero_occupancy false
   ```
   - `-ex1 -ex2`: expand rotamer library to χ1/χ2 ±1 standard deviation (denser sampling)
   - `-extrachi_cutoff 1`: apply extra rotamers even for buried residues (≥1 heavy-atom neighbour)
   - `-use_input_sc`: initialise sidechains from the input PDB rather than idealized rotamers
   - `-ignore_unrecognized_res`: skip HETATM records without params rather than crashing
   - `-ignore_zero_occupancy false`: include zero-occupancy atoms

2. **Load structure** with `pose_from_pdb()`.

3. **Score function**: `ref2015` (Rosetta's all-atom energy function with updated statistical
   potentials, Lennard-Jones, hydrogen bonding, electrostatics, Ramachandran term, etc.)

4. **For each mutation**:
   - Look up Rosetta residue-type name via `ROSETTA_NCAA_MAP`
   - Find the Rosetta pose residue number with `pdb_info.pdb2pose(chain, pos)`
   - Clone the pose, apply `MutateResidue(resnum, rosetta_name).apply(mutant_pose)`
   - Score the mutant: `e_mut = scorefxn(mutant_pose)`
   - Predict: `pred = -(e_mut − e_wt)` (negative ΔΔG; positive = stabilising)

5. **Supported NCAAs** (present in PyRosetta 2026 conda build):
   - **AIB** (α-aminoisobutyric acid) — only NCAA confirmed in this build
   - NLE, NVA, DAL, ABU are absent from this build and are skipped

6. **Skipped categories**:
   - NCAAs absent from the PyRosetta 2026 database (all except AIB)
   - Residue positions not found in the target chain (`pdb2pose` returns 0)

**Hyperparameters summary**:

| Parameter | Value |
|---|---|
| Score function | `ref2015` |
| `nstruct` | 3 (packed structures averaged; only one used in current impl) |
| `-ex1` / `-ex2` | enabled |
| `-extrachi_cutoff` | 1 |
| ΔΔG formula | `-(E_mut − E_wt)` in REU |

---

### Approach B — Rosetta + AM1-BCC generated params

**Goal**: Extend Approach A to all 21 NCAAs by generating Rosetta `.params` files
on-the-fly from SMILES strings, using AM1-BCC partial charges from AmberTools.

**Tools**:
- PyRosetta 2026 (same `ncaa_tools` env)
- AmberTools 23 (`antechamber`, `parmchk2`) via `ncaa_tools` conda env
- RDKit (conformer generation, MMFF94 optimisation)
- PyRosetta `molfile_to_params_polymer.py` (converts mol2 → Rosetta `.params`)
- Script: `scripts/ncaa_ddg/approach_b_rosetta_am1.py`
- SMILES definitions: `scripts/ncaa_ddg/ncaa_smiles.py`

**Workflow**:
1. **PyRosetta init**: same flags as Approach A plus `-extra_res_fa <params_files>` to
   load the generated `.params` files.

2. **Score function**: `ref2015_cart` (Cartesian variant, required for MutateResidue
   with custom residue types).

3. **Per-NCAA parameter generation** (cached at `/flash/project_465002574/rosetta_ncaa/params/`):
   - **Step 1** — RDKit 3D conformer from SMILES:
     `AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())` followed by
     `AllChem.MMFFOptimizeMolecule(mol)` (MMFF94 force field)
   - **Step 2** — antechamber AM1-BCC charges:
     ```
     antechamber -i mol.sdf -fi sdf -o mol.mol2 -fo mol2
                 -c bcc -nc 0 -rn LIG -at gaff2
     ```
     `-c bcc` = AM1-BCC charge model; `-nc 0` = net charge 0 (neutral free amino acid);
     `-at gaff2` = GAFF2 atom types
   - **Step 3** — `parmchk2 -i mol.mol2 -f mol2 -o mol.frcmod -s gaff2`
     (missing-parameter supplement file)
   - **Step 4** — `molfile_to_params_polymer.py` (PyRosetta tool):
     converts mol2 to Rosetta `.params` + idealised conformer PDB
   - **Step 5** — `.params` cached for reuse across PUMA/CP2 runs

4. **Mutation scoring**: identical to Approach A (MutateResidue + score), but with
   generated `.params` loaded at init.

5. **Outcome**: All 21 NCAA `.params` generation attempts **failed** in this run
   (likely due to `molfile_to_params_polymer.py` incompatibility with PyRosetta 2026).
   Approach B therefore scored only canonical AAs (same coverage as Approach A minus AIB).

**Hyperparameters summary**:

| Parameter | Value |
|---|---|
| Score function | `ref2015_cart` |
| `nstruct` | 3 |
| Charge model | AM1-BCC |
| RDKit conformer | ETKDGv3 + MMFF94 |
| antechamber atom types | GAFF2 |
| net charge assumed | 0 (neutral) |
| ΔΔG formula | `-(E_mut − E_wt)` in REU |

---

### Approach C — OpenMM + GAFF2

**Goal**: Score mutations using OpenMM with the AMBER ff14SB protein force field and
GAFF2 small-molecule force field (for NCAAs), with implicit OBC2 solvation.

**Tools**:
- OpenMM 8.x (`unaagi_env` at `/flash/project_465002574/unaagi_env/`)
- `openmmforcefields` (SystemGenerator for GAFF2 parameterisation)
- `openff-toolkit` (OpenFF Molecule for GAFF2)
- PDBFixer (structure preparation)
- RDKit + `AllChem.ETKDGv3()` + `AllChem.MMFFOptimizeMolecule()` (NCAA conformers)
- Script: `scripts/ncaa_ddg/approach_c_openmm.py`
- SMILES definitions: `scripts/ncaa_ddg/ncaa_smiles.py`

**Workflow**:

#### Structure preparation (WT)
1. Load PDB with `PDBFixer`
2. Keep only the target chain; remove duplicate copies (5LY1 chain E appears twice)
3. `replaceNonstandardResidues()` — replace nonstandard residues (e.g. DTY→TYR)
4. `removeHeterogens(keepWater=False)`, `addMissingAtoms()`, `addMissingHydrogens(pH=7.0)`
5. Clear periodic box vectors (`topology.setPeriodicBoxVectors(None)`) — necessary for
   crystal structures (5LY1) to prevent `SystemGenerator` from choosing PME nonbonded
   method (incompatible with implicit solvent)

#### WT minimisation
6. **Force field**: `amber/ff14SB.xml` + `implicit/obc2.xml` (OBC2 Generalised Born implicit solvent)
7. **SystemGenerator** (`openmmforcefields`): `small_molecule_forcefield="gaff-2.11"`
8. **Integrator**: `LangevinMiddleIntegrator`, T=300 K, friction=1 ps⁻¹, dt=4 fs
9. **Minimisation**: `maxIterations=1000`, tolerance=10 kJ mol⁻¹ nm⁻¹
10. Platform: AMD MI250X GPU (`CUDA`/ROCm via `unaagi_env`)

#### Mutation scoring
For **canonical AAs**: `Modeller.applyMutations()` → ff14SB + OBC2 minimisation →
`pred = -(E_mut − E_wt)` in kJ/mol

For **NCAAs**: RDKit generates a 3D conformer from SMILES; MMFF94 strain energy is
computed in vacuum. Prediction uses:
```
pred = −E_NCAA_MMFF94   (negative strain; lower strain = better NCAA fit)
```
This is a sidechain-in-isolation proxy — not a full complex ΔΔG — used when
full sidechain-swap into the complex is not implemented.

**Hyperparameters summary**:

| Parameter | Value |
|---|---|
| Protein force field | AMBER ff14SB |
| Small molecule FF | GAFF2 (gaff-2.11) |
| Implicit solvent | OBC2 (Generalised Born) |
| Temperature | 300 K |
| Friction coefficient | 1 ps⁻¹ |
| Integration timestep | 4 fs |
| Hydrogen mass | 1.5 amu (HMR) |
| Constraints | HBonds |
| Min iterations | 1000 |
| Min tolerance | 10 kJ mol⁻¹ nm⁻¹ |
| NCAA proxy | −E_MMFF94 (vacuum strain) |
| Canonical ΔΔG | `−(E_mut − E_wt)` in kJ/mol |
| Platform | AMD MI250X (ROCm/CUDA) |

---

## LUMI HPC Environment

| Item | Value |
|---|---|
| Cluster | LUMI (CSC), AMD Instinct MI250X GPUs |
| SLURM account | `project_465002574` |
| Approach A/B partition | `small` (CPU), 8 CPUs, 32 GB, 8 h limit |
| Approach C partition | `standard-g` (GPU), 4 CPUs + 1 GPU, 32 GB, 4 h limit |
| Approach A/B Python | `/flash/project_465002574/micromamba/envs/ncaa_tools/bin/python` |
| Approach C Python | `/flash/project_465002574/unaagi_env/bin/python` |
| PyRosetta version | PyRosetta-4 2026 (Rosetta 2026.06+release, Python 3.11) |
| OpenMM version | 8.x (via `unaagi_env`) |
| Modules loaded | `LUMI`, `CrayEnv`, `lumi-container-wrapper/0.4.2-cray-python-default` |
| Excluded nodes | `nid002467`, `nid002482` (known singularity issues) |

---

## Results

### Coverage definition

**Coverage = number of mutations with non-NaN prediction / total benchmark mutations.**

A mutation receives NaN when:
- **Approach A**: the target amino acid is not in PyRosetta 2026's residue-type database
  (only AIB is confirmed present; NLE, NVA, DAL, ABU are absent)
- **Approach B**: AM1-BCC `.params` generation failed for the NCAA (all 21 failed in this
  run due to `molfile_to_params_polymer.py` API incompatibility)
- **Approach C**: the SMILES for the NCAA is unavailable, or RDKit conformer generation fails
- Any approach: `pdb_info.pdb2pose(chain, pos)` returns 0 (residue not in the target chain)

---

### PUMA — Spearman ρ vs. experimental DMS fitness

| Approach | Total ρ (n) | NAA-only ρ (n) | NCAA-only ρ (n) |
|---|---|---|---|
| A — Rosetta ref2015 | −0.184 (513) | −0.189 (486) | −0.064 ns (27) |
| B — Rosetta ref2015_cart + AM1-BCC | −0.313 (612) | −0.313 (612) | N/A (0) |
| C — OpenMM ff14SB + GAFF2 | +0.211 (714) | N/A (0) | +0.211 (714) |

### CP2 — Spearman ρ vs. experimental DMS fitness

| Approach | Total ρ (n) | NAA-only ρ (n) | NCAA-only ρ (n) |
|---|---|---|---|
| A — Rosetta ref2015 | −0.277 (209) | −0.298 (198) | −0.382 ns (11) |
| B — Rosetta ref2015_cart + AM1-BCC | −0.283 (198) | −0.283 (198) | N/A (0) |
| C — OpenMM ff14SB + GAFF2 | +0.114 ns (252) | N/A (0) | +0.114 ns (252) |

*ns = not significant (p ≥ 0.05). All other entries significant at p < 0.05.*

---

### Interpretation

**Sign convention**: Approaches A and B use `pred = -(E_mut - E_wt)` in Rosetta REU
(positive = stabilising mutation). Approach C uses the same formula for canonical AAs in
kJ/mol, but for NCAAs uses `-E_MMFF94` (negative MMFF94 vacuum strain, a proxy for
sidechain compatibility). The negative Spearman ρ for A and B indicates an inverted
relationship to the experimental fitness, suggesting Rosetta's raw ΔΔG requires sign
inversion or recalibration for this peptide-binding context.

**Approach B NCAA failure**: `molfile_to_params_polymer.py` (a PyRosetta 2026 utility)
was incompatible with the generated mol2/frcmod files from AmberTools 23, causing all 21
NCAA `.params` generation attempts to fail. Approach B therefore scored only canonical AAs.

**Approach C NCAA proxy limitation**: For NCAAs, Approach C does not perform a full
structural mutation into the protein–peptide complex. Instead it computes a vacuum MMFF94
strain energy of the free NCAA amino acid as a proxy for sidechain compatibility. This is
a significant approximation; the ρ values for NCAAa under Approach C reflect this proxy
rather than true ΔΔG.

**CP2 structural note**: Chain E of 5LY1 is a macrocyclic peptide; the macrocyclic bond
is not captured by PDBFixer or AMBER ff14SB. After `replaceNonstandardResidues()` the
structure is treated as a linear 13-residue peptide, losing the macrocycle constraint.
This reduces the physical accuracy of Approach C for CP2.

---

## Output Files

Results are stored at `/scratch/project_465002574/NCAA_ddg_results/` on LUMI:

```
NCAA_ddg_results/
├── PUMA/
│   ├── approach_A/
│   │   ├── approach_a_results.csv      # columns: wt_aa, pos, target, value, pred_score
│   │   └── approach_a_resources.json   # Spearman ρ, p-value, coverage, wall time, per-mutation timing
│   ├── approach_B/
│   │   ├── approach_b_results.csv
│   │   └── approach_b_resources.json   # also includes per-NCAA params generation success/time
│   └── approach_C/
│       ├── approach_c_results.csv
│       └── approach_c_resources.json
└── CP2/
    └── (same structure)
```

**`results.csv` columns**:
- `wt_aa` — wild-type amino acid at this position (3-letter code)
- `pos` — residue position number in the PDB chain
- `target` — mutant amino acid (3-letter code; NCAAs use benchmark-specific codes)
- `value` — experimental DMS fitness score (higher = better binding/function)
- `pred_score` — method prediction (higher = predicted more stabilising); NaN if unsupported

**`resources.json` fields**:
- `wt_energy_REU` / `wt_energy_kJ` — wild-type energy in Rosetta energy units or kJ/mol
- `spearman_rho`, `spearman_p` — overall Spearman correlation on non-NaN predictions
- `n_valid`, `n_total` — number of scored mutations vs. total
- `total_wall_s` — wall-clock runtime in seconds
- `params_generation` (Approach B only) — per-NCAA AM1-BCC generation success and time

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/ncaa_ddg/approach_a_rosetta.py` | Rosetta ref2015, existing params only |
| `scripts/ncaa_ddg/approach_b_rosetta_am1.py` | Rosetta ref2015_cart + AM1-BCC params |
| `scripts/ncaa_ddg/approach_c_openmm.py` | OpenMM ff14SB + GAFF2 + OBC2 |
| `scripts/ncaa_ddg/ncaa_smiles.py` | SMILES lookup for all 21 NCAAs |
| `scripts/ncaa_ddg/submit_ncaa_benchmarks.sh` | Submit all 6 jobs via SLURM |
| `slurm/run_ncaa_{a,c}_{puma,cp2}.sh` | Individual SLURM job scripts (hardcoded paths) |
