# UAAG2 Flagship Model Cards

This document records the full data processing pipeline, input graph format, and training configuration for the Flagship model generations (v0.0 → v0.3). Intended as a reference for reproducibility and cross-run comparison.

> **Current flagship:** `Flagship_v0.3_ring_weighted_8gpu_20260531_cont` (§11). It restores both features lost in the v0.2 regression — the 10× PDBBind/NCAA sampling weights (Fix A) and the `is_in_ring` node channel.

---

## Table of Contents

1. [Shared Architecture](#1-shared-architecture)
2. [Shared Data Processing Pipeline](#2-shared-data-processing-pipeline)
3. [Shared Input Graph Format](#3-shared-input-graph-format)
4. [Model: flagship_400gpu_alldata_pocketdropout_0.2 (v0.0)](#4-model-flagship_400gpu_alldata_pocketdropout_02-v00)
5. [Model: Flagship_400gpu_alldata_pocketdropout_0.2v0.1 (v0.1)](#5-model-flagship_400gpu_alldata_pocketdropout_02v01-v01)
6. [Model: Flagship_400gpu_alldata_pocketdropout_0.2v0.1_20260419 (v0.1 resume)](#6-model-flagship_400gpu_alldata_pocketdropout_02v01_20260419-v01-resume)
7. [Model: Flagship_400gpu_alldata_pocketdropout_0.2v0.11 (v0.11)](#7-model-flagship_400gpu_alldata_pocketdropout_02v011-v011)
8. [Model: Flagship_400gpu_alldata_pocketdropout_{0.2,0.5}v0.2 (v0.2)](#8-model-flagship_400gpu_alldata_pocketdropout_0205v02-v02)
9. [Model: Flagship_v0.3_weighted_8gpu_20260601 (v0.3 weighted — Fix A)](#9-model-flagship_v03_weighted_8gpu_20260601-v03-weighted--fix-a)
10. [Model: Flagship_v0.3_ring_weighted_8gpu_20260531 (v0.3 ring-weighted — current flagship)](#10-model-flagship_v03_ring_weighted_8gpu_20260531-v03-ring-weighted--current-flagship)
11. [Key Differences Summary](#11-key-differences-summary)

---

## 1. Shared Architecture

Both models use the same EQGAT-based equivariant diffusion backbone. All architectural hyperparameters are identical.

| Parameter | Value |
|---|---|
| Backbone | EQGAT (equivariant graph attention) |
| Layers | 7 |
| Scalar dim (`sdim`) | 256 |
| Vector dim (`vdim`) | 64 |
| Edge dim (`edim`) | 32 |
| Diffusion timesteps | 500 |
| Noise scheduler | Cosine |
| Loss weighting | SNR-weighted (`snr_t`) |
| Atom type classes | 8 (C, N, O, S, P, Cl, F, Br) |
| Bond type classes | 5 (none, single, double, aromatic, triple) |
| Charge classes | 6 |
| EMA decay | 0.9999 |
| Dropout | 0.3 |
| Grad clip | 10.0 (v0.0–v0.11) → **1.0** (v0.2+, after the v0.11 LR-0.005 collapse; 0.5 in the v0.2 diag run) |
| Optimizer | Adam (weight decay 0.9999) |
| LR scheduler | ReduceOnPlateau (factor 0.75, patience 20, cooldown 5) |
| LR min | 5e-5 |

---

## 2. Shared Data Processing Pipeline

### 2.1 Training Datasets

All three sources are used with equal weight (1.0), mixed by the `CombinedDataset` sampler.

| Source | Type | Path | Description |
|---|---|---|---|
| PDB | On-the-fly PDB parsing | `/scratch/project_465002574/PDB/PDB_processed` | Non-redundant PDB structures, pre-cleaned |
| PDBBind | LMDB | `/scratch/project_465002574/PDB/PDBBind.lmdb` | Protein-ligand complexes from PDBBind |
| NCAA | LMDB | `/scratch/project_465002574/PDB/NCAA/NCAA.lmdb` | Non-canonical amino acid structures |

### 2.2 PDB On-the-Fly Graph Construction (`UAAG2DatasetPDB`)

Each training sample is one (protein structure, target residue) pair. The pipeline runs entirely at worker load time — no pre-built graph cache.

**Step 1 — PDB parsing (cached per worker)**

- Read `.pdb` file using RDKit (`Chem.MolFromPDBFile`, `removeHs=True`) and BioPython (`PDBParser`)
- If the PDB file lacks element columns or is otherwise malformed, it is repaired in a temp file before parsing
- Atom identity is resolved by matching RDKit `MonomerInfo` (chain, resseq, icode, atom name) to BioPython atoms — robust to altloc discrepancies; 'A' conformation is preferred
- Parsed result is cached per worker via `@lru_cache(maxsize=64)` so multiple residues from the same PDB pay parse cost only once
- Bonds are extracted from RDKit; 3D coordinates come from BioPython (crystallographic positions)

**Step 2 — Residue identification**

- All residues in the structure are enumerated as `ResidueRecord` objects
- Each record holds: residue name, chain, resseq, atom indices, mass center, and an `aa_order_index` (set only for standard/non-standard amino acids)
- Residues with fewer than 4 atoms, or with atoms outside the supported element/hybridization vocabulary, are skipped

**Step 3 — Pocket selection**

- For a chosen center residue, all neighbouring residues whose mass center is within `pocket_radius = 10.0 Å` are collected as the pocket context
- Neighbour search is vectorised with a bounding-box pre-filter then exact L2 norm check

**Step 4 — Graph node features**

Atoms from the center residue (`is_ligand=1`) and pocket neighbours (`is_ligand=0`) are merged into a single node set. Maximum 400 atoms total; graphs exceeding this are discarded.

| Field | Dtype | Encoding |
|---|---|---|
| `x` | long | Atom type: C=0, N=1, O=2, S=3, P=4, Cl=5, F=6, Br=7 |
| `pos` | float | 3D coordinates (Å), before centering |
| `charges` | float | Formal charge mapped: {-1→0, 0→1, 1→2, 2→3} |
| `hybridization` | float | SP=0, SP2=1, SP3=2, SP3D=3 |
| `degree` | float | RDKit bond degree |
| `is_aromatic` | float | 1.0 if aromatic, else 0.0 |
| `is_ligand` | float | 1.0 for center residue atoms, 0.0 for pocket |
| `is_backbone` | float | 1.0 for first 4 atoms of the center residue (backbone proxy) |

**Step 5 — Graph edge construction**

Three classes of edges are built; all edges are bidirectional:

| Edge class | Rule | `edge_ligand` flag |
|---|---|---|
| Ligand–ligand | Fully connected among all center residue atoms | 1.0 |
| Pocket–pocket | All pairs within `edge_radius = 8.0 Å` (vectorised `scipy.cdist`) | 0.0 |
| Ligand–pocket cross | All (ligand atom, pocket atom) pairs, regardless of distance | 0.0 |

Edge attribute `edge_attr` carries bond type (0=no covalent bond, 1=single, 2=double, 3=aromatic, 4=triple) read from RDKit.

**Step 6 — Post-processing (`_post_process_graph`)**

Applied to every sample after graph construction, in this exact order:

1. **Pocket centering** — subtract the mean position of all pocket atoms (`is_ligand==0`) from all atom positions. If no pocket atoms exist (pure ligand graph), subtract the overall mean. This step always runs, even when pocket dropout fires.

2. **Compute ligand CoM** — centre-of-mass of center residue atoms after centering. Used as virtual node seed position.

3. **Pocket dropout** (prob=0.2) — with probability 0.2, remove all pocket atoms (`is_ligand==0`). Edges incident to removed atoms are also removed. Because centering already ran, the residue retains its correct position relative to the (now absent) pocket.

4. **Charge remapping** — integer formal charges remapped to class indices via `CHARGE_EMB = {-1:0, 0:1, 1:2, 2:3}`.

5. **Backbone masking** — of the non-backbone ligand atoms (`is_ligand - is_backbone == 1`), each is independently masked (set `is_backbone=1`) with probability `mask_rate`. This is the atom identity masking for the diffusion training objective.

6. **Virtual nodes** — with `virtual_node=True` and `mask_rate=0.5`, a random number of virtual (placeholder) nodes is appended: `sample_n ~ Uniform(1, max_virtual_nodes)`. Virtual nodes have atom type 8, placed at the ligand CoM, `is_ligand=1`, `is_backbone=0`, charge=1, degree/aromaticity/hybridization=0. They are fully connected to all existing nodes.

---

## 3. Shared Input Graph Format

The `Data` object passed to the model has the following fields:

### Node fields (shape `[N, ...]`)

| Field | Shape | Description |
|---|---|---|
| `x` | `[N]` float | Atom type (0–8; 8 = virtual) |
| `pos` | `[N, 3]` float | 3D coordinates, pocket-centred (Å) |
| `charges` | `[N]` float | Charge class index (0–3) |
| `hybridization` | `[N]` float | Hybridization class (0–3) |
| `degree` | `[N]` float | Bond degree |
| `is_aromatic` | `[N]` float | Aromaticity flag |
| `is_ligand` | `[N]` float | 1 = center residue or virtual, 0 = pocket |
| `is_backbone` | `[N]` float | 1 = backbone atom (masked during training) |

### Edge fields (shape `[E, ...]`)

| Field | Shape | Description |
|---|---|---|
| `edge_index` | `[2, E]` long | Bidirectional edge list |
| `edge_attr` | `[E]` float | Bond type (0–4) |
| `edge_ligand` | `[E]` float | 1 = ligand–ligand edge, 0 = other |

### Metadata

| Field | Type | Description |
|---|---|---|
| `compound_id` | str | PDB accession / source identifier |
| `source_name` | str | Path stem of source PDB file |
| `center_residue` | str | `RES_resseq_chain` of the target residue |
| `residue_name` | str | 3-letter residue code |

### Context fields (original Flagship v0.0 only)

| Field | Shape | Description |
|---|---|---|
| `protein_mpnn_latent_node_128` | `[N, 128]` float | ProteinMPNN 128-dim embedding, broadcast to all atoms |
| `protein_mpnn_latent_node_20` | `[N, 20]` float | ProteinMPNN 20-dim embedding, broadcast to all atoms |

These fields are absent in v0.1 (ProteinMPNN disabled).

---

## 4. Model: `flagship_400gpu_alldata_pocketdropout_0.2` (v0.0)

**Checkpoint:** `/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runflagship_400gpu_alldata_pocketdropout_0.2/last.ckpt`  
**Launch script:** `run_train_100nodes.sh`  
**WandB run ID:** `flagship_400gpu_alldata_pocketdropout_0.2`  
**Approx. training date:** 2026-04-14 (final checkpoint)

### Training hyperparameters

| Parameter | Value |
|---|---|
| Nodes × GPUs | 50 × 8 = 400 GPUs |
| Batch size per GPU | 8 |
| Effective batch size | 3,200 |
| Learning rate | 0.0005 |
| Gradient steps (est.) | ~10,000 |
| Max epochs | 300 |
| Mask rate | 0.5 |
| Max virtual nodes | 11 |
| Pocket dropout prob | 0.2 |

### Context conditioning

- **ProteinMPNN enabled**: `use_protein_mpnn_context_128=true`, `context_mapping=true`
- `num_context_features=128`
- ProteinMPNN 128-dim per-residue embeddings loaded from `/scratch/project_465002574/PDB/PDB_128` via `LatentStore`, broadcast to all atoms as `protein_mpnn_latent_node_128`
- At inference: benchmark `.pt` graphs **do not** contain this field → `_get_context_from_batch` falls back to `torch.zeros(N, 128)` → completely out-of-distribution input

### Known issues

1. **Zero context at inference**: The model was conditioned on ProteinMPNN embeddings during training, but benchmark graphs have no such embeddings. The model receives all-zero context at inference, which is OOD and causes near-random performance.
2. **LR under-scaled**: Effective batch 3,200 is 50× larger than the Prior model (batch 64), but the same LR (0.0005) was used. By the linear scaling rule, LR should have been ~0.025.
3. **Too few gradient steps**: ~10,000 steps vs ~1.1M for the Prior model despite more GPU-hours.

---

## 5. Model: `Flagship_400gpu_alldata_pocketdropout_0.2v0.1` (v0.1)

**Checkpoint (target):** `/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/Flagship_400gpu_alldata_pocketdropout_0.2v0.1/`  
**Launch script:** `run_train_flagship_nompnn.sh`  
**WandB run ID:** `Flagship_400gpu_alldata_pocketdropout_0.2v0.1`

### Training hyperparameters

| Parameter | Value |
|---|---|
| Nodes × GPUs | 50 × 8 = 400 GPUs |
| Batch size per GPU | 2 |
| Effective batch size | 800 |
| Learning rate | 0.005 |
| Max epochs | 300 (5,000 steps/epoch cap) |
| Mask rate | 0.5 |
| Max virtual nodes | 11 |
| Pocket dropout prob | 0.2 |

LR 0.005 derived from linear scaling rule: `0.0005 × (800 / 64) ≈ 0.006`, rounded conservatively to 0.005.

### Context conditioning

- **ProteinMPNN fully disabled**
- `use_protein_mpnn_context_128` flag removed
- `--latent-root-128` and `--latent-root-20` paths not passed
- `num_context_features` defaults to 0; `context_mapping` defaults to `False`
- `_get_context_from_batch` returns `None` → context vector is never constructed or fed to EQGAT
- Model is fully unconditional — consistent between training and inference

### Motivation for changes vs v0.0

| Issue in v0.0 | Fix in v0.1 |
|---|---|
| ProteinMPNN context used in training but absent at inference (OOD zeros) | ProteinMPNN removed entirely — unconditional model |
| Effective batch 3,200 → LR severely under-scaled | Effective batch 800, LR scaled to 0.005 |
| ~10,000 gradient steps despite 400-GPU scale | Smaller per-GPU batch → 4× more gradient steps per epoch |
| GPU utilisation poor at batch=1/GPU | batch=2/GPU — better wavefront utilisation vs batch=1 while still 4× more steps than v0.0 |

---

## 6. Model: `Flagship_400gpu_alldata_pocketdropout_0.2v0.1_20260419` (v0.1 resume)

**Checkpoint (target):** `/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFlagship_400gpu_alldata_pocketdropout_0.2v0.1_20260419/`
**Resumed from:** `/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFlagship_400gpu_alldata_pocketdropout_0.2v0.1/last.ckpt`
**Launch script:** `run_train_flagship_v0.1_resume.sh`
**WandB run ID:** `Flagship_400gpu_alldata_pocketdropout_0.2v0.1_20260419`
**Created:** 2026-04-19

### Purpose

Continue v0.1 training from its latest checkpoint into a fresh project folder so the source v0.1 directory is preserved unchanged. All hyperparameters and dataset weights match v0.1 exactly.

### Training hyperparameters

| Parameter | Value |
|---|---|
| Nodes × GPUs | 12 × 8 = 96 GPUs |
| Batch size per GPU | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 768 (≈ v0.1's 800) |
| Learning rate | 0.005 |
| Mask rate | 0.5 |
| Max virtual nodes | 11 |
| Pocket dropout prob | 0.2 |
| ProteinMPNN context | Disabled |

Node count reduced from 50 → 12 to fit tighter compute budget while keeping effective batch ≈ v0.1's 800 via `--accum-batch 4`, which preserves the optimizer/LR dynamics the checkpoint was trained under.

### Resume mechanics

- `--load-ckpt` passes `last.ckpt` to `pl.Trainer.fit(ckpt_path=…)` → full resume of optimizer state, LR scheduler state, global step, and epoch counter.
- LR kept at 0.005 (== stored LR in ckpt) so `run_train.py` skips writing a `retraining_with_lr*.ckpt` sidecar into the source v0.1 folder — source folder remains untouched.
- New checkpoints write into `runFlagship_400gpu_alldata_pocketdropout_0.2v0.1_20260419/`.

---

## 7. Model: `Flagship_400gpu_alldata_pocketdropout_0.2v0.11` (v0.11)

**Checkpoint (target):** `/flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/runFlagship_400gpu_alldata_pocketdropout_0.2v0.11/`
**Launch script:** `run_train_flagship_v0.11.sh`
**WandB run ID:** `Flagship_400gpu_alldata_pocketdropout_0.2v0.11`
**Created:** 2026-04-19

### Purpose

Train from scratch on a small PDB subset (1,000 files) combined with the full PDBBind and full NCAA corpora. Goal: test whether restricting PDB breadth but keeping PDBBind/NCAA coverage changes fitness-prediction generalisation, and get cheaper gradient passes over PDBBind/NCAA by collapsing the dominant PDB dataset.

### Training hyperparameters

| Parameter | Value |
|---|---|
| Nodes × GPUs | 12 × 8 = 96 GPUs |
| Batch size per GPU | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 768 (matches v0.1 dynamics) |
| Learning rate | 0.005 |
| Mask rate | 0.5 |
| Max virtual nodes | 11 |
| Pocket dropout prob | 0.2 |
| ProteinMPNN context | Disabled |

### Dataset composition

- **PDB**: 1,000 files drawn at random (seeded) from `/scratch/project_465002574/PDB/PDB_processed`. The chosen basenames are persisted to `/flash/project_465002574/UAAG2_main/data/pdb_subset_1000_v0.11.txt` on first run. All subsequent runs — continue training, reruns, evaluation — read the identical list from that file, so the subset is fixed for the full life of the experiment.
  - CLI: `--pdb-max-files 1000 --pdb-subset-file .../pdb_subset_1000_v0.11.txt --pdb-subset-seed 42`
  - First run: writes the file and uses `random.Random(42).sample(sorted_all_pdbs, 1000)`.
  - Later runs: file exists → read basenames verbatim, `--pdb-max-files` and seed are ignored.
  - Index cache: `pdb_subset_1000_v0.11.txt.index_cache.pkl` (sibling of subset file), isolated from the full-dataset cache.
- **PDBBind**: full LMDB at `/scratch/project_465002574/PDB/PDBBind.lmdb`.
- **NCAA**: full LMDB at `/scratch/project_465002574/PDB/NCAA/NCAA.lmdb`.
- All three sources mixed every epoch (equal weight = 1.0 each, per-item weighted sampler).

---

## 8. Model: `Flagship_400gpu_alldata_pocketdropout_{0.2,0.5}v0.2` (v0.2)

**Checkpoint dirs:** `runFlagship_400gpu_alldata_pocketdropout_0.2v0.2/` and `…_0.5v0.2/` (+ dated `_20260521` and `_resume_*` continuations)
**WandB run IDs:** `Flagship_400gpu_alldata_pocketdropout_0.2v0.2`, `…_0.5v0.2`
**Created:** 2026-05-21 → 2026-05-25

### Purpose

Move off the fixed PDB subset of v0.11 back to the **full live all-data mix** at a more economical scale, and switch the training objective from masked-virtual-node generation (`mask_rate=0.5`) to **pure denoising of the full side chain** (`mask_rate=0.0`). Two pocket-dropout settings (0.2, 0.5) were trained in parallel.

### Training hyperparameters

| Parameter | Value |
|---|---|
| GPUs | 48 |
| Batch size per GPU | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 768 |
| Learning rate | 0.005 |
| **Grad clip** | **1.0** (tightened from 10.0 after the v0.11 collapse) |
| **Mask rate** | **0.0** (no virtual-node masking — denoise the full residue) |
| Max virtual nodes | 11 |
| Pocket dropout prob | 0.2 / 0.5 (two variants) |
| ProteinMPNN context | Disabled |

### Dataset composition

- Full **PDB + PDBBind + NCAA**, built **on-the-fly** (`use_lmdb=false`).
- **All three sources weighted equally (1.0 / 1.0 / 1.0).** ← **Regression vs. the prior SOTA**, which up-weighted PDBBind and NCAA 10×.

### Known regressions (fixed in v0.3)

1. **Uniform sampling weights** — PDBBind/NCAA (the high-value small corpora) are drowned out by the large PDB pool. The earlier SOTA used a 10× up-weight; this was dropped here.
2. **`is_in_ring` node feature dropped** — the ring-membership channel present in the prior SOTA graph format is absent from the v0.2 data build.

> **`Flagship_v0.2_diag_e32_20260526`** is a short diagnostic resume off `…0.2v0.2_resume_20260521/epoch=31` (LR 0.0025, grad-clip 0.5, 33 epochs) to probe training stability; not a release candidate.

---

## 9. Model: `Flagship_v0.3_weighted_8gpu_20260601` (v0.3 weighted — Fix A)

**Checkpoint dir:** `runFlagship_v0.3_weighted_8gpu_20260601/`
**WandB run ID:** `Flagship_v0.3_weighted_8gpu_20260601`
**Initialisation:** from scratch (`load_ckpt=null`)
**Created:** 2026-06-01

### Purpose

**Fix A** for the v0.2 regression: restore the **10× up-weighting of PDBBind and NCAA** via the per-item `WeightedRandomSampler`, at an 8-GPU scale for fast iteration.

### Training hyperparameters

| Parameter | Value |
|---|---|
| GPUs | 8 |
| Batch size per GPU | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 128 |
| Learning rate | 0.0005 (restored from v0.2's 0.005 to suit the smaller batch) |
| Grad clip | 1.0 |
| Mask rate | 0.0 |
| Max virtual nodes | 11 |
| Pocket dropout prob | 0.5 |
| ProteinMPNN context | Disabled |
| Max epochs | 50 |

### Dataset composition

- Full **PDB + PDBBind + NCAA**, on-the-fly (`use_lmdb=false`).
- **Sampling weights: PDB 1.0, PDBBind 10.0, NCAA 10.0** — the restored SOTA up-weighting.
- Still uses the v0.2 graph build (no `is_in_ring`); that is added in v0.3 ring-weighted (§10).

---

## 10. Model: `Flagship_v0.3_ring_weighted_8gpu_20260531` (v0.3 ring-weighted — current flagship)

**Checkpoint dir:** `runFlagship_v0.3_ring_weighted_8gpu_20260531/` → continued in `…_cont/`
**Latest checkpoint:** `runFlagship_v0.3_ring_weighted_8gpu_20260531_cont/epochepoch=021.ckpt` (epoch 21, 2026-06-05); `last.ckpt` = epoch 19 / step 25000
**WandB run IDs:** `Flagship_v0.3_ring_weighted_8gpu_20260531`, `…_cont`
**Initialisation:** base run from scratch; `_cont` resumes from `…_20260531/last.ckpt`
**Created:** 2026-05-31 (base) → 2026-06-05 (cont)
**Source branch:** `ring_membership`

### Purpose

The **current flagship**. Combines **Fix A** (10× PDBBind/NCAA up-weighting, as in §9) with the **re-added `is_in_ring` node feature** — restoring *both* features lost in the v0.2 regression. The `is_in_ring` channel was re-plumbed through the on-the-fly PDB build and the PDBBind/NCAA LMDBs were rebuilt to carry it (originals backed up as `*.bak_pre_ring_20260531`).

### Training hyperparameters

| Parameter | Value |
|---|---|
| GPUs | 8 |
| Batch size per GPU | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 128 |
| Learning rate | 0.0005 |
| Grad clip | 1.0 |
| Mask rate | 0.0 |
| Max virtual nodes | 11 |
| Pocket dropout prob | 0.5 |
| ProteinMPNN context | Disabled |
| Max epochs | 50 |

### Dataset composition & graph format

- Full **PDB + PDBBind + NCAA**, on-the-fly (`use_lmdb=false`).
- **Sampling weights: PDB 1.0, PDBBind 10.0, NCAA 10.0.**
- **`is_in_ring`** added as a per-atom node feature (1.0 if the atom is in a ring, else 0.0), extending the node-feature set in §3.

---

## 11. Key Differences Summary

| Dimension | v0.0 | v0.1 | v0.1_20260419 | v0.11 | v0.2 | v0.3 weighted | **v0.3 ring-weighted (flagship)** |
|---|---|---|---|---|---|---|---|
| **ProteinMPNN context** | 128-dim, on in training; zeros at inference | Off | Off | Off | Off | Off | Off |
| **Sampling weights (PDB/PDBBind/NCAA)** | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | **1 / 10 / 10** | **1 / 10 / 10** |
| **`is_in_ring` feature** | — | — | — | — | dropped | dropped | **re-added** |
| **Learning rate** | 0.0005 | 0.005 | 0.005 | 0.005 | 0.005 | 0.0005 | 0.0005 |
| **Effective batch** | 3,200 | 800 | 768 (accum=4) | 768 (accum=4) | 768 (accum=4) | 128 (accum=4) | 128 (accum=4) |
| **GPUs** | 400 | 400 | 96 | 96 | 48 | 8 | 8 |
| **Mask rate** | 0.5 | 0.5 | 0.5 | 0.5 | **0.0** | 0.0 | 0.0 |
| **Pocket dropout** | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 / 0.5 | 0.5 | 0.5 |
| **Grad clip** | 10.0 | 10.0 | 10.0 | 10.0 | 1.0 | 1.0 | 1.0 |
| **Training data** | PDB+PDBBind+NCAA (full) | Same | Same | PDB (1,000 seeded) + PDBBind + NCAA | Full, on-the-fly | Full, on-the-fly | Full, on-the-fly |
| **Initialisation** | scratch | scratch | resume v0.1 | scratch | scratch | scratch | scratch (+ `_cont` resume) |
| **Architecture** | EQGAT, 7L, sdim=256 | Same | Same | Same | Same | Same | Same |
| **Max virtual nodes** | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| **Checkpoint dir** | `runflagship_…0.2/` | `runFlagship_…0.2v0.1/` | `…0.2v0.1_20260419/` | `…0.2v0.11/` | `…{0.2,0.5}v0.2/` | `runFlagship_v0.3_weighted_8gpu_20260601/` | `runFlagship_v0.3_ring_weighted_8gpu_20260531_cont/` |
