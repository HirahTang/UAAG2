# UAAG2 / UNAAGI — Script Catalog

> Single index of **what every script does**, so we stop re-writing scripts that already exist.
> Before creating a new script, search this file. Companion to `CLAUDE.md` (architecture) and `eval/RUNBOOK.md` (one-button eval).
> Last audited: 2026-06-20 across local Mac, LUMI, Hendrix.

## STATUS legend
- **CANON** — the maintained version; use this one.
- **SUPERSEDED→X** — kept for history; use X instead.
- **ONE-OFF** — single debugging/inspection use; safe to ignore, do not extend.
- **VARIANT** — same job, different cluster/scale knob (collapse into a parametrized script when touched).
- **CLUSTER-ONLY** — exists only on a cluster, not in git.

---

## 0. Where the code lives (read this first)

| Location | Path | Branch | Notes |
|---|---|---|---|
| Local Mac (reference) | `~/UAAG2` | `main` (clean) | Canonical committed state; has `eval/` one-button pipeline |
| Local Mac (analysis sandbox) | `~/pdb_nglyc_dataset` | not git | Ad-hoc plotting/analysis (`compare_models.py`, modeldata CSVs) |
| LUMI (production source) | `/flash/project_465002574/UAAG2_main` | **`ring_membership`** | 57 `.sh`, 37 `.py` — **diverged from main**; active training/run scripts |
| LUMI (ADiT dev) | `/flash/project_465002574/uaag2-adit-src` | `main` | Latent-diffusion / ADiT backbone work |
| LUMI (mlops) | `/flash/project_465002574/UAAG2` | `mlops` | Older checkout, 8 `.py` |
| LUMI (scratch one-offs) | `/scratch/project_465002574/*.sh` | — | ~20 throwaway run scripts, NOT in git (see §7) |
| Hendrix (eval) | `/home/qcx679/hantang/UAAG2` | `main` + uncommitted | Same git repo; CUDA eval host |

**Action item:** three LUMI checkouts on three branches (`ring_membership`/`main`/`mlops`) is the main source of "which script is real?". Pick one production checkout; treat the others as feature worktrees.

All clusters track `github.com/HirahTang/UAAG2.git`.

---

## 1. One-button evaluation — `eval/`  (USE THIS)

The consolidated pipeline: **sampling → post-analysis → aggregate+Spearman → visualize group.**
Procedure doc an agent can follow: **`eval/PIPELINE.md`**. Everything in §5–§7 predates it.

| Script | Status | Purpose |
|---|---|---|
| `eval/PIPELINE.md` | CANON | **Follow this** — full one-button procedure, all 5 stages, LUMI+Hendrix |
| `eval/run_pipeline.sh` | CANON | **THE one button**: sample group → afterok finalize+plot. `run_pipeline.sh <m> [...]｜all` |
| `eval/finalize_and_plot.sh` | CANON | Stages 3–5: CP2/PUMA per-iter scoring → modeldata → `compare_models.py` |
| `eval/make_modeldata.py` | CANON | Stage 4 glue: per-assay outputs → `<TAG>_modeldata.csv` (+collate baselines). `--selftest` |
| `eval/RUNBOOK.md` | CANON | Cluster cheatsheet + arch/code gotcha |
| `eval/models.tsv` | CANON | LUMI manifest: one row per evaluatable checkpoint (single source of truth) |
| `eval/models_hendrix.tsv` | CANON | Hendrix (CUDA) counterpart of `models.tsv` |
| `eval/slurm_sample.sh` | CANON | Parametrized sampling array (CTMC + 20 NFE, 1000 samples × 5 iters) — LUMI |
| `eval/slurm_sample_hendrix.sh` | CANON | Hendrix port of the sampling array |
| `eval/run_eval.sh` | CANON | Sampling-only launcher (no downstream); use `run_pipeline.sh` for the full chain |
| `eval/run_eval_hendrix.sh` | CANON | Driver for Hendrix |

---

## 2. Core eval / scoring engines — `scripts/`  (CANON, called by the pipeline)

| Script | Status | Purpose |
|---|---|---|
| `aggregate_splits.py` | CANON | Merge per-iter `aa_distribution_split0.csv` → **per-iter** Spearman → `spearman_per_iter.csv` (mean/std) + `results.csv`. Use the per-iter path; running eval on the merged 5-iter CSV collapses to last-iter. |
| `result_eval_uniform.py` | CANON | ProteinGym Spearman (canonical assays); needs `--baselines` |
| `result_eval_uniform_uaa.py` | CANON | UAA/NCAA Spearman for CP2/PUMA; score = **logP_wt − logP_mut**, 1/total_num floor, skip pos 156 + WT-Pro |
| `post_analysis.py` | CANON | Per-sample chirality/AA-identity classification → AA counts (called inline by sampler) |
| `generate_ligand_dpm.py` | CANON | The sampler. Inode-safe (mol→/tmp→tar). Flags: `--ctmc --every-k-step 25 --dpm-solver-pp` |
| `run_train.py` | CANON | Training entry point (dpm-solver-pp branch) |

Eval env (Hendrix): conda `targetdiff` with `LD_LIBRARY_PATH=$env/lib` (see memory `reference_unaagi_eval_env`).

---

## 3. Plotting — consolidated (done 2026-06-20)

**Gold-standard reference output:** `~/unaagi_v03_plots/` (4 SVGs). Any plotting must match that style.

| Script | Location | Status | Purpose |
|---|---|---|---|
| `compare_models.py` | repo `scripts/` | **CANON** | **N-model** plotter; reproduces `~/unaagi_v03_plots/`. Args `--data-dir --output-dir`. Reads `<model>_modeldata.csv` + `baselines_pg/`; stars on a shared vertical line per assay, per-iter error bars, all baselines. Example inputs in `scripts/example_data/v03_compare/`. |
| `generate_comparison_plots.py` | repo `scripts/` | CANON (CLI path) | Original multi-model CLI; N models via `--extra-results-dirs/--extra-labels`. Documented in CLAUDE.md. Keep until `compare_models.py` grows a results-dir mode. |
| `compare_ctmc_vs_ddpm.py` | `scripts/legacy/` | RETIRED→compare_models | 2-model hardcoded; the styling base |
| `generate_clean_plots.py` | `scripts/legacy/` | RETIRED→compare_models | made original `figures/p0203_ctmc` |
| `generate_desktop_compare_reports.py` | `scripts/legacy/` | RETIRED→compare_models | desktop HTML report + NCAA coverage |
| `plot_model_comparison.py` | — | DELETED | earlier reinvention (removed 2026-06-20) |

See `scripts/legacy/README.md`.

---

## 4. Data / LMDB building — `scripts/` + build `.sh`  (CANON, run rarely)

| Script | Status | Purpose |
|---|---|---|
| `build_eqgat_lmdb_from_pdb.py` | CANON | Build EQGAT graph LMDB from PDB folder (shardable) |
| `merge_lmdb_shards.py` | CANON | Merge sharded LMDBs into one |
| `build_pdbbind_lmdb.py` | CANON | Build `PDBBind.lmdb` from PDBBind 2020 |
| `build_ncaa_lmdb.py` | CANON | Convert NCAA `.pt` → `NCAA.lmdb` |
| `pdb_clean.py` | CANON | Strip hydrogens / clean PDB atoms |
| `process_cif.py` | CANON | mmCIF → cleaned PDB |
| `enumerate_pdb_atom_types.py` | CANON | Enumerate atom names/elements across a PDB folder |
| `split_config_files.py` | CANON | Split `slurm_config.txt` → 25 per-assay configs |
| `check_benchmarks.py` | CANON | Validate benchmark `.pt` files |
| `generate_ncaa_coverage.py` | CANON | NCAA mutation coverage per position per model |
| sbatch wrappers: `run_build_eqgat_lmdb*.sh`, `submit_eqgat_lmdb_60jobs.sh`, `run_merge_eqgat_lmdb_sbatch.sh`, `run_build_pdbbind.sh`, `run_pdb_clean_sbatch.sh`, `run_remove_pdb_water_sbatch.sh` / `extract_pdb_water.sh`, `unzip_pdb_sbatch.sh`, `run_enumerate_pdb_atom_types_sbatch.sh` | VARIANT/CANON | LUMI batch wrappers for the builders above |

---

## 5. Training launchers — top-level `run_train_*.sh`  (17 VARIANTs)

All call `scripts/run_train.py`; they differ only by node/GPU count and data subset. Collapse into one parametrized launcher when next touched.

| Script | Knob |
|---|---|
| `run_train_1node.sh` / `_8lumi.sh` / `_10nodes.sh` / `_16lumi.sh` / `_100nodes.sh` | node/GPU scale |
| `run_train_DDP_{1node_4gpus_demo,2nodes,5nodes}.sh`, `run_train_single_demo.sh`, `run_train_lumi.sh`, `run_train_8lumi_test.sh` | DDP/demo variants |
| `run_train_8lumi_exclude_{VAL,PHE,ILE,SER}.sh` | leave-one-canonical-AA-out ablations |
| `run_train_flagship_v0.11.sh`, `run_train_flagship_v0.1_resume.sh` | flagship runs / resume |

(Newer v0.2/v0.3 launchers live on LUMI `ring_membership`, see §7 — not all are in `main`.)

---

## 6. Sampling / assay launchers — SUPERSEDED by `eval/` (§1)

| Script(s) | Status |
|---|---|
| `submit_assay_{ctmc,ctmc_1k,dpm_swift,job_swift,jobs}.sh` (5) | SUPERSEDED→eval/slurm_sample.sh |
| `run_sampling_{CP2,PUMA,prior_puma,prior}_sbatch.sh` (4) | SUPERSEDED→eval/slurm_sample.sh |
| `run_assay_array.sh`, `run_assay_postprocess.sh`, `run_pipeline_monitored.sh`, `test_pipeline.sh`, `test_single_assay.sh` | SUPERSEDED→eval/run_eval.sh |
| `compare_ctmc_ddpm.sh` | SUPERSEDED→(plotting §3) |

---

## 7. Diagnostics, demos, cleanup, scratch one-offs — ONE-OFF (don't extend)

**Repo diagnostics/demos:** `diagnose2.py`, `diagnose_mismatch.py`, `diagnose_pdbbind.py` (+`run_diagnose_pdbbind.sh`), `aa_conversion_stats_demo.py`, `demo_direct.py`, `demo_lmdb_graph.py`, `demo_pdb_dataset.py`, `test_format_vs_lmdb.py`, `evaluate_mol_samples.py` (PoseBusters), `spearman_mut_v_summary.py`, `inspect_demo_lmdb.ipynb`, `lmdb_check.ipynb`.

**Cleanup / posebuster `.sh`:** `cleanup_mol_files{,_array}.sh`, `cleanup_summary.sh`, `posebuster_array.sh`, `run_posebuster_array.sh`, `run_prior_posebuster_cleanup.sh`, `read_lmdb_test.sh`, `install_rocm.sh` (CANON for ROCm setup).

**LUMI `/scratch/*.sh` (CLUSTER-ONLY, not in git):** `run_cp2_full.sh`, `run_cp2_v2_full.sh`, `run_puma_v2_full{,2,3}.sh`, `run_ncaa_{a,c}_{cp2,puma}.sh`, `submit_prior_5iter.sh`, `submit_prior_hi_splits.sh`, `sbatch_{mv,rm,unzip,unzippdb}.sh`, `extract_demo{,_sg}.sh`, `count_files.sh`, `test_c_v2_11b.sh`, `run_sbatch_10h.sh`.
→ These are throwaway per-run scripts. They are the main cause of script re-creation. When one proves useful, fold it into `eval/` and delete the rest.
