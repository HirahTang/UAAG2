# UNAAGI one-button evaluation pipeline

**Purpose:** evaluate a *group* of trained UNAAGI/UAAG2 checkpoints end-to-end and produce
the comparison figures, in one command. An agent can follow this file top-to-bottom.

```
sampling ─▶ post-analysis (aa_distribution) ─▶ aggregate + Spearman ─▶ visualize group
   stage 1            stage 1 (inline)              stages 3–4              stage 5
```

The reference output (what the figures must look like): the 4 SVGs in `~/unaagi_v03_plots/`
(`proteingym_comparison.svg` + `barplot_{all,ncaa,naa}_mutations.svg`).

---

## TL;DR — the one button

```bash
# LUMI, from /flash/project_465002574/UAAG2_main   (ring_membership checkout)
eval/run_pipeline.sh v0.3ring_base v0.3ring_cont v0.3weighted_end p0203_ctmc
# or
eval/run_pipeline.sh all
```

That submits sampling for every model, then submits the finalize+plot job with a SLURM
`afterok` dependency on all of them. When it finishes, the figures are in
`/scratch/project_465002574/UNAAGI_result/figures/pipeline_<timestamp>/`.

Nothing else is required. The sections below explain each stage and how to run it
manually / debug it.

> **Compute note (2026-06):** LUMI `project_465002574` GPU budget is exhausted; GPU sbatch
> is rejected until topped up (fallback `project_465002988`). Hendrix is the active compute —
> use the Hendrix equivalents in §6.

---

## Prerequisites

1. **Model must be in the manifest** `eval/models.tsv` (one row: `name·ckpt·arch·code_ref·notes`).
   To evaluate a new checkpoint, add one row. `arch` (`ring`/`prering`) selects the code
   checkout automatically — see `eval/RUNBOOK.md` "the one gotcha" for why this matters.
2. **Env**: LUMI `unaagi_env` container (loaded by the SLURM scripts). For manual python runs
   on Hendrix use conda `targetdiff` + `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib`.
3. **Benchmarks/baselines present** (already on scratch):
   - `UNAAGI_benchmarks/{<PROTEIN>.pt, 5ly1_cp2.pt, 2roc_puma.pt}`
   - `UNAAGI_benchmark_values/baselines/<PROTEIN>_*.csv` (ProteinGym)
   - `UNAAGI_benchmark_values/uaa_benchmark_csv/{CP2,PUMA}_reframe.csv` (truth)

---

## Stage 1 — sampling (+ inline post-analysis & ProteinGym aggregate)

Script: `eval/run_pipeline.sh` → `eval/slurm_sample.sh` (135-task array = 27 assays × 5 iters).
Per task it runs `scripts/generate_ligand_dpm.py` with `--ctmc --every-k-step 25 --dpm-solver-pp
--num-samples 1000`, which:
- samples 1000 side chains, writes mol files to node-local `/tmp`, tars to scratch (inode quota),
- runs `post_analysis.py` inline → **`aa_distribution_split0.csv`** (the AA frequency count),
- for ProteinGym assays only, runs `aggregate_splits.py --baselines` inline →
  **`spearman_per_iter.csv`** (per-iter mean/std) + **`results.csv`** (baseline table).

Outputs:
```
ProteinGymSampling/run<TAG>/<ASSAY>_<TAG>_1000_iter{0..4}/aa_distribution_split0.csv
UNAAGI_result/results/<TAG>/<PROTEIN>_<TAG>_1000/{spearman_per_iter.csv,results.csv}
```
Manual single-assay sampling: see `CLAUDE.md` → "Single benchmark, accelerated".

## Stages 3–4 — CP2/PUMA scoring + build modeldata

Script: `eval/finalize_and_plot.sh` (CPU job, runs under the `afterok` dependency).
CP2/PUMA are **not** aggregated inline (they need a different scorer and the all/NAA/NCAA split),
so finalize does, **per iter**:
```
scripts/result_eval_uniform_uaa.py --benchmark <ASSAY>_reframe.csv \
    --aa_output .../<ASSAY>_<TAG>_1000_iter<it>/aa_distribution_split0.csv \
    --output_dir out --total_num 1000
```
score = `logP_wt − logP_mut`; floor `1/total_num`; skip pos 156 and WT-Pro. The scorer writes
`all_benchmark_results_raw.csv`; finalize stashes it as
`results/<TAG>/<ASSAY>_<TAG>_iter<it>/all_benchmark_results_raw.csv`.

> Why per-iter and not one merged file: running the scorer on the concatenated 5-iter CSV
> overwrites by `(pos,residue)` → collapses to ~the last iteration. Always score each iter,
> then average. `eval/make_modeldata.py` does this aggregation.

Then `eval/make_modeldata.py` assembles **`<TAG>_modeldata.csv`**:
`model,assay,subset(all|naa|ncaa),mean,std` — 25 ProteinGym (all) from `spearman_per_iter.csv`
+ CP2/PUMA (all/naa/ncaa) from the per-iter raw csvs — and collates the ProteinGym baseline
tables into `<modeldata-dir>/baselines_pg/`.
```
python eval/make_modeldata.py --result-base UNAAGI_result/results/<TAG> \
    --model-tag <TAG> --out-dir UNAAGI_result/modeldata
python eval/make_modeldata.py --selftest      # offline sanity check, no cluster
```

## Stage 5 — visualize the group

Script: `scripts/compare_models.py` (the **canonical** plotter; reproduces `~/unaagi_v03_plots/`).
```
python scripts/compare_models.py \
    --data-dir UNAAGI_result/modeldata \
    --models v0.3ring_base,v0.3ring_cont,v0.3weighted_end,p0203_ctmc \
    --output-dir UNAAGI_result/figures/<name>
```
- `--models` = the assigned group (must each have `<tag>_modeldata.csv` in `--data-dir`).
  Omit to use the v0.3 reference group, or auto-discover all `*_modeldata.csv`.
- The first model is the reference for assay sort order; `p0203_ctmc` is always black.
- Produces `proteingym_comparison.svg` (stars per model on a shared vertical line per assay,
  per-iter error bars, faint baseline scatter) + `barplot_{all,ncaa,naa}_mutations.svg`
  (models solid, literature baselines hatched).

Example inputs for an offline dry run live in `scripts/example_data/v03_compare/`:
```
python scripts/compare_models.py --data-dir scripts/example_data/v03_compare \
    --models p0203_ctmc,v0.3ring_cont --output-dir /tmp/fig   # (needs baselines_pg too)
```

---

## §6 — Hendrix (CUDA) equivalents

| | LUMI | Hendrix |
|---|---|---|
| repo | `/flash/.../UAAG2_main` (ring) | `/home/qcx679/hantang/UAAG2` |
| sampling | `eval/run_pipeline.sh` | `eval/run_eval_hendrix.sh` + `eval/slurm_sample_hendrix.sh` |
| manifest | `eval/models.tsv` | `eval/models_hendrix.tsv` |
| env | `unaagi_env` container | `conda activate targetdiff; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib` |
| sbatch flags | `--account=project_465002574 --partition=standard-g` | `--account=boomsma --partition=gpu --gres=gpu:1 --exclude=hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl` |

Stages 3–5 (`finalize_and_plot.sh` logic, `make_modeldata.py`, `compare_models.py`) are
cluster-agnostic python — run them under `targetdiff` on Hendrix with the paths from
`models_hendrix.tsv`. (A `finalize_and_plot_hendrix.sh` wrapper can be added when Hendrix
becomes the primary path; the python it calls is identical.)

---

## Files in this pipeline

| File | Stage | Role |
|---|---|---|
| `eval/run_pipeline.sh` | orchestrator | one button: submit sampling group → afterok finalize+plot |
| `eval/run_eval.sh` | 1 | sampling-only launcher (no downstream) |
| `eval/slurm_sample.sh` | 1 | 135-task sampling array (CTMC 1k×5) |
| `eval/finalize_and_plot.sh` | 3–5 | CP2/PUMA scoring → modeldata → plot group |
| `eval/make_modeldata.py` | 4 | per-assay outputs → `<TAG>_modeldata.csv` (+ collate baselines) |
| `scripts/generate_ligand_dpm.py` | 1 | sampler (inline post_analysis) |
| `scripts/aggregate_splits.py` | 3 | per-iter Spearman → `spearman_per_iter.csv` |
| `scripts/result_eval_uniform_uaa.py` | 4 | CP2/PUMA scorer (logP_wt−logP_mut) |
| `scripts/result_eval_uniform.py` | 4 | ProteinGym scorer |
| `scripts/compare_models.py` | 5 | canonical N-model plotter |

See also `eval/RUNBOOK.md` (cluster cheatsheet + arch gotcha) and `SCRIPT_CATALOG.md`
(every script in the repo, with status tags).

## Status

`make_modeldata.py` and `compare_models.py` are verified locally (`--selftest`, gold figures
reproduced). The SLURM orchestration (`run_pipeline.sh`, `finalize_and_plot.sh`) is
syntax-checked but **not yet cluster-run** — LUMI GPU budget is exhausted; first live run
should be a single model to confirm the dependency chain before `all`.
