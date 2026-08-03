# UAAG2 — Agent guide: from sampled results → benchmark plots (one button)

This file is for AI agents working on UAAG2 evaluation/plotting. Goal: never re-derive the
eval+plot pipeline again. (Repo is duplicated across clusters & branches — see "Repo hygiene".)

## 0. Environment (Hendrix) — REQUIRED
Eval/plot scripts need pandas/scipy/seaborn/matplotlib. Use the **targetdiff** env, but its
python must be run with the env's libstdc++ on the path or pandas fails (`GLIBCXX_3.4.29 not found`):
```
export LD_LIBRARY_PATH=/home/qcx679/.conda/envs/targetdiff/lib:$LD_LIBRARY_PATH
export MPLBACKEND=Agg
PY=/home/qcx679/.conda/envs/targetdiff/bin/python   # (geodiff env also works, no LD fix needed)
```
Connect to Hendrix by IP if DNS fails: `ssh ... -o HostKeyAlias=hendrixgate qcx679@10.84.3.168`.

## 1. Path conventions
- Samples (per assay, 5 iters): `/datasets/biochem/unaagi/ProteinGymSampling/run<MODEL>/<ASSAY>_<MODEL>_1000_iter{0..4}/aa_distribution_split0.csv`
- Eval results:                  `/datasets/biochem/unaagi/results/<MODEL>/<ASSAY>_<MODEL>_1000/`
- UAA ground truth (ΔΔG):        `data/uaa_benchmark_csv/{CP2,PUMA}_reframe.csv` (cols aa,pos,target=mut,value)
- ProteinGym ground truth+baselines: per-assay baseline CSV (passed via --baselines)
- Figures: `/scratch/project_465002574/UNAAGI_result/figures/<MODEL_or_comparison>/`

## 2. ONE-BUTTON: sampled results → plots
### Step A — score each assay (samples → Spearman)  [scripts/aggregate_splits.py]
ProteinGym assay (canonical): writes spearman_per_iter.csv (per-iter mean±std) + results.csv
```
$PY scripts/aggregate_splits.py \
  --glob "<SAMPLES>/run<MODEL>/<ASSAY>_<MODEL>_1000_iter*/aa_distribution_split0.csv" \
  --output-dir "<RESULTS>/<MODEL>/<ASSAY>_<MODEL>_1000" --total-num 1000 \
  --baselines <proteingym_assay_baseline.csv>
```
CP2 / PUMA (NCAA peptide): writes <ASSAY>/all_benchmark_results.csv (pred,value) + naa/uaa splits
```
$PY scripts/aggregate_splits.py \
  --glob "<SAMPLES>/run<MODEL>/<ASSAY>_<MODEL>_1000_iter*/aa_distribution_split0.csv" \
  --output-dir "<RESULTS>/<MODEL>/<ASSAY>_<MODEL>_1000" --total-num 1000 \
  --uaa-benchmarks "<ASSAY>:data/uaa_benchmark_csv/<ASSAY>_reframe.csv"
```
### Step B — plot (scores → SVGs)  [scripts/compare_models.py]  <- THE ONE canonical plotter (N models)
**Do NOT write new plotting code.** `compare_models.py` is the single source of truth: same fixed
style + ALL baselines (SaProt/ESM/Progen/… dots; PepINVENT/NCFlow/Rosetta/OpenMM bars), 5-iter
error bars, all model dots on each assay's vertical line. Add ANY model = one more `--model` arg.
Produces proteingym_comparison.svg + barplot_{all,ncaa,naa}_mutations.svg.
```
$PY scripts/compare_models.py --output-dir <FIGURES>/<comparison> \
  --baseline-from <label of model whose results.csv has the ProteinGym baselines, e.g. p0203_SOTA> \
  --total-num 1000 \
  --cp2-benchmark data/uaa_benchmark_csv/CP2_reframe.csv --puma-benchmark data/uaa_benchmark_csv/PUMA_reframe.csv \
  --model "LABEL:<RESULTS>/<MODEL>:<SAMPLES>/run<MODEL>/CP2_<MODEL>_1000_iter*:<SAMPLES>/run<MODEL>/PUMA_<MODEL>_1000_iter*" \
  --model "..."   # repeat per model; quote each --model so the shell doesn't expand the globs
```
(SOTA `p0203_ctmc` results on LUMI `/scratch/.../UNAAGI_result/results/p0203_ctmc` + samples
`runctmc_1k_5iter`; pull to Hendrix to include it. Old 2-model `compare_ctmc_vs_ddpm.py` and the
one-off `compare_models_uaa.py`/`*_multi.svg` scripts are SUPERSEDED — don't use/recreate them.)

## 3. Correctness notes (easy to get wrong)
- UNAAGI fitness score = **log P_wt − log P_mut** (per result_eval_uniform_uaa.py), with a 1/total_num
  floor for unsampled residues; combined eval uses total_num × n_iters; skips pos156 & WT-Pro.
- Evaluate Spearman **PER ASSAY** (CP2 and PUMA separately) — never pool the two (different ΔΔG scales).
- Report **per-iteration mean ± std** (5 iters), as spearman_per_iter.csv does. WARNING: running
  result_eval_uniform_uaa.py on the *merged* (concatenated 5-iter) aa_distribution OVERWRITES per
  (pos,residue) → effectively uses the LAST iter, not an average. For CP2/PUMA per-iter mean use the
  compute_uaa_per_iter() logic (in the plot scripts), not the merged eval.
- NCAA = the 19 UAA codes (ABU,NVA,...); NAA = 20 canonical. ProteinGym assays are canonical → NCAA N/A.

## 4. Script catalog (scripts/)
TRAINING:        run_train.py (entry, dpm-solver-pp), run_train_ablation_pdbbind.py
SAMPLING:        generate_ligand_dpm.py (DPM-Solver++ sampling), run_generate_ligand_prior.sh
SAMPLES→aa_dist: post_analysis.py, run_post_analysis_standalone_hendrix.py (Hendrix, no-torch)
EVAL:            aggregate_splits.py (ORCHESTRATOR), result_eval_uniform.py (ProteinGym),
                 result_eval_uniform_uaa.py (CP2/PUMA UAA)
PLOTTING:        compare_models.py  <- CANONICAL, N models, --model "label:results_dir:cp2_glob:puma_glob" (use THIS only).
                 (superseded/do-not-use: compare_ctmc_vs_ddpm.py, compare_models_uaa.py, generate_clean_plots.py, generate_comparison_plots.py)
DATA BUILD:      build_eqgat_lmdb_from_pdb.py, build_pdbbind_lmdb.py, build_ncaa_lmdb.py,
                 construct_pdbbind_data_group{,_exclude_aa}.py, merge_lmdb_shards.py, process_cif.py, pdb_clean.py
DIAGNOSTICS:     check_benchmarks.py, diagnose{2,_mismatch,_pdbbind}.py, demo_{direct,pdb_dataset}.py,
                 test_format_vs_lmdb.py, enumerate_pdb_atom_types.py
NCAA ΔΔG baselines: ncaa_ddg/{approach_a_rosetta,approach_b_rosetta_am1,approach_c_openmm,ncaa_smiles}.py + submit_ncaa_benchmarks.sh
MISC:            split_config_files.py (slurm cfg per assay), generate_ncaa_coverage.py

## 5. TODO — reduce redundancy (the cleanup)
- [DONE 2026-06-25] Plot consolidation: `compare_models.py` is the ONE N-model plotter
  (`--model "label:results_dir:cp2_glob:puma_glob"`, repeatable; same style + all baselines +
  5-iter error bars + per-assay aligned dots). New models = new `--model` args, never new code.
- Have aggregate_splits.py also write per-iter all/naa/ncaa for UAA assays (avoid the last-iter merge pitfall).
- Repo hygiene: same GitHub repo (HirahTang/UAAG2) is checked out 3–4× on different branches:
  Hendrix `main` (heavy uncommitted), LUMI UAAG2=`mlops`, UAAG2_main=`ring_membership`, UAAG2_ring=not-git.
  Pick ONE canonical checkout per cluster, commit eval/plot scripts, sync via git.

## 6. Baselines kept in plots
CP2/PUMA bar baselines (hardcoded in the plot scripts): PepINVENT, NCFlow(AEV-PLIG/ATM), OpenMM+GAFF2.
ProteinGym dot-plot baseline: SaProt (+ per-assay baselines from results.csv).
