# UNAAGI Evaluation Runbook

**Goal:** evaluate a trained UAAG2/UNAAGI checkpoint on the UNAAGI benchmark
(25 ProteinGym assays + the CP2 & PUMA NCAA assays) and get per-assay Spearman ρ
against ground-truth DMS / ΔG fitness.

**If you are a fresh agent, read this first.** Everything you need is here; you do
not need to reverse-engineer the pipeline from the SLURM scripts.

---

## TL;DR — one button

```bash
# on LUMI, from the repo root (/flash/project_465002574/UAAG2_main)
eval/run_eval.sh v0.3ring_cont        # one model
eval/run_eval.sh all                  # every model in the manifest
```

This submits the **sampling** stage. Then run the downstream stages below.

---

## The model manifest — `eval/models.tsv`

Single source of truth for *what* can be evaluated and *which code* each checkpoint
needs. To evaluate a new checkpoint, **add one row**. Columns: `name · ckpt · arch ·
code_ref · notes`.

### ⚠️ The one gotcha: architecture must match code

`equivariant_diffusion.py` on the `ring_membership` branch **always** adds
`num_is_in_ring` to the model input *and* prediction head (`is_in_ring` is a diffused
categorical variable, not just an input flag). So:

| Checkpoint arch | params | `atom_mapping` | needs |
|---|---|---|---|
| **ring** (e.g. v0.3 ring-weighted) | 249 | 256×26 | `ring_membership` code (live `UAAG2_main`) |
| **prering** (e.g. v0.2, v0.3-weighted) | 240 | 256×24 | pre-ring code (`d7d0165` worktree) |

Loading a prering checkpoint on ring code (or vice-versa) → **state_dict size
mismatch crash**. `run_eval.sh` reads the `arch`/`code_ref` columns and points
`SRCDIR` at the right code automatically (provisioning the pre-ring worktree at
`/flash/project_465002574/UAAG2_preRing` on demand).

---

## Pipeline stages

| # | Stage | Script | Output |
|---|---|---|---|
| 1 | **Sample** 1000 × 5 iter, CTMC + 20 NFE + DPM-Solver++ | `eval/run_eval.sh` → `eval/slurm_sample.sh` (135-task array) | mol tars + `aa_distribution_split0.csv` under `/scratch/.../ProteinGymSampling/run<TAG>/` |
| 2 | **QC** PoseBusters over sampled molecules | `run_posebuster_array.sh` | validity stats |
| 3 | **Count** match each sample → AA, frequency table | `scripts/post_analysis.py` | `aa_distribution.csv` |
| 4 | **Score** frequency vs ground truth → Spearman | `scripts/result_eval_uniform.py` (ProteinGym), `scripts/result_eval_uniform_uaa.py` (CP2/PUMA) | per-assay ρ |

Stage 1 also runs `aggregate_splits.py` inline for a quick ProteinGym Spearman; the
`result_eval_uniform*` scripts in stage 4 are the canonical scoring (and the only
path for CP2/PUMA).

### Key paths
```
CKPTS       /flash/project_465002574/UAAG2_main/3DcoordsAtomsBonds_0/<run>/
benchmarks  /scratch/project_465002574/UNAAGI_benchmarks/{<PROTEIN>.pt,5ly1_cp2.pt,2roc_puma.pt}
baselines   /scratch/project_465002574/UNAAGI_benchmark_values/baselines/<PROTEIN>_*.csv
uaa truth   /scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/{CP2,PUMA}_reframe.csv
sampling    /scratch/project_465002574/ProteinGymSampling/run<TAG>/
results     /scratch/project_465002574/UNAAGI_result/results/<TAG>/
job ids     /scratch/project_465002574/uaag2_adit_results/eval_jobids.txt
```

## Notes / gotchas
- **Submit limit:** the account hits `AssocMaxSubmitJobLimit`. The array is throttled
  to `%30` concurrent tasks; launch models one at a time if the queue is full.
- Sampling writes mol files to node-local `/tmp` then tars to scratch (2M inode quota).
- `--account=project_465002574 --partition=standard-g`, 1 GCD/task.
