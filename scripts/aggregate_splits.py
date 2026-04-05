"""Aggregate aa_distribution CSVs across multiple iterations.

With the single-split design (total_partition=1), each iteration job writes:
  {save_dir}/run{model}/{protein}_{model}_{n}_iter{I}/aa_distribution_split0.csv

This script:
  1. Collects all matching CSVs via a glob pattern
  2. Concatenates them → aa_distribution.csv  (all iters, all positions)
  3. Computes per-iteration Spearman ρ and reports mean ± std
  4. Runs result_eval_uniform.py on the combined CSV
  5. Optionally runs result_eval_uniform_uaa.py

Usage:
  python aggregate_splits.py \\
      --glob "/scratch/.../run{model}/{protein}_*_iter*/aa_distribution_split0.csv" \\
      --output-dir /scratch/.../results/MODEL/PROTEIN \\
      --baselines /scratch/.../baselines/PROTEIN.csv \\
      --total-num 1000
"""
import argparse
import glob as globmod
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Scoring logic (mirrors result_eval_uniform.py)
# ---------------------------------------------------------------------------

AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'UNK': 'X', 'INV': 'Z',
}


def _divide_mutant(mutant):
    return mutant[0], int(mutant[1:-1]), mutant[-1]


def score_aa_distribution(aa_csv: str, baselines_csv: str, total_num: int):
    """Compute UNAAGI Spearman ρ for one aa_distribution CSV.

    Returns (spearmanr, ndcg) or (nan, nan) on failure.
    """
    try:
        df_gen = pd.read_csv(aa_csv)
        df_base = pd.read_csv(baselines_csv)
        df_base['wt'], df_base['pos'], df_base['mut'] = zip(
            *df_base['mutant'].map(_divide_mutant)
        )

        rows = {'aa': [], 'pred': [], 'wt': []}
        for _, row in df_gen.iterrows():
            mt_aa = AA_MAP.get(row['aa'])
            if mt_aa is None or row['aa'] == 'PRO':
                continue
            pos = row['pos']
            wt_val = row[row['aa']] if not np.isnan(row[row['aa']]) else 1
            for aa_name, aa_letter in AA_MAP.items():
                if aa_name == 'PRO':
                    continue
                mut_key = mt_aa + str(pos) + aa_letter
                count = row[aa_name] if not np.isnan(row[aa_name]) else 1
                rows['aa'].append(mut_key)
                rows['pred'].append(count)
                rows['wt'].append(wt_val)

        out = pd.DataFrame(rows)
        out['pred'] = np.log(out['pred'] / total_num)
        out['wt']   = np.log(out['wt']   / total_num)
        out['UNAAGI'] = -out['wt'] + out['pred']

        merged = df_base.merge(out, left_on='mutant', right_on='aa')
        if merged.empty:
            return np.nan, np.nan

        r = spearmanr(merged['UNAAGI'], merged['DMS_score']).correlation
        return r, np.nan   # ndcg omitted for per-iter (expensive, covered by combined run)
    except Exception as e:
        print(f"[WARN] score_aa_distribution failed for {aa_csv}: {e}")
        return np.nan, np.nan


# ---------------------------------------------------------------------------

def concat_csvs(paths: list, output: str) -> bool:
    """Concatenate a list of CSV paths → output. Returns True on success."""
    if not paths:
        print("[WARN] No CSV files to concatenate.")
        return False
    dfs = []
    for f in paths:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
    if not dfs:
        return False
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output, index=False)
    print(f"[INFO] Aggregated {len(paths)} CSVs → {output}  ({len(combined)} rows)")
    return True


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Expand glob and collect per-iteration CSV paths
    aa_files = sorted(globmod.glob(args.glob))
    print(f"[INFO] Found {len(aa_files)} aa_distribution CSVs:")
    for f in aa_files:
        print(f"       {f}")

    if not aa_files:
        print("[ERROR] No aa_distribution CSVs found — aborting.")
        sys.exit(1)

    # 2. Merge all iterations → combined aa_distribution.csv
    aa_csv = os.path.join(args.output_dir, "aa_distribution.csv")
    aa_ok = concat_csvs(aa_files, aa_csv)
    if not aa_ok:
        sys.exit(1)

    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    # 3. Per-iteration Spearman ρ → mean ± std
    if args.baselines and os.path.isfile(args.baselines):
        print("[INFO] Computing per-iteration Spearman ρ ...")
        iter_spearman = []
        iter_rows = []
        for i, f in enumerate(aa_files):
            r, _ = score_aa_distribution(f, args.baselines, args.total_num)
            iter_spearman.append(r)
            iter_rows.append({'iter': i, 'aa_csv': f, 'spearmanr': r})
            print(f"       iter{i}: ρ = {r:.4f}  ({os.path.basename(os.path.dirname(f))})")

        valid = [x for x in iter_spearman if not np.isnan(x)]
        if valid:
            mean_r = np.mean(valid)
            std_r  = np.std(valid, ddof=1) if len(valid) > 1 else 0.0
            print(f"[INFO] Spearman ρ  mean={mean_r:.4f}  std={std_r:.4f}  "
                  f"(n={len(valid)} iters)")
        else:
            mean_r, std_r = np.nan, np.nan
            print("[WARN] All per-iteration Spearman values are NaN.")

        # Save per-iter table
        iter_df = pd.DataFrame(iter_rows)
        iter_df.loc[len(iter_df)] = {
            'iter': 'mean', 'aa_csv': '', 'spearmanr': mean_r
        }
        iter_df.loc[len(iter_df)] = {
            'iter': 'std',  'aa_csv': '', 'spearmanr': std_r
        }
        iter_csv = os.path.join(args.output_dir, "spearman_per_iter.csv")
        iter_df.to_csv(iter_csv, index=False)
        print(f"[INFO] Per-iter stats → {iter_csv}")

    # 4. Run result_eval_uniform.py on the combined CSV (Spearman + NDCG + plots)
    if args.baselines:
        result_eval = os.path.join(scripts_dir, "result_eval_uniform.py")
        if os.path.isfile(result_eval):
            print("[INFO] Running result_eval_uniform (combined) ...")
            ret = subprocess.run(
                [sys.executable, result_eval,
                 "--generated", aa_csv,
                 "--baselines", args.baselines,
                 "--total_num", str(args.total_num * len(aa_files)),
                 "--output_dir", args.output_dir],
                check=False,
            )
            if ret.returncode != 0:
                print(f"[WARN] result_eval_uniform returned {ret.returncode}")
        else:
            print(f"[WARN] {result_eval} not found — skipping.")

    # 5. Run result_eval_uniform_uaa.py for each UAA benchmark
    if args.uaa_benchmarks:
        uaa_eval = os.path.join(scripts_dir, "result_eval_uniform_uaa.py")
        if os.path.isfile(uaa_eval):
            for uaa_spec in args.uaa_benchmarks:
                if ":" not in uaa_spec:
                    print(f"[WARN] UAA spec must be NAME:PATH — got {uaa_spec}")
                    continue
                uaa_name, uaa_csv_path = uaa_spec.split(":", 1)
                uaa_out = os.path.join(args.output_dir, uaa_name)
                os.makedirs(uaa_out, exist_ok=True)
                print(f"[INFO] Running UAA eval for {uaa_name} ...")
                ret = subprocess.run(
                    [sys.executable, uaa_eval,
                     "--benchmark", uaa_csv_path,
                     "--aa_output", aa_csv,
                     "--output_dir", uaa_out,
                     "--total_num", str(args.total_num * len(aa_files))],
                    check=False,
                )
                if ret.returncode != 0:
                    print(f"[WARN] result_eval_uniform_uaa returned {ret.returncode} "
                          f"for {uaa_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate per-iteration aa_distribution CSVs and run result_eval"
    )
    parser.add_argument("--glob", required=True,
                        help="Glob pattern matching aa_distribution CSVs from all iterations "
                             "(quote the pattern to prevent shell expansion)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write aa_distribution.csv and result_eval outputs")
    parser.add_argument("--baselines", default=None,
                        help="ProteinGym baseline CSV for result_eval_uniform.py")
    parser.add_argument("--total-num", default=1000, type=int,
                        help="Samples per position per iteration (used for per-iter scoring; "
                             "combined run uses total_num * n_iters automatically)")
    parser.add_argument("--uaa-benchmarks", nargs="*", default=[],
                        help="UAA specs as NAME:/path/to/csv (repeatable)")
    args = parser.parse_args()
    main(args)
