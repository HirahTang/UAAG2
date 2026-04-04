"""Aggregate aa_distribution CSVs across multiple iterations.

With the single-split design (total_partition=1), each iteration job writes:
  {save_dir}/run{model}/{protein}_{model}_{n}_iter{I}/aa_distribution_split0.csv

This script:
  1. Collects all matching CSVs via a glob pattern
  2. Concatenates them → aa_distribution.csv  (all iters, all positions)
  3. Optionally runs result_eval_uniform.py / result_eval_uniform_uaa.py

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

import pandas as pd


def concat_csvs(paths: list, output: str) -> bool:
    """Concatenate a list of CSV paths → output. Returns True on success."""
    if not paths:
        print(f"[WARN] No CSV files to concatenate.")
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

    # 1. Expand glob and merge aa_distribution CSVs
    aa_files = sorted(globmod.glob(args.glob))
    print(f"[INFO] Found {len(aa_files)} aa_distribution CSVs:")
    for f in aa_files:
        print(f"       {f}")

    aa_csv = os.path.join(args.output_dir, "aa_distribution.csv")
    aa_ok = concat_csvs(aa_files, aa_csv)

    if not aa_ok:
        print("[ERROR] No aa_distribution CSVs found — aborting.")
        sys.exit(1)

    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Run result_eval_uniform.py (ProteinGym Spearman / NDCG)
    if args.baselines:
        result_eval = os.path.join(scripts_dir, "result_eval_uniform.py")
        if os.path.isfile(result_eval):
            print("[INFO] Running result_eval_uniform ...")
            ret = subprocess.run(
                [sys.executable, result_eval,
                 "--generated", aa_csv,
                 "--baselines", args.baselines,
                 "--total_num", str(args.total_num),
                 "--output_dir", args.output_dir],
                check=False,
            )
            if ret.returncode != 0:
                print(f"[WARN] result_eval_uniform returned {ret.returncode}")
        else:
            print(f"[WARN] {result_eval} not found — skipping result eval.")

    # 3. Run result_eval_uniform_uaa.py for each UAA benchmark
    if args.uaa_benchmarks:
        uaa_eval = os.path.join(scripts_dir, "result_eval_uniform_uaa.py")
        if os.path.isfile(uaa_eval):
            for uaa_spec in args.uaa_benchmarks:
                # format: NAME:/path/to/CSV
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
                     "--total_num", str(args.total_num)],
                    check=False,
                )
                if ret.returncode != 0:
                    print(f"[WARN] result_eval_uniform_uaa returned {ret.returncode} for {uaa_name}")


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
                        help="Total samples per position across all iterations (for normalization)")
    parser.add_argument("--uaa-benchmarks", nargs="*", default=[],
                        help="UAA specs as NAME:/path/to/csv (repeatable)")
    args = parser.parse_args()
    main(args)
