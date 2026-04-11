"""Generate a CSV showing NCAA mutation coverage per position per model.

Coverage = model has a valid (non-NaN / non-zero) prediction for that mutation.

Columns:
  protein, pos, wt_aa, ncaa_target, UNAAGI_count, UNAAGI_covered,
  RosettaA_covered, RosettaC_covered

Usage (on LUMI):
  python scripts/generate_ncaa_coverage.py \\
      --output /scratch/project_465002574/UNAAGI_result/figures/ctmc_1k_5iter/ncaa_coverage.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------

NCAA_UPPER = [x.upper() for x in [
    'Abu', 'Nva', 'Nle', 'Ahp', 'Aoc', 'Tme', 'hSM', 'tBu',
    'Cpa', 'Aib', 'MeG', 'MeA', 'MeB', 'MeF', '2th', '3th', 'YMe', '2Np', 'Bzt',
]]

# aa_distribution CSV uses original mixed-case names; build upper→mixed map
NCAA_UPPER_TO_MIXED = {x.upper(): x for x in [
    'Abu', 'Nva', 'Nle', 'Ahp', 'Aoc', 'Tme', 'hSM', 'tBu',
    'Cpa', 'Aib', 'MeG', 'MeA', 'MeB', 'MeF', '2th', '3th', 'YMe', '2Np', 'Bzt',
]}

PROTEINS = ["CP2", "PUMA"]

DEFAULT_PATHS = {
    "CP2": {
        "benchmark":   "/scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/CP2_reframe.csv",
        "unaagi_dist": "/scratch/project_465002574/UNAAGI_result/results/ctmc_1k_5iter/CP2_ctmc_1k_5iter_1000/aa_distribution.csv",
        "rosetta_a":   "/scratch/project_465002574/NCAA_ddg_results/CP2/approach_A/approach_a_results.csv",
        "rosetta_c":   "/scratch/project_465002574/NCAA_ddg_results/CP2/approach_C/approach_c_results.csv",
    },
    "PUMA": {
        "benchmark":   "/scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/PUMA_reframe.csv",
        "unaagi_dist": "/scratch/project_465002574/UNAAGI_result/results/ctmc_1k_5iter/PUMA_ctmc_1k_5iter_1000/aa_distribution.csv",
        "rosetta_a":   "/scratch/project_465002574/NCAA_ddg_results/PUMA/approach_A/approach_a_results.csv",
        "rosetta_c":   "/scratch/project_465002574/NCAA_ddg_results/PUMA/approach_C/approach_c_results.csv",
    },
}


def build_unaagi_lookup(dist_csv: str) -> dict:
    """Return {(pos, ncaa_upper): total_count} from aa_distribution CSV."""
    df = pd.read_csv(dist_csv)
    lookup = {}
    for _, row in df.iterrows():
        pos = int(row['pos'])
        for upper, mixed in NCAA_UPPER_TO_MIXED.items():
            col = mixed  # original case column name
            if col not in row.index:
                continue
            val = row[col]
            count = 0 if (pd.isna(val) or val == 0) else int(val)
            lookup[(pos, upper)] = lookup.get((pos, upper), 0) + count
    return lookup


def build_rosetta_lookup(results_csv: str) -> dict:
    """Return {(pos, ncaa_upper): pred_score_or_nan} from Rosetta results CSV."""
    df = pd.read_csv(results_csv)
    df_ncaa = df[df['target'].isin(NCAA_UPPER)]
    lookup = {}
    for _, row in df_ncaa.iterrows():
        key = (int(row['pos']), row['target'])
        lookup[key] = row['pred_score']
    return lookup


def build_coverage_table(protein: str, paths: dict) -> pd.DataFrame:
    bench      = pd.read_csv(paths["benchmark"])
    unaagi_lut = build_unaagi_lookup(paths["unaagi_dist"])
    ra_lut     = build_rosetta_lookup(paths["rosetta_a"])
    rc_lut     = build_rosetta_lookup(paths["rosetta_c"])

    bench_ncaa = bench[bench['target'].isin(NCAA_UPPER)].copy()
    rows = []
    for _, row in bench_ncaa.iterrows():
        pos    = int(row['pos'])
        ncaa   = row['target']
        wt_aa  = row['aa']
        key    = (pos, ncaa)

        unaagi_count   = unaagi_lut.get(key, 0)
        unaagi_covered = int(unaagi_count > 0)

        ra_score   = ra_lut.get(key, np.nan)
        ra_covered = int(not (isinstance(ra_score, float) and np.isnan(ra_score)))

        rc_score   = rc_lut.get(key, np.nan)
        rc_covered = int(not (isinstance(rc_score, float) and np.isnan(rc_score)))

        rows.append({
            "protein":        protein,
            "pos":            pos,
            "wt_aa":          wt_aa,
            "ncaa_target":    ncaa,
            "UNAAGI_count":   unaagi_count,
            "UNAAGI_covered": unaagi_covered,
            "RosettaA_covered": ra_covered,
            "RosettaC_covered": rc_covered,
        })

    return pd.DataFrame(rows).sort_values(["pos", "ncaa_target"]).reset_index(drop=True)


def print_summary(df: pd.DataFrame, protein: str):
    print(f"\n{'='*60}")
    print(f"  {protein} — NCAA coverage summary")
    print(f"{'='*60}")
    total = len(df)
    for col in ["UNAAGI_covered", "RosettaA_covered", "RosettaC_covered"]:
        n = df[col].sum()
        print(f"  {col:<22}: {n:>4} / {total}  ({100*n/total:.1f}%)")

    print(f"\n  Per-NCAA-type coverage:")
    pivot = df.groupby("ncaa_target")[["UNAAGI_covered", "RosettaA_covered", "RosettaC_covered"]].sum()
    pivot["n_benchmark"] = df.groupby("ncaa_target")["ncaa_target"].count()
    pivot = pivot[["n_benchmark", "UNAAGI_covered", "RosettaA_covered", "RosettaC_covered"]]
    print(pivot.to_string())


def main(args):
    frames = []
    for protein in PROTEINS:
        paths = DEFAULT_PATHS[protein]
        # Override with CLI args if provided
        if args.cp2_benchmark   and protein == "CP2":  paths["benchmark"]   = args.cp2_benchmark
        if args.puma_benchmark  and protein == "PUMA": paths["benchmark"]   = args.puma_benchmark
        if args.cp2_unaagi_dist and protein == "CP2":  paths["unaagi_dist"] = args.cp2_unaagi_dist
        if args.puma_unaagi_dist and protein == "PUMA": paths["unaagi_dist"] = args.puma_unaagi_dist

        df = build_coverage_table(protein, paths)
        frames.append(df)
        print_summary(df, protein)

    combined = pd.concat(frames, ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\n[OK] Coverage CSV saved to {args.output}")
    print(f"     Columns: {list(combined.columns)}")
    print(f"     Total rows: {len(combined)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=(
        "/scratch/project_465002574/UNAAGI_result/figures/ctmc_1k_5iter/ncaa_coverage.csv"
    ))
    parser.add_argument("--cp2-benchmark",    default=None)
    parser.add_argument("--puma-benchmark",   default=None)
    parser.add_argument("--cp2-unaagi-dist",  default=None)
    parser.add_argument("--puma-unaagi-dist", default=None)
    args = parser.parse_args()
    main(args)
