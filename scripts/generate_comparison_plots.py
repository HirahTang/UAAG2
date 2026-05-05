"""Generate comparison plots for UNAAGI benchmark results.

Produces four figures:
  1. ProteinGym dot plot — all baselines (transparent) + UNAAGI iterations (gold stars ±std)
  2. CP2 + PUMA bar chart — all mutations
  3. CP2 + PUMA bar chart — NCAA (non-canonical AA) mutations only
  4. CP2 + PUMA bar chart — NAA (natural AA) mutations only

Usage (on LUMI after aggregate_splits.py has completed):
  python scripts/generate_comparison_plots.py \\
      --results-dir /scratch/project_465002574/UNAAGI_result/results/ctmc_1k_5iter \\
      --cp2-iter-dirs /scratch/project_465002574/ProteinGymSampling/runctmc_1k_5iter/CP2_ctmc_1k_5iter_1000_iter{0,1,2,3,4} \\
      --puma-iter-dirs /scratch/project_465002574/ProteinGymSampling/runctmc_1k_5iter/PUMA_ctmc_1k_5iter_1000_iter{0,1,2,3,4} \\
      --cp2-benchmark /scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/CP2_reframe.csv \\
      --puma-benchmark /scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/PUMA_reframe.csv \\
      --output-dir ./figures
"""
import argparse
import glob as globmod
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Hardcoded baselines (from prior notebooks / Rosetta README)
# ---------------------------------------------------------------------------

SAPROT_VALUES = {
    "SBI_STAAM":         0.62,
    "VRPI_BPT7":         0.662,
    "ARGR_ECOLI":        0.604,
    "HCP_LAMBD":         0.768,
    "FKBP3_HUMAN":       0.581,
    "OTU7A_HUMAN":       0.642,
    "RS15_GEOSE":        0.433,
    "SQSTM_MOUSE":       0.682,
    "PKN1_HUMAN":        0.324,
    "SCIN_STAAR":        0.62,
    "ENV_HV1B9":         0.15,
    "DLG4_RAT":          0.504,
    "SUMO1_HUMAN":       0.479,
    "ILF3_HUMAN":        0.319,
    "DN7A_SACS2":        0.556,
    "VG08_BPP22":        0.619,
    "A0A247D711_LISMN":  0.427,
    "SOX30_HUMAN":       0.333,
    "IF1_ECOLI":         0.616,
    "B2L11_HUMAN":       0.242,
    "CCDB_ECOLI":        0.438,
    "AICDA_HUMAN":       0.257,
    "TAT_HV1BR":         0.157,
    "ENVZ_ECOLI":        0.148,
    "ERBB2_HUMAN":       0.525,
}

# CP2 / PUMA bar-chart baselines
BARPLOT_BASELINES = {
    # (benchmark, model) -> (spearmanr, std)
    ("CP2",  "PepINVENT"):         (-0.0339,  0),
    ("PUMA", "PepINVENT"):         ( 0.1554,  0),
    ("CP2",  "NCFlow(AEV-PLIG)"):  (-0.08,    0),
    ("PUMA", "NCFlow(AEV-PLIG)"):  ( 0.19,    0),
    ("CP2",  "NCFlow(ATM)"):       ( 0.15,    0),
    # Rosetta A (ref2015)
    ("CP2",  "Rosetta A"):         ( 0.277,   0),
    ("PUMA", "Rosetta A"):         ( 0.184,   0),
    # Rosetta B (ref2015_cart + AM1-BCC)
    ("CP2",  "Rosetta B"):         ( 0.283,   0),
    ("PUMA", "Rosetta B"):         ( 0.313,   0),
    # Rosetta C (OpenMM + GAFF2)
    ("CP2",  "Rosetta C"):         ( 0.114,   0),
    ("PUMA", "Rosetta C"):         ( 0.211,   0),
}

BARPLOT_NCAA_BASELINES = {
    ("CP2",  "PepINVENT"):  ( 0.0984,  0),
    ("PUMA", "PepINVENT"):  ( 0.1644,  0),
    ("CP2",  "Rosetta A"):  ( 0.382,   0),
    ("PUMA", "Rosetta A"):  ( 0.064,   0),
    ("CP2",  "Rosetta C"):  ( 0.114,   0),
    ("PUMA", "Rosetta C"):  ( 0.211,   0),
}

BARPLOT_NAA_BASELINES = {
    ("CP2",  "PepINVENT"):  (-0.0339,  0),   # PepINVENT NAA same as total for CP2 (not separated)
    ("PUMA", "PepINVENT"):  ( 0.1554,  0),
    ("CP2",  "Rosetta A"):  ( 0.298,   0),
    ("PUMA", "Rosetta A"):  ( 0.189,   0),
    ("CP2",  "Rosetta B"):  ( 0.283,   0),
    ("PUMA", "Rosetta B"):  ( 0.313,   0),
}


# ---------------------------------------------------------------------------
# UAA scoring logic (mirrors result_eval_uniform_uaa.py)
# ---------------------------------------------------------------------------

UAA_IDENTITY = [x.upper() for x in [
    'Abu', 'Nva', 'Nle', 'Ahp', 'Aoc', 'Tme', 'hSM', 'tBu',
    'Cpa', 'Aib', 'MeG', 'MeA', 'MeB', 'MeF', '2th', '3th', 'YMe', '2Np', 'Bzt',
]]
NAA_IDENTITY = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR',
]
ALL_AA = UAA_IDENTITY + NAA_IDENTITY  # PRO excluded


def score_uaa_distribution(aa_csv: str, benchmark_csv: str, total_num: int):
    """Compute UNAAGI Spearman ρ for all / NCAA / NAA subsets.

    Returns dict: {'all': r, 'ncaa': r, 'naa': r}  (nan if unavailable)
    """
    df_gen = pd.read_csv(aa_csv)
    df_bench = pd.read_csv(benchmark_csv)

    # Capitalise aa_distribution column names
    df_gen.columns = [c.upper() for c in df_gen.columns]

    df_bench['UAAG'] = np.nan
    df_bench['wt_UAAG'] = np.nan

    for _, row in df_gen.iterrows():
        wt_aa = row['AA']
        pos = int(row['POS'])
        if wt_aa == 'PRO':
            continue
        wt_val = row[wt_aa] if not np.isnan(row[wt_aa]) else 1 / total_num

        for aa_name in ALL_AA:
            if aa_name not in row.index:
                continue
            mt_val = row[aa_name] if not np.isnan(row[aa_name]) else 1 / total_num
            mask = (
                (df_bench['aa'] == wt_aa) &
                (df_bench['pos'] == pos) &
                (df_bench['target'] == aa_name)
            )
            df_bench.loc[mask, 'UAAG']    = mt_val
            df_bench.loc[mask, 'wt_UAAG'] = wt_val

    df_bench['wt_UAAG'] = np.log(df_bench['wt_UAAG'] / total_num)
    df_bench['UAAG']    = np.log(df_bench['UAAG']    / total_num)
    df_bench['pred']    = df_bench['wt_UAAG'] - df_bench['UAAG']
    df_bench = df_bench.dropna(subset=['pred', 'value'])

    df_uaa = df_bench[df_bench['target'].isin(UAA_IDENTITY)]
    df_naa = df_bench[df_bench['target'].isin(NAA_IDENTITY)]

    def spr(df):
        if len(df) < 3:
            return np.nan
        return spearmanr(df['pred'], df['value']).correlation

    return {'all': spr(df_bench), 'ncaa': spr(df_uaa), 'naa': spr(df_naa)}


def compute_uaa_per_iter(iter_dirs: list[str], benchmark_csv: str, total_num: int, label: str):
    """Score each iteration's aa_distribution_split0.csv against the UAA benchmark.

    Returns {'all': (mean, std), 'ncaa': (mean, std), 'naa': (mean, std)}
    """
    all_scores = {'all': [], 'ncaa': [], 'naa': []}
    for d in iter_dirs:
        aa_csv = os.path.join(d, "aa_distribution_split0.csv")
        if not os.path.isfile(aa_csv):
            print(f"[WARN] {aa_csv} not found — skipping")
            continue
        scores = score_uaa_distribution(aa_csv, benchmark_csv, total_num)
        print(f"  {os.path.basename(d)}: all={scores['all']:.4f}  ncaa={scores['ncaa']:.4f}  naa={scores['naa']:.4f}")
        for k, v in scores.items():
            if not np.isnan(v):
                all_scores[k].append(v)

    result = {}
    for k, vals in all_scores.items():
        if vals:
            result[k] = (float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
        else:
            result[k] = (np.nan, np.nan)
        print(f"  [{label}] {k}: mean={result[k][0]:.4f} ± {result[k][1]:.4f}")
    return result


# ---------------------------------------------------------------------------
# Load ProteinGym per-iter results
# ---------------------------------------------------------------------------

def load_proteingym_results(results_dir: str):
    """Load per-iter Spearman and baselines from aggregate_splits.py output dirs.

    Returns:
      summary_df : DataFrame(benchmark, mean_spearman, std_spearman)  — sorted desc
      baseline_df: DataFrame(benchmark_name, model, spearmanr_pred)
    """
    iter_rows = []
    baseline_rows = []

    for subdir in sorted(os.listdir(results_dir)):
        subpath = os.path.join(results_dir, subdir)
        if not os.path.isdir(subpath):
            continue

        per_iter_csv = os.path.join(subpath, "spearman_per_iter.csv")
        results_csv  = os.path.join(subpath, "results.csv")

        # Extract short benchmark name (first two underscore-separated parts)
        bench_name = "_".join(subdir.split("_")[:2])

        # Per-iter Spearman for UNAAGI
        if os.path.isfile(per_iter_csv):
            df_iter = pd.read_csv(per_iter_csv)
            # rows: iter=0..4, mean, std
            num_iters = df_iter[df_iter['iter'].astype(str).str.match(r'^\d+$')]
            mean_row = df_iter[df_iter['iter'].astype(str) == 'mean']
            std_row  = df_iter[df_iter['iter'].astype(str) == 'std']
            if not mean_row.empty and not std_row.empty:
                iter_rows.append({
                    'benchmark':      bench_name,
                    'mean_spearman':  float(mean_row['spearmanr'].values[0]),
                    'std_spearman':   float(std_row['spearmanr'].values[0]),
                })

        # Baselines
        if os.path.isfile(results_csv):
            df_res = pd.read_csv(results_csv)
            for _, row in df_res.iterrows():
                if row['model'] == 'UNAAGI':
                    continue
                baseline_rows.append({
                    'benchmark_name': bench_name,
                    'model':          row['model'],
                    'spearmanr_pred': row['spearmanr_pred'],
                })

    summary_df  = pd.DataFrame(iter_rows).sort_values('mean_spearman', ascending=False).reset_index(drop=True)
    baseline_df = pd.DataFrame(baseline_rows)
    return summary_df, baseline_df


# ---------------------------------------------------------------------------
# Plot 1 — ProteinGym dot plot
# ---------------------------------------------------------------------------

SELECTED_BASELINES = [
    'MSA_Transformer_ensemble', 'ESM2_15B', 'Progen2_xlarge', 'ProtGPT2',
    'Tranception_L', 'MIFST', 'ESM-IF1', 'ProteinMPNN',
]

BASELINE_COLORS = {
    'MSA_Transformer_ensemble': '#1f77b4',
    'ESM2_15B':                 '#ff7f0e',
    'Progen2_xlarge':           '#2ca02c',
    'ProtGPT2':                 '#d62728',
    'Tranception_L':            '#9467bd',
    'MIFST':                    '#8c564b',
    'ESM-IF1':                  '#e377c2',
    'ProteinMPNN':              '#17becf',
    'SaProt(650M)':             '#bcbd22',
    'PepINVENT':                '#FF1493',
}


def plot_proteingym(summary_df, baseline_df, output_path):
    plt.rcParams.update({
        "font.size": 16, "font.weight": "bold",
        "axes.labelweight": "bold", "axes.titlesize": 15, "axes.titleweight": "bold",
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    })

    benchmarks = summary_df['benchmark'].tolist()
    x_pos = np.arange(len(benchmarks))

    fig, ax = plt.subplots(figsize=(22, 10))

    # Background baselines
    for bm in SELECTED_BASELINES:
        bm_data = baseline_df[baseline_df['model'] == bm].set_index('benchmark_name')['spearmanr_pred']
        y = [bm_data.get(b, np.nan) for b in benchmarks]
        ax.scatter(x_pos, y, s=80, alpha=0.45,
                   color=BASELINE_COLORS.get(bm, '#999999'),
                   edgecolor='black', linewidth=0.6,
                   label=bm, zorder=1)

    # SaProt(650M) — hardcoded
    saprot_y = [SAPROT_VALUES.get(b, np.nan) for b in benchmarks]
    ax.scatter(x_pos, saprot_y, s=80, alpha=0.45,
               color=BASELINE_COLORS['SaProt(650M)'],
               edgecolor='black', linewidth=0.6,
               label='SaProt(650M)', zorder=1)

    # UNAAGI mean ± std (gold stars)
    ax.errorbar(x_pos, summary_df['mean_spearman'],
                yerr=summary_df['std_spearman'],
                fmt='*', markersize=16,
                color='#FFD700',
                ecolor='black', elinewidth=2, capsize=5, capthick=2,
                alpha=1.0, markeredgecolor='black', markeredgewidth=1.5,
                label='UNAAGI', zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(benchmarks, rotation=75, ha='right')
    ax.set_xlabel('Benchmark', fontweight='bold', fontsize=16)
    ax.set_ylabel(r'Spearman $\rho$', fontweight='bold', fontsize=16)
    ax.set_ylim(-0.35, 0.95)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.legend(frameon=True, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


# ---------------------------------------------------------------------------
# Plots 2-4 — bar charts
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "UNAAGI":          "#E63946",
    "PepINVENT":       "#457B9D",
    "NCFlow(AEV-PLIG)":"#2A9D8F",
    "NCFlow(ATM)":     "#F4A261",
    "Rosetta A":       "#6A4C93",
    "Rosetta B":       "#1982C4",
    "Rosetta C":       "#8AC926",
}

BENCHMARKS = ["CP2", "PUMA"]


def _build_barplot_data(unaagi_cp2, unaagi_puma, baselines_dict):
    """Build a unified DataFrame for one bar chart variant.

    unaagi_cp2 / unaagi_puma: (mean, std)
    baselines_dict: {(benchmark, model): (spearmanr, std)}
    """
    rows = [
        {"benchmark": "CP2",  "model": "UNAAGI", "spearmanr": unaagi_cp2[0],  "std": unaagi_cp2[1]},
        {"benchmark": "PUMA", "model": "UNAAGI", "spearmanr": unaagi_puma[0], "std": unaagi_puma[1]},
    ]
    for (bm, model), (spr, std) in baselines_dict.items():
        rows.append({"benchmark": bm, "model": model, "spearmanr": spr, "std": std})
    return pd.DataFrame(rows)


def _bar_chart(df, title, output_path):
    plt.rcParams.update({
        "font.size": 14, "font.weight": "bold",
        "axes.labelweight": "bold", "axes.titlesize": 16, "axes.titleweight": "bold",
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    })

    models = list(dict.fromkeys(df['model'].tolist()))   # preserve insertion order
    x = np.arange(len(BENCHMARKS))
    n = len(models)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('#f8f9fa')

    for i, model in enumerate(models):
        pos = x + (i - n / 2 + 0.5) * width
        values, errors = [], []
        for bm in BENCHMARKS:
            row = df[(df['benchmark'] == bm) & (df['model'] == model)]
            if not row.empty:
                values.append(float(row['spearmanr'].values[0]))
                errors.append(float(row['std'].values[0]))
            else:
                values.append(np.nan)
                errors.append(0.0)

        eb = errors if model == "UNAAGI" else None
        ax.bar(pos, values, width,
               label=model,
               color=MODEL_COLORS.get(model, None),
               alpha=0.9, edgecolor='white', linewidth=1.5,
               yerr=eb, capsize=6,
               error_kw={'linewidth': 2.5, 'ecolor': '#2b2d42', 'alpha': 0.8})

    ax.set_xlabel('Benchmark', fontweight='bold', fontsize=15)
    ax.set_ylabel(r'Spearman $\rho$', fontweight='bold', fontsize=15)
    ax.set_title(title, fontweight='bold', fontsize=17, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, fontsize=13)
    ax.axhline(y=0, color='#495057', linestyle='-', linewidth=1.5, alpha=0.6)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#adb5bd', linewidth=1)
    ax.legend(frameon=True, loc='upper left', bbox_to_anchor=(1.01, 1),
              fancybox=True, shadow=True, framealpha=0.95)

    for spine in ax.spines.values():
        spine.set_edgecolor('#dee2e6')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # --- ProteinGym data ---
    print("[1/4] Loading ProteinGym results ...")
    summary_df, baseline_df = load_proteingym_results(args.results_dir)
    print(f"      {len(summary_df)} benchmarks, {len(baseline_df['benchmark_name'].unique())} with baselines")
    print(summary_df[['benchmark', 'mean_spearman', 'std_spearman']].to_string(index=False))

    # --- CP2/PUMA per-iter UAA scoring ---
    print("[2/4] Scoring CP2 per iteration ...")
    cp2_scores = compute_uaa_per_iter(
        args.cp2_iter_dirs, args.cp2_benchmark, args.total_num, "CP2"
    )
    print("[3/4] Scoring PUMA per iteration ...")
    puma_scores = compute_uaa_per_iter(
        args.puma_iter_dirs, args.puma_benchmark, args.total_num, "PUMA"
    )

    # --- Figure 1: ProteinGym dot plot ---
    print("[4/4] Generating plots ...")
    plot_proteingym(
        summary_df, baseline_df,
        os.path.join(args.output_dir, "proteingym_comparison.svg"),
    )

    # --- Figures 2-4: Bar charts ---
    for subset_key, title, baselines_dict, fname in [
        (
            "all",
            "Protein Fitness Prediction — All Mutations",
            BARPLOT_BASELINES,
            "barplot_all_mutations.svg",
        ),
        (
            "ncaa",
            "Protein Fitness Prediction — NCAA Mutations Only",
            BARPLOT_NCAA_BASELINES,
            "barplot_ncaa_mutations.svg",
        ),
        (
            "naa",
            "Protein Fitness Prediction — NAA Mutations Only",
            BARPLOT_NAA_BASELINES,
            "barplot_naa_mutations.svg",
        ),
    ]:
        df_bar = _build_barplot_data(
            cp2_scores[subset_key],
            puma_scores[subset_key],
            baselines_dict,
        )
        _bar_chart(df_bar, title, os.path.join(args.output_dir, fname))

    print(f"\nAll figures saved to: {args.output_dir}")


# ---------------------------------------------------------------------------

def _expand_dirs(paths: list[str]) -> list[str]:
    """Expand glob patterns and sort."""
    expanded = []
    for p in paths:
        matches = sorted(globmod.glob(p))
        if matches:
            expanded.extend(matches)
        elif os.path.isdir(p):
            expanded.append(p)
        else:
            print(f"[WARN] No match for: {p}")
    return expanded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="/scratch/project_465002574/UNAAGI_result/results/ctmc_1k_5iter",
        help="Aggregate results directory (contains per-assay subdirs with spearman_per_iter.csv)",
    )
    parser.add_argument(
        "--cp2-iter-dirs", nargs="+",
        default=[
            f"/scratch/project_465002574/ProteinGymSampling/runctmc_1k_5iter/CP2_ctmc_1k_5iter_1000_iter{i}"
            for i in range(5)
        ],
        help="Directories containing per-iteration aa_distribution_split0.csv for CP2",
    )
    parser.add_argument(
        "--puma-iter-dirs", nargs="+",
        default=[
            f"/scratch/project_465002574/ProteinGymSampling/runctmc_1k_5iter/PUMA_ctmc_1k_5iter_1000_iter{i}"
            for i in range(5)
        ],
        help="Directories containing per-iteration aa_distribution_split0.csv for PUMA",
    )
    parser.add_argument(
        "--cp2-benchmark",
        default="/scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/CP2_reframe.csv",
        help="CP2 benchmark CSV for UAA scoring",
    )
    parser.add_argument(
        "--puma-benchmark",
        default="/scratch/project_465002574/UNAAGI_benchmark_values/uaa_benchmark_csv/PUMA_reframe.csv",
        help="PUMA benchmark CSV for UAA scoring",
    )
    parser.add_argument(
        "--total-num", type=int, default=1000,
        help="Samples per position per iteration",
    )
    parser.add_argument(
        "--output-dir", default="./figures",
        help="Directory to save SVG figures",
    )

    args = parser.parse_args()
    args.cp2_iter_dirs  = _expand_dirs(args.cp2_iter_dirs)
    args.puma_iter_dirs = _expand_dirs(args.puma_iter_dirs)
    main(args)
