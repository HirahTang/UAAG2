from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path('/Users/qcx679/Desktop/UNAAGI_results')

BENCHMARK_ORDER = [
    'SBI_STAAM',
    'VRPI_BPT7',
    'ARGR_ECOLI',
    'HCP_LAMBD',
    'FKBP3_HUMAN',
    'OTU7A_HUMAN',
    'RS15_GEOSE',
    'SQSTM_MOUSE',
    'PKN1_HUMAN',
    'SCIN_STAAR',
    'ENV_HV1B9',
    'DLG4_RAT',
    'SUMO1_HUMAN',
    'ILF3_HUMAN',
    'DN7A_SACS2',
    'VG08_BPP22',
    'A0A247D711_LISMN',
    'SOX30_HUMAN',
    'IF1_ECOLI',
    'B2L11_HUMAN',
    'CCDB_ECOLI',
    'AICDA_HUMAN',
    'TAT_HV1BR',
    'ENVZ_ECOLI',
    'ERBB2_HUMAN',
]

AA_ONE_TO_THREE = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR',
}

COMPARISONS = [
    {
        'target_name': 'ILE',
        'target_mut': 'I',
        'coverage_col': 'ile',
        'left_root': ROOT / 'runFull_mask_8_gpu_UAAG_model_official_8_EXCLUDE_ILE_0212',
        'left_label': 'EXCLUDE_ILE',
        'right_root': ROOT / 'UAAG_MODEL_SWIFT',
        'right_label': 'MODEL_SWIFT',
        'coverage_left_root': ROOT / 'runrunFull_mask_8_gpu_UAAG_model_official_8_EXCLUDE_ILE_0212_full',
        'coverage_right_root': ROOT / 'runUAAG_MODEL_SWIFT',
        'out_dir': ROOT / 'UAAG_model_compare_EXCLUDE_ILE_vs_MODEL_SWIFT',
    },
    {
        'target_name': 'SER',
        'target_mut': 'S',
        'coverage_col': 'ser',
        'left_root': ROOT / 'UAAG_model_EXCLUDE_VER',
        'left_label': 'EXCLUDE_SER',
        'right_root': ROOT / 'UAAG_MODEL_SWIFT',
        'right_label': 'MODEL_SWIFT',
        'coverage_left_root': ROOT / 'runUAAG_model_EXCLUDE_VER_full',
        'coverage_right_root': ROOT / 'runUAAG_MODEL_SWIFT',
        'out_dir': ROOT / 'UAAG_model_compare_EXCLUDE_SER_vs_MODEL_SWIFT',
    },
    {
        'target_name': 'PHE',
        'target_mut': 'F',
        'coverage_col': 'phe',
        'left_root': ROOT / 'UAAG_model_PHE',
        'left_label': 'PHE',
        'right_root': ROOT / 'UAAG_MODEL_SWIFT',
        'right_label': 'MODEL_SWIFT',
        'coverage_left_root': ROOT / 'runUAAG_model_PHE_full',
        'coverage_right_root': ROOT / 'runUAAG_MODEL_SWIFT',
        'out_dir': ROOT / 'UAAG_model_compare_PHE_vs_MODEL_SWIFT',
    },
]


def benchmark_from_folder(folder_name: str) -> str:
    for benchmark in BENCHMARK_ORDER:
        if folder_name.startswith(f'{benchmark}_'):
            return benchmark
    return folder_name.split('_', 1)[0]


def collect_mut_spearman(model_root: Path, model_label: str, mut_letter: str) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(model_root.glob('*/full_table.csv')):
        folder = csv_path.parent.name
        benchmark = benchmark_from_folder(folder)

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            rows.append(
                {
                    'model': model_label,
                    'folder': folder,
                    'benchmark': benchmark,
                    f'n_rows_{mut_letter.lower()}': 0,
                    'spearman_rho': None,
                    'spearman_p': None,
                    'status': 'read_error',
                }
            )
            continue

        required = {'mut', 'DMS_score', 'UNAAGI'}
        if not required.issubset(df.columns):
            rows.append(
                {
                    'model': model_label,
                    'folder': folder,
                    'benchmark': benchmark,
                    f'n_rows_{mut_letter.lower()}': 0,
                    'spearman_rho': None,
                    'spearman_p': None,
                    'status': 'missing_columns',
                }
            )
            continue

        subset = df[df['mut'] == mut_letter].copy()
        subset['DMS_score'] = pd.to_numeric(subset['DMS_score'], errors='coerce')
        subset['UNAAGI'] = pd.to_numeric(subset['UNAAGI'], errors='coerce')
        subset = subset.dropna(subset=['DMS_score', 'UNAAGI'])

        n_rows = len(subset)
        if n_rows < 2:
            rows.append(
                {
                    'model': model_label,
                    'folder': folder,
                    'benchmark': benchmark,
                    f'n_rows_{mut_letter.lower()}': n_rows,
                    'spearman_rho': None,
                    'spearman_p': None,
                    'status': 'insufficient_rows',
                }
            )
            continue

        rho, p_val = spearmanr(subset['DMS_score'], subset['UNAAGI'])
        rows.append(
            {
                'model': model_label,
                'folder': folder,
                'benchmark': benchmark,
                f'n_rows_{mut_letter.lower()}': n_rows,
                'spearman_rho': float(rho),
                'spearman_p': float(p_val),
                'status': 'ok',
            }
        )

    return pd.DataFrame(rows)


def collect_overall_from_results(model_root: Path, model_label: str) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(model_root.glob('*/results.csv')):
        folder = csv_path.parent.name
        benchmark = benchmark_from_folder(folder)

        try:
            df = pd.read_csv(csv_path)
            unaagi = df[df['model'] == 'UNAAGI']
            if unaagi.empty:
                rows.append(
                    {
                        'model_name': model_label,
                        'folder': folder,
                        'benchmark': benchmark,
                        'unaagi_spearmanr_overall': None,
                        'status': 'missing_unaagi',
                    }
                )
            else:
                val = pd.to_numeric(unaagi.iloc[0]['spearmanr_pred'], errors='coerce')
                rows.append(
                    {
                        'model_name': model_label,
                        'folder': folder,
                        'benchmark': benchmark,
                        'unaagi_spearmanr_overall': None if pd.isna(val) else float(val),
                        'status': 'ok',
                    }
                )
        except Exception:
            rows.append(
                {
                    'model_name': model_label,
                    'folder': folder,
                    'benchmark': benchmark,
                    'unaagi_spearmanr_overall': None,
                    'status': 'read_error',
                }
            )

    return pd.DataFrame(rows)


def collect_coverage(model_root: Path, model_label: str, mut_letter: str, coverage_col: str) -> pd.DataFrame:
    rows = []
    target_col = AA_ONE_TO_THREE.get(mut_letter)
    if target_col is None:
        raise ValueError(f'Unsupported mut letter for coverage: {mut_letter}')

    # Coverage source priority:
    # 1) run folders: */Samples/aa_distribution.csv
    # 2) model folders: */full_table.csv
    coverage_files = sorted(model_root.glob('*/Samples/aa_distribution.csv'))
    source_kind = 'aa_distribution'
    if not coverage_files:
        coverage_files = sorted(model_root.glob('*/full_table.csv'))
        source_kind = 'full_table'

    for csv_path in coverage_files:
        folder = csv_path.parent.parent.name if source_kind == 'aa_distribution' else csv_path.parent.name
        benchmark = benchmark_from_folder(folder)

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            rows.append(
                {
                    'model': model_label,
                    'folder': folder,
                    'benchmark': benchmark,
                    'total_positions': 0,
                    f'{coverage_col}_sampled_positions': 0,
                    f'{coverage_col}_coverage_pct': 0.0,
                    'status': 'read_error',
                }
            )
            continue

        if source_kind == 'aa_distribution':
            if 'pos' not in df.columns or target_col not in df.columns:
                rows.append(
                    {
                        'model': model_label,
                        'folder': folder,
                        'benchmark': benchmark,
                        'total_positions': 0,
                        f'{coverage_col}_sampled_positions': 0,
                        f'{coverage_col}_coverage_pct': 0.0,
                        'status': 'missing_columns',
                    }
                )
                continue

            total_positions = int(pd.to_numeric(df['pos'], errors='coerce').dropna().nunique())
            target_counts = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
            sampled_positions = int(pd.to_numeric(df.loc[target_counts > 0, 'pos'], errors='coerce').dropna().nunique())
        else:
            if 'pos' not in df.columns or 'mut' not in df.columns or 'pred' not in df.columns:
                rows.append(
                    {
                        'model': model_label,
                        'folder': folder,
                        'benchmark': benchmark,
                        'total_positions': 0,
                        f'{coverage_col}_sampled_positions': 0,
                        f'{coverage_col}_coverage_pct': 0.0,
                        'status': 'missing_columns',
                    }
                )
                continue

            total_positions = int(pd.to_numeric(df['pos'], errors='coerce').dropna().nunique())
            all_pred = pd.to_numeric(df['pred'], errors='coerce')
            pred_floor = float(all_pred.min()) if all_pred.notna().any() else np.nan

            subset = df[df['mut'] == mut_letter].copy()
            subset['pred'] = pd.to_numeric(subset['pred'], errors='coerce')

            if subset.empty or np.isnan(pred_floor):
                sampled_positions = 0
            else:
                sampled_positions = int(subset.loc[subset['pred'] > pred_floor + 1e-12, 'pos'].nunique())

        coverage_pct = (100.0 * sampled_positions / total_positions) if total_positions > 0 else 0.0

        rows.append(
            {
                'model': model_label,
                'folder': folder,
                'benchmark': benchmark,
                'total_positions': total_positions,
                f'{coverage_col}_sampled_positions': sampled_positions,
                f'{coverage_col}_coverage_pct': coverage_pct,
                'status': 'ok',
            }
        )

    return pd.DataFrame(rows)


def plot_compare(
    df: pd.DataFrame,
    benchmark_col: str,
    y_col: str,
    left_label: str,
    right_label: str,
    title: str,
    ylabel: str,
    out_png: Path,
    out_svg: Path,
) -> None:
    order = list(BENCHMARK_ORDER)
    extras = sorted(set(df[benchmark_col].dropna()) - set(order))
    order.extend(extras)

    pivot = df.pivot_table(index=benchmark_col, columns='model_like', values=y_col, aggfunc='first')
    pivot = pivot.reindex(order)

    x = np.arange(len(pivot.index))
    fig, ax = plt.subplots(figsize=(14, 7))

    if left_label in pivot.columns:
        left_vals = pivot[left_label]
        left_mask = left_vals.notna().values
        ax.scatter(
            x[left_mask],
            left_vals[left_mask],
            s=90,
            marker='o',
            edgecolor='black',
            linewidth=0.6,
            label=left_label,
            zorder=3,
        )
    if right_label in pivot.columns:
        right_vals = pivot[right_label]
        right_mask = right_vals.notna().values
        ax.scatter(
            x[right_mask],
            right_vals[right_mask],
            s=90,
            marker='D',
            edgecolor='black',
            linewidth=0.6,
            label=right_label,
            zorder=3,
        )

    if 'coverage' not in y_col:
        ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=75, ha='right')
    ax.set_title(title)
    ax.set_xlabel('Benchmark')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_svg)
    plt.close(fig)


def main() -> None:
    for cfg in COMPARISONS:
        out_dir = cfg['out_dir']
        out_dir.mkdir(parents=True, exist_ok=True)

        mut_df = pd.concat(
            [
                collect_mut_spearman(cfg['left_root'], cfg['left_label'], cfg['target_mut']),
                collect_mut_spearman(cfg['right_root'], cfg['right_label'], cfg['target_mut']),
            ],
            ignore_index=True,
        )
        mut_csv = out_dir / f"mut_{cfg['target_mut']}_spearman_summary_combined.csv"
        mut_df.to_csv(mut_csv, index=False)

        mut_plot_df = mut_df.copy()
        mut_plot_df['model_like'] = mut_plot_df['model']
        plot_compare(
            mut_plot_df,
            benchmark_col='benchmark',
            y_col='spearman_rho',
            left_label=cfg['left_label'],
            right_label=cfg['right_label'],
            title=f"mut == '{cfg['target_mut']}': DMS_score vs UNAAGI",
            ylabel='Spearman rho',
            out_png=out_dir / f"mut_{cfg['target_mut']}_spearman_compare.png",
            out_svg=out_dir / f"mut_{cfg['target_mut']}_spearman_compare.svg",
        )

        overall_df = pd.concat(
            [
                collect_overall_from_results(cfg['left_root'], cfg['left_label']),
                collect_overall_from_results(cfg['right_root'], cfg['right_label']),
            ],
            ignore_index=True,
        )
        overall_csv = out_dir / 'overall_unaagi_spearman_from_results_combined.csv'
        overall_df.to_csv(overall_csv, index=False)

        overall_plot_df = overall_df.copy()
        overall_plot_df['model_like'] = overall_plot_df['model_name']
        plot_compare(
            overall_plot_df,
            benchmark_col='benchmark',
            y_col='unaagi_spearmanr_overall',
            left_label=cfg['left_label'],
            right_label=cfg['right_label'],
            title='Overall UNAAGI Spearman from results.csv',
            ylabel='Spearman rho',
            out_png=out_dir / 'overall_unaagi_spearman_compare.png',
            out_svg=out_dir / 'overall_unaagi_spearman_compare.svg',
        )

        coverage_left_root = cfg.get('coverage_left_root', cfg['left_root'])
        coverage_right_root = cfg.get('coverage_right_root', cfg['right_root'])

        cov_df = pd.concat(
            [
                collect_coverage(coverage_left_root, cfg['left_label'], cfg['target_mut'], cfg['coverage_col']),
                collect_coverage(coverage_right_root, cfg['right_label'], cfg['target_mut'], cfg['coverage_col']),
            ],
            ignore_index=True,
        )
        cov_csv = out_dir / f"{cfg['coverage_col']}_coverage_summary_combined.csv"
        cov_df.to_csv(cov_csv, index=False)

        cov_plot_df = cov_df.copy()
        cov_plot_df['model_like'] = cov_plot_df['model']
        plot_compare(
            cov_plot_df,
            benchmark_col='benchmark',
            y_col=f"{cfg['coverage_col']}_coverage_pct",
            left_label=cfg['left_label'],
            right_label=cfg['right_label'],
            title=f"{cfg['target_name']} coverage comparison",
            ylabel='Coverage (%)',
            out_png=out_dir / f"{cfg['coverage_col']}_coverage_compare.png",
            out_svg=out_dir / f"{cfg['coverage_col']}_coverage_compare.svg",
        )

        print(f'[done] {out_dir}')
        print(f'  - {mut_csv.name}')
        print(f'  - {overall_csv.name}')
        print(f'  - {cov_csv.name}')


if __name__ == '__main__':
    main()
