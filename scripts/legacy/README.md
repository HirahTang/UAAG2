# Legacy plotters — retired

Kept for history only. **Do not extend these.** Use `scripts/compare_models.py` (the canonical
N-model plotter; reproduces the gold-standard figures in `~/unaagi_v03_plots/`) or, for the
documented CLI path, `scripts/generate_comparison_plots.py`.

| File | Was | Superseded by |
|---|---|---|
| `compare_ctmc_vs_ddpm.py` | 2-model hardcoded (prior_ctmc vs prior_ddpm); the styling base | `compare_models.py` |
| `generate_clean_plots.py` | "clean" variant; made original `figures/p0203_ctmc` | `compare_models.py` |
| `generate_desktop_compare_reports.py` | desktop HTML report + NCAA coverage | `compare_models.py` |

See `../../SCRIPT_CATALOG.md` §3.
