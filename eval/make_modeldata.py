#!/usr/bin/env python3
"""Stage-4 glue: turn per-assay eval outputs into one <MODEL_TAG>_modeldata.csv
that scripts/compare_models.py consumes.

Output columns: model,assay,subset(all|naa|ncaa),mean,std   (per-iteration mean/std over 5 iters)
  - 25 ProteinGym assays: subset=all only, read from each assay's spearman_per_iter.csv
  - CP2, PUMA: subset in {all,naa,ncaa}, computed per-iter from the UAA scorer's
    *_benchmark_results_raw.csv files (pred = logP_wt - logP_mut), then mean/std over iters.

Also collates the ProteinGym baseline tables into <out-dir>/baselines_pg/<assay>_<TAG>_1000/results.csv
so the plotter finds them.

Run as part of eval/finalize_and_plot.sh; or standalone:
  python eval/make_modeldata.py --result-base /scratch/.../UNAAGI_result/results/<TAG> \
      --model-tag <TAG> --out-dir /scratch/.../UNAAGI_result/modeldata
Self-test (no cluster):  python eval/make_modeldata.py --selftest
"""
import os, glob, argparse, sys, shutil
import numpy as np, pandas as pd
from scipy.stats import spearmanr

PROTEINGYM = ["A0A247D711_LISMN","AICDA_HUMAN","ARGR_ECOLI","B2L11_HUMAN","CCDB_ECOLI","DLG4_RAT",
    "DN7A_SACS2","ENVZ_ECOLI","ENV_HV1B9","ERBB2_HUMAN","FKBP3_HUMAN","HCP_LAMBD","IF1_ECOLI",
    "ILF3_HUMAN","OTU7A_HUMAN","PKN1_HUMAN","RS15_GEOSE","SBI_STAAM","SCIN_STAAR","SOX30_HUMAN",
    "SQSTM_MOUSE","SUMO1_HUMAN","TAT_HV1BR","VG08_BPP22","VRPI_BPT7"]
# canonical 20 (NAA); everything else in a UAA benchmark is NCAA
NAA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())


def proteingym_rows(result_base, tag, n):
    rows = []
    for p in PROTEINGYM:
        f = os.path.join(result_base, f"{p}_{tag}_{n}", "spearman_per_iter.csv")
        if not os.path.isfile(f):
            print(f"[warn] missing {f}"); continue
        d = pd.read_csv(f)
        mean = float(d.loc[d["iter"] == "mean", "spearmanr"].values[0])
        std = float(d.loc[d["iter"] == "std", "spearmanr"].values[0])
        rows.append((tag, p, "all", mean, std))
    return rows


def _spr_subset(df, subset):
    if subset == "naa":  df = df[df["target"].isin(NAA)]
    elif subset == "ncaa": df = df[~df["target"].isin(NAA)]
    if len(df) < 3: return np.nan
    return spearmanr(df["pred"], df["value"]).statistic


def uaa_rows(result_base, tag, assay, n_iters=5):
    """Per-iter all/naa/ncaa Spearman from the UAA scorer's raw output, aggregated to mean/std."""
    rows = []
    per = {"all": [], "naa": [], "ncaa": []}
    for it in range(n_iters):
        raw = os.path.join(result_base, f"{assay}_{tag}_iter{it}", "all_benchmark_results_raw.csv")
        if not os.path.isfile(raw):
            print(f"[warn] missing {raw}"); continue
        df = pd.read_csv(raw)
        if "target" in df: df["target"] = df["target"].astype(str).str.upper()
        for s in per: per[s].append(_spr_subset(df, s))
    for s in ("all", "naa", "ncaa"):
        v = [x for x in per[s] if x == x]  # drop nan
        if not v: continue
        rows.append((tag, assay, s, float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)))
    return rows


def collate_baselines(result_base, tag, n, out_dir):
    dst = os.path.join(out_dir, "baselines_pg"); os.makedirs(dst, exist_ok=True)
    cnt = 0
    for p in PROTEINGYM:
        src = os.path.join(result_base, f"{p}_{tag}_{n}", "results.csv")
        if os.path.isfile(src):
            d = os.path.join(dst, f"{p}_{tag}_{n}"); os.makedirs(d, exist_ok=True)
            shutil.copy(src, os.path.join(d, "results.csv")); cnt += 1
    print(f"[info] collated {cnt} baseline results.csv -> {dst}")


def build(result_base, tag, n, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = proteingym_rows(result_base, tag, n)
    for assay in ("CP2", "PUMA"):
        rows += uaa_rows(result_base, tag, assay)
    df = pd.DataFrame(rows, columns=["model", "assay", "subset", "mean", "std"])
    out = os.path.join(out_dir, f"{tag}_modeldata.csv")
    df.to_csv(out, index=False)
    print(f"[ok] wrote {out}  ({len(df)} rows: {df[df.subset=='all'].shape[0]} all, "
          f"{df[df.subset=='ncaa'].shape[0]} ncaa, {df[df.subset=='naa'].shape[0]} naa)")
    collate_baselines(result_base, tag, n, out_dir)
    return out


def selftest():
    import tempfile
    tmp = tempfile.mkdtemp(); rb = os.path.join(tmp, "results", "TESTM"); tag = "TESTM"
    # fake ProteinGym spearman_per_iter for 2 assays
    for p in PROTEINGYM[:2]:
        d = os.path.join(rb, f"{p}_{tag}_1000"); os.makedirs(d)
        pd.DataFrame({"iter": [0, 1, 2, 3, 4, "mean", "std"],
                      "aa_csv": [""] * 7,
                      "spearmanr": [0.40, 0.42, 0.39, 0.41, 0.43, 0.41, 0.015]}).to_csv(
            os.path.join(d, "spearman_per_iter.csv"), index=False)
        pd.DataFrame({"model": ["ESM2_15B", "ProteinMPNN"], "spearmanr_pred": [0.5, 0.3],
                      "ndcg_pred": [0.8, 0.7]}).to_csv(os.path.join(d, "results.csv"), index=False)
    # fake CP2 raw per iter (4 NAA + 4 NCAA targets, >=3 per subset)
    rng = np.random.default_rng(0)
    for it in range(5):
        d = os.path.join(rb, f"CP2_{tag}_iter{it}"); os.makedirs(d)
        tgt = ["ALA", "VAL", "LEU", "SER", "ABU", "NVA", "NLE", "AIB"]
        pd.DataFrame({"target": tgt,
                      "pred": rng.normal(0, 1, len(tgt)),
                      "value": [-0.5, -0.2, -0.3, 0.1, 0.3, 0.1, 0.4, -0.1]}).to_csv(
            os.path.join(d, "all_benchmark_results_raw.csv"), index=False)
    out = build(rb, tag, 1000, os.path.join(tmp, "modeldata"))
    df = pd.read_csv(out); print(df.to_string(index=False))
    assert set(df[df.assay == "CP2"]["subset"]) == {"all", "naa", "ncaa"}, "CP2 must have 3 subsets"
    assert df[df.subset == "all"].shape[0] == 3, "2 ProteinGym + CP2 all"
    print("\nSELFTEST OK ->", tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-base"); ap.add_argument("--model-tag")
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(); sys.exit(0)
    if not (a.result_base and a.model_tag and a.out_dir):
        ap.error("need --result-base --model-tag --out-dir (or --selftest)")
    build(a.result_base, a.model_tag, a.num_samples, a.out_dir)
