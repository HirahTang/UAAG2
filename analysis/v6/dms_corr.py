"""Do the v6 conditional-sampling metrics correlate with CP2 DMS fitness?

Granularity: one row per (wt_aa, pos, target) DMS measurement. For the forced
condition == target, the aa_distribution row keyed on (aa,pos) gives that
position's counts. Joined on keys, never on row order.
"""
import pandas as pd, numpy as np, os, json, itertools

META = {"aa","pos","UNK","INV","chir_L","chir_D","chir_achiral","chir_unknown"}
B = "/datasets/biochem/unaagi/samples_v6"
DMS = "/home/qcx679/hantang/UAAG2/data/uaa_benchmark_csv/CP2_reframe.csv"
NAT = {"ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU",
       "MET","ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"}

STAGES = {
    "grover w1.0":  "grover_cond/runv6_grover_cond",
    "grover w2.0":  "grover_cfg/runv6_grover_cfg_w2.0",
    "grover w3.0":  "grover_cfg/runv6_grover_cfg_w3.0",
    "grover nopkt": "grover_nopocket/runv6_grover_nopocket",
    "morgan w1.0":  "morgan_cond/runv6_morgan_cond",
    "morgan w2.0":  "morgan_cfg/runv6_morgan_cfg_w2.0",
    "morgan w3.0":  "morgan_cfg/runv6_morgan_cfg_w3.0",
    "morgan nopkt": "morgan_nopocket/runv6_morgan_nopocket",
}

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 8 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, len(x)
    rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
    r = np.corrcoef(rx, ry)[0, 1]
    return r, len(x)

def load_stage(sub):
    """-> dict: (COND, WT, POS) -> dict(hits, named, total); plus per-position marginals."""
    root = os.path.join(B, sub)
    tbl, marg = {}, {}
    for cond in sorted(os.listdir(root)):
        f = os.path.join(root, cond, "aa_distribution_split0.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        res = [c for c in d.columns if c not in META]
        lut = {c.upper(): c for c in res}
        num = d[res].apply(pd.to_numeric, errors="coerce").fillna(0)
        inv = pd.to_numeric(d["INV"], errors="coerce").fillna(0)
        unk = pd.to_numeric(d["UNK"], errors="coerce").fillna(0)
        col = lut.get(cond.upper())
        for i in range(len(d)):
            wt, pos = str(d["aa"][i]).strip().upper(), int(d["pos"][i])
            named = float(num.iloc[i].sum())
            total = named + float(inv[i]) + float(unk[i])
            hits = float(num[col][i]) if col else 0.0
            tbl[(cond.upper(), wt, pos)] = dict(hits=hits, named=named, total=total)
            # marginal: how often does EACH residue appear at this position, over all conditions
            m = marg.setdefault((wt, pos), {"tot": 0.0, "cnt": {}})
            m["tot"] += total
            for rc in res:
                m["cnt"][rc.upper()] = m["cnt"].get(rc.upper(), 0.0) + float(num[rc][i])
    return tbl, marg

dms = pd.read_csv(DMS)
dms["target"] = dms["target"].astype(str).str.strip().str.upper()
dms["aa"] = dms["aa"].astype(str).str.strip().str.upper()
print("DMS rows: %d   distinct targets: %d   positions: %d"
      % (len(dms), dms["target"].nunique(), dms.groupby(["aa","pos"]).ngroups))
print("target classes: %d natural, %d NCAA"
      % (dms["target"].isin(NAT).sum(), (~dms["target"].isin(NAT)).sum()))

rows = []
for label, sub in STAGES.items():
    tbl, marg = load_stage(sub)
    rec = {"e2e": [], "self": [], "hit": [], "marg": [], "val": [], "kind": []}
    miss = 0
    for _, r in dms.iterrows():
        k = (r["target"], r["aa"], int(r["pos"]))
        s = tbl.get(k)
        if s is None or s["total"] == 0:
            miss += 1; continue
        m = marg.get((r["aa"], int(r["pos"])))
        rec["e2e"].append(100*s["hits"]/s["total"])
        rec["self"].append(100*s["hits"]/s["named"] if s["named"] else np.nan)
        rec["hit"].append(1.0 if s["hits"] > 0 else 0.0)
        rec["marg"].append(100*m["cnt"].get(r["target"],0.0)/m["tot"] if m and m["tot"] else np.nan)
        rec["val"].append(float(r["value"]))
        rec["kind"].append("nat" if r["target"] in NAT else "NCAA")
    df = pd.DataFrame(rec)
    for subset in ("all", "NCAA", "nat"):
        g = df if subset == "all" else df[df["kind"] == subset]
        out = {"stage": label, "subset": subset, "n": len(g)}
        for met in ("e2e", "self", "hit", "marg"):
            r, n = spearman(g[met], g["val"])
            out[met] = r
        rows.append(out)
    if label == "grover w1.0":
        print("unjoined DMS rows (no sample for that target): %d" % miss)

R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n=== Spearman rho vs CP2 DMS fitness ===")
print("  e2e  = per-position end-to-end rate under the target's own condition")
print("  self = per-position self-rate (named denominator)")
print("  hit  = binary, was the target produced at all at that position")
print("  marg = marginal density of the target at that position, pooled over all 40 conditions\n")
print(R.to_string(index=False, float_format=lambda v: "%+.3f" % v))
R.to_csv("/home/qcx679/hantang/v6_analysis/v6_dms_corr.csv", index=False)
print("\nsaved v6_dms_corr.csv")
