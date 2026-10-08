"""Adds the off-diagonal ('accidental') metric: how often the target residue is
produced at a position when some OTHER residue was requested. That is the closest
proxy to an unconditional preference obtainable from forced-condition runs, and it
is what the canonical DMS evaluator scores on (an unconditional count per position).
"""
import pandas as pd, numpy as np, os

META = {"aa","pos","UNK","INV","chir_L","chir_D","chir_achiral","chir_unknown"}
B = "/datasets/biochem/unaagi/samples_v6"
DMS = "/home/qcx679/hantang/UAAG2/data/uaa_benchmark_csv/CP2_reframe.csv"
NAT = {"ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU",
       "MET","ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"}
STAGES = {
    "grover w1.0":  "grover_cond/runv6_grover_cond",
    "grover w2.0":  "grover_cfg/runv6_grover_cfg_w2.0",
    "grover nopkt": "grover_nopocket/runv6_grover_nopocket",
    "morgan w1.0":  "morgan_cond/runv6_morgan_cond",
    "morgan w2.0":  "morgan_cfg/runv6_morgan_cfg_w2.0",
    "morgan nopkt": "morgan_nopocket/runv6_morgan_nopocket",
}

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    if len(x) < 8 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, len(x)
    return np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1], len(x)

def load(sub):
    root = os.path.join(B, sub)
    own, offc, offt = {}, {}, {}
    for cond in sorted(os.listdir(root)):
        f = os.path.join(root, cond, "aa_distribution_split0.csv")
        if not os.path.exists(f): continue
        d = pd.read_csv(f)
        res = [c for c in d.columns if c not in META]
        lut = {c.upper(): c for c in res}
        num = d[res].apply(pd.to_numeric, errors="coerce").fillna(0)
        inv = pd.to_numeric(d["INV"], errors="coerce").fillna(0)
        unk = pd.to_numeric(d["UNK"], errors="coerce").fillna(0)
        C = cond.upper(); col = lut.get(C)
        for i in range(len(d)):
            wt, pos = str(d["aa"][i]).strip().upper(), int(d["pos"][i])
            named = float(num.iloc[i].sum()); total = named + float(inv[i]) + float(unk[i])
            own[(C, wt, pos)] = dict(hits=float(num[col][i]) if col else 0.0,
                                     named=named, total=total)
            for rc in res:
                RC = rc.upper()
                if RC == C:      # skip the forced diagonal
                    continue
                offc[(RC, wt, pos)] = offc.get((RC, wt, pos), 0.0) + float(num[rc][i])
                offt[(RC, wt, pos)] = offt.get((RC, wt, pos), 0.0) + total
    return own, offc, offt

dms = pd.read_csv(DMS)
for c in ("aa", "target"):
    dms[c] = dms[c].astype(str).str.strip().str.upper()
dms = dms.dropna(subset=["value"])

rows = []
for label, sub in STAGES.items():
    own, offc, offt = load(sub)
    rec = {k: [] for k in ("e2e","self","offdiag","val","kind")}
    for _, r in dms.iterrows():
        k = (r["target"], r["aa"], int(r["pos"]))
        s = own.get(k)
        if s is None or s["total"] == 0: continue
        rec["e2e"].append(100*s["hits"]/s["total"])
        rec["self"].append(100*s["hits"]/s["named"] if s["named"] else np.nan)
        t = offt.get(k, 0.0)
        rec["offdiag"].append(100*offc.get(k,0.0)/t if t else np.nan)
        rec["val"].append(float(r["value"]))
        rec["kind"].append("nat" if r["target"] in NAT else "NCAA")
    df = pd.DataFrame(rec)
    for sset in ("all","NCAA","nat"):
        g = df if sset=="all" else df[df["kind"]==sset]
        o = {"stage": label, "subset": sset, "n": len(g)}
        for m in ("e2e","self","offdiag"):
            o[m], _ = spearman(g[m], g["val"])
        rows.append(o)

R = pd.DataFrame(rows)
print("=== Spearman rho vs CP2 DMS fitness (higher value = fitter) ===")
print("  e2e     : forced-condition end-to-end rate at that position")
print("  self    : forced-condition self-rate at that position")
print("  offdiag : rate of the target at that position when a DIFFERENT residue was requested")
print("            (preference proxy; what the canonical evaluator scores, unconditionally)\n")
print(R.to_string(index=False, float_format=lambda v: "%+.3f" % v))
R.to_csv("/home/qcx679/hantang/v6_analysis/v6_dms_corr2.csv", index=False)

# how much signal is even available off-diagonal?
own, offc, offt = load(STAGES["grover w1.0"])
nz = [v for v in offc.values() if v > 0]
print("\noff-diagonal support (grover w1.0): %d of %d (residue,position) cells nonzero"
      % (len(nz), len(offc)))
