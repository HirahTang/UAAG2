"""Is the negative off-diagonal correlation a residue-size confound?

If small residues are (a) produced accidentally most often and (b) deleterious at
CP2 binding positions, the anti-correlation is size, not a preference inversion.
"""
import pandas as pd, numpy as np, os
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")

META = {"aa","pos","UNK","INV","chir_L","chir_D","chir_achiral","chir_unknown"}
B = "/datasets/biochem/unaagi/samples_v6"
DMS = "/home/qcx679/hantang/UAAG2/data/uaa_benchmark_csv/CP2_reframe.csv"
CE = "/datasets/biochem/unaagi/condition_embeddings/cp2_experiment_conditions.npz"
NAT = {"ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU",
       "MET","ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"}

z = np.load(CE, allow_pickle=True)
SIZE = {}
for c, s in zip(z["codes"], z["smiles"]):
    m = Chem.MolFromSmiles(str(s))
    if m: SIZE[str(c).upper()] = m.GetNumHeavyAtoms()

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    if len(x) < 8: return np.nan, len(x)
    return np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0,1], len(x)

# off-diagonal rates from grover w1.0
root = os.path.join(B, "grover_cond/runv6_grover_cond")
offc, offt = {}, {}
for cond in sorted(os.listdir(root)):
    f = os.path.join(root, cond, "aa_distribution_split0.csv")
    if not os.path.exists(f): continue
    d = pd.read_csv(f)
    res = [c for c in d.columns if c not in META]
    num = d[res].apply(pd.to_numeric, errors="coerce").fillna(0)
    inv = pd.to_numeric(d["INV"], errors="coerce").fillna(0)
    unk = pd.to_numeric(d["UNK"], errors="coerce").fillna(0)
    C = cond.upper()
    for i in range(len(d)):
        wt, pos = str(d["aa"][i]).strip().upper(), int(d["pos"][i])
        total = float(num.iloc[i].sum()) + float(inv[i]) + float(unk[i])
        for rc in res:
            RC = rc.upper()
            if RC == C: continue
            offc[(RC,wt,pos)] = offc.get((RC,wt,pos),0.0) + float(num[rc][i])
            offt[(RC,wt,pos)] = offt.get((RC,wt,pos),0.0) + total

dms = pd.read_csv(DMS)
for c in ("aa","target"): dms[c] = dms[c].astype(str).str.strip().str.upper()
dms = dms.dropna(subset=["value"])
dms["size"] = dms["target"].map(SIZE)
dms["off"] = [100*offc.get((r.target,r.aa,int(r.pos)),0.0)/offt[(r.target,r.aa,int(r.pos))]
              if (r.target,r.aa,int(r.pos)) in offt and offt[(r.target,r.aa,int(r.pos))] else np.nan
              for r in dms.itertuples()]
dms = dms.dropna(subset=["off","size"])

print("n = %d mutations" % len(dms))
for lbl, sub in (("all", dms), ("NCAA", dms[~dms.target.isin(NAT)]), ("natural", dms[dms.target.isin(NAT)])):
    a,_ = spearman(sub["size"], sub["off"])
    b,_ = spearman(sub["size"], sub["value"])
    c,_ = spearman(sub["off"], sub["value"])
    print("%-8s n=%3d   rho(size, offdiag)=%+.3f   rho(size, DMS)=%+.3f   rho(offdiag, DMS)=%+.3f"
          % (lbl, len(sub), a, b, c))

print("\nper-residue mean off-diagonal rate vs size (all positions pooled):")
g = dms.groupby("target").agg(size=("size","first"), off=("off","mean"), dms=("value","mean")).sort_values("size")
print(g.to_string(float_format=lambda v: "%.2f" % v))

# partial: does offdiag still predict DMS once size is regressed out (rank-space)?
from itertools import product
r_od, _ = spearman(dms["off"], dms["value"])
r_os, _ = spearman(dms["off"], dms["size"])
r_sd, _ = spearman(dms["size"], dms["value"])
den = np.sqrt((1-r_os**2)*(1-r_sd**2))
print("\nSpearman partial rho(offdiag, DMS | size) = %+.3f   (raw %+.3f)"
      % ((r_od - r_os*r_sd)/den if den else np.nan, r_od))
