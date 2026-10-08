import pandas as pd, os, json
META = {"aa","pos","UNK","INV","chir_L","chir_D","chir_achiral","chir_unknown"}
B = "/datasets/biochem/unaagi/samples_v6"
NAT = {"ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU",
       "MET","ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"}

def stats(sub, code):
    p = "%s/%s/%s/aa_distribution_split0.csv" % (B, sub, code)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    res = [c for c in d.columns if c not in META]
    num = d[res].apply(pd.to_numeric, errors="coerce").fillna(0)
    m = [c for c in res if c.upper() == code.upper()]
    per_pos = num[m[0]] if m else None
    hits = float(per_pos.sum()) if m else 0.0
    pos_hit = int((per_pos > 0).sum()) if m else 0
    named = float(num.values.sum())
    inv = float(pd.to_numeric(d["INV"], errors="coerce").fillna(0).sum())
    unk = float(pd.to_numeric(d["UNK"], errors="coerce").fillna(0).sum())
    return dict(hits=hits, named=named, total=named + inv + unk,
                pos_hit=pos_hit, npos=int(len(d)))

SUB = {}
for arm in ("grover", "morgan"):
    SUB[(arm, "w1.0")]  = "%s_cond/runv6_%s_cond" % (arm, arm)
    SUB[(arm, "w1.5")]  = "%s_cfg/runv6_%s_cfg_w1.5" % (arm, arm)
    SUB[(arm, "w2.0")]  = "%s_cfg/runv6_%s_cfg_w2.0" % (arm, arm)
    SUB[(arm, "w3.0")]  = "%s_cfg/runv6_%s_cfg_w3.0" % (arm, arm)
    SUB[(arm, "w5.0")]  = "%s_cfg/runv6_%s_cfg_w5.0" % (arm, arm)
    SUB[(arm, "nopkt")] = "%s_nopocket/runv6_%s_nopocket" % (arm, arm)

root = B + "/grover_cond/runv6_grover_cond"
codes = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))

out = {}
print("=== AGGREGATE ===")
print("%-16s %10s %10s %10s   %10s %10s" % ("arm|stage","NCAA e2e","NCAA self","NCAA cov","nat e2e","nat cov"))
for (arm, st), sub in SUB.items():
    A = {"NCAA": [0,0,0,0,0], "nat": [0,0,0,0,0]}   # hits,total,named,pos_hit,npos
    for c in codes:
        s = stats(sub, c)
        if not s or s["total"] == 0:
            continue
        out["%s|%s|%s" % (arm, st, c)] = s
        if c == "CHA":
            continue
        k = "nat" if c.upper() in NAT else "NCAA"
        A[k][0] += s["hits"]; A[k][1] += s["total"]; A[k][2] += s["named"]
        A[k][3] += s["pos_hit"]; A[k][4] += s["npos"]
    f = lambda k: (100*A[k][0]/max(A[k][1],1), 100*A[k][0]/max(A[k][2],1), 100*A[k][3]/max(A[k][4],1))
    ne, ns, nc = f("NCAA"); ae, _, ac = f("nat")
    print("%-16s %9.1f%% %9.1f%% %9.1f%%   %9.1f%% %9.1f%%" % (arm+"|"+st, ne, ns, nc, ae, ac))

json.dump(out, open("/home/qcx679/hantang/v6_analysis/v6_e2e.json", "w"))
print("\nsaved v6_e2e.json (%d cells)" % len(out))
print("npos values seen:", sorted({v["npos"] for v in out.values()}))
