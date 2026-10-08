import json
NAT = {"ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU",
       "MET","ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"}
d = json.load(open("/home/qcx679/hantang/v6_analysis/v6_e2e.json"))
codes = [c for c in sorted({k.split("|")[2] for k in d}) if c != "CHA"]
ncaa = sorted(c for c in codes if c.upper() not in NAT)
nat  = sorted(c for c in codes if c.upper() in NAT)
STAGES = ("w1.0", "w2.0", "w3.0", "nopkt")

def get(arm, st, c):
    return d.get("%s|%s|%s" % (arm, st, c))

def e2e_cell(arm, st, c):
    s = get(arm, st, c)
    if not s or not s["total"]:
        return '<td class="w">&mdash;</td>'
    v = 100.0 * s["hits"] / s["total"]
    cls = ' class="w best"' if v == 0 else (' class="w"' if v < 1 else (' class="best"' if v >= 60 else ""))
    return '<td%s>%s</td>' % (cls, ("%.1f" % v) if v >= 1 else ("%.2f" % v))

def self_cell(arm, st, c):
    """self-rate with its own denominator shown when that denominator is small."""
    s = get(arm, st, c)
    if not s:
        return '<td class="w">&mdash;</td>'
    n = s["named"]
    if n == 0:
        return '<td class="w">&mdash;</td>'
    v = 100.0 * s["hits"] / n
    small = n < 30
    cls = ' class="w"' if (small or v < 30) else (' class="best"' if v >= 90 else "")
    txt = "%.0f%%" % v
    if small:
        txt += '<span style="opacity:.6"> (n=%d)</span>' % int(n)
    return '<td%s>%s</td>' % (cls, txt)

def cov_cell(arm, st, c):
    s = get(arm, st, c)
    if not s:
        return '<td class="w">&mdash;</td>'
    n, N = s["pos_hit"], s["npos"]
    cls = ' class="w best"' if n == 0 else (' class="w"' if n <= 3 else (' class="best"' if n == N else ""))
    return '<td%s>%d/%d</td>' % (cls, n, N)

def build(fn, with_cov):
    out = []
    for grp, lst in (("N", ncaa), ("A", nat)):
        for i, c in enumerate(lst):
            cls = ' class="grp"' if (grp == "A" and i == 0) else ""
            tds = ["<td>%s</td>" % c]
            for arm in ("grover", "morgan"):
                if with_cov:
                    tds.append(cov_cell(arm, "w1.0", c))
                for st in STAGES:
                    tds.append(fn(arm, st, c))
            out.append("      <tr%s>%s</tr>" % (cls, "".join(tds)))
    return "\n".join(out) + "\n"

open("/home/qcx679/hantang/v6_analysis/tbody_e2e.html", "w").write(build(e2e_cell, True))
open("/home/qcx679/hantang/v6_analysis/tbody_self.html", "w").write(build(self_cell, False))
print("e2e rows/cells:", build(e2e_cell, True).count("<tr"), build(e2e_cell, True).split("\n")[0].count("<td"))
print("self rows/cells:", build(self_cell, False).count("<tr"), build(self_cell, False).split("\n")[0].count("<td"))
print(build(self_cell, False).split("\n")[0])
