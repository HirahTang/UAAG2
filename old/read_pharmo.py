#!/usr/bin/env python3
# pip install rdkit-pypi py3Dmol
import argparse, os
from pathlib import Path
import py3Dmol

COLOR = {
    "HD": "0x1f77b4",  # donor -> blue
    "HA": "0xd62728",  # acceptor -> red
    "PO": "0x377eb8",  # positive ionizable -> blue-ish
    "NE": "0xe41a1c",  # negative ionizable -> red-ish
    "EX": "0x7f7f7f",  # excluded volume -> grey
}

def parse_phore(path):
    feats = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s == "$$$$" or s.endswith((".xyz",".sdf",".mol2")):
            # header or ligand path line
            continue
        toks = s.split()
        if len(toks) < 7:
            continue
        ftype = toks[0]
        # common columns we care about:
        # 0:type 1:weight(?) 2:radius 3:on(?) 4:cx 5:cy 6:cz 7:dir_on 8:dx 9:dy 10:dz ...
        try:
            radius = float(toks[2])
            cx, cy, cz = map(float, toks[4:7])
        except Exception:
            continue
        dir_on = 0
        dx = dy = dz = None
        if len(toks) >= 11:
            # some lines set dir_on to 1 for directional features
            try:
                dir_on = int(float(toks[7]))
            except Exception:
                dir_on = 0
            if dir_on == 1:
                try:
                    dx, dy, dz = map(float, toks[8:11])
                except Exception:
                    dx = dy = dz = None
        feats.append({
            "type": ftype,
            "radius": radius,
            "center": (cx, cy, cz),
            "dirpt": (dx, dy, dz) if (dx is not None and dy is not None and dz is not None) else None
        })
    return feats

def main():
    ap = argparse.ArgumentParser(description="Visualize pharmacophore + ligand")
    ap.add_argument("--ligand", required=True, help="Ligand file (.sdf/.mol2/.mol/.xyz)")
    ap.add_argument("--phore", required=True, help="Pharmacophore file")
    ap.add_argument("--out", default="pharmacophore_view.html", help="Output HTML")
    args = ap.parse_args()

    # Load ligand text for py3Dmol (use format by extension)
    lig_text = Path(args.ligand).read_text()
    ext = args.ligand.split(".")[-1].lower()
    fmt = "sdf" if ext in ("sdf","mol") else ("mol2" if ext=="mol2" else "xyz")

    feats = parse_phore(args.phore)

    view = py3Dmol.view(width=900, height=650)
    view.addModel(lig_text, fmt)
    view.setStyle({"stick": {"radius": 0.2}})

    # draw features
    for i, f in enumerate(feats, 1):
        col = COLOR.get(f["type"], "0xaaaaaa")
        x, y, z = f["center"]
        rad = max(0.3, float(f["radius"]))  # ensure visible
        alpha = 0.45 if f["type"] != "EX" else 0.25

        # sphere
        view.addSphere({
            "center": {"x": x, "y": y, "z": z},
            "radius": rad,
            "color": col,
            "alpha": alpha
        })

        # arrow (directional features only)
        if f["dirpt"] is not None and f["type"] in ("HD","HA"):
            dx, dy, dz = f["dirpt"]
            # if dirpt equals center, skip
            if (dx,dy,dz) != (x,y,z):
                view.addArrow({
                    "start": {"x": x, "y": y, "z": z},
                    "end": {"x": dx, "y": dy, "z": dz},
                    "color": col,
                    "radius": 0.15
                })

        # label
        view.addLabel(f'{f["type"]}{i}', {
            "position": {"x": x, "y": y, "z": z},
            "backgroundOpacity": 0.5, "fontSize": 10
        })

    view.zoomTo()
    html = view._make_html()
    Path(args.out).write_text(html)
    print(f"Wrote {args.out} — open it in a browser.")

if __name__ == "__main__":
    main()
