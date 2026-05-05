"""Validate benchmark .pt files.

For every graph in every benchmark:
  - is_backbone.sum() == 4   (N, Ca, C, O always present)
  - is_ligand.sum()  >= 4    (at least the 4 backbone atoms)

Reports any violations and prints a summary.
"""
import glob
import sys
import torch

BENCH_DIR = "/scratch/project_465002574/UNAAGI_benchmarks"
UAA_BENCH = [
    "/scratch/project_465002574/UNAAGI_benchmarks/2roc_puma.pt",
    "/scratch/project_465002574/UNAAGI_benchmarks/5ly1_cp2.pt",
]

pt_files = sorted(glob.glob(f"{BENCH_DIR}/*.pt"))
pt_files += [f for f in UAA_BENCH if f not in pt_files]

total_graphs = 0
violations = []

for pt_path in pt_files:
    data = torch.load(pt_path, weights_only=False)
    name = pt_path.split("/")[-1]
    file_ok = True
    for idx, graph in enumerate(data):
        total_graphs += 1
        bb = int(graph.is_backbone.sum().item())
        lig = int(graph.is_ligand.sum().item())
        cid = getattr(graph, "compound_id", f"graph_{idx}")
        if bb != 4 or lig < 4:
            violations.append({
                "file": name, "idx": idx, "compound_id": cid,
                "is_backbone_sum": bb, "is_ligand_sum": lig,
            })
            file_ok = False
    status = "OK" if file_ok else f"VIOLATIONS ({sum(1 for v in violations if v['file']==name)})"
    print(f"  {name:<45} {len(data):>5} graphs  {status}")

print()
print(f"Total graphs checked : {total_graphs}")
print(f"Total violations     : {len(violations)}")

if violations:
    print()
    print("Violation details:")
    for v in violations:
        print(f"  {v['file']}  idx={v['idx']}  {v['compound_id']}"
              f"  is_backbone={v['is_backbone_sum']}  is_ligand={v['is_ligand_sum']}")
    sys.exit(1)
else:
    print("All benchmarks passed.")
