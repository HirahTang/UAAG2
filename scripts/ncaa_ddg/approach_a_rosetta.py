"""
Approach A: Rosetta cartesian_ddg with EXISTING params only.

Covers NCAAs that are already in Rosetta's NCAA residue type database.
All others are skipped (reported as NaN).

Requires PyRosetta (academic license from https://graylab.jhu.edu/PyRosetta4/).
Install: pip install pyrosetta --index-url https://graylab.jhu.edu/download/PyRosetta4/archive/pip/release/

Coverage among benchmark NCAAs:
  Likely parameterized: AIB, NLE, NVA, DAL, ABU
  Need params (skipped): 2NP, 2TH, 3TH, AHP, AOC, BZT, CHA, CPA, HSM,
                          MEA, MEB, MEF, MEG, TBU, TME, YME
"""
from __future__ import annotations

import faulthandler
faulthandler.enable()

import argparse
import csv
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from ncaa_smiles import CANONICAL_AAS

# Rosetta 3-letter → Rosetta residue name (standard + known NCAA)
ROSETTA_NCAA_MAP = {
    # Standard canonical
    **{aa: aa for aa in CANONICAL_AAS},
    # NCAAs with known Rosetta params (PyRosetta 2026 build)
    "AIB": "AIB",    # alpha-aminoisobutyric acid (verified present)
    # NLE, NVA, DAL, ABU: not in this PyRosetta 2026 build -> moved to UNSUPPORTED
}

# NCAAs we CANNOT score (no existing params)
UNSUPPORTED_NCAA = {
    "2NP","2TH","3TH","AHP","AOC","BZT","CHA","CPA",
    "HSM","MEA","MEB","MEF","MEG","TBU","TME","YME",
    # Not in PyRosetta 2026 conda build:
    "NLE","NVA","DAL","ABU",
}


def run_benchmark(
    pdb_path: str,
    chain: str,
    benchmark_csv: str,
    output_dir: str,
    nstruct: int = 1,
):
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    try:
        import pyrosetta
        from pyrosetta import init, pose_from_pdb
    except ImportError:
        print("[A] PyRosetta not installed. Install from https://graylab.jhu.edu/PyRosetta4/")
        print("    Approach A requires: pip install pyrosetta "
              "--index-url https://graylab.jhu.edu/download/PyRosetta4/archive/pip/release/")
        sys.exit(1)

    print("[A] Initializing PyRosetta...")
    t_init = time.time()
    init("-mute all -ex1 -ex2 -extrachi_cutoff 1 -use_input_sc -ignore_unrecognized_res -ignore_zero_occupancy false")
    t_init = time.time() - t_init
    print(f"    PyRosetta init: {t_init:.1f}s")

    print(f"[A] Loading structure: {pdb_path}")
    t_load = time.time()
    pose = pose_from_pdb(pdb_path)
    scorefxn = pyrosetta.create_score_function("ref2015")
    e_wt = scorefxn(pose)
    t_load = time.time() - t_load
    print(f"    WT energy: {e_wt:.2f} REU  t={t_load:.1f}s")

    # Read mutations
    mutations = []
    with open(benchmark_csv) as f:
        for row in csv.DictReader(f):
            if not row.get("value", "").strip():
                continue
            mutations.append({
                "wt_aa": row["aa"],
                "pos": int(float(row["pos"])),
                "target": row["target"],
                "value": float(row["value"].strip()),
            })

    results = []
    resources = {
        "wt_energy_REU": e_wt,
        "wt_load_s": t_load,
        "init_s": t_init,
        "approach": "A_rosetta_existing_params",
        "mutations": [],
    }

    for i, mut in enumerate(mutations):
        target = mut["target"]
        pos = mut["pos"]

        if target in UNSUPPORTED_NCAA:
            pred_score = float("nan")
            t_mut = 0.0
            mem_peak = 0.0
            print(f"[{i+1}/{len(mutations)}] {mut['wt_aa']}{pos}→{target}  SKIPPED (no params)")
        else:
            # Find Rosetta residue number for chain+position
            pdb_info = pose.pdb_info()
            try:
                resnum = pdb_info.pdb2pose(chain, pos)
            except Exception:
                pred_score = float("nan")
                results.append({**mut, "pred_score": pred_score})
                continue
            if resnum == 0:
                print(f"[{i+1}/{len(mutations)}] {mut['wt_aa']}{pos}→{target}  SKIP (pos not in chain {chain})")
                pred_score = float("nan")
                results.append({**mut, "pred_score": pred_score})
                continue

            t_mut = time.time()
            tracemalloc.start()
            try:
                from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
                mutant_pose = pose.clone()
                rosetta_name = ROSETTA_NCAA_MAP.get(target, target)
                MutateResidue(resnum, rosetta_name).apply(mutant_pose)
                e_mut = scorefxn(mutant_pose)
                pred_score = -(e_mut - e_wt)  # negative ΔΔG: positive = stabilizing
                print(f"[{i+1}/{len(mutations)}] {mut['wt_aa']}{pos}→{target}  "
                      f"ΔΔG={e_mut-e_wt:.2f} REU  pred={pred_score:.3f}")
            except Exception as e:
                print(f"[{i+1}/{len(mutations)}] {mut['wt_aa']}{pos}→{target}  ERROR: {e}")
                pred_score = float("nan")
            mem_peak = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
            t_mut = time.time() - t_mut

        results.append({**mut, "pred_score": pred_score})
        resources["mutations"].append({
            "mutation": f"{mut['wt_aa']}{pos}{target}",
            "wall_s": round(t_mut, 3),
            "mem_mb": round(mem_peak, 1),
            "skipped": target in UNSUPPORTED_NCAA,
        })

    # Spearman on supported mutations only
    valid = [(r["value"], r["pred_score"]) for r in results
             if not np.isnan(r["pred_score"])]
    rho, pval = (float("nan"), float("nan"))
    if len(valid) > 2:
        y_true, y_pred = zip(*valid)
        rho, pval = spearmanr(y_true, y_pred)

    n_supported = sum(1 for r in results if not np.isnan(r["pred_score"]))
    print(f"\n[A] Spearman ρ = {rho:.4f}  (p={pval:.4g})")
    print(f"    Coverage: {n_supported}/{len(mutations)} mutations scored")

    resources.update({
        "total_wall_s": round(time.time() - t0, 1),
        "spearman_rho": rho,
        "spearman_p": pval,
        "n_valid": n_supported,
        "n_total": len(mutations),
    })

    out_csv = os.path.join(output_dir, "approach_a_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with open(os.path.join(output_dir, "approach_a_resources.json"), "w") as f:
        json.dump(resources, f, indent=2)

    print(f"[A] Results saved to {out_csv}")
    return rho


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approach A: Rosetta existing params")
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nstruct", type=int, default=3)
    args = parser.parse_args()
    run_benchmark(args.pdb, args.chain, args.benchmark_csv,
                  args.output_dir, args.nstruct)
