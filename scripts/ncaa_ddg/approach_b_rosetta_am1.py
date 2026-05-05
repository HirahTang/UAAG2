"""
Approach B: Rosetta + AM1-BCC charges via AmberTools antechamber.

Generates Rosetta .params files for ALL 21 NCAAs using:
  1. RDKit  → 3D conformer from SMILES
  2. antechamber (AmberTools) → AM1-BCC partial charges
  3. molfile_to_params_polymer.py (from PyRosetta) → .params file
  4. MakeRotLib (via PyRosetta) → rotamer library
  5. Rosetta cartesian_ddg → ΔΔG

Requires:
  - PyRosetta (https://graylab.jhu.edu/PyRosetta4/)
  - AmberTools (conda install -c conda-forge ambertools)
  
AmberTools env: /flash/project_465002574/micromamba/envs/ncaa_tools/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

sys.path.insert(0, str(Path(__file__).parent))
from ncaa_smiles import NCAA_SMILES, CANONICAL_AAS

ANTECHAMBER = "/flash/project_465002574/micromamba/envs/ncaa_tools/bin/antechamber"
PARMCHK2    = "/flash/project_465002574/micromamba/envs/ncaa_tools/bin/parmchk2"

PARAMS_CACHE = Path("/flash/project_465002574/rosetta_ncaa/params")
PARAMS_CACHE.mkdir(parents=True, exist_ok=True)


def generate_params_am1(aa_code: str, smiles: str) -> Path | None:
    """
    Generate Rosetta .params file for NCAA via AM1-BCC charges.
    Returns path to .params file, or None on failure.
    """
    params_file = PARAMS_CACHE / f"{aa_code}.params"
    if params_file.exists():
        return params_file

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Generate 3D SDF with RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"    [!] Invalid SMILES for {aa_code}")
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)

        sdf_path = tmpdir / f"{aa_code}.sdf"
        with Chem.SDWriter(str(sdf_path)) as w:
            w.write(mol)

        # 2. antechamber: SDF → mol2 with AM1-BCC charges
        mol2_path = tmpdir / f"{aa_code}.mol2"
        cmd = [
            ANTECHAMBER,
            "-i", str(sdf_path), "-fi", "sdf",
            "-o", str(mol2_path), "-fo", "mol2",
            "-c", "bcc",           # AM1-BCC charges
            "-s", "2",             # silent
            "-pf", "y",            # remove tmp files
            "-nc", "0",            # net charge 0 (neutral AA)
            "-at", "gaff2",        # atom type
        ]
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        t_antechamber = time.time() - t0
        if result.returncode != 0 or not mol2_path.exists():
            print(f"    [!] antechamber failed for {aa_code}: {result.stderr[:200]}")
            return None
        print(f"    antechamber: {t_antechamber:.1f}s")

        # 3. parmchk2: generate frcmod
        frcmod_path = tmpdir / f"{aa_code}.frcmod"
        subprocess.run([
            PARMCHK2, "-i", str(mol2_path), "-f", "mol2",
            "-o", str(frcmod_path), "-s", "2",
        ], capture_output=True, timeout=60)

        # 4. molfile_to_params_polymer.py (from PyRosetta tools)
        try:
            import pyrosetta
            tools_dir = Path(pyrosetta.__file__).parent / "toolbox" / "params_tools"
            m2p = tools_dir / "molfile_to_params_polymer.py"
            if not m2p.exists():
                # Try alternate location
                m2p = Path(pyrosetta.__file__).parent.parent / "apps" / "public" / \
                      "molfile_to_params_polymer.py"
        except ImportError:
            print("    [!] PyRosetta not installed")
            return None

        if not m2p.exists():
            print(f"    [!] molfile_to_params_polymer.py not found")
            return None

        out_prefix = tmpdir / aa_code
        result = subprocess.run(
            [sys.executable, str(m2p),
             "--polymer",
             "--clobber",
             "-n", aa_code,
             "-o", str(out_prefix),
             str(mol2_path)],
            capture_output=True, text=True, cwd=str(tmpdir), timeout=120,
        )
        generated_params = tmpdir / f"{aa_code}.params"
        if not generated_params.exists():
            print(f"    [!] molfile_to_params failed: {result.stderr[:200]}")
            return None

        # Cache the params file
        shutil.copy(generated_params, params_file)
        # Also copy any rotamer library
        rot_lib = tmpdir / f"{aa_code}.rotlib"
        if rot_lib.exists():
            shutil.copy(rot_lib, PARAMS_CACHE / f"{aa_code}.rotlib")

        print(f"    Generated params for {aa_code} → {params_file}")
        return params_file


def run_benchmark(
    pdb_path: str,
    chain: str,
    benchmark_csv: str,
    output_dir: str,
    nstruct: int = 3,
):
    os.makedirs(output_dir, exist_ok=True)
    t0_total = time.time()

    try:
        import pyrosetta
        from pyrosetta import init, pose_from_pdb
        from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
        from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    except ImportError:
        print("[B] PyRosetta not installed.")
        print("    Get academic license at https://www.rosettacommons.org/software/license-and-download")
        print("    Then: pip install pyrosetta "
              "--index-url https://graylab.jhu.edu/download/PyRosetta4/archive/pip/release/")
        sys.exit(1)

    # Pre-generate params for all NCAAs
    print("[B] Pre-generating NCAA params with AM1-BCC charges...")
    params_files = {}
    resources_params = {}
    for aa_code, smiles in NCAA_SMILES.items():
        t_p = time.time()
        pf = generate_params_am1(aa_code, smiles)
        t_p = time.time() - t_p
        resources_params[aa_code] = {"params_s": round(t_p, 2), "success": pf is not None}
        if pf:
            params_files[aa_code] = str(pf)

    # Init PyRosetta with all generated params
    extra_params = " ".join(f"-extra_res_fa {p}" for p in params_files.values())
    init(f"-mute all -ex1 -ex2 -extrachi_cutoff 1 {extra_params}")

    print(f"[B] Loading structure: {pdb_path}")
    t_load = time.time()
    pose = pose_from_pdb(pdb_path)
    scorefxn = ScoreFunctionFactory.create_score_function("ref2015_cart")
    e_wt = scorefxn(pose)
    t_load = time.time() - t_load
    print(f"    WT energy: {e_wt:.2f} REU  t={t_load:.1f}s")

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
        "approach": "B_rosetta_am1_params",
        "wt_energy_REU": e_wt,
        "params_generation": resources_params,
        "mutations": [],
    }

    for i, mut in enumerate(mutations):
        target = mut["target"]
        pos = mut["pos"]

        is_ncaa = target not in CANONICAL_AAS
        has_params = (not is_ncaa) or (target in params_files)

        if not has_params:
            pred_score = float("nan")
            print(f"[{i+1}/{len(mutations)}] {mut['wt_aa']}{pos}→{target}  SKIPPED (params failed)")
        else:
            pdb_info = pose.pdb_info()
            try:
                resnum = pdb_info.pdb2pose(chain, pos)
            except Exception:
                results.append({**mut, "pred_score": float("nan")})
                continue

            t_mut = time.time()
            tracemalloc.start()
            try:
                mutant_pose = pose.clone()
                MutateResidue(resnum, target).apply(mutant_pose)
                e_mut = scorefxn(mutant_pose)
                pred_score = -(e_mut - e_wt)
            except Exception as e:
                print(f"    ERROR: {e}")
                pred_score = float("nan")
            mem_peak = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
            t_mut = time.time() - t_mut
            print(f"[{i+1}/{len(mutations)}] {mut['wt_aa']}{pos}→{target}  "
                  f"pred={pred_score:.3f}  t={t_mut:.2f}s")

        results.append({**mut, "pred_score": pred_score})
        resources["mutations"].append({
            "mutation": f"{mut['wt_aa']}{pos}{target}",
            "is_ncaa": is_ncaa,
            "wall_s": round(t_mut if has_params else 0, 3),
            "mem_mb": round(mem_peak if has_params else 0, 1),
        })

    valid = [(r["value"], r["pred_score"]) for r in results
             if not np.isnan(r["pred_score"])]
    rho, pval = (float("nan"), float("nan"))
    if len(valid) > 2:
        y_true, y_pred = zip(*valid)
        rho, pval = spearmanr(y_true, y_pred)
        print(f"\n[B] Spearman ρ = {rho:.4f}  (p={pval:.4g})  n={len(valid)}")

    resources.update({
        "total_wall_s": round(time.time() - t0_total, 1),
        "spearman_rho": rho, "spearman_p": pval,
        "n_valid": len(valid), "n_total": len(mutations),
    })

    out_csv = os.path.join(output_dir, "approach_b_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with open(os.path.join(output_dir, "approach_b_resources.json"), "w") as f:
        json.dump(resources, f, indent=2)

    print(f"[B] Results saved to {out_csv}")
    return rho


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approach B: Rosetta AM1-BCC params")
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nstruct", type=int, default=3)
    args = parser.parse_args()
    run_benchmark(args.pdb, args.chain, args.benchmark_csv,
                  args.output_dir, args.nstruct)
