"""Join residue identity with PoseBusters geometry, per molecule.

Reports, per (stage, condition):
  total       - molecules scored
  topo        - heavy-atom graph isomorphic to the requested residue (what the
                report's end-to-end rate counts today)
  geom        - passes the real PoseBusters checks (no_radicals and the
                pass_all_criteria summary excluded)
  both        - topo AND geom  <- the metric that was missing
Env: STAGE_SUB, CODE, OUT
"""
import os, sys, glob, tarfile, tempfile, shutil, json
import inspect as _inspect
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "/home/qcx679/hantang/UAAG2_conditional/src")
import uaag2.utils as U
from posebusters import PoseBusters

B = "/datasets/biochem/unaagi/samples_v6"
SUB, CODE, OUT = os.environ["STAGE_SUB"], os.environ["CODE"], os.environ["OUT"]
EXCLUDE = {"no_radicals", "pass_all_criteria", "file_name", "molecule",
           "mol_pred_loaded", "bust_available", "pass_mol_loaded"}

tar_path = os.path.join(B, SUB, CODE, "samples_split0.tar.gz")
tmp = tempfile.mkdtemp(prefix="pbjoin_")
try:
    with tarfile.open(tar_path) as tf:
        names = [m for m in tf.getnames() if m.endswith("final/ligand.mol")]
        tf.extractall(tmp, members=[tf.getmember(n) for n in names])
    mols = sorted(glob.glob(os.path.join(tmp, "**", "final", "ligand.mol"), recursive=True))

    # --- identity, and a single combined SDF for PoseBusters ---
    ident, sdf_paths = [], []
    sdf_dir = os.path.join(tmp, "_sdf"); os.makedirs(sdf_dir, exist_ok=True)
    for i, f in enumerate(mols):
        m = Chem.MolFromMolFile(f, sanitize=False)
        name = None
        if m is not None:
            try:
                name = str(U.aa_check(m))
            except Exception:
                name = None
        ident.append(name)
        p = os.path.join(sdf_dir, "m%06d.sdf" % i)
        if m is not None:
            try:
                w = Chem.SDWriter(p); w.write(m); w.close()
                sdf_paths.append((i, p))
            except Exception:
                pass

    kw = {"config": "mol"}
    if "max_workers" in _inspect.signature(PoseBusters.__init__).parameters:
        kw["max_workers"] = 1
    buster = PoseBusters(**kw)

    geom = [False] * len(mols)
    for i, p in sdf_paths:
        try:
            r = buster.bust(mol_pred=p, mol_true=p, mol_cond=p)
            if r is None or len(r) == 0:
                continue
            row = r.iloc[0].to_dict()
            checks = [v for k, v in row.items() if isinstance(v, bool) and k not in EXCLUDE]
            geom[i] = bool(checks) and all(checks)
        except Exception:
            pass

    up = CODE.upper()
    topo = [(x is not None and x.upper() == up) for x in ident]
    rec = dict(stage=SUB, code=CODE, total=len(mols),
               topo=int(sum(topo)), geom=int(sum(geom)),
               both=int(sum(1 for a, b in zip(topo, geom) if a and b)),
               inv=int(sum(1 for x in ident if x == "INV")),
               unk=int(sum(1 for x in ident if x == "UNK")),
               scored_geom=len(sdf_paths))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(rec, fh)
    print(json.dumps(rec))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
