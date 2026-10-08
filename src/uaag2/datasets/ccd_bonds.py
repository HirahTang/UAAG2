"""CCD bond-order repair for PDB-parsed structures.

`Chem.MolFromPDBFile` carries bond-order templates for the 20 standard amino acids
only. Every non-standard residue -- NCAAs, PTMs, HETATM ligands -- comes back with
**all bonds SINGLE and no aromaticity**, which silently strips the chemistry from
exactly the residues an NCAA model is supposed to learn.

This module re-assigns those bond orders from the wwPDB Chemical Component Dictionary,
matched on (residue code, unordered atom-name pair). Standard residues are left
untouched, so behaviour for ordinary CATH data is bit-identical to before.

Bond orders use the CCD `value_order` field, which is Kekule (alternating SING/DOUB for
aromatic rings). That matches how RDKit returns standard residues under
``sanitize=False``, so both paths stay on the same convention.

Table is built by ``scripts/build_ccd_bond_table.py``; point ``UAAG2_CCD_BONDS`` at the
pickle. If the table is absent this is a no-op, so the loader never hard-fails.
"""
import os
import pickle
import re

from rdkit import Chem

STANDARD_AA = frozenset(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split())

_ORDER_TO_RDKIT = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
    "QUAD": Chem.BondType.QUADRUPLE,
}

_TABLE = None      # {"bonds": {...}, "charges": {...}}
_WARNED = False


def _default_path():
    return os.environ.get("UAAG2_CCD_BONDS", "")


def load_table(path=None):
    """Load (and memoise) the CCD bond table. Returns {} when unavailable."""
    global _TABLE, _WARNED
    if _TABLE is not None:
        return _TABLE
    p = path or _default_path()
    if not p or not os.path.exists(p):
        if not _WARNED:
            print("[ccd_bonds] no CCD table (UAAG2_CCD_BONDS=%r); non-standard residues "
                  "keep RDKit's all-SINGLE bonds" % p, flush=True)
            _WARNED = True
        _TABLE = {}
        return _TABLE
    with open(p, "rb") as fh:
        raw = pickle.load(fh)
    # tolerate the first-generation table, which was a bare {comp: bonds} dict
    _TABLE = raw if isinstance(raw, dict) and "bonds" in raw else {"bonds": raw, "charges": {}}
    print("[ccd_bonds] loaded %d components (%d with charges) from %s"
          % (len(_TABLE["bonds"]), len(_TABLE["charges"]), p), flush=True)
    return _TABLE


def _residue_key(atom):
    ri = atom.GetPDBResidueInfo()
    if ri is None:
        return None, None
    return (ri.GetChainId(), ri.GetResidueNumber(), ri.GetInsertionCode(),
            ri.GetResidueName().strip()), ri.GetName().strip()


def apply_ccd_bond_orders(mol, table=None, skip=()):
    """Re-assign intra-residue bond orders for non-standard residues, in place.

    Returns (mol, stats) where stats counts what happened -- callers can log it to
    verify the repair is actually firing rather than silently no-opping.
    """
    tbl = table if table is not None else load_table()
    stats = {"residues": 0, "no_template": 0, "bonds_seen": 0,
             "bonds_changed": 0, "bonds_unmatched": 0, "charges_set": 0}
    if not tbl:
        return mol, stats
    bonds_tbl, charge_tbl = tbl.get("bonds", {}), tbl.get("charges", {})
    if not bonds_tbl:
        return mol, stats

    # formal charges first: a repaired DOUBLE bond onto e.g. a quaternary N fails
    # RDKit's valence check unless the CCD charge is applied alongside it.
    for atom in mol.GetAtoms():
        k, name = _residue_key(atom)
        if k is None or k[3] in STANDARD_AA or k in skip:
            continue
        q = charge_tbl.get(k[3], {}).get(name)
        if q and atom.GetFormalCharge() != q:
            atom.SetFormalCharge(int(q))
            stats["charges_set"] += 1

    seen_res = set()
    for bond in mol.GetBonds():
        k1, n1 = _residue_key(bond.GetBeginAtom())
        k2, n2 = _residue_key(bond.GetEndAtom())
        if k1 is None or k1 != k2 or k1 in skip:   # inter-residue / no info / blocked
            continue
        resname = k1[3]
        if resname in STANDARD_AA:          # RDKit already got these right
            continue
        if k1 not in seen_res:
            seen_res.add(k1)
            stats["residues"] += 1
            if resname not in bonds_tbl:
                stats["no_template"] += 1
        comp = bonds_tbl.get(resname)
        if comp is None:
            continue
        stats["bonds_seen"] += 1
        entry = comp.get(tuple(sorted((n1, n2))))
        if entry is None:
            stats["bonds_unmatched"] += 1
            continue
        bt = _ORDER_TO_RDKIT.get(entry[0])
        if bt is not None and bond.GetBondType() != bt:
            bond.SetBondType(bt)
            stats["bonds_changed"] += 1
    return mol, stats


def _offending_atom(exc):
    """Pull the atom index out of an RDKit sanitization error, if it names one."""
    idx = getattr(exc, "GetAtomIdx", None)
    if callable(idx):
        try:
            return int(idx())
        except Exception:
            pass
    m = re.search(r"atom #\s*(\d+)", str(exc))
    return int(m.group(1)) if m else None


def repair_and_sanitize(mol, table=None, max_reverts=25):
    """Apply the CCD repair, then sanitize -- reverting only the residues that fail.

    A handful of residues carry charges the CCD does not record, so a corrected DOUBLE
    bond can push an atom over its permitted valence. Rather than drop the structure
    (which would silently shrink the dataset) or drop the whole repair (which would
    waste it on every other residue in the file), revert just the offending residue and
    retry. Returns (mol, stats) with ``reverted_residues`` and ``sanitized`` recorded.
    """
    pristine = Chem.Mol(mol)
    mol, stats = apply_ccd_bond_orders(mol, table)
    stats["reverted_residues"] = 0
    stats["sanitized"] = True

    blocked = set()
    for _ in range(max_reverts):
        try:
            Chem.SanitizeMol(mol)
            return mol, stats
        except Exception as exc:
            ai = _offending_atom(exc)
            if ai is None or ai >= mol.GetNumAtoms():
                break
            key, _ = _residue_key(mol.GetAtomWithIdx(ai))
            if key is None or key in blocked:
                break
            blocked.add(key)
            # rebuild from pristine, repairing every residue except the blocked ones
            mol = Chem.Mol(pristine)
            mol, stats2 = apply_ccd_bond_orders(mol, table, skip=blocked)
            stats2["reverted_residues"] = len(blocked)
            stats2["sanitized"] = True
            stats = stats2

    # last resort: unrepaired molecule, sanitized as before this change
    mol = Chem.Mol(pristine)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        stats["sanitized"] = False
    stats["reverted_residues"] = -1        # -1 == whole-file revert
    return mol, stats
