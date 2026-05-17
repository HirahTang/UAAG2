"""
Approach C: OpenMM + GAFF2 ΔΔG for canonical and NCAA mutations.

Protocol:
  1. Prepare WT structure with PDBFixer (fill missing, protonate, solvate → vacuum GB)
  2. Minimize WT energy (implicit solvent, OBC2 GB model)
  3. For each mutation (WT_AA, pos, target_AA):
     - Canonical AA: rename residue in PDB, rebuild with PDBFixer, minimize
     - NCAA: align NCAA conformer (from SMILES) onto WT backbone N-CA-C, insert as
       HETATM in place of the WT residue, minimize with backbone restraints
     - ΔΔG_approx = -(E_mut - E_wt)  (positive = stabilising)
  4. Save per-mutation results + timing

NCAA sidechain swap protocol:
  - Extract N, CA, C backbone coordinates from WT residue
  - Generate NCAA conformer (ETKDGv3 + MMFF94) from SMILES
  - Identify backbone atoms (N-CA-C) in NCAA via SMARTS
  - Kabsch-align NCAA conformer onto WT backbone
  - Remove WT residue from PDB; insert aligned NCAA heavy atoms as HETATM
  - Load combined structure; apply GAFF2 to NCAA via SystemGenerator
  - Minimize with harmonic restraints on all protein backbone heavy atoms

Resource tracking: wall time per mutation, memory via tracemalloc.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Optional

import numpy as np

# ── OpenMM ────────────────────────────────────────────────────────────────────
import openmm
from openmm import LangevinMiddleIntegrator, Platform, unit
from openmm.app import (
    ForceField, Modeller, PDBFile, Simulation,
    HBonds, NoCutoff,
)
from pdbfixer import PDBFixer
from openmmforcefields.generators import SystemGenerator, GAFFTemplateGenerator

# ── Stub openff.units so GAFFTemplateGenerator works without openff.toolkit ───
# openff.units is not installable via pip on LUMI; inject a minimal shim before
# any import inside template_generators.py tries to load it.
import sys as _sys, types as _types

def _install_openff_units_stub():
    """Install a minimal openff.units shim into sys.modules."""
    if "openff.units" in _sys.modules:
        return  # already present (real or stub)

    class _Quantity:
        """Minimal stand-in for openff.units.Quantity (scalar or array)."""
        def __init__(self, m):
            import numpy as np
            self._m = np.atleast_1d(np.asarray(m, dtype=float)) if hasattr(m, '__len__') else float(m)
        @property
        def m(self):
            return self._m
        def m_as(self, unit):
            return self._m   # units are already in e (elementary charge)
        def __iter__(self):
            if hasattr(self._m, '__iter__'):
                for v in self._m:
                    yield _Quantity(v)
            else:
                yield self
        def __len__(self):
            if hasattr(self._m, '__len__'):
                return len(self._m)
            return 1

    class _UnitNS:
        """Attribute access returns a sentinel unit object."""
        class _AnyUnit:
            pass
        def __getattr__(self, name):
            return self._AnyUnit()

    _mod = _types.ModuleType("openff.units")
    _mod.unit = _UnitNS()
    _mod.Quantity = _Quantity
    if "openff" not in _sys.modules:
        _sys.modules["openff"] = _types.ModuleType("openff")
    _sys.modules["openff.units"] = _mod

_install_openff_units_stub()

# ── RDKit ─────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms


# ── Minimal RDKit→GAFFTemplateGenerator adapter ──────────────────────────────
# GAFFTemplateGenerator needs: molecule.to_smiles(), molecule.to_file(path, file_format),
# molecule.atoms (iterable with .gaff_type, .atomic_number, .name),
# and molecule.bonds (iterable with .atom1_index, .atom2_index, .atom1.name, .atom2.name).
# This wrapper satisfies that interface without requiring openff.toolkit.

from collections import defaultdict as _defaultdict

def _assign_mol2_names(rdmol: "Chem.Mol") -> list[str]:
    """Return antechamber-style atom names (N1, C1, C2, O1, H1, …) for every atom."""
    counter: dict[str, int] = _defaultdict(int)
    names = []
    for atom in rdmol.GetAtoms():
        sym = atom.GetSymbol()
        counter[sym] += 1
        names.append(f"{sym}{counter[sym]}")
    return names


class _RDKitAtomProxy:
    """Proxy for a single RDKit atom; satisfies GAFFTemplateGenerator duck-typed interface.

    Required by GAFFTemplateGenerator:
      .gaff_type     — GAFF atom type string (set by _read_gaff_atom_types_from_mol2)
      .atomic_number — integer atomic number (for template graph matching)
      .name          — atom name string (set by _generate_unique_atom_names, used in XML)
      .symbol        — element symbol string (used by _generate_unique_atom_names)
      .typename      — GAFF type name for XML AtomTypes section (set by generator)
      .mass          — atomic mass (used in XML AtomTypes section)
      .partial_charge — openff.units.Quantity-like with .m_as() (charge in XML)
    """
    class _ElementProxy:
        """Minimal element proxy satisfying atom.element.atomic_number access."""
        __slots__ = ("atomic_number",)
        def __init__(self, n):
            self.atomic_number = n

    __slots__ = ("gaff_type", "atomic_number", "element", "name", "symbol", "typename", "mass", "partial_charge")
    def __init__(self, rdatom, name: str, charge: float = 0.0):
        self.gaff_type = ""
        self.atomic_number = rdatom.GetAtomicNum()
        self.element = _RDKitAtomProxy._ElementProxy(rdatom.GetAtomicNum())
        self.symbol = rdatom.GetSymbol()
        self.name = name
        self.typename = ""   # set later by GAFFTemplateGenerator.generator()
        self.mass = rdatom.GetMass()
        # Wrap charge as a stub Quantity so .m_as(unit.elementary_charge) works
        from sys import modules as _mods
        _Q = _mods.get("openff.units", None)
        if _Q is not None and hasattr(_Q, "Quantity"):
            self.partial_charge = _Q.Quantity(charge)
        else:
            self.partial_charge = charge


class _BondProxy:
    """Proxy for a bond; exposes atom1_index/atom2_index and atom1/atom2 with .name."""
    __slots__ = ("atom1_index", "atom2_index", "atom1", "atom2")
    def __init__(self, idx1: int, idx2: int, atoms: list):
        self.atom1_index = idx1
        self.atom2_index = idx2
        self.atom1 = atoms[idx1]
        self.atom2 = atoms[idx2]


class RDKitMolForGAFF:
    """Thin wrapper around an RDKit Mol that satisfies GAFFTemplateGenerator's interface.

    Attributes used by GAFFTemplateGenerator:
      .atoms           → list of _RDKitAtomProxy (gaff_type, atomic_number, name, partial_charge)
      .bonds           → list of _BondProxy (atom1_index, atom2_index, atom1.name, atom2.name)
      .partial_charges → non-None signals user-provided charges (skips AM1-BCC)
      .to_smiles()     → canonical SMILES string
      .to_file(path)   → writes SDF for antechamber
      .generate_conformers() → no-op (conformer already embedded)
    """

    def __init__(self, rdmol: "Chem.Mol", smiles: str):
        self._mol = rdmol
        self._smiles = smiles

        # Gasteiger charges (fast approximate charges; bypasses openff AM1-BCC)
        from rdkit.Chem import rdPartialCharges
        rdPartialCharges.ComputeGasteigerCharges(rdmol)

        names = _assign_mol2_names(rdmol)
        charges = [
            float(a.GetDoubleProp("_GasteigerCharge") or 0.0)
            for a in rdmol.GetAtoms()
        ]
        self.atoms = [
            _RDKitAtomProxy(a, names[a.GetIdx()], charges[a.GetIdx()])
            for a in rdmol.GetAtoms()
        ]
        self.bonds = [
            _BondProxy(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), self.atoms)
            for b in rdmol.GetBonds()
        ]
        # partial_charges must be a single array Quantity (not a list) for .m_as() to work.
        # GAFFTemplateGenerator checks molecule.partial_charges.m_as(unit.elementary_charge).
        import numpy as _np
        from sys import modules as _mods
        _Q = _mods.get("openff.units", None)
        if _Q is not None and hasattr(_Q, "Quantity"):
            self.partial_charges = _Q.Quantity([float(c) for c in charges])
            self.total_charge = _Q.Quantity(0.0)  # neutral amino acid (NH2+COOH = 0)
        else:
            self.partial_charges = _np.array(charges, dtype=float)
            self.total_charge = 0.0

    def to_smiles(self) -> str:
        return self._smiles

    def to_file(self, path: str, file_format: str = "sdf") -> None:
        if file_format.lower() == "sdf":
            writer = Chem.SDWriter(path)
            writer.write(self._mol)
            writer.close()
        else:
            raise ValueError(f"Unsupported file_format '{file_format}' for RDKitMolForGAFF")

    def generate_conformers(self, n_conformers: int = 1) -> None:
        """No-op: molecule already has an embedded conformer."""
        pass

    def assign_partial_charges(self, partial_charge_method: str = "am1bcc",
                               normalize_partial_charges: bool = True) -> None:
        """No-op: user-provided Gasteiger charges are used instead."""
        pass

    @classmethod
    def from_smiles(cls, smiles: str) -> "RDKitMolForGAFF":
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == -1:
            AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        return cls(mol, smiles)

sys.path.insert(0, str(Path(__file__).parent))
from ncaa_smiles import NCAA_SMILES, CANONICAL_AAS, get_smiles

# ─────────────────────────────────────────────────────────────────────────────
ROSETTA_ONE_TO_THREE = {
    "A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","E":"GLU","Q":"GLN",
    "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
    "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL",
}

# ─── backbone SMARTS patterns (tried in order) ───────────────────────────────
_BACKBONE_SMARTS = [
    # Standard alpha-AA: primary or secondary amine - alpha-C - carboxylate
    "[N;H1,H2,H3;!$(NC=O)]-[C;H1,H2]-C(=O)[O;H1,H0-,H0]",
    # N-methylated (Sar / MeGly / Pro-like): tertiary N not amide
    "[N;H0;!$(NC=O)]-[C;H1,H2]-C(=O)[O;H1,H0-]",
    # Permissive fallback
    "[N]-[C]-C(=O)[O,OH]",
]

# Canonical three-letter residue names (for backbone restraint selection)
_CANONICAL_RES = {
    "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
}


def _find_backbone_atoms(mol: Chem.Mol) -> tuple[int, int, int]:
    """Return (N_idx, CA_idx, C_idx) for the alpha-amino acid backbone.

    Tries multiple SMARTS patterns to handle canonical AAs, N-methylated
    NCAAs (Sar, MeGly…) and other alpha-amino acid scaffolds.
    Raises ValueError if no pattern matches.
    """
    for smarts in _BACKBONE_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            n_idx, ca_idx, c_idx = matches[0][:3]
            return n_idx, ca_idx, c_idx
    raise ValueError("Cannot identify alpha-amino acid backbone (N-CA-C) in SMILES")


def _align_conformer_to_backbone(
    mol: Chem.Mol,
    n_idx: int, ca_idx: int, c_idx: int,
    n_wt: np.ndarray, ca_wt: np.ndarray, c_wt: np.ndarray,
) -> None:
    """In-place Kabsch alignment of mol's conformer so that its N, CA, C atoms
    superimpose onto the WT residue backbone positions (coordinates in Å).
    """
    conf = mol.GetConformer(0)
    src = np.array([
        list(conf.GetAtomPosition(n_idx)),
        list(conf.GetAtomPosition(ca_idx)),
        list(conf.GetAtomPosition(c_idx)),
    ], dtype=float)
    dst = np.array([n_wt, ca_wt, c_wt], dtype=float)

    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    H = (src - src_center).T @ (dst - dst_center)
    U, _, Vt = np.linalg.svd(H)
    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dst_center - R @ src_center

    all_pos = conf.GetPositions()
    new_pos = (R @ all_pos.T).T + t
    for i, p in enumerate(new_pos):
        conf.SetAtomPosition(i, p.tolist())


def build_ncaa_sidechain_mutant(
    wt_modeller: Modeller,
    chain_id: str,
    residue_number: int,
    ncaa_smiles: str,
    ncaa_name: str,
) -> tuple[Optional[Modeller], Optional[any]]:
    """Place NCAA at the target position and return a Modeller + OpenFF molecule.

    Steps:
      1. Extract N, CA, C backbone coordinates from the WT residue (in Å).
      2. Generate NCAA conformer from SMILES (ETKDGv3 + MMFF94).
      3. Find backbone atoms in NCAA SMARTS and Kabsch-align to WT backbone.
      4. Write protein (WT residue removed) + aligned NCAA heavy atoms (HETATM)
         to a temp PDB, load with PDBFixer for H-addition on the protein part.
      5. Return (Modeller, openff_molecule) ready for GAFF2 parameterization.

    The NCAA is inserted as a standalone HETATM (not covalently linked to the
    chain in the OpenMM topology). Backbone restraints during minimization keep
    the surrounding protein fixed while the NCAA sidechain relaxes in the pocket.
    """
    # ── 1. Extract WT backbone atom positions (in Å) ──────────────────────────
    wt_positions = wt_modeller.positions
    backbone_pos: dict[str, np.ndarray] = {}

    for res in wt_modeller.topology.residues():
        if res.chain.id == chain_id and res.id == str(residue_number):
            for atom in res.atoms():
                if atom.name in ("N", "CA", "C", "O"):
                    pos = wt_positions[atom.index].value_in_unit(unit.angstrom)
                    backbone_pos[atom.name] = np.array(pos)
            break

    if not all(k in backbone_pos for k in ("N", "CA", "C")):
        print(f"    [!] Could not find backbone N/CA/C for residue {residue_number} "
              f"in chain {chain_id}")
        return None, None, None

    n_wt, ca_wt, c_wt = backbone_pos["N"], backbone_pos["CA"], backbone_pos["C"]

    # ── 2. Generate NCAA conformer ─────────────────────────────────────────────
    mol = Chem.MolFromSmiles(ncaa_smiles)
    if mol is None:
        print(f"    [!] Invalid SMILES for {ncaa_name}: {ncaa_smiles}")
        return None, None, None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        AllChem.EmbedMolecule(mol)   # fallback
    AllChem.MMFFOptimizeMolecule(mol)

    # ── 3. Find backbone atoms and align ──────────────────────────────────────
    try:
        n_idx, ca_idx, c_idx = _find_backbone_atoms(mol)
    except ValueError as exc:
        print(f"    [!] {exc}")
        return None, None, None

    _align_conformer_to_backbone(mol, n_idx, ca_idx, c_idx, n_wt, ca_wt, c_wt)

    # ── 4. Write protein (minus WT residue) + NCAA HETATM to temp PDB ─────────
    tmp_pdb: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as fh:
            tmp_pdb = fh.name
            PDBFile.writeFile(wt_modeller.topology, wt_positions, fh)

        with open(tmp_pdb) as fh:
            pdb_lines = fh.readlines()

        # Remove WT residue lines
        kept_lines = []
        for line in pdb_lines:
            if line.startswith(("ATOM", "HETATM")):
                if line[21] == chain_id and line[22:26].strip() == str(residue_number):
                    continue
            kept_lines.append(line)

        # Build HETATM + CONECT lines for the aligned NCAA (all atoms including H).
        # Use the same antechamber-style names (N1, C1, C2, H1, …) as RDKitMolForGAFF
        # so the GAFF2 residue template atom names match the topology atom names.
        # CONECT records are required so OpenMM topology has bonds for the NCAA residue;
        # without bonds, template graph isomorphism always fails.
        conf = mol.GetConformer(0)
        atom_names = _assign_mol2_names(mol)
        # Offset serials past the MAXIMUM existing serial, not just the count.
        # kept_lines still has the original PDB serials (we removed residue lines from text
        # but did NOT renumber), so using count would cause serial collisions.
        max_existing_serial = 0
        for l in kept_lines:
            if l.startswith(("ATOM", "HETATM")):
                try:
                    max_existing_serial = max(max_existing_serial, int(l[6:11]))
                except ValueError:
                    pass
        hetatm_lines: list[str] = []
        ncaa_serials: dict[int, int] = {}  # rdkit atom idx → pdb serial
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            elem = atom.GetSymbol()
            aname = atom_names[atom.GetIdx()]
            serial = max_existing_serial + atom.GetIdx() + 1
            ncaa_serials[atom.GetIdx()] = serial
            hetatm_lines.append(
                f"HETATM{serial:5d} {aname:<4s} {ncaa_name[:3]:<3s} "
                f"{chain_id}{residue_number:4d}    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                f"  1.00  0.00          {elem:>2s}\n"
            )
        # CONECT records for bonds within the NCAA (bidirectional)
        for bond in mol.GetBonds():
            s1 = ncaa_serials[bond.GetBeginAtomIdx()]
            s2 = ncaa_serials[bond.GetEndAtomIdx()]
            hetatm_lines.append(f"CONECT{s1:5d}{s2:5d}\n")
            hetatm_lines.append(f"CONECT{s2:5d}{s1:5d}\n")

        # Insert before END record
        insert_at = next(
            (i for i, l in enumerate(kept_lines) if l.strip() == "END"),
            len(kept_lines),
        )
        kept_lines = kept_lines[:insert_at] + hetatm_lines + kept_lines[insert_at:]

        with open(tmp_pdb, "w") as fh:
            fh.writelines(kept_lines)

        # Load combined PDB. Protein H are already present from the WT preparation;
        # NCAA H are written into the HETATM block from the RDKit conformer.
        # Skip addMissingHydrogens to avoid PDBFixer adding duplicate H to the NCAA.
        fixer = PDBFixer(filename=tmp_pdb)
        mut_modeller = Modeller(fixer.topology, fixer.positions)
        mut_modeller.topology.setPeriodicBoxVectors(None)
        os.unlink(tmp_pdb)
        tmp_pdb = None

    except Exception as exc:
        print(f"    [!] Failed to build NCAA mutant structure: {exc}")
        if tmp_pdb:
            try:
                os.unlink(tmp_pdb)
            except Exception:
                pass
        return None, None, None

    # ── 5. Build RDKit-wrapped molecule for GAFF2 using the SAME aligned mol ────
    # atom ordering must match the HETATM atom names written above.
    canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(ncaa_smiles))
    gaff_mol = RDKitMolForGAFF(mol, canonical_smiles)

    # Return backbone rdkit atom indices (n_idx, ca_idx, c_idx) so the caller can
    # identify which NCAA topology atoms to restrain during minimization.
    return mut_modeller, gaff_mol, (n_idx, ca_idx, c_idx)


def build_system_and_minimize_restrained(
    modeller: Modeller,
    small_mols: list | None = None,
    k_restraint: float = 1000.0,
    ncaa_res_name: str | None = None,
    platform_name: str = "CUDA",
) -> float:
    """Like build_system_and_minimize but with harmonic restraints on protein
    backbone heavy atoms (N, CA, C, O) AND all NCAA heavy atoms.

    Restraining NCAA heavy atoms prevents the unconnected NCAA molecule from
    drifting to unphysical positions during minimization (since it has no peptide
    bond tethering it to the protein chain).  Sidechain H-atoms are free to relax.
    Returns energy in kJ/mol.
    """
    ff = ForceField("amber/ff14SB.xml", "implicit/obc2.xml")

    if small_mols:
        # Register each NCAA with GAFFTemplateGenerator using our RDKit wrapper
        # (avoids openff.toolkit dependency)
        gaff = GAFFTemplateGenerator(forcefield="gaff-2.11")
        for mol in small_mols:
            gaff.add_molecules(mol)
        ff.registerTemplateGenerator(gaff.generator)

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
        rigidWater=True,
        removeCMMotion=False,
        hydrogenMass=1.5 * unit.amu,
    )

    # ── Close-contact exception fix ─────────────────────────────────────────────
    # Because the NCAA is inserted as a standalone HETATM (not peptide-bonded to the
    # protein), its backbone N, CA, C=O atoms sit at bonding distances (~1.3–1.5 Å)
    # from adjacent protein backbone atoms.  Without a bonded exclusion between them,
    # the 12-6 LJ potential at these sub-Ångström distances diverges to +10^24 kJ/mol.
    # Fix: find all NCAA–protein atom pairs within 2.2 Å (0.22 nm) of each other and
    # add explicit NonbondedForce exceptions that zero out their LJ and Coulomb
    # interactions.  This mimics the 1-2 / 1-3 bond exclusions that would exist if the
    # NCAA were properly bonded in the topology.
    _ncaa_indices = [a.index for a in modeller.topology.atoms()
                     if a.residue.name not in _CANONICAL_RES]
    _prot_indices  = [a.index for a in modeller.topology.atoms()
                     if a.residue.name in _CANONICAL_RES]

    if _ncaa_indices and _prot_indices:
        _pos_arr = np.array([[p.x, p.y, p.z] for p in modeller.positions])
        _nb_force = None
        for _fi in range(system.getNumForces()):
            _f = system.getForce(_fi)
            if type(_f).__name__ == "NonbondedForce":
                _nb_force = _f
                break
        if _nb_force is not None:
            _CLASH_NM = 0.22   # 2.2 Å — catches peptide-bond-length contacts
            for _ni in _ncaa_indices:
                for _pi in _prot_indices:
                    if np.linalg.norm(_pos_arr[_ni] - _pos_arr[_pi]) < _CLASH_NM:
                        try:
                            _nb_force.addException(_ni, _pi, 0.0, 0.1, 0.0)
                        except Exception:
                            pass  # exception already exists; leave as-is

    # Harmonic restraints: k * [(x-x0)^2 + (y-y0)^2 + (z-z0)^2]  (units: kJ/mol/nm²)
    restraint = openmm.CustomExternalForce(
        f"{k_restraint} * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    positions = modeller.positions
    backbone_names = {"N", "CA", "C", "O"}
    hydrogen_elements = {"H", "D"}
    for atom in modeller.topology.atoms():
        if atom.residue.name in _CANONICAL_RES and atom.name in backbone_names:
            # Protein backbone heavy atoms: strong restraint
            pos = positions[atom.index]
            restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
        elif atom.residue.name not in _CANONICAL_RES:
            # NCAA heavy atoms: restrain to prevent unphysical drift.
            # The NCAA has no peptide bond to the protein, so without restraints
            # it is free to wander to configurations with astronomical energies.
            if atom.element is not None and atom.element.symbol not in hydrogen_elements:
                pos = positions[atom.index]
                restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
    system.addForce(restraint)

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.004 * unit.picoseconds
    )
    try:
        platform = Platform.getPlatformByName(platform_name)
    except Exception:
        platform = Platform.getPlatformByName("CPU")

    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=1000, tolerance=10 * unit.kilojoules_per_mole / unit.nanometer)
    state = sim.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


def prepare_structure(pdb_path: str, chain: str = "A") -> PDBFixer:
    """Load PDB, keep only one chain, fill missing atoms, add hydrogens."""
    fixer = PDBFixer(filename=pdb_path)
    # Keep only the target chain
    # removeChains expects integer indices, not chain ID strings
    all_chains = list(fixer.topology.chains())
    chains_to_remove = [i for i, c in enumerate(all_chains) if c.id != chain]
    fixer.removeChains(chains_to_remove)
    # If multiple copies of the chain exist, keep only the first
    all_remaining = list(fixer.topology.chains())
    target_chains = [i for i, c in enumerate(all_remaining) if c.id == chain]
    if len(target_chains) > 1:
        fixer.removeChains(target_chains[1:])
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    return fixer


def build_system_and_minimize(
    modeller: Modeller,
    small_mols: list | None = None,
    platform_name: str = "CUDA",
) -> tuple[float, openmm.System]:
    """Build OpenMM system (ff14SB + GAFF2) and minimize. Returns (energy_kJ, system)."""
    forcefield_kwargs = {
        "constraints": HBonds,
        "rigidWater": True,
        "removeCMMotion": False,
        "hydrogenMass": 1.5 * unit.amu,
    }
    # nonbondedMethod must go in nonperiodic_forcefield_kwargs for SystemGenerator;
    # NoCutoff is required for implicit solvent (OBC2) — omitting causes SystemGenerator
    # to default to PME which is incompatible with implicit solvent.
    nonperiodic_kwargs = {"nonbondedMethod": NoCutoff}
    generator = SystemGenerator(
        forcefields=["amber/ff14SB.xml", "implicit/obc2.xml"],
        small_molecule_forcefield="gaff-2.11",
        molecules=small_mols or [],
        forcefield_kwargs=forcefield_kwargs,
        nonperiodic_forcefield_kwargs=nonperiodic_kwargs,
    )

    system = generator.create_system(modeller.topology, molecules=small_mols or [])
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                          0.004 * unit.picoseconds)

    try:
        platform = Platform.getPlatformByName(platform_name)
    except Exception:
        platform = Platform.getPlatformByName("CPU")

    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    # Minimize
    sim.minimizeEnergy(maxIterations=1000, tolerance=10 * unit.kilojoules_per_mole / unit.nanometer)
    state = sim.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    return energy, system


def mutate_residue_openmm(
    wt_modeller: Modeller,
    chain_id: str,
    residue_number: int,
    target_aa: str,
    smiles: Optional[str],
) -> Optional[Modeller]:
    """
    Build a mutant by replacing residue sidechain. 
    For canonical AAs: use OpenMM Modeller.applyMutations.
    For NCAAs: truncate to Gly-like (remove sidechain), attach NCAA from SMILES.
    """
    if target_aa in CANONICAL_AAS:
        # applyMutations is not available in OpenMM 8.5.
        # Workaround: write WT to a temp PDB, rename the target residue, rebuild with PDBFixer.
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as fh:
                tmp_pdb = fh.name
                PDBFile.writeFile(wt_modeller.topology, wt_modeller.positions, fh)

            # Replace the residue name in the PDB text
            with open(tmp_pdb) as fh:
                lines = fh.readlines()
            mutant_lines = []
            for line in lines:
                if line.startswith(("ATOM", "HETATM")):
                    rec_chain = line[21]           # chain ID column
                    rec_resnum = line[22:26].strip()
                    if rec_chain == chain_id and rec_resnum == str(residue_number):
                        line = line[:17] + f"{target_aa:<3}" + line[20:]
                mutant_lines.append(line)
            with open(tmp_pdb, "w") as fh:
                fh.writelines(mutant_lines)

            fixer = PDBFixer(filename=tmp_pdb)
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            mut = Modeller(fixer.topology, fixer.positions)
            mut.topology.setPeriodicBoxVectors(None)
            os.unlink(tmp_pdb)
            return mut, None, None
        except Exception as e:
            print(f"    [!] Canonical mutation failed: {e}")
            try:
                os.unlink(tmp_pdb)
            except Exception:
                pass
            return None, None, None
    else:
        if smiles is None:
            print(f"    [!] No SMILES for {target_aa}, skipping")
            return None, None, None
        # NCAA: align conformer to WT backbone and insert into pocket
        return build_ncaa_sidechain_mutant(
            wt_modeller, chain_id, residue_number, smiles, target_aa
        )


def run_benchmark(
    pdb_path: str,
    chain: str,
    benchmark_csv: str,
    output_dir: str,
    platform: str = "CUDA",
    max_mutations: int = 0,
    resume: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    t0_total = time.time()

    # Prepare WT
    print(f"[C] Preparing WT structure: {pdb_path} chain {chain}")
    t_prep = time.time()
    fixer = prepare_structure(pdb_path, chain)
    wt_modeller = Modeller(fixer.topology, fixer.positions)
    # Crystal structures have periodic box info → SystemGenerator would use PME
    # which conflicts with implicit solvent. Clear box vectors to force NoCutoff.
    wt_modeller.topology.setPeriodicBoxVectors(None)
    t_prep = time.time() - t_prep
    print(f"    WT prep: {t_prep:.1f}s")

    # Minimize WT
    print("[C] Minimizing WT...")
    t_wt = time.time()
    tracemalloc.start()
    e_wt, _ = build_system_and_minimize(wt_modeller, platform_name=platform)
    mem_wt = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    t_wt = time.time() - t_wt
    print(f"    WT energy: {e_wt:.2f} kJ/mol  t={t_wt:.1f}s  mem_peak={mem_wt:.0f} MB")

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

    if max_mutations > 0:
        mutations = mutations[:max_mutations]

    # ── Resume from checkpoint ────────────────────────────────────────────────
    # If --resume and a partial results CSV exists, skip mutations already done.
    results: list[dict] = []
    done_keys: set[tuple] = set()
    ckpt_csv = os.path.join(output_dir, "approach_c_results.csv")
    if resume and os.path.exists(ckpt_csv):
        with open(ckpt_csv) as _f:
            for _row in csv.DictReader(_f):
                results.append({
                    "wt_aa": _row["wt_aa"],
                    "pos": int(float(_row["pos"])),
                    "target": _row["target"],
                    "value": float(_row["value"]),
                    "pred_score": float(_row["pred_score"]),
                })
                done_keys.add((_row["wt_aa"], int(float(_row["pos"])), _row["target"]))
        print(f"[C] Resuming: {len(results)} mutations already done, "
              f"{len(mutations) - len(results)} remaining.")
        mutations = [m for m in mutations
                     if (m["wt_aa"], m["pos"], m["target"]) not in done_keys]

    resources = {"wt_energy_kJmol": e_wt, "wt_prep_s": t_prep, "wt_min_s": t_wt,
                 "wt_mem_mb": mem_wt, "mutations": []}

    total = len(done_keys) + len(mutations)
    for i, mut in enumerate(mutations):
        target = mut["target"]
        pos = mut["pos"]
        smiles = get_smiles(target)
        is_ncaa = target not in CANONICAL_AAS

        print(f"[{len(done_keys)+i+1}/{total}] {mut['wt_aa']}{pos}→{target}"
              f"  {'(NCAA)' if is_ncaa else '(canonical)'}")

        t_mut = time.time()
        tracemalloc.start()

        mut_modeller, off_mol, _bb_idx = mutate_residue_openmm(
            wt_modeller, chain, pos, target, smiles
        )
        if mut_modeller is None:
            pred_score = float("nan")
        else:
            try:
                if is_ncaa:
                    # NCAA: restrained minimization.  Protein backbone heavy atoms AND
                    # NCAA heavy atoms are harmonically restrained at their initial positions
                    # to prevent the unconnected NCAA molecule from drifting to unphysical
                    # energy minima.  Only H-atoms are free to relax.
                    e_mut = build_system_and_minimize_restrained(
                        mut_modeller,
                        small_mols=[off_mol] if off_mol is not None else [],
                        ncaa_res_name=target,
                        platform_name=platform,
                    )
                else:
                    # Canonical: standard unrestrained minimization
                    e_mut, _ = build_system_and_minimize(
                        mut_modeller, platform_name=platform
                    )
                # Sanity check: astronomically large energies indicate a force-field
                # parameterization failure (e.g. serial collision, duplicate bonds).
                if abs(e_mut) > 1e10:
                    print(f"    [!] Unrealistic energy {e_mut:.3e} kJ/mol — treating as NaN")
                    pred_score = float("nan")
                else:
                    pred_score = -(e_mut - e_wt)  # positive = stabilising
            except Exception as e:
                print(f"    [!] Minimization failed: {e}")
                pred_score = float("nan")

        mem_peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        t_mut = time.time() - t_mut

        results.append({**mut, "pred_score": pred_score})
        resources["mutations"].append({
            "mutation": f"{mut['wt_aa']}{pos}{target}",
            "is_ncaa": is_ncaa,
            "wall_s": round(t_mut, 3),
            "mem_mb": round(mem_peak, 1),
        })
        print(f"    pred={pred_score:.3f}  t={t_mut:.2f}s  mem={mem_peak:.0f}MB")

        # Checkpoint: flush results to disk after every mutation so partial
        # progress is recoverable if the job is cancelled by the scheduler.
        _ckpt_csv = os.path.join(output_dir, "approach_c_results.csv")
        with open(_ckpt_csv, "w", newline="") as _f:
            _w = csv.DictWriter(_f, fieldnames=list(results[0].keys()))
            _w.writeheader()
            _w.writerows(results)

    # Compute Spearman correlation
    from scipy.stats import spearmanr
    valid = [(r["value"], r["pred_score"]) for r in results
             if not np.isnan(r["pred_score"])]
    if len(valid) > 2:
        y_true, y_pred = zip(*valid)
        rho, pval = spearmanr(y_true, y_pred)
        print(f"\n[C] Spearman ρ = {rho:.4f}  (p={pval:.4g})  n={len(valid)}")
    else:
        rho, pval = float("nan"), float("nan")
        print("[C] Not enough valid predictions for Spearman.")

    resources["total_wall_s"] = round(time.time() - t0_total, 1)
    resources["spearman_rho"] = rho
    resources["spearman_p"] = pval
    resources["n_valid"] = len(valid)
    resources["n_total"] = total

    # Save outputs
    out_csv = os.path.join(output_dir, "approach_c_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    out_resources = os.path.join(output_dir, "approach_c_resources.json")
    with open(out_resources, "w") as f:
        json.dump(resources, f, indent=2)

    print(f"[C] Results saved to {out_csv}")
    print(f"[C] Resources saved to {out_resources}")
    return rho


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approach C: OpenMM+GAFF2 ΔΔG")
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--max-mutations", type=int, default=0,
                        help="0 = all mutations")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial results CSV")
    args = parser.parse_args()
    run_benchmark(args.pdb, args.chain, args.benchmark_csv,
                  args.output_dir, args.platform, args.max_mutations,
                  resume=args.resume)
