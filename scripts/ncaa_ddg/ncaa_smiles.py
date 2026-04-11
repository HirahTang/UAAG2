"""
NCAA → SMILES lookup table.

Three-letter codes match the benchmark CSV 'target' column (case-insensitive).
SMILES derived from atom-count analysis of UAAG2's aa_graph.json + peptide
chemistry literature (Checco et al. 2015, Reznik et al. 2020 on PUMA/CP2 benchmarks).
"""

# Mapping from benchmark code (uppercase) → canonical SMILES as free amino acid
NCAA_SMILES = {
    # --- Aliphatic chain extensions ---
    "ABU": "CC[C@@H](N)C(=O)O",            # L-α-aminobutyric acid (2-aminobutyric)
    "NVA": "CCC[C@@H](N)C(=O)O",           # L-norvaline
    "NLE": "CCCC[C@@H](N)C(=O)O",          # L-norleucine
    "AHP": "N[C@@H](CCCCCO)C(=O)O",        # α-amino-ε-hydroxyhexanoic acid (6-OH-Nle)
    "AOC": "CCCCCC[C@@H](N)C(=O)O",        # L-α-aminooctanoic acid

    # --- α-Branched / quaternary ---
    "AIB": "CC(C)(N)C(=O)O",               # α-aminoisobutyric acid (achiral)
    "TME": "C[C@@H](O)[C@](C)(N)C(=O)O",  # α-methylthreonine

    # --- Cyclic sidechains ---
    "CPA": "N[C@@H](CC1CCCC1)C(=O)O",     # β-cyclopentylalanine
    "CHA": "N[C@@H](CC1CCCCC1)C(=O)O",    # β-cyclohexylalanine
    "TBU": "N[C@@H](CC(C)(C)C)C(=O)O",    # β-tert-butylalanine

    # --- Aromatic ---
    "2NP": "N[C@@H](Cc1ccc2ccccc2c1)C(=O)O",    # β-(2-naphthyl)-L-alanine
    "2TH": "N[C@@H](Cc1cccs1)C(=O)O",            # β-(2-thienyl)-L-alanine
    "3TH": "N[C@@H](Cc1ccsc1)C(=O)O",            # β-(3-thienyl)-L-alanine
    "BZT": "N[C@@H](Cc1cc2ccccc2s1)C(=O)O",      # β-(benzothiophen-2-yl)-L-alanine

    # --- D-amino acids ---
    "DAL": "C[C@H](N)C(=O)O",              # D-alanine

    # --- O-modified ---
    "HSM": "N[C@@H](CCOC)C(=O)O",          # O-methylhomoserine
    "YME": "N[C@@H](Cc1ccc(OC)cc1)C(=O)O", # O-methyltyrosine

    # --- N-methylated backbone ---
    "MEG": "CNCC(=O)O",                    # N-methylglycine (sarcosine)
    "MEA": "CN[C@@H](C)C(=O)O",            # N-methylalanine
    "MEB": "CN[C@@H](CC)C(=O)O",           # N-methyl-α-aminobutyric acid
    "MEF": "CN[C@@H](Cc1ccccc1)C(=O)O",   # N-methylphenylalanine
}

# Standard canonical AAs (Rosetta/OpenMM handle these natively)
CANONICAL_AAS = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
    "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
    "THR","TRP","TYR","VAL",
}

def get_smiles(aa_code: str) -> str | None:
    """Return SMILES for given 3-letter AA code, or None if canonical/unknown."""
    code = aa_code.upper()
    if code in CANONICAL_AAS:
        return None  # handled natively
    return NCAA_SMILES.get(code)


def validate_smiles():
    """Check all SMILES parse with RDKit."""
    from rdkit import Chem
    print("Validating NCAA SMILES:")
    for code, smi in NCAA_SMILES.items():
        mol = Chem.MolFromSmiles(smi)
        status = "OK" if mol else "INVALID"
        natoms = mol.GetNumHeavyAtoms() if mol else "?"
        print(f"  {code:6s}: {status}  heavy_atoms={natoms}  {smi}")


if __name__ == "__main__":
    validate_smiles()
