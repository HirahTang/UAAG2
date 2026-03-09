#!/usr/bin/env python3
"""Enumerate atom names and element types across PDB files in a folder.

Usage examples:
  python scripts/enumerate_pdb_atom_types.py --root /scratch/project_465002574/PDB/PDB_cleaned
  python scripts/enumerate_pdb_atom_types.py --root /scratch/project_465002574/PDB/PDB_cleaned --out-json report.json

The script scans recursively for `*.pdb` and `*.pdb.gz` files and reports
counts of atom names (column 13-16) and element symbols (column 77-78),
with a simple fallback when element column is empty.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple


def iter_pdb_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.pdb"))
    yield from sorted(root.rglob("*.pdb.gz"))


def parse_pdb_lines(path: Path) -> Iterable[Tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            # atom name: columns 13-16 (1-based) -> [12:16]
            atom_name = line[12:16].strip()
            # element: columns 77-78 (1-based) -> [76:78]
            element = line[76:78].strip() if len(line) >= 78 else ""
            if not element:
                # fallback: extract leading letters from atom name
                m = re.match(r"([A-Za-z]+)", atom_name)
                if m:
                    s = m.group(1)
                    # prefer 1-letter element (uppercase) when ambiguous
                    element = s[0].upper()
                else:
                    element = ""
            yield atom_name, element


def aggregate(root: Path) -> Tuple[Counter, Counter, int]:
    atom_counter: Counter = Counter()
    element_counter: Counter = Counter()
    files_scanned = 0
    for p in iter_pdb_files(root):
        files_scanned += 1
        for atom_name, element in parse_pdb_lines(p):
            if atom_name:
                atom_counter[atom_name] += 1
            if element:
                element_counter[element] += 1
    return atom_counter, element_counter, files_scanned


def write_csv(counter: Counter, path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["type", "count"])
        for k, v in counter.most_common():
            writer.writerow([k, v])


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate atom types in PDB files")
    parser.add_argument("--root", type=Path, required=False, default=Path("/scratch/project_465002574/PDB/PDB_cleaned"), help="Root folder containing PDB files")
    parser.add_argument("--show", choices=["both", "elements", "atom_names"], default="both", help="Which sets to show")
    parser.add_argument("--top", type=int, default=0, help="Show top N types (0 = all)")
    parser.add_argument("--out-csv", type=Path, default=None, help="Write atom names to CSV (path)")
    parser.add_argument("--out-json", type=Path, default=None, help="Write full JSON report (path)")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    atom_counter, element_counter, files_scanned = aggregate(root)

    report = {
        "root": str(root),
        "files_scanned": files_scanned,
        "atom_name_counts": dict(atom_counter),
        "element_counts": dict(element_counter),
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)

    if args.out_csv:
        # write atom names by default
        write_csv(atom_counter, args.out_csv)

    def print_top(counter: Counter, title: str) -> None:
        print(f"\n{title} (unique: {len(counter)})")
        items = counter.most_common(args.top or None)
        for name, count in items:
            print(f"{name}: {count}")

    if args.show in ("both", "atom_names"):
        print_top(atom_counter, "Atom names")
    if args.show in ("both", "elements"):
        print_top(element_counter, "Element symbols")


if __name__ == "__main__":
    main()
