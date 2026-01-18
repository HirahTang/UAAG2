from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lmdb
import typer

HUGGINGFACE_REPO_ID = "yhsure/uaag2-data"
DATA_FILES = (
    "aa_graph.json",
    "statistic.pkl",
    "pdb_subset.lmdb",
    "benchmarks/ENVZ_ECOLI.pt",
)


def human_readable_size(num_bytes: int) -> str:
    """Convert a byte count into a human-readable size string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        Human-readable size string.
    """
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def get_lmdb_length(path: Path) -> int | None:
    """Read the number of entries from an LMDB dataset.

    Args:
        path: Path to the LMDB file.

    Returns:
        The number of entries if the LMDB file exists, otherwise None.
    """
    if not path.exists():
        return None

    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        subdir=False,
        readahead=False,
        meminit=False,
    )
    try:
        with env.begin() as txn:
            length = pickle.loads(txn.get(b"__len__"))
    finally:
        env.close()
    return int(length)


def get_statistics_keys(path: Path) -> list[str]:
    """Load the statistic pickle and return its top-level keys.

    Args:
        path: Path to the statistic.pkl file.

    Returns:
        List of keys if the file exists and contains a mapping; otherwise an empty list.
    """
    if not path.exists():
        return []

    with path.open("rb") as handle:
        data: Any = pickle.load(handle)

    if isinstance(data, dict):
        return sorted(str(key) for key in data.keys())
    return []


def build_report(data_dir: str = "data") -> str:
    """Build a markdown report with basic dataset statistics.

    Args:
        data_dir: Directory where dataset files are stored.

    Returns:
        Markdown formatted report string.
    """
    data_path = Path(data_dir)
    lines = [
        "# Hugging Face dataset report",
        "",
        f"- Repo: `{HUGGINGFACE_REPO_ID}`",
        f"- Data directory: `{data_path.resolve()}`",
        "",
        "| File | Status | Size |",
        "| --- | --- | --- |",
    ]

    for filename in DATA_FILES:
        file_path = data_path / filename
        if file_path.exists():
            size = human_readable_size(file_path.stat().st_size)
            status = "present"
        else:
            size = "n/a"
            status = "missing"
        lines.append(f"| `{filename}` | {status} | {size} |")

    lmdb_length = get_lmdb_length(data_path / "pdb_subset.lmdb")
    if lmdb_length is not None:
        lines.extend(["", f"- LMDB entries: **{lmdb_length:,}**"])

    statistic_keys = get_statistics_keys(data_path / "statistic.pkl")
    if statistic_keys:
        joined_keys = ", ".join(statistic_keys)
        lines.extend(["", f"- Statistic keys: {joined_keys}"])

    return "\n".join(lines)


def dataset_statistics(data_dir: str = "data") -> None:
    """Print dataset statistics as a markdown report.

    Args:
        data_dir: Directory where dataset files are stored.
    """
    print(build_report(data_dir))


if __name__ == "__main__":
    typer.run(dataset_statistics)
