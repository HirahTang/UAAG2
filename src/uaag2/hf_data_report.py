from __future__ import annotations

from pathlib import Path

import typer

HUGGINGFACE_REPO_ID = "yhsure/uaag2-data"
DATA_FILES = (
    "aa_graph.json",
    "statistic.pkl",
    "pdb_subset.lmdb",
    "benchmarks/ENVZ_ECOLI.pt",
)
BYTES_PER_UNIT = 1024
SIZE_UNITS = ("B", "KB", "MB", "GB", "TB")


def human_readable_size(num_bytes: int) -> str:
    """Convert a byte count into a human-readable size string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        Human-readable size string with unit suffix (e.g., "1.5 MB", "512 B").
    """
    size = float(num_bytes)
    for unit in SIZE_UNITS:
        if size < BYTES_PER_UNIT:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= BYTES_PER_UNIT
    return f"{size:.1f} {SIZE_UNITS[-1]}"


def build_report(data_dir: str | Path = "data") -> str:
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

    return "\n".join(lines)


def dataset_statistics(data_dir: str | Path = "data") -> None:
    """Print dataset statistics as a markdown report.

    Args:
        data_dir: Directory where dataset files are stored.
    """
    print(build_report(data_dir))


if __name__ == "__main__":
    typer.run(dataset_statistics)
