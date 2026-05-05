import argparse
import csv
import os
import pickle
from collections import Counter

import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from tqdm import tqdm


DEFAULT_DATA_PATH = "/scratch/project_465002574/unaagi_whole_v1.lmdb"
NATURAL_AA = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AA_SET = set(NATURAL_AA)


def write_counts_csv(csv_out: str, aa_counter: Counter) -> None:
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    total = sum(aa_counter.values())
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["amino_acid", "count", "fraction"])
        for aa in NATURAL_AA:
            count = aa_counter[aa]
            fraction = (count / total) if total > 0 else 0.0
            writer.writerow([aa, count, f"{fraction:.6f}"])


def write_summary_csv(
    summary_out: str,
    total_graphs: int,
    missing_compound_id: int,
    no_match: int,
    multi_match: int,
    aa_counter: Counter,
) -> None:
    os.makedirs(os.path.dirname(summary_out) or ".", exist_ok=True)
    total_matches = sum(aa_counter.values())
    nonzero_aa = sum(1 for aa in NATURAL_AA if aa_counter[aa] > 0)

    with open(summary_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_graph_objects_scanned", total_graphs])
        writer.writerow(["missing_compound_id", missing_compound_id])
        writer.writerow(["compound_id_with_no_aa_match", no_match])
        writer.writerow(["compound_id_with_multiple_aa_matches", multi_match])
        writer.writerow(["total_aa_token_matches_counted", total_matches])
        writer.writerow(["natural_aa_with_nonzero_count", nonzero_aa])


def save_histogram(hist_out: str, aa_counter: Counter) -> None:
    os.makedirs(os.path.dirname(hist_out) or ".", exist_ok=True)
    labels = NATURAL_AA
    values = [aa_counter[aa] for aa in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(labels, values)
    ax.set_title("Natural Amino Acid Frequency from LMDB compound_id")
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(hist_out, dpi=200)
    plt.close(fig)


def count_natural_aa_from_lmdb(
    data_path: str,
    csv_out: str,
    hist_out: str,
    summary_out: str,
) -> None:
    env = lmdb.open(
        data_path,
        readonly=True,
        lock=False,
        subdir=False,
        readahead=False,
        meminit=False,
    )

    aa_counter = Counter({aa: 0 for aa in NATURAL_AA})
    total_graphs = 0
    missing_compound_id = 0
    no_match = 0
    multi_match = 0

    with env.begin() as txn:
        cursor = txn.cursor()
        for _, byteflow in tqdm(cursor, desc="Scanning LMDB"):
            graph_data = pickle.loads(byteflow)
            if not isinstance(graph_data, Data):
                continue

            total_graphs += 1
            compound_id = getattr(graph_data, "compound_id", None)
            if not compound_id:
                missing_compound_id += 1
                continue

            tokens = str(compound_id).split("_")
            matched = [tok for tok in tokens if tok in AA_SET]

            if len(matched) == 0:
                no_match += 1
            elif len(matched) > 1:
                multi_match += 1

            for aa in matched:
                aa_counter[aa] += 1

    env.close()

    print("=" * 72)
    print("Natural amino acid frequency from compound_id")
    print("=" * 72)
    print(f"LMDB path: {data_path}")
    print(f"Total graph objects scanned: {total_graphs}")
    print(f"Missing compound_id: {missing_compound_id}")
    print(f"compound_id with no AA match: {no_match}")
    print(f"compound_id with multiple AA matches: {multi_match}")
    print("-" * 72)
    for aa in NATURAL_AA:
        print(f"{aa}: {aa_counter[aa]}")
    print("-" * 72)
    print(f"Total AA token matches counted: {sum(aa_counter.values())}")

    write_counts_csv(csv_out, aa_counter)
    write_summary_csv(
        summary_out,
        total_graphs,
        missing_compound_id,
        no_match,
        multi_match,
        aa_counter,
    )
    save_histogram(hist_out, aa_counter)
    print(f"CSV written to: {csv_out}")
    print(f"Summary CSV written to: {summary_out}")
    print(f"Histogram written to: {hist_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count 20 natural amino-acid frequencies from LMDB compound_id tokens."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to LMDB file.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="outputs/lmdb_aa_frequency.csv",
        help="Output CSV path for amino-acid counts.",
    )
    parser.add_argument(
        "--hist-out",
        type=str,
        default="outputs/lmdb_aa_frequency_hist.png",
        help="Output histogram image path.",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="outputs/lmdb_aa_summary.csv",
        help="Output CSV path for run-level summary metrics.",
    )
    args = parser.parse_args()

    count_natural_aa_from_lmdb(
        args.data_path,
        args.csv_out,
        args.hist_out,
        args.summary_out,
    )
