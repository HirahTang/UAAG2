import argparse
import glob
import lmdb
import os
import pickle
from typing import Dict


def read_lmdb_length(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        raw = txn.get(b"__len__")
        if raw is None:
            return 0
        return int(pickle.loads(raw))


def merge_lmdb_shards(shard_glob: str, output_lmdb: str, output_metadata: str):
    shard_paths = sorted(glob.glob(shard_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No shard LMDB found for glob: {shard_glob}")

    os.makedirs(os.path.dirname(output_lmdb), exist_ok=True)

    out_env = lmdb.open(
        output_lmdb,
        map_size=1 << 42,
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    out_meta: Dict[bytes, str] = {}
    out_txn = out_env.begin(write=True)

    total = 0
    for shard_path in shard_paths:
        shard_meta_path = shard_path.replace(".lmdb", ".metadata.pkl")
        shard_meta = {}
        if os.path.exists(shard_meta_path):
            with open(shard_meta_path, "rb") as handle:
                shard_meta = pickle.load(handle)

        in_env = lmdb.open(shard_path, subdir=False, readonly=True, lock=False, readahead=False)
        shard_len = read_lmdb_length(in_env)

        with in_env.begin() as in_txn:
            for i in range(shard_len):
                in_key = f"{i:08}".encode("ascii")
                payload = in_txn.get(in_key)
                if payload is None:
                    continue

                out_key = f"{total:08}".encode("ascii")
                out_txn.put(out_key, payload)

                src = shard_meta.get(in_key)
                if src is not None:
                    out_meta[out_key] = src

                total += 1
                if total % 10000 == 0:
                    out_txn.commit()
                    out_txn = out_env.begin(write=True)

        in_env.close()
        print(f"Merged {shard_path} ({shard_len} entries)")

    out_txn.put(b"__len__", pickle.dumps(total))
    out_txn.commit()
    out_env.sync()
    out_env.close()

    with open(output_metadata, "wb") as handle:
        pickle.dump(out_meta, handle)

    print("\nMerge complete")
    print(f"  Output LMDB: {output_lmdb}")
    print(f"  Output metadata: {output_metadata}")
    print(f"  Total graphs: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge sharded LMDB files into one LMDB")
    parser.add_argument(
        "--shard_glob",
        type=str,
        required=True,
        help="Glob for shard LMDBs, e.g. /path/to/out/uaag_shard_*.lmdb",
    )
    parser.add_argument("--output_lmdb", type=str, required=True, help="Final merged LMDB path")
    parser.add_argument(
        "--output_metadata",
        type=str,
        required=True,
        help="Final merged metadata pickle path",
    )
    args = parser.parse_args()

    merge_lmdb_shards(
        shard_glob=args.shard_glob,
        output_lmdb=args.output_lmdb,
        output_metadata=args.output_metadata,
    )
