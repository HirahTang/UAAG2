import os
import requests
from pathlib import Path

def get_pdb_ids_after_date(date_str="2024-12-31", max_rows=100):
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_accession_info.deposit_date",
                "operator": "greater",
                "value": date_str
            }
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_rows
            }
        }
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query?json"
    response = requests.post(url, json=query)
    response.raise_for_status()

    pdb_ids = [entry["identifier"] for entry in response.json()["result_set"]]
    return pdb_ids

from Bio.PDB import PDBList

def download_pdb(pdb_id, save_dir="structures_pdb"):
    """Download PDB file for a given PDB ID."""
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    save_path = Path(save_dir) / f"{pdb_id}.pdb"

    if save_path.exists():
        print(f"Already downloaded: {pdb_id}")
        return

    response = requests.get(url)
    if response.status_code == 200:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {pdb_id}")
    else:
        print(f"Failed to download: {pdb_id} (status code {response.status_code})")



row_ids = get_pdb_ids_after_date(date_str="2023-12-31", max_rows=100)
for row_id in row_ids:
    download_pdb(row_id, save_dir = "/home/qcx679/hantang/UAAG2/data/benchmark_row")
# from IPython import embed; embed()