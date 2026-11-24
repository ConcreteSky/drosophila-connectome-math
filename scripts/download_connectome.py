"""Download and ingest the FlyWire v783 Drosophila whole-brain connectome.

Downloads:
- Completeness_783.csv (philshiu/Drosophila_brain_model)
- Connectivity_783.parquet (philshiu/Drosophila_brain_model)
- Supplemental_file1_neuron_annotations.tsv (flyconnectome/flywire_annotations)

Outputs:
- data/connectome.duckdb (DuckDB database with `neurons` and `edges` tables)
- data/processed/neurons.parquet
- data/processed/edges.parquet
- data/processed/connectome_csr.pt (PyTorch CSR sparse weight matrix + metadata)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
import requests
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DUCKDB_PATH = ROOT_DIR / "data" / "connectome.duckdb"

URLS = {
    "Completeness_783.csv": "https://github.com/philshiu/Drosophila_brain_model/raw/main/Completeness_783.csv",
    "Connectivity_783.parquet": "https://github.com/philshiu/Drosophila_brain_model/raw/main/Connectivity_783.parquet",
    "Supplemental_file1_neuron_annotations.tsv": (
        "https://github.com/flyconnectome/flywire_annotations/raw/main/"
        "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
    ),
}


def download_file(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size / 1e6:.2f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[done] {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)")


def build_duckdb_and_parquet() -> tuple[pd.DataFrame, dict]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    comp_path = RAW_DIR / "Completeness_783.csv"
    conn_path = RAW_DIR / "Connectivity_783.parquet"
    ann_path = RAW_DIR / "Supplemental_file1_neuron_annotations.tsv"

    print("[ingest] Loading Completeness_783.csv...")
    comp_df = pd.read_csv(comp_path)
    # First column is root_id (unnamed in CSV)
    root_col = comp_df.columns[0]
    comp_df = comp_df.rename(columns={root_col: "root_id"})
    comp_df["idx"] = np.arange(len(comp_df), dtype=np.int32)
    comp_df["root_id"] = comp_df["root_id"].astype(np.int64)

    con = duckdb.connect(str(DUCKDB_PATH))
    con.register("comp_df", comp_df)

    print("[ingest] Joining with FlyWire neuron annotations in DuckDB...")
    ann_str = str(ann_path).replace("\\", "/")
    conn_str = str(conn_path).replace("\\", "/")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE raw_annotations AS
        SELECT
            CAST(root_id AS BIGINT) AS root_id,
            FIRST(CAST(COALESCE(pos_x, soma_x) AS DOUBLE)) AS pos_x,
            FIRST(CAST(COALESCE(pos_y, soma_y) AS DOUBLE)) AS pos_y,
            FIRST(CAST(COALESCE(pos_z, soma_z) AS DOUBLE)) AS pos_z,
            FIRST(COALESCE(super_class, 'unknown')) AS super_class,
            FIRST(COALESCE(cell_class, 'unknown')) AS cell_class,
            FIRST(COALESCE(cell_type, 'unknown')) AS cell_type,
            FIRST(COALESCE(top_nt, 'ACH')) AS top_nt,
            FIRST(COALESCE(side, 'unknown')) AS side
        FROM read_csv('{ann_str}', delim='\t', header=true, ignore_errors=true)
        WHERE root_id IS NOT NULL
        GROUP BY root_id
        """
    )

    # Determine presynaptic neurotransmitter sign from Connectivity_783.parquet as fallback/check
    con.execute(
        f"""
        CREATE OR REPLACE TABLE pre_signs AS
        SELECT
            CAST(Presynaptic_Index AS INTEGER) AS idx,
            CAST(SIGN(AVG(Excitatory)) AS INTEGER) AS edge_sign
        FROM read_parquet('{conn_str}')
        GROUP BY Presynaptic_Index
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE neurons AS
        WITH joined AS (
            SELECT
                c.idx,
                c.root_id,
                a.pos_x,
                a.pos_y,
                a.pos_z,
                COALESCE(a.super_class, 'unknown') AS super_class,
                COALESCE(a.cell_class, 'unknown') AS cell_class,
                COALESCE(a.cell_type, 'unknown') AS cell_type,
                UPPER(COALESCE(a.top_nt, 'ACH')) AS top_nt,
                COALESCE(a.side, 'unknown') AS side,
                COALESCE(
                    p.edge_sign,
                    CASE
                        WHEN UPPER(COALESCE(a.top_nt, 'ACH')) IN ('GABA', 'GLUT', 'SER') THEN -1
                        ELSE 1
                    END
                ) AS sign
            FROM comp_df c
            LEFT JOIN raw_annotations a ON c.root_id = a.root_id
            LEFT JOIN pre_signs p ON c.idx = p.idx
        ),
        bounds AS (
            SELECT
                PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY pos_x) AS x_min,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY pos_x) AS x_max,
                PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY pos_y) AS y_min,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY pos_y) AS y_max,
                MEDIAN(pos_x) AS x_med,
                MEDIAN(pos_y) AS y_med,
                MEDIAN(pos_z) AS z_med
            FROM joined
            WHERE pos_x IS NOT NULL AND pos_y IS NOT NULL
        )
        SELECT
            j.idx,
            j.root_id,
            CAST(COALESCE(j.pos_x, b.x_med) AS FLOAT) AS x,
            CAST(COALESCE(j.pos_y, b.y_med) AS FLOAT) AS y,
            CAST(COALESCE(j.pos_z, b.z_med) AS FLOAT) AS z,
            CAST(LEAST(1.0, GREATEST(0.0, (COALESCE(j.pos_x, b.x_med) - b.x_min) / NULLIF(b.x_max - b.x_min, 0))) AS FLOAT) AS x_2d,
            CAST(LEAST(1.0, GREATEST(0.0, (COALESCE(j.pos_y, b.y_med) - b.y_min) / NULLIF(b.y_max - b.y_min, 0))) AS FLOAT) AS y_2d,
            j.super_class,
            j.cell_class,
            j.cell_type,
            j.top_nt,
            j.side,
            CAST(CASE WHEN j.sign < 0 THEN -1 ELSE 1 END AS TINYINT) AS sign,
            CASE
                WHEN j.cell_class ILIKE '%Kenyon%'
                  OR j.cell_class ILIKE '%MBON%'
                  OR j.cell_class ILIKE '%MBIN%'
                  OR j.cell_class ILIKE '%DAN%'
                  OR j.cell_type LIKE 'KC%'
                  OR j.cell_type LIKE 'MBON%'
                  OR j.cell_type LIKE 'PAM%'
                  OR j.cell_type LIKE 'PPL%' THEN 'Mushroom Body'
                WHEN j.cell_class ILIKE '%CX%'
                  OR j.cell_class ILIKE '%central_complex%'
                  OR j.cell_type LIKE 'EPG%'
                  OR j.cell_type LIKE 'PFN%'
                  OR j.cell_type LIKE 'PFL%'
                  OR j.cell_type LIKE 'PEN%'
                  OR j.cell_type LIKE 'hDelta%'
                  OR j.cell_type LIKE 'vDelta%' THEN 'Central Complex'
                WHEN j.super_class IN ('sensory', 'optic', 'visual_projection', 'visual_centrifugal') THEN 'Sensory/Optic'
                WHEN j.super_class IN ('motor', 'descending', 'endocrine') THEN 'Motor/Descending'
                ELSE 'Central/Other'
            END AS region
        FROM joined j
        CROSS JOIN bounds b
        ORDER BY j.idx
        """
    )

    print("[ingest] Building edges table in DuckDB...")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE edges AS
        SELECT
            CAST(Presynaptic_Index AS INTEGER) AS pre_idx,
            CAST(Postsynaptic_Index AS INTEGER) AS post_idx,
            CAST(Connectivity AS INTEGER) AS syn_count,
            CAST(Excitatory AS TINYINT) AS nt_sign,
            CAST("Excitatory x Connectivity" AS FLOAT) AS weight
        FROM read_parquet('{conn_str}')
        WHERE Connectivity > 0
        """
    )

    con.execute("DROP TABLE raw_annotations")
    con.execute("DROP TABLE pre_signs")

    neurons_parquet = str(PROCESSED_DIR / "neurons.parquet").replace("\\", "/")
    edges_parquet = str(PROCESSED_DIR / "edges.parquet").replace("\\", "/")
    con.execute(f"COPY neurons TO '{neurons_parquet}' (FORMAT PARQUET)")
    con.execute(f"COPY edges TO '{edges_parquet}' (FORMAT PARQUET)")

    stats = con.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM neurons) AS num_neurons,
            (SELECT COUNT(*) FROM edges) AS num_edges,
            (SELECT SUM(syn_count) FROM edges) AS total_synapses
        """
    ).fetchone()
    region_counts = con.execute(
        "SELECT region, COUNT(*) FROM neurons GROUP BY region ORDER BY 2 DESC"
    ).fetchall()
    print(f"[stats] Neurons: {stats[0]:,}, Directed edges: {stats[1]:,}, Total synapses: {stats[2]:,}")
    print(f"[stats] Regions: {region_counts}")

    neurons_df = con.execute("SELECT * FROM neurons ORDER BY idx").fetchdf()
    edges_df = con.execute("SELECT post_idx, pre_idx, weight FROM edges ORDER BY post_idx, pre_idx").fetchdf()
    con.close()

    return neurons_df, edges_df


def compile_csr_checkpoint(neurons_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    print("[csr] Compiling PyTorch CSR sparse matrix (Dale's Law signed weights)...")
    t0 = time.perf_counter()
    num_neurons = len(neurons_df)
    post = torch.from_numpy(edges_df["post_idx"].to_numpy(dtype=np.int64))
    pre = torch.from_numpy(edges_df["pre_idx"].to_numpy(dtype=np.int64))
    val = torch.from_numpy(edges_df["weight"].to_numpy(dtype=np.float32))

    # Build CSR crow_indices from sorted post_idx
    counts = torch.bincount(post, minlength=num_neurons)
    crow = torch.zeros(num_neurons + 1, dtype=torch.int64)
    torch.cumsum(counts, dim=0, out=crow[1:])

    # Normalize weights slightly for stable LIF membrane dynamics (preserving Dale's sign)
    # Raw synapse counts range from 1 to ~2000; scale by 1/20 as base conductance factor
    # while keeping raw integer/float values accessible.
    csr_path = PROCESSED_DIR / "connectome_csr.pt"
    region_names = ["Sensory/Optic", "Mushroom Body", "Central Complex", "Central/Other", "Motor/Descending"]
    region_to_id = {name: i for i, name in enumerate(region_names)}
    region_ids = torch.tensor(
        [region_to_id.get(r, 3) for r in neurons_df["region"].tolist()],
        dtype=torch.int8,
    )
    coords_2d = torch.from_numpy(neurons_df[["x_2d", "y_2d"]].to_numpy(dtype=np.float32))
    signs = torch.from_numpy(neurons_df["sign"].to_numpy(dtype=np.int8))

    torch.save(
        {
            "num_neurons": num_neurons,
            "num_edges": len(val),
            "crow_indices": crow,
            "col_indices": pre.to(torch.int32),
            "values": val,
            "region_names": region_names,
            "region_ids": region_ids,
            "coords_2d": coords_2d,
            "signs": signs,
        },
        csr_path,
    )
    dt = time.perf_counter() - t0
    print(f"[csr] Saved {csr_path} in {dt:.2f}s ({csr_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and ingest FlyWire v783 connectome.")
    parser.add_argument("--force", action="store_true", help="Re-download raw files even if present.")
    args = parser.parse_args()

    for name, url in URLS.items():
        download_file(url, RAW_DIR / name, force=args.force)

    neurons_df, edges_df = build_duckdb_and_parquet()
    compile_csr_checkpoint(neurons_df, edges_df)
    print("[complete] Connectome ingestion finished.")


if __name__ == "__main__":
    main()
