"""DuckDB/Parquet connectome loader, sparse CSR builder, and degree-preserving shuffler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import torch

from flybrain_rl.config import DATA_DIR, REGION_CENTERS_2D, REGION_NAMES


def map_region(super_class: str, cell_class: str, cell_type: str) -> str:
    """Map FlyWire annotations into the 6 canonical dorsal regions."""
    sc = (super_class or "").lower()
    cc = (cell_class or "").lower()
    ct = (cell_type or "")

    if "kenyon" in cc or "mbon" in cc or "mbin" in cc or "dan" in cc or ct.startswith(("KC", "MBON", "PAM", "PPL")):
        return "Mushroom Body"
    if "cx" in cc or "central_complex" in cc or ct.startswith(("EPG", "PFN", "PFL", "PEN", "hDelta", "vDelta")):
        return "Central Complex"
    if "alpn" in cc or "alln" in cc or "antennal" in cc or ct.startswith(("AL", "PN", "lLN", "vPN")):
        return "Antennal Lobe"
    if "lh" in cc or "lateral_horn" in cc or ct.startswith(("LHON", "LHLN", "LH")):
        return "Lateral Horn"
    if sc in ("motor", "descending", "endocrine"):
        return "Motor/Output"
    if sc in ("sensory", "optic", "visual_projection", "visual_centrifugal"):
        return "Sensory"
    # Remaining central/ascending interneurons map to Lateral Horn / Motor / Central Complex
    if sc == "ascending":
        return "Motor/Output"
    return "Lateral Horn"


@dataclass
class ConnectomeGraph:
    num_neurons: int
    num_edges: int
    crow_indices: torch.Tensor
    col_indices: torch.Tensor
    values: torch.Tensor
    signs: torch.Tensor
    region_names: list[str]
    region_ids: torch.Tensor
    coords_2d: torch.Tensor

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR) -> "ConnectomeGraph":
        """Load compiled CSR connectome from Parquet/DuckDB or cached `.pt`."""
        csr_v2 = data_dir / "connectome_csr_v2.pt"
        if csr_v2.exists():
            d = torch.load(csr_v2, map_location="cpu", weights_only=False)
            return cls(
                num_neurons=int(d["num_neurons"]),
                num_edges=int(d["num_edges"]),
                crow_indices=d["crow_indices"].to(torch.int64),
                col_indices=d["col_indices"].to(torch.int64),
                values=d["values"].to(torch.float32),
                signs=d["signs"].to(torch.int8),
                region_names=list(d["region_names"]),
                region_ids=d["region_ids"].to(torch.int8),
                coords_2d=d["coords_2d"].to(torch.float32),
            )

        neurons_pq = data_dir / "neurons.parquet"
        edges_pq = data_dir / "edges.parquet"
        if not neurons_pq.exists():
            neurons_pq = data_dir / "processed" / "neurons.parquet"
        if not edges_pq.exists():
            edges_pq = data_dir / "processed" / "edges.parquet"

        if not neurons_pq.exists() or not edges_pq.exists():
            raise FileNotFoundError(
                f"Connectome Parquet files not found in {data_dir}. Run scripts/download_connectome.py first."
            )

        con = duckdb.connect()
        n_df = con.execute(f"SELECT idx, x_2d, y_2d, super_class, cell_class, cell_type, sign FROM '{neurons_pq}' ORDER BY idx").fetchdf()
        e_df = con.execute(f"SELECT post_idx, pre_idx, weight FROM '{edges_pq}' ORDER BY post_idx, pre_idx").fetchdf()
        con.close()

        num_neurons = len(n_df)
        reg_to_id = {name: i for i, name in enumerate(REGION_NAMES)}
        regions = [
            map_region(sc, cc, ct)
            for sc, cc, ct in zip(n_df["super_class"], n_df["cell_class"], n_df["cell_type"])
        ]
        region_ids = torch.tensor([reg_to_id[r] for r in regions], dtype=torch.int8)

        # Use real FlyWire 2D coords if valid, otherwise canonical region centers with deterministic jitter
        coords = n_df[["x_2d", "y_2d"]].to_numpy(dtype=np.float32).copy()
        coords = np.clip(coords, 0.02, 0.98)
        coords_2d = torch.from_numpy(coords)
        signs = torch.from_numpy(n_df["sign"].to_numpy(dtype=np.int8).copy())

        post = torch.from_numpy(e_df["post_idx"].to_numpy(dtype=np.int64).copy())
        pre = torch.from_numpy(e_df["pre_idx"].to_numpy(dtype=np.int64).copy())
        val = torch.from_numpy(e_df["weight"].to_numpy(dtype=np.float32).copy())

        counts = torch.bincount(post, minlength=num_neurons)
        crow = torch.zeros(num_neurons + 1, dtype=torch.int64)
        torch.cumsum(counts, dim=0, out=crow[1:])

        graph = cls(
            num_neurons=num_neurons,
            num_edges=len(val),
            crow_indices=crow,
            col_indices=pre,
            values=val,
            signs=signs,
            region_names=list(REGION_NAMES),
            region_ids=region_ids,
            coords_2d=coords_2d,
        )

        torch.save(
            {
                "num_neurons": graph.num_neurons,
                "num_edges": graph.num_edges,
                "crow_indices": graph.crow_indices,
                "col_indices": graph.col_indices.to(torch.int32),
                "values": graph.values,
                "signs": graph.signs,
                "region_names": graph.region_names,
                "region_ids": graph.region_ids,
                "coords_2d": graph.coords_2d,
            },
            csr_v2,
        )
        return graph

    @classmethod
    def make_synthetic(
        cls,
        num_neurons: int = 360,
        num_edges: int = 2200,
        seed: int = 42,
    ) -> "ConnectomeGraph":
        """Create a small deterministic synthetic connectome across all 6 regions for tests."""
        rng = torch.Generator().manual_seed(seed)
        post = torch.randint(0, num_neurons, (num_edges,), generator=rng)
        pre = torch.randint(0, num_neurons, (num_edges,), generator=rng)
        keys = torch.unique(post * num_neurons + pre)
        post = torch.div(keys, num_neurons, rounding_mode="floor")
        pre = keys % num_neurons

        signs = torch.where(
            torch.rand(num_neurons, generator=rng) > 0.25,
            torch.ones(num_neurons, dtype=torch.int8),
            -torch.ones(num_neurons, dtype=torch.int8),
        )
        mag = torch.randint(1, 12, (len(keys),), generator=rng).to(torch.float32)
        val = mag * signs[pre].to(torch.float32)

        counts = torch.bincount(post, minlength=num_neurons)
        crow = torch.zeros(num_neurons + 1, dtype=torch.int64)
        torch.cumsum(counts, dim=0, out=crow[1:])

        # Balanced across all 6 regions
        region_ids = (torch.arange(num_neurons) % len(REGION_NAMES)).to(torch.int8)
        coords = torch.zeros((num_neurons, 2), dtype=torch.float32)
        for i in range(num_neurons):
            cx, cy = REGION_CENTERS_2D[REGION_NAMES[int(region_ids[i])]]
            coords[i, 0] = cx
            coords[i, 1] = cy
        coords = (coords + 0.05 * torch.randn((num_neurons, 2), generator=rng)).clamp(0.02, 0.98)

        return cls(
            num_neurons=num_neurons,
            num_edges=len(keys),
            crow_indices=crow,
            col_indices=pre.to(torch.int64),
            values=val,
            signs=signs,
            region_names=list(REGION_NAMES),
            region_ids=region_ids,
            coords_2d=coords,
        )

    def make_degree_preserving_shuffled(self, seed: int = 123) -> "ConnectomeGraph":
        """Configuration-model rewiring preserving exact in-degree, out-degree, and E/I sign."""
        rng = torch.Generator().manual_seed(seed)
        perm_col = torch.randperm(self.num_edges, generator=rng)
        col_shuffled = self.col_indices[perm_col]

        perm_mag = torch.randperm(self.num_edges, generator=rng)
        mag = self.values.abs()[perm_mag]
        val_shuffled = mag * self.signs[col_shuffled].to(torch.float32)

        return ConnectomeGraph(
            num_neurons=self.num_neurons,
            num_edges=self.num_edges,
            crow_indices=self.crow_indices.clone(),
            col_indices=col_shuffled.to(torch.int64),
            values=val_shuffled,
            signs=self.signs.clone(),
            region_names=list(self.region_names),
            region_ids=self.region_ids.clone(),
            coords_2d=self.coords_2d.clone(),
        )


def query_connectome_summary(data_dir: Path = DATA_DIR) -> dict:
    """Query DuckDB/Parquet for connectome summary stats and top hubs."""
    neurons_pq = data_dir / "neurons.parquet"
    edges_pq = data_dir / "edges.parquet"
    if not neurons_pq.exists():
        neurons_pq = data_dir / "processed" / "neurons.parquet"
    if not edges_pq.exists():
        edges_pq = data_dir / "processed" / "edges.parquet"

    if not neurons_pq.exists() or not edges_pq.exists():
        return {}

    con = duckdb.connect()
    total_neurons = int(con.execute(f"SELECT COUNT(*) FROM '{neurons_pq}'").fetchone()[0])
    ei_counts = con.execute(
        f"SELECT SUM(CASE WHEN sign > 0 THEN 1 ELSE 0 END), SUM(CASE WHEN sign < 0 THEN 1 ELSE 0 END) FROM '{neurons_pq}'"
    ).fetchone()
    edge_stats = con.execute(
        f"SELECT COUNT(*), COALESCE(SUM(syn_count), 0) FROM '{edges_pq}'"
    ).fetchone()
    top_hubs = con.execute(
        f"""
        SELECT pre_idx, COUNT(*) AS out_degree, SUM(syn_count) AS total_synapses
        FROM '{edges_pq}'
        GROUP BY pre_idx
        ORDER BY out_degree DESC
        LIMIT 10
        """
    ).fetchall()
    sc_counts = con.execute(
        f"SELECT super_class, cell_class, cell_type, COUNT(*) AS cnt FROM '{neurons_pq}' GROUP BY 1, 2, 3"
    ).fetchall()
    con.close()

    region_totals = {r: 0 for r in REGION_NAMES}
    for sc, cc, ct, cnt in sc_counts:
        region_totals[map_region(sc, cc, ct)] += int(cnt)

    return {
        "total_neurons": total_neurons,
        "excitatory_neurons": int(ei_counts[0] or 0),
        "inhibitory_neurons": int(ei_counts[1] or 0),
        "num_edges": int(edge_stats[0]),
        "total_synapses": int(edge_stats[1]),
        "regions": [{"region": r, "count": c} for r, c in region_totals.items()],
        "top_hubs": [
            {"neuron_idx": int(idx), "out_degree": int(deg), "total_synapses": int(syn)}
            for idx, deg, syn in top_hubs
        ],
    }
