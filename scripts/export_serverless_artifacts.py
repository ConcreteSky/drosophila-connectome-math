"""Export lightweight NumPy/SciPy `.npz` & `.json` artifacts for Vercel serverless inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from flybrain_rl.config import ARTIFACTS_DIR, DATA_DIR, EXPERIMENTS_DIR
from flybrain_rl.connectome import ConnectomeGraph, query_connectome_summary
from flybrain_rl.encoding import PopulationEncoderDecoder
from flybrain_rl.projection import DorsalProjection
from flybrain_rl.tasks import MathTaskEnv
from flybrain_rl.train import RSTDPTrainer
from flybrain_rl.brain import SpikingFlyBrain


def main() -> None:
    out_dir = ARTIFACTS_DIR / "serverless"
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = ConnectomeGraph.load(DATA_DIR)
    brain = SpikingFlyBrain(graph, device="cpu")
    enc = PopulationEncoderDecoder(
        num_neurons=brain.num_neurons,
        sensory_indices=brain.get_top_out_degree_indices("Sensory", top_k=1500),
        output_indices=brain.get_region_indices("Motor/Output"),
        device=torch.device("cpu"),
        neurons_per_pool=64,
        seed=42,
    )
    proj = DorsalProjection.build(graph, enc, num_landmarks=3000, seed=42)

    # Extract induced 3,000-neuron FlyWire subgraph in SciPy CSR
    lm_idx = proj.landmark_indices.numpy()
    lm_map = {int(orig): i for i, orig in enumerate(lm_idx)}
    N_sub = len(lm_idx)

    # Build SciPy CSR of full graph then slice [lm_idx, :][:, lm_idx]
    full_csr = sp.csr_matrix(
        (
            (graph.values * brain.cfg.weight_scale).numpy(),
            graph.col_indices.numpy(),
            graph.crow_indices.numpy(),
        ),
        shape=(graph.num_neurons, graph.num_neurons),
    )
    sub_csr = full_csr[lm_idx, :][:, lm_idx].tocsr()
    sub_csr.sort_indices()

    def _remap_pools(pools: list[torch.Tensor]) -> np.ndarray:
        remapped = [[lm_map[int(x)] for x in p.tolist() if int(x) in lm_map] for p in pools]
        max_len = max((len(r) for r in remapped), default=1)
        arr = np.full((len(remapped), max_len), -1, dtype=np.int32)
        for i, r in enumerate(remapped):
            if r:
                arr[i, : len(r)] = r
        return arr

    # Save trained readout weights for each task (or fresh if no ckpt)
    env_xor = MathTaskEnv(enc, task="delayed_xor", split_seed=42)
    tr_base = RSTDPTrainer(brain, env_xor, seed=42)

    np.savez_compressed(
        out_dir / "subnetwork.npz",
        indptr=sub_csr.indptr.astype(np.int32),
        indices=sub_csr.indices.astype(np.int32),
        data=sub_csr.data.astype(np.float32),
        signs=graph.signs.numpy()[lm_idx].astype(np.int8),
        region_ids=graph.region_ids.numpy()[lm_idx].astype(np.int8),
        a_pools=_remap_pools(enc.a_pools),
        b_pools=_remap_pools(enc.b_pools),
        go_pool=np.array([lm_map[int(x)] for x in enc.go_pool.tolist() if int(x) in lm_map], dtype=np.int32),
        output_pools=_remap_pools(enc.output_pools),
    )

    summary = query_connectome_summary(DATA_DIR)
    (out_dir / "layout_and_summary.json").write_text(
        json.dumps(
            {
                "region_names": list(graph.region_names),
                "landmarks": proj.landmarks,
                "summary": summary,
            }
        )
    )
    print(
        f"[exported] subnetwork.npz ({(out_dir / 'subnetwork.npz').stat().st_size / 1e3:.1f} KB, "
        f"{N_sub} neurons, {sub_csr.nnz} edges) & layout_and_summary.json "
        f"({(out_dir / 'layout_and_summary.json').stat().st_size / 1e3:.1f} KB)"
    )


if __name__ == "__main__":
    main()
