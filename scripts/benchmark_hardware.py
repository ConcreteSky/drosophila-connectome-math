"""CPU vs. CUDA GPU throughput benchmark across batch_size in [1, 16, 64]."""

from __future__ import annotations

import json
import time

import torch

from flybrain_rl.brain import SpikingFlyBrain
from flybrain_rl.config import ARTIFACTS_DIR, DATA_DIR, LIFConfig
from flybrain_rl.connectome import ConnectomeGraph


def run_benchmark(batch_sizes: tuple[int, ...] = (1, 16, 64), trial_ms: int = 60) -> dict:
    graph = ConnectomeGraph.load(DATA_DIR)
    cfg = LIFConfig(trial_ms=trial_ms)
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    results: dict = {
        "num_neurons": graph.num_neurons,
        "num_edges": graph.num_edges,
        "trial_ms": trial_ms,
        "runs": [],
    }

    for dev in devices:
        brain = SpikingFlyBrain(graph, config=cfg, device=dev)
        sensory_idx = brain.get_top_out_degree_indices("Sensory", top_k=1500)
        bs_list = batch_sizes if dev == "cuda" else tuple(b for b in batch_sizes if b <= 16)
        for bs in bs_list:
            ext = torch.zeros((trial_ms, bs, graph.num_neurons), dtype=torch.float32, device=dev)
            if len(sensory_idx) > 0:
                ext[:, :, sensory_idx] = 18.0

            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            # Warmup
            brain.run(ext[:10])
            if dev == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            out = brain.run(ext)
            if dev == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            mem_mb = (
                float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                if dev == "cuda"
                else 0.0
            )
            total_spikes = out["total_spikes"]
            results["runs"].append(
                {
                    "device": dev,
                    "device_name": "local_gpu" if dev == "cuda" else "local_cpu",
                    "batch_size": bs,
                    "trial_latency_ms": round(elapsed * 1000.0, 2),
                    "ms_per_episode": round((elapsed * 1000.0) / bs, 3),
                    "total_spikes": total_spikes,
                    "spikes_per_sec": round(total_spikes / max(elapsed, 1e-9), 1),
                    "gpu_mem_peak_mb": round(mem_mb, 2),
                }
            )

    return results


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    res = run_benchmark()
    out_path = ARTIFACTS_DIR / "hardware_benchmark.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
