"""Phase 2 experiment runner: Real Fly vs. Shuffled vs. GRU + Lesions & Recovery.

Usage:
    python scripts/run_experiments.py --smoke-test
    python scripts/run_experiments.py --task mod_add --seeds 3 --episodes 200 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
import torch

from flybrain_rl.brain import SpikingFlyBrain
from flybrain_rl.config import DATA_DIR, EXPERIMENTS_DIR, TrainingConfig
from flybrain_rl.connectome import ConnectomeGraph
from flybrain_rl.encoding import PopulationEncoderDecoder
from flybrain_rl.tasks import MathTaskEnv, TaskName
from flybrain_rl.train import GRUBaselineTrainer, REINFORCETrainer, RSTDPTrainer


def run_suite(
    real_graph: ConnectomeGraph,
    task: TaskName,
    seeds: list[int],
    episodes: int,
    batch_size: int,
    recovery_episodes: int,
    device: str,
    out_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    dev_name = "local_gpu" if device == "cuda" else "local_cpu"
    cfg = TrainingConfig(batch_size=batch_size)

    for seed in seeds:
        shuf_graph = real_graph.make_degree_preserving_shuffled(seed=seed + 1000)

        def _make_env(brain: SpikingFlyBrain) -> MathTaskEnv:
            enc = PopulationEncoderDecoder(
                num_neurons=brain.num_neurons,
                sensory_indices=brain.get_top_out_degree_indices("Sensory", top_k=1500),
                output_indices=brain.get_region_indices("Motor/Output"),
                device=brain.device,
                neurons_per_pool=64,
                seed=seed,
            )
            return MathTaskEnv(enc, task=task, split_seed=seed)

        b_rstdp = SpikingFlyBrain(real_graph, device=device)
        e_rstdp = _make_env(b_rstdp)
        t_rstdp = RSTDPTrainer(b_rstdp, e_rstdp, cfg=cfg, seed=seed)

        b_rf = SpikingFlyBrain(real_graph, device=device)
        e_rf = _make_env(b_rf)
        t_rf = REINFORCETrainer(b_rf, e_rf, cfg=cfg, seed=seed)

        b_shuf = SpikingFlyBrain(shuf_graph, device=device)
        e_shuf = _make_env(b_shuf)
        t_shuf = RSTDPTrainer(b_shuf, e_shuf, cfg=cfg, seed=seed)

        t_gru = GRUBaselineTrainer(e_rstdp, hidden_size=64, lr=0.01, seed=seed)

        runners = [
            ("real_fly", "rstdp", t_rstdp),
            ("real_fly", "reinforce", t_rf),
            ("shuffled_brain", "rstdp", t_shuf),
            ("pytorch_gru", "reinforce", t_gru),
        ]

        for model_type, plasticity, trainer in runners:
            for ep in range(1, episodes + 1):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                tr = trainer.train_step(batch_size=batch_size)
                if device == "cuda":
                    torch.cuda.synchronize()
                ms_per_ep = ((time.perf_counter() - t0) * 1000.0) / max(1, batch_size)
                ev = trainer.evaluate(split="test", batch_size=batch_size)

                rows.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "plasticity": plasticity,
                        "task": task,
                        "lesion": "none",
                        "episode": ep,
                        "train_acc": round(tr["train_acc"], 4),
                        "test_acc": round(ev["accuracy"], 4),
                        "mean_reward": round(ev["mean_reward"], 4),
                        "mean_spikes": round(ev["mean_spikes"], 1),
                        "ms_per_episode": round(ms_per_ep, 3),
                        "device": dev_name,
                    }
                )

        # Save trained R-STDP checkpoint
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task": task,
                "seed": seed,
                "W_out": t_rstdp.W_out.cpu(),
                "b_out": t_rstdp.b_out.cpu(),
                "W_kc": t_rstdp.W_kc.cpu(),
                "mb_sample": t_rstdp.mb_sample.cpu(),
                "cx_sample": t_rstdp.cx_sample.cpu(),
                "lh_sample": t_rstdp.lh_sample.cpu(),
            },
            ckpt_dir / f"{task}_seed{seed}.pt",
        )

        # Lesion & recovery sweep on trained real_fly R-STDP
        base_W, base_b = t_rstdp.W_out.clone(), t_rstdp.b_out.clone()
        base_vals = b_rstdp.values.clone()
        lesion_specs = [
            ("mushroom_body", ["Mushroom Body"], False),
            ("central_complex", ["Central Complex"], False),
            ("hub_top_5pct", None, True),
        ]
        for lname, regs, hub_flag in lesion_specs:
            t_rstdp.W_out.copy_(base_W)
            t_rstdp.b_out.copy_(base_b)
            b_rstdp.values.copy_(base_vals)
            b_rstdp.rebuild_csr()
            b_rstdp.set_lesion(regions=regs, hub_top_5pct=hub_flag)

            imm = t_rstdp.evaluate(split="test", batch_size=batch_size)
            rows.append(
                {
                    "seed": seed,
                    "model_type": "real_fly",
                    "plasticity": "rstdp",
                    "task": task,
                    "lesion": f"{lname}_immediate",
                    "episode": episodes,
                    "train_acc": round(imm["accuracy"], 4),
                    "test_acc": round(imm["accuracy"], 4),
                    "mean_reward": round(imm["mean_reward"], 4),
                    "mean_spikes": round(imm["mean_spikes"], 1),
                    "ms_per_episode": 0.0,
                    "device": dev_name,
                }
            )

            for rec_ep in range(1, recovery_episodes + 1):
                t_rstdp.train_step(batch_size=batch_size)
            rec = t_rstdp.evaluate(split="test", batch_size=batch_size)
            rows.append(
                {
                    "seed": seed,
                    "model_type": "real_fly",
                    "plasticity": "rstdp",
                    "task": task,
                    "lesion": f"{lname}_recovered",
                    "episode": episodes + recovery_episodes,
                    "train_acc": round(rec["accuracy"], 4),
                    "test_acc": round(rec["accuracy"], 4),
                    "mean_reward": round(rec["mean_reward"], 4),
                    "mean_spikes": round(rec["mean_spikes"], 1),
                    "ms_per_episode": 0.0,
                    "device": dev_name,
                }
            )
        b_rstdp.set_lesion(regions=None, hub_top_5pct=False)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Drosophila connectome RL experiments.")
    parser.add_argument("--task", choices=["delayed_xor", "add_3bit", "mod_add", "all"], default="all")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--recovery-episodes", type=int, default=20)
    parser.add_argument("--smoke-test", action="store_true", help="Fast <15s local smoke verification.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph = ConnectomeGraph.load(DATA_DIR)

    if args.smoke_test:
        tasks: list[TaskName] = ["delayed_xor"]
        seeds = [0]
        episodes = 10
        batch_size = 1
        recovery_episodes = 2
    else:
        tasks = ["delayed_xor", "add_3bit", "mod_add"] if args.task == "all" else [args.task]
        seeds = list(range(args.seeds))
        episodes = args.episodes
        batch_size = args.batch_size
        recovery_episodes = args.recovery_episodes

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for task in tasks:
        t0 = time.perf_counter()
        rows = run_suite(
            real_graph=graph,
            task=task,
            seeds=seeds,
            episodes=episodes,
            batch_size=batch_size,
            recovery_episodes=recovery_episodes,
            device=device,
            out_dir=EXPERIMENTS_DIR,
        )
        all_rows.extend(rows)
        print(f"[done] task={task} in {time.perf_counter() - t0:.2f}s ({len(rows)} rows)")

    df = pd.DataFrame(all_rows)
    df.to_csv(EXPERIMENTS_DIR / "results.csv", index=False)

    summary = {
        "device": "local_gpu" if device == "cuda" else "local_cpu",
        "smoke_test": args.smoke_test,
        "tasks": tasks,
        "seeds": seeds,
        "episodes": episodes,
        "by_task_model": (
            df[df["lesion"] == "none"]
            .groupby(["task", "model_type", "plasticity"])[["test_acc", "mean_spikes"]]
            .mean()
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
        "lesions": (
            df[df["lesion"] != "none"]
            .groupby(["task", "lesion"])[["test_acc", "mean_spikes"]]
            .mean()
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    (EXPERIMENTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[saved] {EXPERIMENTS_DIR / 'results.csv'} & summary.json")


if __name__ == "__main__":
    main()
