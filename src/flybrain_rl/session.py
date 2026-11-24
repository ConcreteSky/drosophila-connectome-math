"""Asyncio-locked live brain session for inference, manual feedback, training, and lesions."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Literal, Optional

import pandas as pd
import torch

from flybrain_rl.brain import SpikingFlyBrain
from flybrain_rl.config import ARTIFACTS_DIR, DATA_DIR, EXPERIMENTS_DIR, TrainingConfig
from flybrain_rl.connectome import ConnectomeGraph, query_connectome_summary
from flybrain_rl.encoding import PopulationEncoderDecoder
from flybrain_rl.projection import DorsalProjection
from flybrain_rl.tasks import MathTaskEnv, TaskName
from flybrain_rl.train import REINFORCETrainer, RSTDPTrainer


class BrainSession:
    """Thread/async-safe live brain controller protected by `asyncio.Lock`."""

    def __init__(
        self,
        graph: Optional[ConnectomeGraph] = None,
        load_checkpoints: bool = True,
    ) -> None:
        self.lock = asyncio.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.graph = graph if graph is not None else ConnectomeGraph.load(DATA_DIR)
        self.brain = SpikingFlyBrain(self.graph, device=self.device)
        self.encoder = PopulationEncoderDecoder(
            num_neurons=self.brain.num_neurons,
            sensory_indices=self.brain.get_top_out_degree_indices("Sensory", top_k=1500),
            output_indices=self.brain.get_region_indices("Motor/Output"),
            device=self.brain.device,
            neurons_per_pool=64,
            seed=42,
        )
        self.projection = DorsalProjection.build(self.graph, self.encoder, num_landmarks=3000, seed=42)
        self.cfg = TrainingConfig()

        self.envs: dict[TaskName, MathTaskEnv] = {}
        self.rstdp: dict[TaskName, RSTDPTrainer] = {}
        self.reinforce: dict[TaskName, REINFORCETrainer] = {}
        self.checkpoint_loaded = False

        for task in ("delayed_xor", "add_3bit", "sub_3bit", "mod_add"):
            env_t = MathTaskEnv(self.encoder, task=task, split_seed=42)
            tr_r = RSTDPTrainer(self.brain, env_t, cfg=self.cfg, seed=42)
            tr_f = REINFORCETrainer(self.brain, env_t, cfg=self.cfg, seed=42)
            if load_checkpoints:
                ckpt_file = EXPERIMENTS_DIR / "checkpoints" / f"{task}_seed0.pt"
                if ckpt_file.exists():
                    ck = torch.load(ckpt_file, map_location=self.brain.device, weights_only=False)
                    if ck["W_out"].shape == tr_r.W_out.shape:
                        tr_r.W_out.copy_(ck["W_out"].to(self.brain.device))
                        tr_r.b_out.copy_(ck["b_out"].to(self.brain.device))
                        tr_r.W_kc.copy_(ck["W_kc"].to(self.brain.device))
                        tr_r.mb_sample = ck["mb_sample"].to(self.brain.device)
                        tr_r.cx_sample = ck["cx_sample"].to(self.brain.device)
                        tr_r.lh_sample = ck["lh_sample"].to(self.brain.device)
                        self.checkpoint_loaded = True
            self.envs[task] = env_t
            self.rstdp[task] = tr_r
            self.reinforce[task] = tr_f

        self.active_task: TaskName = "delayed_xor"
        self._summary_cache: Optional[dict] = None

    def health(self) -> dict:
        mem_alloc = float(torch.cuda.memory_allocated() / (1024 ** 2)) if self.device == "cuda" else 0.0
        mem_total = (
            float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
            if self.device == "cuda"
            else 0.0
        )
        return {
            "status": "ok",
            "device": "local_gpu" if self.device == "cuda" else "local_cpu",
            "gpu_name": "local_gpu" if self.device == "cuda" else "local_cpu",
            "gpu_mem_allocated_mb": round(mem_alloc, 2),
            "gpu_mem_total_mb": round(mem_total, 2),
            "checkpoint_loaded": self.checkpoint_loaded,
        }

    def solve(self, task: TaskName, a: int, b: int) -> dict:
        self.active_task = task
        trainer = self.rstdp[task]
        prob = MathTaskEnv.make_problem(
            task, a, b, num_a_channels=self.encoder.num_a_channels, num_b_channels=self.encoder.num_b_channels
        )
        t0 = time.perf_counter()
        out = trainer.solve_single(prob, landmark_indices=self.projection.landmark_indices)
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        pred = out.predicted - 7 if task == "sub_3bit" else out.predicted
        true_ans = out.true_answer - 7 if task == "sub_3bit" else out.true_answer
        labels = [str(i - 7) for i in range(15)] if task == "sub_3bit" else [str(i) for i in range(len(out.output_spike_counts))]
        return {
            "predicted_answer": pred,
            "true_answer": true_ans,
            "is_correct": out.is_correct,
            "answer_labels": labels,
            "output_spike_counts": out.output_spike_counts,
            "total_spikes": out.total_spikes,
            "latency_ms": round(latency_ms, 2),
            "region_activity": out.region_activity,
            "spike_events": out.spike_events,
        }

    def feedback(self, reward: int) -> dict:
        trainer = self.rstdp[self.active_task]
        delta_norm = trainer.apply_feedback(float(reward))
        return {
            "reward_applied": int(reward),
            "weight_delta_norm": round(delta_norm, 6),
            "total_weight_shift": round(trainer.total_weight_shift, 4),
            "dopamine_pulse": "reward" if reward > 0 else "punish",
            "total_updates": trainer.total_updates,
        }

    def train_episode(
        self,
        task: TaskName,
        batch_size: int,
        plasticity: Literal["rstdp", "reinforce"],
    ) -> dict:
        self.active_task = task
        trainer = self.rstdp[task] if plasticity == "rstdp" else self.reinforce[task]
        tr = trainer.train_step(batch_size=batch_size)
        ev = trainer.evaluate(split="test", batch_size=batch_size)
        prob = self.envs[task].test_pairs[0]
        return {
            "train_acc": round(tr["train_acc"], 4),
            "test_acc": round(ev["accuracy"], 4),
            "mean_reward": round(tr["mean_reward"], 4),
            "last_a": int(prob[0]),
            "last_b": int(prob[1]),
            "predicted": int(tr.get("last_pred", 0)),
            "true_answer": int(tr.get("last_target", 0)),
            "total_spikes": int(round(tr["mean_spikes"])),
        }

    def set_lesion(self, mushroom_body: bool, central_complex: bool, hub_top_5pct: bool) -> dict:
        regs = []
        if mushroom_body:
            regs.append("Mushroom Body")
        if central_complex:
            regs.append("Central Complex")
        disabled = self.brain.set_lesion(regions=regs, hub_top_5pct=hub_top_5pct)
        return {
            "mushroom_body": mushroom_body,
            "central_complex": central_complex,
            "hub_top_5pct": hub_top_5pct,
            "disabled_neurons": disabled,
        }

    def reset(self, mode: Literal["checkpoint", "fresh", "shuffled"]) -> dict:
        if mode == "shuffled":
            self.brain.reset_weights(self.graph.make_degree_preserving_shuffled(seed=123))
        else:
            self.brain.reset_weights(self.graph)

        for task, tr in self.rstdp.items():
            tr.W_out.zero_()
            tr.b_out.zero_()
            tr.r_bar = 0.0
            tr.reset_memory()
            if mode == "checkpoint":
                ckpt_file = EXPERIMENTS_DIR / "checkpoints" / f"{task}_seed0.pt"
                if ckpt_file.exists():
                    ck = torch.load(ckpt_file, map_location=self.brain.device, weights_only=False)
                    if ck["W_out"].shape == tr.W_out.shape:
                        tr.W_out.copy_(ck["W_out"].to(self.brain.device))
                        tr.b_out.copy_(ck["b_out"].to(self.brain.device))
        self.brain.set_lesion(regions=None, hub_top_5pct=False)
        return {"status": "reset", "mode": mode}

    def connectome_summary(self) -> dict:
        if self._summary_cache is None:
            summary = query_connectome_summary(DATA_DIR)
            if not summary:
                # Fallback for synthetic test graphs
                summary = {
                    "total_neurons": self.graph.num_neurons,
                    "excitatory_neurons": int((self.graph.signs > 0).sum().item()),
                    "inhibitory_neurons": int((self.graph.signs < 0).sum().item()),
                    "num_edges": self.graph.num_edges,
                    "total_synapses": int(self.graph.values.abs().sum().item()),
                    "regions": [
                        {"region": r, "count": int((self.graph.region_ids == i).sum().item())}
                        for i, r in enumerate(self.graph.region_names)
                    ],
                    "top_hubs": [],
                }
            self._summary_cache = summary
        return self._summary_cache

    def experiments(self) -> dict:
        hw_path = ARTIFACTS_DIR / "hardware_benchmark.json"
        sum_path = EXPERIMENTS_DIR / "summary.json"
        csv_path = EXPERIMENTS_DIR / "results.csv"

        hw = json.loads(hw_path.read_text()) if hw_path.exists() else None
        summary = json.loads(sum_path.read_text()) if sum_path.exists() else None
        rows = pd.read_csv(csv_path).to_dict(orient="records") if csv_path.exists() else []
        return {
            "hardware_benchmark": hw,
            "summary": summary,
            "results": rows,
        }
