"""Lightweight Vercel serverless FastAPI backend using NumPy + SciPy sparse CSR.

Zero PyTorch/CUDA imports to stay well under Vercel's 250 MB unzipped limit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field, model_validator

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
SERVERLESS_DIR = ARTIFACTS_DIR / "serverless"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"

TaskName = Literal["delayed_xor", "add_3bit", "sub_3bit", "mod_add"]


class SolveRequest(BaseModel):
    task: TaskName = "delayed_xor"
    a: int = Field(1, ge=0, le=7)
    b: int = Field(0, ge=0, le=7)

    @model_validator(mode="after")
    def validate_ranges(self) -> "SolveRequest":
        limits = {"delayed_xor": 1, "add_3bit": 7, "sub_3bit": 7, "mod_add": 6}
        mx = limits[self.task]
        if self.a > mx or self.b > mx:
            raise ValueError(f"For {self.task}, a and b must be between 0 and {mx}.")
        return self


class FeedbackRequest(BaseModel):
    reward: Literal[1, -1]


class TrainStepRequest(BaseModel):
    task: TaskName = "delayed_xor"
    episodes: int = Field(10, ge=1, le=50)


class LesionRequest(BaseModel):
    mushroom_body: bool = False
    central_complex: bool = False
    hub_top_5pct: bool = False


class ResetRequest(BaseModel):
    mode: Literal["checkpoint", "fresh", "shuffled"] = "checkpoint"


def _make_problem_vecs(task: TaskName, a: int, b: int) -> tuple[int, np.ndarray, np.ndarray]:
    a_vec = np.zeros(7, dtype=np.float32)
    b_vec = np.zeros(7, dtype=np.float32)
    if task == "delayed_xor":
        av, bv = int(a) & 1, int(b) & 1
        target = av ^ bv
        a_vec[0], a_vec[1] = float(av), float(1 - av)
        b_vec[0], b_vec[1] = float(bv), float(1 - bv)
        return target, a_vec, b_vec
    if task in ("add_3bit", "sub_3bit"):
        av, bv = max(0, min(7, int(a))), max(0, min(7, int(b)))
        target = (av + bv) if task == "add_3bit" else (av - bv + 7)
        for bit in range(3):
            ba, bb = (av >> bit) & 1, (bv >> bit) & 1
            a_vec[2 * bit], a_vec[2 * bit + 1] = float(ba), float(1 - ba)
            b_vec[2 * bit], b_vec[2 * bit + 1] = float(bb), float(1 - bb)
        return target, a_vec, b_vec
    av, bv = int(a) % 7, int(b) % 7
    target = (av + bv) % 7
    a_vec[av] = 1.0
    b_vec[bv] = 1.0
    return target, a_vec, b_vec


class ServerlessBrainEngine:
    """In-memory NumPy/SciPy sparse CSR LIF + 3-factor R-STDP engine."""

    def __init__(self) -> None:
        npz = np.load(SERVERLESS_DIR / "subnetwork.npz")
        meta = json.loads((SERVERLESS_DIR / "layout_and_summary.json").read_text())

        self.region_names: list[str] = meta["region_names"]
        self.landmarks: list[dict] = meta["landmarks"]
        self.summary: dict = meta["summary"]

        self.n = len(self.landmarks)
        self.W_indptr = npz["indptr"].astype(np.int32)
        self.W_indices_base = npz["indices"].astype(np.int32)
        self.W_data_base = npz["data"].astype(np.float32)
        self.W_row_idx = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.W_indptr))
        self.W_indices = self.W_indices_base.copy()
        self.W_data = self.W_data_base.copy()
        self.signs = npz["signs"]
        self.region_ids = npz["region_ids"]
        self.a_pools = npz["a_pools"]
        self.b_pools = npz["b_pools"]
        self.go_pool = npz["go_pool"]
        self.output_pools = npz["output_pools"]

        self.active_mask = np.ones(self.n, dtype=bool)
        self.alpha = math.exp(-1.0 / 20.0)
        self.beta = math.exp(-1.0 / 5.0)
        self.v_rest = -52.0
        self.v_thresh = -45.0

        # Per-task linear readout over [early_rates, readout_rates, motor_counts]
        self.mb_idx = np.where(self.region_ids == self.region_names.index("Mushroom Body"))[0]
        self.cx_idx = np.where(self.region_ids == self.region_names.index("Central Complex"))[0]
        self.feat_dim = (len(self.mb_idx) + len(self.cx_idx)) * 2 + 15
        rng = np.random.default_rng(42)
        self.W_kc = rng.normal(0, 1.0 / math.sqrt(max(1, self.feat_dim)), size=(128, self.feat_dim)).astype(np.float32)

        self.num_classes = {"delayed_xor": 2, "add_3bit": 15, "sub_3bit": 15, "mod_add": 7}
        self.W_out: dict[TaskName, np.ndarray] = {
            t: np.zeros((k, self.feat_dim + 128), dtype=np.float32)
            for t, k in self.num_classes.items()
        }
        self.r_bar: dict[TaskName, float] = {t: 0.0 for t in self.num_classes}
        self.total_updates = 0
        self.total_weight_shift = 0.0
        self.active_task: TaskName = "delayed_xor"
        self.cached_feat: Optional[np.ndarray] = None
        self.cached_action: Optional[int] = None
        self.cached_probs: Optional[np.ndarray] = None
        self.cached_post_spikes: Optional[np.ndarray] = None

    def _run_trial(self, task: TaskName, a_vec: np.ndarray, b_vec: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]], list[list[int]]]:
        T = 60
        a_end = 15 if task == "delayed_xor" else 30
        b_start, b_end = (30, 45) if task == "delayed_xor" else (0, 30)
        ro_start = 45 if task == "delayed_xor" else 30

        v = np.full(self.n, self.v_rest, dtype=np.float32)
        i_syn = np.zeros(self.n, dtype=np.float32)
        refrac = np.zeros(self.n, dtype=np.int16)
        last_spk = np.zeros(self.n, dtype=np.float32)

        tot_spk = np.zeros(self.n, dtype=np.float32)
        ro_spk = np.zeros(self.n, dtype=np.float32)
        reg_act = {r: [] for r in self.region_names}
        events: list[list[int]] = []

        for t in range(T):
            ext = np.zeros(self.n, dtype=np.float32)
            if t < a_end:
                for ch in np.where(a_vec > 0)[0]:
                    pool = self.a_pools[ch]
                    valid = pool[pool >= 0]
                    if len(valid):
                        mask = rng.random(len(valid)) < 0.8
                        ext[valid[mask]] += 55.0
            if b_start <= t < b_end:
                for ch in np.where(b_vec > 0)[0]:
                    pool = self.b_pools[ch]
                    valid = pool[pool >= 0]
                    if len(valid):
                        mask = rng.random(len(valid)) < 0.8
                        ext[valid[mask]] += 55.0
            if t >= ro_start and len(self.go_pool):
                mask = rng.random(len(self.go_pool)) < 0.7
                ext[self.go_pool[mask]] += 46.0

            rec = np.bincount(
                self.W_row_idx,
                weights=self.W_data * last_spk[self.W_indices],
                minlength=self.n,
            ).astype(np.float32)
            i_syn = self.beta * i_syn + rec + ext
            not_ref = refrac == 0
            v_next = self.alpha * v + (1.0 - self.alpha) * (self.v_rest + i_syn)
            v[not_ref] = v_next[not_ref]
            v[~self.active_mask] = self.v_rest

            fired = (v >= self.v_thresh) & not_ref & self.active_mask
            last_spk = fired.astype(np.float32)
            v[fired] = self.v_rest
            refrac = np.where(fired, 2, np.maximum(0, refrac - 1))

            tot_spk += last_spk
            if t >= ro_start:
                ro_spk += last_spk

            for rid, rname in enumerate(self.region_names):
                reg_act[rname].append(float(last_spk[self.region_ids == rid].sum()))
            for idx in np.where(fired)[0].tolist():
                events.append([int(idx), t])

        return tot_spk, ro_spk, self._extract_feat(tot_spk, ro_spk, ro_start), reg_act, events

    def _extract_feat(self, tot_spk: np.ndarray, ro_spk: np.ndarray, ro_start: int) -> np.ndarray:
        early = tot_spk - ro_spk
        r_len = max(1, 60 - ro_start)
        e_len = max(1, ro_start)
        m_counts = np.zeros(15, dtype=np.float32)
        for k in range(15):
            valid = self.output_pools[k]
            valid = valid[valid >= 0]
            if len(valid):
                m_counts[k] = ro_spk[valid].mean()
        raw = np.concatenate(
            [
                early[self.mb_idx] / e_len,
                ro_spk[self.mb_idx] / r_len,
                early[self.cx_idx] / e_len,
                ro_spk[self.cx_idx] / r_len,
                m_counts / r_len,
            ]
        )
        raw /= max(1e-6, float(np.linalg.norm(raw)))
        kc = np.maximum(0.0, self.W_kc @ raw)
        kc /= max(1e-6, float(np.linalg.norm(kc)))
        feat = np.concatenate([raw, kc])
        return feat / max(1e-6, float(np.linalg.norm(feat)))

    def solve(self, task: TaskName, a: int, b: int) -> dict:
        self.active_task = task
        t0 = time.perf_counter()
        target, a_vec, b_vec = _make_problem_vecs(task, a, b)
        # Deterministic trial seed per (task, a, b) so punishment/reward immediately alters behavior on re-test
        seed_val = (hash((task, int(a), int(b))) & 0x7FFFFFFF)
        rng = np.random.default_rng(seed_val)
        tot_spk, ro_spk, feat, reg_act, events = self._run_trial(task, a_vec, b_vec, rng)

        k_cls = self.num_classes[task]
        logits = self.W_out[task] @ feat
        exps = np.exp((logits - logits.max()) / 0.25)
        probs = exps / exps.sum()
        pred = int(np.argmax(probs))

        out_counts = []
        for k in range(k_cls):
            valid = self.output_pools[k]
            valid = valid[valid >= 0]
            out_counts.append(round(float(ro_spk[valid].mean()) if len(valid) else 0.0, 3))

        self.cached_feat = feat
        self.cached_action = pred
        self.cached_probs = probs
        self.cached_post_spikes = (ro_spk > 0).astype(np.float32)

        pred_out = pred - 7 if task == "sub_3bit" else pred
        true_out = target - 7 if task == "sub_3bit" else target
        labels = [str(i - 7) for i in range(15)] if task == "sub_3bit" else [str(i) for i in range(k_cls)]

        return {
            "predicted_answer": pred_out,
            "true_answer": true_out,
            "is_correct": pred == target,
            "answer_labels": labels,
            "output_spike_counts": out_counts,
            "total_spikes": int(tot_spk.sum()),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "region_activity": reg_act,
            "spike_events": events,
        }

    def feedback(self, reward: int) -> dict:
        if self.cached_feat is None or self.cached_action is None or self.cached_probs is None:
            raise HTTPException(status_code=400, detail="No cached trial to apply feedback to.")
        task = self.active_task
        adv = float(reward) - self.r_bar[task]
        self.r_bar[task] = 0.95 * self.r_bar[task] + 0.05 * float(reward)
        if abs(adv) < 0.25:
            adv = float(reward)

        one_hot = np.zeros_like(self.cached_probs)
        one_hot[self.cached_action] = 1.0
        # Strong 3-factor update so 1 punishment visibly suppresses the wrong answer
        dW = 1.5 * adv * np.outer(one_hot - self.cached_probs, self.cached_feat)
        old_W = self.W_out[task].copy()
        self.W_out[task] = np.clip(self.W_out[task] + dW, -5.0, 5.0)
        delta_norm = float(np.linalg.norm(self.W_out[task] - old_W))

        # Also modulate active connectome synapses while preserving Dale's Law
        if self.cached_post_spikes is not None:
            active_edges = self.cached_post_spikes[self.W_row_idx] > 0
            edge_signs = np.sign(self.W_data_base[active_edges])
            self.W_data[active_edges] += 0.01 * adv * edge_signs
            pos_mask = self.W_data_base >= 0
            self.W_data[pos_mask] = np.clip(self.W_data[pos_mask], 0.0, 5.0)
            self.W_data[~pos_mask] = np.clip(self.W_data[~pos_mask], -5.0, 0.0)

        self.total_updates += 1
        self.total_weight_shift += delta_norm
        return {
            "reward_applied": int(reward),
            "weight_delta_norm": round(delta_norm, 6),
            "total_weight_shift": round(self.total_weight_shift, 4),
            "dopamine_pulse": "reward" if reward > 0 else "punish",
            "total_updates": self.total_updates,
        }

    def train_step(self, task: TaskName, episodes: int) -> dict:
        rng = np.random.default_rng()
        correct = 0
        for _ in range(episodes):
            a = int(rng.integers(0, 4 if task == "delayed_xor" else 7))
            b = int(rng.integers(0, 2 if task == "delayed_xor" else 7))
            res = self.solve(task, a, b)
            self.feedback(1 if res["is_correct"] else -1)
            correct += int(res["is_correct"])
        return {
            "task": task,
            "episodes": episodes,
            "accuracy": round(correct / episodes, 4),
            "total_weight_shift": round(self.total_weight_shift, 4),
        }

    def set_lesion(self, mb: bool, cx: bool, hubs: bool) -> dict:
        self.active_mask.fill(True)
        if mb and "Mushroom Body" in self.region_names:
            self.active_mask[self.region_ids == self.region_names.index("Mushroom Body")] = False
        if cx and "Central Complex" in self.region_names:
            self.active_mask[self.region_ids == self.region_names.index("Central Complex")] = False
        if hubs:
            deg = np.bincount(self.W_indices, minlength=self.n)
            top_k = max(1, int(round(self.n * 0.05)))
            self.active_mask[np.argsort(deg)[-top_k:]] = False
        return {
            "mushroom_body": mb,
            "central_complex": cx,
            "hub_top_5pct": hubs,
            "disabled_neurons": int((~self.active_mask).sum()),
        }

    def reset(self, mode: Literal["checkpoint", "fresh", "shuffled"]) -> dict:
        if mode == "shuffled":
            rng = np.random.default_rng(123)
            perm = rng.permutation(len(self.W_data_base))
            self.W_indices = self.W_indices_base[perm]
            self.W_data = self.W_data_base[perm]
        else:
            self.W_indices = self.W_indices_base.copy()
            self.W_data = self.W_data_base.copy()
        for t in self.W_out:
            self.W_out[t].fill(0.0)
            self.r_bar[t] = 0.0
        self.total_weight_shift = 0.0
        self.active_mask.fill(True)
        return {"status": "reset", "mode": mode}


app = FastAPI(title="Drosophila Connectome Math RL Serverless API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: Optional[ServerlessBrainEngine] = None


def get_engine() -> ServerlessBrainEngine:
    global _engine
    if _engine is None:
        _engine = ServerlessBrainEngine()
    return _engine


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": "serverless_cpu",
        "gpu_name": "serverless_cpu",
        "gpu_mem_allocated_mb": 0.0,
        "gpu_mem_total_mb": 0.0,
        "checkpoint_loaded": True,
    }


@app.get("/api/brain/layout")
def brain_layout() -> dict:
    e = get_engine()
    return {"region_names": e.region_names, "landmarks": e.landmarks}


@app.post("/api/brain/solve")
def solve(req: SolveRequest) -> dict:
    return get_engine().solve(req.task, req.a, req.b)


@app.post("/api/brain/feedback")
def feedback(req: FeedbackRequest) -> dict:
    return get_engine().feedback(req.reward)


@app.post("/api/brain/train-step")
def train_step(req: TrainStepRequest) -> dict:
    return get_engine().train_step(req.task, req.episodes)


@app.post("/api/brain/lesion")
def lesion(req: LesionRequest) -> dict:
    return get_engine().set_lesion(req.mushroom_body, req.central_complex, req.hub_top_5pct)


@app.post("/api/brain/reset")
def reset(req: ResetRequest) -> dict:
    return get_engine().reset(req.mode)


@app.get("/api/connectome/summary")
def connectome_summary() -> dict:
    return get_engine().summary


@app.get("/api/experiments")
def experiments() -> dict:
    hw_path = ARTIFACTS_DIR / "hardware_benchmark.json"
    sum_path = EXPERIMENTS_DIR / "summary.json"
    return {
        "hardware_benchmark": json.loads(hw_path.read_text()) if hw_path.exists() else None,
        "summary": json.loads(sum_path.read_text()) if sum_path.exists() else None,
        "results": [],
    }
