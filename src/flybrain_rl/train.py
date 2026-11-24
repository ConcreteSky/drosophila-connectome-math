"""Dual plasticity engines (3-factor R-STDP & REINFORCE) + PyTorch GRU baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from flybrain_rl.brain import SpikingFlyBrain
from flybrain_rl.config import TrainingConfig
from flybrain_rl.tasks import MathProblem, MathTaskEnv, TaskName


@dataclass
class TrialOutput:
    task: TaskName
    label: str
    predicted: int
    true_answer: int
    is_correct: bool
    output_spike_counts: list[float]
    total_spikes: int
    region_activity: Optional[dict[str, list[float]]] = None
    spike_events: Optional[list[list[int]]] = None


class SpikingPolicyBase:
    """Shared MB/CX/Motor population readout and plastic CSR edge indexing."""

    def __init__(
        self,
        brain: SpikingFlyBrain,
        env: MathTaskEnv,
        cfg: Optional[TrainingConfig] = None,
        seed: int = 42,
    ) -> None:
        self.brain = brain
        self.env = env
        self.cfg = cfg or TrainingConfig()
        self.device = brain.device
        self.num_classes = env.num_classes
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

        mb_all = brain.get_region_indices("Mushroom Body").cpu()
        cx_all = brain.get_region_indices("Central Complex").cpu()
        other_all = brain.get_region_indices("Lateral Horn").cpu()

        def _sample(idx: torch.Tensor, k: int) -> torch.Tensor:
            if len(idx) == 0:
                return torch.empty(0, dtype=torch.long, device=self.device)
            perm = idx[torch.randperm(len(idx), generator=self.rng)[: min(k, len(idx))]]
            return perm.to(self.device)

        self.mb_sample = _sample(mb_all, 512)
        self.cx_sample = _sample(cx_all, 256)
        self.lh_sample = _sample(other_all, 128)

        pop_dim = len(self.mb_sample) + len(self.cx_sample) + len(self.lh_sample)
        self.raw_dim = pop_dim * 2 + self.num_classes
        self.kc_dim = 256
        kc_gen = torch.Generator(device="cpu").manual_seed(seed + 101)
        self.W_kc = (
            torch.randn((self.kc_dim, self.raw_dim), generator=kc_gen)
            / max(1.0, self.raw_dim ** 0.5)
        ).to(self.device)
        self.feat_dim = self.raw_dim + self.kc_dim

        self.W_out = torch.zeros((self.num_classes, self.feat_dim), dtype=torch.float32, device=self.device)
        self.b_out = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)
        self.r_bar = 0.0
        self.total_updates = 0

        # Plastic synapses: Mushroom Body & Motor/Output postsynaptic targets
        mb_id = brain.region_names.index("Mushroom Body") if "Mushroom Body" in brain.region_names else -1
        mot_id = brain.region_names.index("Motor/Output") if "Motor/Output" in brain.region_names else -1
        plastic_post = (brain.region_ids == mb_id) | (brain.region_ids == mot_id)
        row_lengths = brain.crow_indices[1:] - brain.crow_indices[:-1]
        plastic_mask = torch.repeat_interleave(plastic_post, row_lengths)
        self.plastic_edge_idx = torch.nonzero(plastic_mask, as_tuple=True)[0]
        post_all = torch.repeat_interleave(
            torch.arange(brain.num_neurons, device=self.device, dtype=torch.long),
            row_lengths,
        )
        self.plastic_post = post_all[self.plastic_edge_idx]
        self.plastic_pre = brain.col_indices[self.plastic_edge_idx]

        # Cached trial state for interactive POST /api/brain/feedback
        self._cached_feat: Optional[torch.Tensor] = None
        self._cached_action: Optional[int] = None
        self._cached_probs: Optional[torch.Tensor] = None
        self._cached_elig_conn: Optional[torch.Tensor] = None

    def extract_features(self, total_spikes: torch.Tensor, readout_spikes: torch.Tensor) -> torch.Tensor:
        sched = self.env.schedule
        r_len = max(1, sched.readout_end - sched.readout_start)
        e_len = max(1, sched.total_ms - r_len)
        early_spikes = total_spikes - readout_spikes

        parts = []
        for pop in (self.mb_sample, self.cx_sample, self.lh_sample):
            if len(pop) > 0:
                parts.append(early_spikes[:, pop] / e_len)
                parts.append(readout_spikes[:, pop] / r_len)

        motor_counts = self.env.encoder.decode_counts(readout_spikes, self.num_classes)
        parts.append(motor_counts / r_len)

        raw = torch.cat(parts, dim=1)
        raw = raw / raw.norm(dim=1, keepdim=True).clamp(min=1e-6)
        kc = torch.relu(raw @ self.W_kc.t())
        kc = kc / kc.norm(dim=1, keepdim=True).clamp(min=1e-6)
        feat = torch.cat([raw, kc], dim=1)
        return feat / feat.norm(dim=1, keepdim=True).clamp(min=1e-6)


class RSTDPTrainer(SpikingPolicyBase):
    """Three-factor dopamine R-STDP with decaying eligibility traces & Dale's Law."""

    def _run_with_traces(
        self,
        ext: torch.Tensor,
        record_regions: bool = False,
        landmark_indices: Optional[torch.Tensor] = None,
        track_traces: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor], Optional[dict], list]:
        T, B, N = ext.shape
        sched = self.env.schedule
        self.brain.reset_state(batch_size=B)

        total_counts = torch.zeros((B, N), dtype=torch.float32, device=self.device)
        readout_counts = torch.zeros((B, N), dtype=torch.float32, device=self.device)

        use_conn = track_traces and self.cfg.conn_lr_rstdp > 0 and len(self.plastic_edge_idx) > 0
        if use_conn:
            x_pre = torch.zeros(N, dtype=torch.float32, device=self.device)
            y_post = torch.zeros(N, dtype=torch.float32, device=self.device)
            elig_conn = torch.zeros(len(self.plastic_edge_idx), dtype=torch.float32, device=self.device)
        else:
            elig_conn = None

        region_activity = {r: [] for r in self.brain.region_names} if record_regions else None
        spike_events: list[list[int]] = []
        lm_dev = landmark_indices.to(self.device) if landmark_indices is not None else None

        for t in range(T):
            spikes = self.brain.step(ext[t])
            total_counts += spikes
            if sched.readout_start <= t < sched.readout_end:
                readout_counts += spikes

            if use_conn:
                s_mean = spikes.mean(dim=0)
                x_pre = 0.9 * x_pre + s_mean
                y_post = 0.9 * y_post + s_mean
                ltp = self.cfg.a_plus * x_pre[self.plastic_pre] * s_mean[self.plastic_post]
                ltd = self.cfg.a_minus * s_mean[self.plastic_pre] * y_post[self.plastic_post]
                elig_conn = self.cfg.lambda_e * elig_conn + (ltp - ltd)

            if record_regions and region_activity is not None:
                s0 = spikes[0]
                for rid, rname in enumerate(self.brain.region_names):
                    region_activity[rname].append(float(s0[self.brain.region_ids == rid].sum().item()))

            if lm_dev is not None:
                active_lm = torch.nonzero(spikes[0, lm_dev] > 0, as_tuple=True)[0].cpu().tolist()
                for lm_idx in active_lm:
                    spike_events.append([int(lm_idx), t])

        mean_spikes = float(total_counts.sum().item() / max(1, B))
        return total_counts, readout_counts, mean_spikes, elig_conn, region_activity, spike_events

    def _clamp_dales_law(self) -> None:
        w_max = self.cfg.w_max
        self.brain.values = torch.where(
            self.brain.edge_signs > 0,
            self.brain.values.clamp(min=0.0, max=w_max),
            self.brain.values.clamp(min=-w_max, max=0.0),
        )
        self.brain.rebuild_csr()

    def solve_single(
        self,
        problem: MathProblem,
        landmark_indices: Optional[torch.Tensor] = None,
    ) -> TrialOutput:
        a_t = torch.tensor([problem.a_vec], dtype=torch.float32)
        b_t = torch.tensor([problem.b_vec], dtype=torch.float32)
        ext = self.env.encoder.encode_batch(a_t, b_t, self.env.schedule)
        tot, ro, _, elig_conn, reg_act, spk_ev = self._run_with_traces(
            ext, record_regions=True, landmark_indices=landmark_indices, track_traces=True
        )
        feat = self.extract_features(tot, ro)[0]
        logits = self.W_out @ feat + self.b_out
        probs = torch.softmax(logits / self.cfg.temperature, dim=0)
        pred = int(torch.argmax(probs).item())
        out_counts = self.env.encoder.decode_counts(ro, self.num_classes)[0].cpu().tolist()

        self._cached_feat = feat
        self._cached_action = pred
        self._cached_probs = probs
        self._cached_elig_conn = elig_conn

        return TrialOutput(
            task=problem.task,
            label=problem.label,
            predicted=pred,
            true_answer=problem.target,
            is_correct=(pred == problem.target),
            output_spike_counts=[round(float(x), 3) for x in out_counts],
            total_spikes=int(tot.sum().item()),
            region_activity=reg_act,
            spike_events=spk_ev,
        )

    def apply_feedback(self, reward: float) -> float:
        if self._cached_feat is None or self._cached_action is None or self._cached_probs is None:
            return 0.0
        adv = reward - self.r_bar
        self.r_bar = (1.0 - self.cfg.reward_baseline_alpha) * self.r_bar + self.cfg.reward_baseline_alpha * reward
        if abs(adv) < 1e-6:
            adv = reward

        one_hot = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)
        one_hot[self._cached_action] = 1.0
        dW = self.cfg.lr_rstdp * adv * torch.outer(one_hot - self._cached_probs, self._cached_feat)
        db = 0.2 * self.cfg.lr_rstdp * adv * (one_hot - self._cached_probs)
        old_W = self.W_out.clone()
        self.W_out = (self.W_out + dW).clamp(-self.cfg.w_max, self.cfg.w_max)
        self.b_out = (self.b_out + db).clamp(-self.cfg.w_max, self.cfg.w_max)

        if self._cached_elig_conn is not None and self.cfg.conn_lr_rstdp > 0:
            self.brain.values[self.plastic_edge_idx] += (
                self.cfg.conn_lr_rstdp
                * adv
                * self._cached_elig_conn
                * self.brain.edge_signs[self.plastic_edge_idx]
            )
            self._clamp_dales_law()

        self.total_updates += 1
        return float((self.W_out - old_W).norm().item())

    def train_step(self, batch_size: int = 32) -> dict:
        _, a_vecs, b_vecs, targets = self.env.sample_batch("train", batch_size=batch_size, rng=self.rng)
        ext = self.env.encoder.encode_batch(a_vecs, b_vecs, self.env.schedule)
        tot, ro, mean_spk, elig_conn, _, _ = self._run_with_traces(ext, track_traces=True)

        feats = self.extract_features(tot, ro)
        logits = feats @ self.W_out.t() + self.b_out
        probs = torch.softmax(logits / self.cfg.temperature, dim=1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(1)

        rewards = MathTaskEnv.compute_reward(actions, targets)
        mean_r = float(rewards.mean().item())
        adv = rewards - self.r_bar
        self.r_bar = (1.0 - self.cfg.reward_baseline_alpha) * self.r_bar + self.cfg.reward_baseline_alpha * mean_r

        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, actions.unsqueeze(1), 1.0)
        diff = (one_hot - probs) * adv.unsqueeze(1)
        self.W_out = (self.W_out + self.cfg.lr_rstdp * (diff.t() @ feats) / batch_size).clamp(
            -self.cfg.w_max, self.cfg.w_max
        )
        self.b_out = (self.b_out + 0.2 * self.cfg.lr_rstdp * diff.mean(dim=0)).clamp(
            -self.cfg.w_max, self.cfg.w_max
        )

        if elig_conn is not None and self.cfg.conn_lr_rstdp > 0:
            mean_adv = float(adv.mean().item())
            self.brain.values[self.plastic_edge_idx] += (
                self.cfg.conn_lr_rstdp * mean_adv * elig_conn * self.brain.edge_signs[self.plastic_edge_idx]
            )
            self._clamp_dales_law()

        self.total_updates += 1
        return {
            "train_acc": float((actions == targets).float().mean().item()),
            "mean_reward": mean_r,
            "mean_spikes": mean_spk,
            "last_pred": int(actions[0].item()),
            "last_target": int(targets[0].item()),
        }

    def evaluate(self, split: Literal["train", "test"] = "test", batch_size: int = 32) -> dict:
        _, a_vecs, b_vecs, targets = self.env.sample_batch(split, batch_size=batch_size, rng=self.rng)
        ext = self.env.encoder.encode_batch(a_vecs, b_vecs, self.env.schedule)
        tot, ro, mean_spk, _, _, _ = self._run_with_traces(ext, track_traces=False)
        feats = self.extract_features(tot, ro)
        preds = torch.argmax(feats @ self.W_out.t() + self.b_out, dim=1)
        rewards = MathTaskEnv.compute_reward(preds, targets)
        return {
            "accuracy": float((preds == targets).float().mean().item()),
            "mean_reward": float(rewards.mean().item()),
            "mean_spikes": mean_spk,
        }


class REINFORCETrainer(SpikingPolicyBase):
    """Spiking Policy Gradient (REINFORCE) baseline."""

    def __init__(
        self,
        brain: SpikingFlyBrain,
        env: MathTaskEnv,
        cfg: Optional[TrainingConfig] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(brain, env, cfg=cfg, seed=seed)
        self.policy = nn.Linear(self.feat_dim, self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr_reinforce)

    def train_step(self, batch_size: int = 32) -> dict:
        _, a_vecs, b_vecs, targets = self.env.sample_batch("train", batch_size=batch_size, rng=self.rng)
        ext = self.env.encoder.encode_batch(a_vecs, b_vecs, self.env.schedule)
        sched = self.env.schedule
        out = self.brain.run(ext, readout_window=(sched.readout_start, sched.readout_end))
        feats = self.extract_features(out["spike_counts"], out["readout_counts"])
        logits = self.policy(feats)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()

        rewards = MathTaskEnv.compute_reward(actions, targets)
        mean_r = float(rewards.mean().item())
        adv = rewards - self.r_bar
        self.r_bar = (1.0 - self.cfg.reward_baseline_alpha) * self.r_bar + self.cfg.reward_baseline_alpha * mean_r

        loss = -(dist.log_prob(actions) * adv).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_updates += 1

        return {
            "train_acc": float((actions == targets).float().mean().item()),
            "mean_reward": mean_r,
            "mean_spikes": out["mean_spikes"],
        }

    @torch.no_grad()
    def evaluate(self, split: Literal["train", "test"] = "test", batch_size: int = 32) -> dict:
        _, a_vecs, b_vecs, targets = self.env.sample_batch(split, batch_size=batch_size, rng=self.rng)
        ext = self.env.encoder.encode_batch(a_vecs, b_vecs, self.env.schedule)
        sched = self.env.schedule
        out = self.brain.run(ext, readout_window=(sched.readout_start, sched.readout_end))
        feats = self.extract_features(out["spike_counts"], out["readout_counts"])
        preds = torch.argmax(self.policy(feats), dim=1)
        rewards = MathTaskEnv.compute_reward(preds, targets)
        return {
            "accuracy": float((preds == targets).float().mean().item()),
            "mean_reward": float(rewards.mean().item()),
            "mean_spikes": out["mean_spikes"],
        }


class GRUBaselineTrainer:
    """Matched-budget non-spiking PyTorch GRU baseline."""

    def __init__(self, env: MathTaskEnv, hidden_size: int = 64, lr: float = 0.01, seed: int = 42) -> None:
        self.env = env
        self.device = env.device
        self.num_classes = env.num_classes
        torch.manual_seed(seed)
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

        in_dim = env.encoder.num_a_channels + env.encoder.num_b_channels + 1
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_size, batch_first=True).to(self.device)
        self.head = nn.Linear(hidden_size, self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.gru.parameters()) + list(self.head.parameters()), lr=lr
        )
        self.r_bar = 0.0

    def _encode_seq(self, a_vecs: torch.Tensor, b_vecs: torch.Tensor) -> torch.Tensor:
        B = a_vecs.shape[0]
        sched = self.env.schedule
        ca, cb = self.env.encoder.num_a_channels, self.env.encoder.num_b_channels
        seq = torch.zeros((B, sched.total_ms, ca + cb + 1), dtype=torch.float32, device=self.device)
        seq[:, sched.a_start : sched.a_end, :ca] = a_vecs.to(self.device).unsqueeze(1)
        seq[:, sched.b_start : sched.b_end, ca : ca + cb] = b_vecs.to(self.device).unsqueeze(1)
        seq[:, sched.readout_start : sched.readout_end, ca + cb] = 1.0
        return seq

    def train_step(self, batch_size: int = 32) -> dict:
        _, a_vecs, b_vecs, targets = self.env.sample_batch("train", batch_size=batch_size, rng=self.rng)
        seq = self._encode_seq(a_vecs, b_vecs)
        out, _ = self.gru(seq)
        dist = torch.distributions.Categorical(logits=self.head(out[:, -1, :]))
        actions = dist.sample()

        rewards = MathTaskEnv.compute_reward(actions, targets)
        mean_r = float(rewards.mean().item())
        adv = rewards - self.r_bar
        self.r_bar = 0.95 * self.r_bar + 0.05 * mean_r

        loss = -(dist.log_prob(actions) * adv).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "train_acc": float((actions == targets).float().mean().item()),
            "mean_reward": mean_r,
            "mean_spikes": 0.0,
        }

    @torch.no_grad()
    def evaluate(self, split: Literal["train", "test"] = "test", batch_size: int = 32) -> dict:
        _, a_vecs, b_vecs, targets = self.env.sample_batch(split, batch_size=batch_size, rng=self.rng)
        seq = self._encode_seq(a_vecs, b_vecs)
        out, _ = self.gru(seq)
        preds = torch.argmax(self.head(out[:, -1, :]), dim=1)
        rewards = MathTaskEnv.compute_reward(preds, targets)
        return {
            "accuracy": float((preds == targets).float().mean().item()),
            "mean_reward": float(rewards.mean().item()),
            "mean_spikes": 0.0,
        }
