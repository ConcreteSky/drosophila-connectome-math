"""Batched PyTorch sparse CSR Leaky Integrate-and-Fire (LIF) brain simulator."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch

from flybrain_rl.config import LIFConfig
from flybrain_rl.connectome import ConnectomeGraph


class SpikingFlyBrain:
    """Batched LIF spiking neural network on CUDA or CPU with Dale's Law & lesions."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        config: Optional[LIFConfig] = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.graph = graph
        self.cfg = config or LIFConfig()
        self.num_neurons = graph.num_neurons
        self.num_edges = graph.num_edges

        self.alpha = math.exp(-self.cfg.dt_ms / self.cfg.tau_mem_ms)
        self.beta = math.exp(-self.cfg.dt_ms / self.cfg.tau_syn_ms)
        self.gamma_adapt = math.exp(-self.cfg.dt_ms / self.cfg.tau_adapt_ms)

        self.crow_indices = graph.crow_indices.to(self.device)
        self.col_indices = graph.col_indices.to(self.device)
        self.values = (graph.values * self.cfg.weight_scale).to(self.device)
        self.initial_values = self.values.clone()
        self.edge_signs = torch.sign(self.initial_values)

        self.region_names = list(graph.region_names)
        self.region_ids = graph.region_ids.to(self.device)

        self.W = torch.sparse_csr_tensor(
            self.crow_indices,
            self.col_indices,
            self.values,
            size=(self.num_neurons, self.num_neurons),
            device=self.device,
        )

        self.active_mask = torch.ones(self.num_neurons, dtype=torch.float32, device=self.device)
        self.lesioned_regions: set[str] = set()
        self.hub_lesion_active: bool = False

        self.v: torch.Tensor
        self.i_syn: torch.Tensor
        self.refrac: torch.Tensor
        self.last_spikes: torch.Tensor
        self.theta_adapt: torch.Tensor
        self.reset_state(batch_size=1)

    def rebuild_csr(self) -> None:
        self.W = torch.sparse_csr_tensor(
            self.crow_indices,
            self.col_indices,
            self.values,
            size=(self.num_neurons, self.num_neurons),
            device=self.device,
        )

    def reset_weights(self, new_graph: Optional[ConnectomeGraph] = None) -> None:
        """Restore initial weights or swap in a new ConnectomeGraph (e.g. shuffled)."""
        if new_graph is not None:
            self.graph = new_graph
            self.crow_indices = new_graph.crow_indices.to(self.device)
            self.col_indices = new_graph.col_indices.to(self.device)
            self.values = (new_graph.values * self.cfg.weight_scale).to(self.device)
            self.initial_values = self.values.clone()
            self.edge_signs = torch.sign(self.initial_values)
        else:
            self.values.copy_(self.initial_values)
        self.rebuild_csr()

    def reset_state(self, batch_size: int = 1) -> None:
        if (
            hasattr(self, "v")
            and self.v.shape == (batch_size, self.num_neurons)
            and self.v.device == self.device
        ):
            self.v.fill_(self.cfg.v_rest_mv)
            self.i_syn.zero_()
            self.refrac.zero_()
            self.last_spikes.zero_()
            self.theta_adapt.zero_()
        else:
            self.v = torch.full(
                (batch_size, self.num_neurons),
                self.cfg.v_rest_mv,
                dtype=torch.float32,
                device=self.device,
            )
            self.i_syn = torch.zeros(
                (batch_size, self.num_neurons), dtype=torch.float32, device=self.device
            )
            self.refrac = torch.zeros(
                (batch_size, self.num_neurons), dtype=torch.int16, device=self.device
            )
            self.last_spikes = torch.zeros(
                (batch_size, self.num_neurons), dtype=torch.float32, device=self.device
            )
            self.theta_adapt = torch.zeros(
                (batch_size, self.num_neurons), dtype=torch.float32, device=self.device
            )

    def get_region_indices(self, region_name: str) -> torch.Tensor:
        if region_name not in self.region_names:
            return torch.empty(0, dtype=torch.long, device=self.device)
        rid = self.region_names.index(region_name)
        return torch.nonzero(self.region_ids == rid, as_tuple=True)[0]

    def get_top_out_degree_indices(
        self, region_name: Optional[str] = None, top_k: Optional[int] = None, fraction: Optional[float] = None
    ) -> torch.Tensor:
        out_deg = torch.bincount(self.col_indices, minlength=self.num_neurons)
        idx = self.get_region_indices(region_name) if region_name else torch.arange(self.num_neurons, device=self.device)
        if len(idx) == 0:
            return idx
        if fraction is not None:
            k = max(1, int(round(len(idx) * fraction)))
        elif top_k is not None:
            k = min(len(idx), top_k)
        else:
            k = len(idx)
        _, order = torch.topk(out_deg[idx], k=k)
        return idx[order]

    def set_lesion(
        self,
        regions: Optional[Iterable[str]] = None,
        hub_top_5pct: bool = False,
    ) -> int:
        """Silence specified regions and/or top 5% out-degree hub neurons."""
        self.active_mask.fill_(1.0)
        self.lesioned_regions.clear()
        self.hub_lesion_active = hub_top_5pct

        if regions:
            for r in regions:
                if r in self.region_names:
                    rid = self.region_names.index(r)
                    self.active_mask[self.region_ids == rid] = 0.0
                    self.lesioned_regions.add(r)

        if hub_top_5pct:
            hubs = self.get_top_out_degree_indices(fraction=0.05)
            self.active_mask[hubs] = 0.0

        return int((self.active_mask == 0).sum().item())

    @torch.no_grad()
    def step(self, external_current: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Advance LIF simulation by dt = 1 ms for batch [B, N]."""
        B = self.v.shape[0]
        if B == 1:
            rec = torch.mv(self.W, self.last_spikes[0]).unsqueeze(0)
        else:
            rec = torch.mm(self.W, self.last_spikes.t().contiguous()).t()

        self.i_syn.mul_(self.beta).add_(rec)
        if external_current is not None:
            if external_current.dim() == 1:
                self.i_syn.add_(external_current.unsqueeze(0))
            else:
                self.i_syn.add_(external_current)

        not_refrac = self.refrac == 0
        drive = self.cfg.v_rest_mv + self.cfg.r_mem * self.i_syn
        v_next = self.alpha * self.v + (1.0 - self.alpha) * drive
        self.v = torch.where(not_refrac, v_next, self.v)

        active = self.active_mask.unsqueeze(0)
        # Lesioned neurons stay clamped at rest and cannot spike
        self.v = torch.where(active > 0, self.v, torch.full_like(self.v, self.cfg.v_rest_mv))

        eff_thresh = self.cfg.v_thresh_mv + self.theta_adapt
        fired = (self.v >= eff_thresh) & not_refrac & (active > 0)
        spikes = fired.to(torch.float32)

        self.theta_adapt.mul_(self.gamma_adapt).add_(spikes * self.cfg.adapt_inc_mv)
        self.v = torch.where(fired, torch.full_like(self.v, self.cfg.v_reset_mv), self.v)
        self.refrac = torch.where(
            fired,
            torch.full_like(self.refrac, self.cfg.refrac_steps),
            torch.clamp(self.refrac - 1, min=0),
        )
        self.last_spikes = spikes
        return spikes

    @torch.no_grad()
    def run(
        self,
        external_currents: torch.Tensor,
        record_regions: bool = False,
        landmark_indices: Optional[torch.Tensor] = None,
        readout_window: Optional[tuple[int, int]] = None,
    ) -> dict:
        """Run T steps on external_currents [T, N] or [T, B, N]."""
        if external_currents.dim() == 2:
            T, B = external_currents.shape[0], 1
        else:
            T, B, _ = external_currents.shape

        self.reset_state(batch_size=B)
        total_counts = torch.zeros((B, self.num_neurons), dtype=torch.float32, device=self.device)
        readout_counts = torch.zeros((B, self.num_neurons), dtype=torch.float32, device=self.device)
        r_start, r_end = readout_window if readout_window is not None else (0, T)

        region_activity = {r: [] for r in self.region_names} if record_regions else None
        spike_events: list[list[int]] = []
        lm_dev = landmark_indices.to(self.device) if landmark_indices is not None else None

        for t in range(T):
            spikes = self.step(external_currents[t])
            total_counts += spikes
            if r_start <= t < r_end:
                readout_counts += spikes
            if record_regions and region_activity is not None:
                s0 = spikes[0]
                for rid, rname in enumerate(self.region_names):
                    region_activity[rname].append(float(s0[self.region_ids == rid].sum().item()))
            if lm_dev is not None:
                active_lm = torch.nonzero(spikes[0, lm_dev] > 0, as_tuple=True)[0].cpu().tolist()
                for lm_idx in active_lm:
                    spike_events.append([int(lm_idx), t])

        return {
            "spike_counts": total_counts,
            "readout_counts": readout_counts,
            "total_spikes": int(total_counts.sum().item()),
            "mean_spikes": float(total_counts.sum().item() / max(1, B)),
            "region_activity": region_activity,
            "spike_events": spike_events,
        }
