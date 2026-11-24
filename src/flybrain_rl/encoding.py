"""Biological population spike encoder and motor/MBON decision decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class TrialSchedule:
    total_ms: int = 60
    a_start: int = 0
    a_end: int = 15
    b_start: int = 30
    b_end: int = 45
    readout_start: int = 45
    readout_end: int = 60


class PopulationEncoderDecoder:
    """Non-overlapping sensory input pools and K output decision pools."""

    def __init__(
        self,
        num_neurons: int,
        sensory_indices: torch.Tensor,
        output_indices: torch.Tensor,
        device: torch.device,
        num_a_channels: int = 7,
        num_b_channels: int = 7,
        neurons_per_pool: int = 64,
        max_classes: int = 15,
        seed: int = 42,
    ) -> None:
        self.num_neurons = num_neurons
        self.device = device
        self.num_a_channels = num_a_channels
        self.num_b_channels = num_b_channels
        self.max_classes = max_classes

        rng = torch.Generator(device="cpu").manual_seed(seed)
        total_in = num_a_channels + num_b_channels + 1  # +1 timing/go cue
        sens_cpu = sensory_indices.cpu()
        perm_s = sens_cpu[torch.randperm(len(sens_cpu), generator=rng)]
        eff_in = max(1, min(neurons_per_pool, len(perm_s) // total_in))
        self.in_pool_size = eff_in

        self.a_pools: list[torch.Tensor] = []
        self.b_pools: list[torch.Tensor] = []
        off = 0
        for _ in range(num_a_channels):
            self.a_pools.append(perm_s[off : off + eff_in].to(device))
            off += eff_in
        for _ in range(num_b_channels):
            self.b_pools.append(perm_s[off : off + eff_in].to(device))
            off += eff_in
        self.go_pool = perm_s[off : off + eff_in].to(device)

        out_cpu = output_indices.cpu()
        perm_o = out_cpu[torch.randperm(len(out_cpu), generator=rng)]
        eff_out = max(1, min(neurons_per_pool, len(perm_o) // max_classes))
        self.out_pool_size = eff_out
        self.output_pools: list[torch.Tensor] = [
            perm_o[c * eff_out : (c + 1) * eff_out].to(device)
            for c in range(max_classes)
        ]

    def encode_batch(
        self,
        a_vecs: torch.Tensor,
        b_vecs: torch.Tensor,
        schedule: TrialSchedule,
        drive_amp: float = 55.0,
        rate_prob: float = 0.80,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Encode [B, C] operand features into [T, B, N] sensory currents."""
        B = a_vecs.shape[0]
        T = schedule.total_ms
        ext = torch.zeros((T, B, self.num_neurons), dtype=torch.float32, device=self.device)

        t_a = schedule.a_end - schedule.a_start
        for ch in range(min(self.num_a_channels, a_vecs.shape[1])):
            active_b = torch.nonzero(a_vecs[:, ch] > 0.0, as_tuple=True)[0]
            if len(active_b) == 0:
                continue
            pool = self.a_pools[ch]
            probs = (a_vecs[active_b, ch].to(self.device) * rate_prob).clamp(0.0, 0.95)
            u = torch.rand((t_a, len(active_b), len(pool)), generator=rng, device=self.device)
            spikes = (u < probs.view(1, -1, 1)).to(torch.float32) * drive_amp
            ext[schedule.a_start : schedule.a_end, active_b.unsqueeze(-1), pool.unsqueeze(0)] += spikes

        t_b = schedule.b_end - schedule.b_start
        for ch in range(min(self.num_b_channels, b_vecs.shape[1])):
            active_b = torch.nonzero(b_vecs[:, ch] > 0.0, as_tuple=True)[0]
            if len(active_b) == 0:
                continue
            pool = self.b_pools[ch]
            probs = (b_vecs[active_b, ch].to(self.device) * rate_prob).clamp(0.0, 0.95)
            u = torch.rand((t_b, len(active_b), len(pool)), generator=rng, device=self.device)
            spikes = (u < probs.view(1, -1, 1)).to(torch.float32) * drive_amp
            ext[schedule.b_start : schedule.b_end, active_b.unsqueeze(-1), pool.unsqueeze(0)] += spikes

        t_g = schedule.readout_end - schedule.readout_start
        if len(self.go_pool) > 0 and t_g > 0:
            u_g = torch.rand((t_g, B, len(self.go_pool)), generator=rng, device=self.device)
            ext[schedule.readout_start : schedule.readout_end, :, self.go_pool] += (
                (u_g < 0.7).to(torch.float32) * (drive_amp * 0.85)
            )

        return ext

    def decode_counts(self, readout_spikes: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Sum spike counts in each output pool over [B, N] -> [B, K]."""
        if readout_spikes.dim() == 1:
            readout_spikes = readout_spikes.unsqueeze(0)
        B = readout_spikes.shape[0]
        counts = torch.zeros((B, num_classes), dtype=torch.float32, device=self.device)
        for k in range(num_classes):
            counts[:, k] = readout_spikes[:, self.output_pools[k]].mean(dim=1)
        return counts
