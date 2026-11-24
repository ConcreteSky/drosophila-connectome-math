"""Dorsal 2D anatomical projection and compact 3,000-landmark spike packing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flybrain_rl.config import REGION_CENTERS_2D
from flybrain_rl.connectome import ConnectomeGraph
from flybrain_rl.encoding import PopulationEncoderDecoder


@dataclass
class DorsalProjection:
    landmark_indices: torch.Tensor
    landmarks: list[dict]

    @classmethod
    def build(
        cls,
        graph: ConnectomeGraph,
        encoder: PopulationEncoderDecoder,
        num_landmarks: int = 3000,
        seed: int = 42,
    ) -> "DorsalProjection":
        """Select ~3,000 landmark neurons proportional across 6 regions + IO pools."""
        rng = torch.Generator(device="cpu").manual_seed(seed)

        # Always include input/output pool neurons so sensory->motor flow is visible
        io_set = set()
        for p in encoder.a_pools + encoder.b_pools + [encoder.go_pool] + encoder.output_pools:
            io_set.update(p.cpu().tolist())
        io_list = sorted(io_set)

        remaining = max(0, min(graph.num_neurons, num_landmarks) - len(io_list))
        chosen = list(io_list)
        chosen_set = set(chosen)

        num_regions = len(graph.region_names)
        per_region = max(1, remaining // max(1, num_regions))
        for rid in range(num_regions):
            r_idx = torch.nonzero(graph.region_ids == rid, as_tuple=True)[0]
            if len(r_idx) == 0:
                continue
            perm = r_idx[torch.randperm(len(r_idx), generator=rng)].tolist()
            added = 0
            for idx in perm:
                if idx not in chosen_set:
                    chosen.append(idx)
                    chosen_set.add(idx)
                    added += 1
                    if added >= per_region or len(chosen) >= min(graph.num_neurons, num_landmarks):
                        break

        lm_tensor = torch.tensor(chosen[: min(graph.num_neurons, num_landmarks)], dtype=torch.long)
        coords = graph.coords_2d[lm_tensor].tolist()
        reg_ids = graph.region_ids[lm_tensor].tolist()
        signs = graph.signs[lm_tensor].tolist()

        # Blend real coordinates with canonical dorsal region anchors for crisp anatomical grouping
        landmarks = []
        for i, n_idx in enumerate(lm_tensor.tolist()):
            rname = graph.region_names[reg_ids[i]]
            cx, cy = REGION_CENTERS_2D.get(rname, (0.5, 0.5))
            x = 0.45 * float(coords[i][0]) + 0.55 * cx
            y = 0.45 * float(coords[i][1]) + 0.55 * cy
            landmarks.append(
                {
                    "id": i,
                    "neuron_idx": int(n_idx),
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "region": rname,
                    "sign": int(signs[i]),
                }
            )

        return cls(landmark_indices=lm_tensor, landmarks=landmarks)
