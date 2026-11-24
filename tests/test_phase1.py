"""Phase 1 unit tests: connectome loader, Dale's Law signs, LIF decay, refractory, and lesions."""

import math
import torch

from flybrain_rl.brain import SpikingFlyBrain
from flybrain_rl.config import LIFConfig
from flybrain_rl.connectome import ConnectomeGraph


def _two_neuron_graph(weight: float) -> ConnectomeGraph:
    return ConnectomeGraph(
        num_neurons=2,
        num_edges=1,
        crow_indices=torch.tensor([0, 0, 1], dtype=torch.int64),
        col_indices=torch.tensor([0], dtype=torch.int64),
        values=torch.tensor([weight], dtype=torch.float32),
        signs=torch.tensor([1 if weight >= 0 else -1, 1], dtype=torch.int8),
        region_names=["Sensory", "Motor/Output"],
        region_ids=torch.tensor([0, 1], dtype=torch.int8),
        coords_2d=torch.tensor([[0.15, 0.5], [0.85, 0.5]], dtype=torch.float32),
    )


def test_synthetic_loader_counts_and_dale_signs() -> None:
    g = ConnectomeGraph.make_synthetic(num_neurons=240, num_edges=1200)
    assert g.num_neurons == 240
    assert g.num_edges == len(g.values)
    assert set(g.signs.tolist()).issubset({-1, 1})
    # Dale's Law: every edge sign matches presynaptic neuron sign
    assert torch.equal(torch.sign(g.values).to(torch.int8), g.signs[g.col_indices])


def test_lif_membrane_decay_and_refractory() -> None:
    cfg = LIFConfig(dt_ms=1.0, tau_mem_ms=20.0, tau_syn_ms=5.0, v_rest_mv=-52.0, v_reset_mv=-52.0, v_thresh_mv=-45.0, refrac_steps=2)
    g = _two_neuron_graph(0.0)
    brain = SpikingFlyBrain(g, config=cfg, device="cpu")

    # Sub-threshold current raises V above V_rest, then decays back toward V_rest
    s0 = brain.step(torch.tensor([50.0, 0.0]))
    assert s0.sum().item() == 0.0
    v1 = brain.v[0, 0].item()
    assert v1 > cfg.v_rest_mv

    brain.step(torch.zeros(2))
    v2 = brain.v[0, 0].item()
    # Since i_syn decays by beta, drive > 0; check exact formula
    alpha = math.exp(-1.0 / 20.0)
    expected_v2 = alpha * v1 + (1.0 - alpha) * (cfg.v_rest_mv + brain.i_syn[0, 0].item())
    assert abs(v2 - expected_v2) < 1e-4

    # Strong current triggers spike + 2-step refractory period
    brain.reset_state(1)
    strong = torch.tensor([200.0, 0.0])
    assert brain.step(strong)[0, 0].item() == 1.0
    assert brain.step(strong)[0, 0].item() == 0.0
    assert brain.step(strong)[0, 0].item() == 0.0
    assert brain.step(strong)[0, 0].item() == 1.0


def test_lesion_zeroing() -> None:
    g = ConnectomeGraph.make_synthetic(num_neurons=180, num_edges=900)
    brain = SpikingFlyBrain(g, device="cpu")
    disabled = brain.set_lesion(regions=["Mushroom Body"])
    assert disabled > 0
    mb_idx = brain.get_region_indices("Mushroom Body")
    ext = torch.full((10, 2, 180), 25.0)
    out = brain.run(ext)
    assert out["spike_counts"][:, mb_idx].sum().item() == 0.0
    assert (brain.v[:, mb_idx] == brain.cfg.v_rest_mv).all()
