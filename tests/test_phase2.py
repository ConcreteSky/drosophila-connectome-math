"""Phase 2 unit tests: encoder determinism, zero leakage, Dale's Law, shuffler, and lesions."""

import torch

from flybrain_rl.brain import SpikingFlyBrain
from flybrain_rl.config import TrainingConfig
from flybrain_rl.connectome import ConnectomeGraph
from flybrain_rl.encoding import PopulationEncoderDecoder, TrialSchedule
from flybrain_rl.tasks import MathTaskEnv
from flybrain_rl.train import GRUBaselineTrainer, REINFORCETrainer, RSTDPTrainer


def _make_setup(num_neurons: int = 360, num_edges: int = 2400) -> tuple[SpikingFlyBrain, PopulationEncoderDecoder]:
    g = ConnectomeGraph.make_synthetic(num_neurons=num_neurons, num_edges=num_edges, seed=42)
    b = SpikingFlyBrain(g, device="cpu")
    enc = PopulationEncoderDecoder(
        num_neurons=b.num_neurons,
        sensory_indices=b.get_region_indices("Sensory"),
        output_indices=b.get_region_indices("Motor/Output"),
        device=b.device,
        neurons_per_pool=4,
        seed=42,
    )
    return b, enc


def test_encoder_determinism_and_delay_gap() -> None:
    _, enc1 = _make_setup()
    _, enc2 = _make_setup()
    for p1, p2 in zip(enc1.a_pools, enc2.a_pools):
        assert torch.equal(p1, p2)

    all_in = torch.cat(enc1.a_pools + enc1.b_pools + [enc1.go_pool]).tolist()
    assert len(all_in) == len(set(all_in))

    sched = TrialSchedule()
    a_v = torch.ones((2, 7))
    b_v = torch.ones((2, 7))
    r1 = torch.Generator().manual_seed(123)
    r2 = torch.Generator().manual_seed(123)
    ext1 = enc1.encode_batch(a_v, b_v, sched, rng=r1)
    ext2 = enc2.encode_batch(a_v, b_v, sched, rng=r2)
    assert torch.equal(ext1, ext2)
    assert ext1[15:30].sum().item() == 0.0


def test_zero_train_test_leakage() -> None:
    _, enc = _make_setup()
    for task in ("delayed_xor", "add_3bit", "mod_add"):
        env = MathTaskEnv(enc, task=task, split_seed=42)
        assert set(env.train_pairs).isdisjoint(set(env.test_pairs))


def test_dales_law_after_100_rstdp_updates() -> None:
    brain, enc = _make_setup(num_neurons=420, num_edges=3000)
    env = MathTaskEnv(enc, task="delayed_xor", split_seed=42)
    cfg = TrainingConfig(lr_rstdp=0.4, conn_lr_rstdp=0.05, w_max=4.0)
    trainer = RSTDPTrainer(brain, env, cfg=cfg, seed=42)
    orig_signs = brain.edge_signs.clone()

    for _ in range(100):
        trainer.train_step(batch_size=4)

    assert (brain.values[orig_signs > 0] >= 0.0).all()
    assert (brain.values[orig_signs < 0] <= 0.0).all()
    assert (brain.values.abs() <= cfg.w_max + 1e-6).all()
    assert (trainer.W_out.abs() <= cfg.w_max + 1e-6).all()


def test_degree_preserving_shuffler_invariants() -> None:
    g = ConnectomeGraph.make_synthetic(num_neurons=240, num_edges=1400, seed=7)
    s = g.make_degree_preserving_shuffled(seed=99)
    assert torch.equal(g.crow_indices[1:] - g.crow_indices[:-1], s.crow_indices[1:] - s.crow_indices[:-1])
    assert torch.equal(
        torch.bincount(g.col_indices, minlength=g.num_neurons),
        torch.bincount(s.col_indices, minlength=s.num_neurons),
    )
    assert torch.equal(torch.sign(s.values).to(torch.int8), s.signs[s.col_indices])


def test_reinforce_gru_and_hub_lesion() -> None:
    brain, enc = _make_setup(num_neurons=360, num_edges=2400)
    env = MathTaskEnv(enc, task="mod_add", split_seed=42)
    rf = REINFORCETrainer(brain, env, seed=42)
    assert 0.0 <= rf.train_step(batch_size=4)["train_acc"] <= 1.0
    assert 0.0 <= rf.evaluate(split="test", batch_size=4)["accuracy"] <= 1.0

    gru = GRUBaselineTrainer(env, hidden_size=32, seed=42)
    assert 0.0 <= gru.train_step(batch_size=4)["train_acc"] <= 1.0

    disabled = brain.set_lesion(hub_top_5pct=True)
    assert disabled == max(1, int(round(360 * 0.05)))
