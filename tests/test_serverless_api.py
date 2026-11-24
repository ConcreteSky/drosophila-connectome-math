"""Smoke tests for Vercel serverless `api/index.py` (NumPy/SciPy, zero PyTorch)."""

import pathlib

from api.index import (
    FeedbackRequest,
    LesionRequest,
    ResetRequest,
    SolveRequest,
    TrainStepRequest,
    brain_layout,
    connectome_summary,
    experiments,
    feedback,
    health,
    lesion,
    reset,
    solve,
    train_step,
)


def test_serverless_health_privacy_and_endpoints() -> None:
    assert "torch" not in pathlib.Path("api/index.py").read_text()
    h = health()
    assert h["status"] == "ok"
    assert h["device"] == "serverless_cpu"
    assert h["gpu_name"] == "serverless_cpu"

    lay = brain_layout()
    assert len(lay["landmarks"]) == 3000

    res = solve(SolveRequest(task="delayed_xor", a=1, b=0))
    assert len(res["region_activity"]["Mushroom Body"]) == 60
    assert res["total_spikes"] > 0

    fb = feedback(FeedbackRequest(reward=1))
    assert fb["weight_delta_norm"] > 0.0

    tr = train_step(TrainStepRequest(task="delayed_xor", episodes=4))
    assert 0.0 <= tr["accuracy"] <= 1.0

    les = lesion(LesionRequest(mushroom_body=True))
    assert les["disabled_neurons"] > 0

    rst = reset(ResetRequest(mode="fresh"))
    assert rst["mode"] == "fresh"

    assert connectome_summary()["total_neurons"] == 138639
    assert "hardware_benchmark" in experiments()
