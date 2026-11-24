"""Phase 3 Gate tests: FastAPI endpoints, feedback weight updates, Dale's Law, lesions, and payload size."""

import json

from fastapi import HTTPException
import pytest
import torch

from flybrain_rl.api import (
    FeedbackRequest,
    LesionRequest,
    ResetRequest,
    SolveRequest,
    create_app,
)
from flybrain_rl.connectome import ConnectomeGraph
from flybrain_rl.session import BrainSession


def _run_coro(coro):
    """Drive a non-blocking async coroutine synchronously (avoids Windows sandbox socketpair)."""
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


def _make_routes():
    g = ConnectomeGraph.make_synthetic(num_neurons=360, num_edges=2400, seed=7)
    sess = BrainSession(graph=g, load_checkpoints=False)
    app = create_app(session=sess)
    routes = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
    return sess, routes


def test_health_layout_summary_and_experiments() -> None:
    _, routes = _make_routes()
    h = _run_coro(routes["/api/health"]())
    assert h["status"] == "ok"

    layout = _run_coro(routes["/api/brain/layout"]())
    assert 0 < len(layout["landmarks"]) <= 360
    assert len(layout["region_names"]) == 6

    summary = _run_coro(routes["/api/connectome/summary"]())
    assert summary["total_neurons"] > 0

    exp = _run_coro(routes["/api/experiments"]())
    assert "results" in exp


def test_solve_feedback_dales_law_lesion_and_payload_size() -> None:
    sess, routes = _make_routes()

    # Feedback before solve raises 400
    with pytest.raises(HTTPException) as exc:
        _run_coro(routes["/api/brain/feedback"](FeedbackRequest(reward=1)))
    assert exc.value.status_code == 400

    # Solve
    solve_res = _run_coro(routes["/api/brain/solve"](SolveRequest(task="delayed_xor", a=1, b=0)))
    payload_bytes = len(json.dumps(solve_res).encode("utf-8"))
    assert payload_bytes < 300 * 1024
    assert len(solve_res["region_activity"]["Mushroom Body"]) == 60

    # Feedback (+1) produces weight_delta_norm > 0 and preserves Dale's Law
    orig_signs = sess.brain.edge_signs.clone()
    fb = _run_coro(routes["/api/brain/feedback"](FeedbackRequest(reward=1)))
    assert fb["weight_delta_norm"] > 0.0
    assert fb["dopamine_pulse"] == "reward"
    assert (sess.brain.values[orig_signs > 0] >= 0.0).all()
    assert (sess.brain.values[orig_signs < 0] <= 0.0).all()

    # Lesion Mushroom Body -> 0 spikes in Mushroom Body
    les = _run_coro(routes["/api/brain/lesion"](LesionRequest(mushroom_body=True)))
    assert les["disabled_neurons"] > 0
    solve_les = _run_coro(routes["/api/brain/solve"](SolveRequest(task="delayed_xor", a=1, b=0)))
    assert sum(solve_les["region_activity"]["Mushroom Body"]) == 0.0

    # Reset
    rst = _run_coro(routes["/api/brain/reset"](ResetRequest(mode="fresh")))
    assert rst["mode"] == "fresh"
