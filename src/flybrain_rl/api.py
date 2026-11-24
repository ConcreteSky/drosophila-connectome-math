"""FastAPI + WebSocket server with Pydantic v2 schemas for `flybrain_rl`."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from flybrain_rl.session import BrainSession
from flybrain_rl.tasks import TaskName


class SolveRequest(BaseModel):
    task: TaskName = "delayed_xor"
    a: int = Field(1, ge=0, le=7)
    b: int = Field(0, ge=0, le=7)


class FeedbackRequest(BaseModel):
    reward: Literal[1, -1]


class LesionRequest(BaseModel):
    mushroom_body: bool = False
    central_complex: bool = False
    hub_top_5pct: bool = False


class ResetRequest(BaseModel):
    mode: Literal["checkpoint", "fresh", "shuffled"] = "checkpoint"


def create_app(session: Optional[BrainSession] = None) -> FastAPI:
    app = FastAPI(title="Drosophila Connectome Math RL API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict[str, BrainSession] = {}

    def get_session() -> BrainSession:
        if "s" not in state:
            state["s"] = session if session is not None else BrainSession()
        return state["s"]

    @app.get("/api/health")
    async def health() -> dict:
        s = get_session()
        async with s.lock:
            return s.health()

    @app.get("/api/brain/layout")
    async def brain_layout() -> dict:
        s = get_session()
        return {
            "region_names": list(s.graph.region_names),
            "landmarks": s.projection.landmarks,
        }

    @app.post("/api/brain/solve")
    async def solve(req: SolveRequest) -> dict:
        s = get_session()
        async with s.lock:
            return s.solve(req.task, req.a, req.b)

    @app.post("/api/brain/feedback")
    async def feedback(req: FeedbackRequest) -> dict:
        s = get_session()
        async with s.lock:
            if s.rstdp[s.active_task]._cached_action is None:
                raise HTTPException(status_code=400, detail="No cached trial to apply feedback to.")
            return s.feedback(req.reward)

    @app.post("/api/brain/lesion")
    async def lesion(req: LesionRequest) -> dict:
        s = get_session()
        async with s.lock:
            return s.set_lesion(req.mushroom_body, req.central_complex, req.hub_top_5pct)

    @app.post("/api/brain/reset")
    async def reset(req: ResetRequest) -> dict:
        s = get_session()
        async with s.lock:
            return s.reset(req.mode)

    @app.get("/api/connectome/summary")
    async def connectome_summary() -> dict:
        s = get_session()
        return s.connectome_summary()

    @app.get("/api/experiments")
    async def experiments() -> dict:
        s = get_session()
        return s.experiments()

    @app.websocket("/api/ws/train")
    async def ws_train(websocket: WebSocket) -> None:
        await websocket.accept()
        s = get_session()
        try:
            cfg = await websocket.receive_json()
            task: TaskName = cfg.get("task", "delayed_xor")
            episodes: int = min(200, max(1, int(cfg.get("episodes", 10))))
            batch_size: int = min(64, max(1, int(cfg.get("batch_size", 4))))
            plasticity: Literal["rstdp", "reinforce"] = cfg.get("plasticity", "rstdp")

            for ep in range(1, episodes + 1):
                async with s.lock:
                    metrics = s.train_episode(task=task, batch_size=batch_size, plasticity=plasticity)
                await websocket.send_json({"episode": ep, **metrics})
        except WebSocketDisconnect:
            return

    return app


app = create_app()
