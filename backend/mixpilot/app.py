"""HTTP API: sessions, uploads, chat messages, and a server-sent event stream.

POST /sessions                      -> {id}
POST /sessions/{id}/files           multipart upload (Excel/CSV)
POST /sessions/{id}/demo-files      load the synthetic demo files
POST /sessions/{id}/messages        {text} -> starts an agent turn
POST /sessions/{id}/approve         {inputs} -> the Train model button
GET  /sessions/{id}/events?after=n  SSE: text_delta, message, tool_start, progress, tool_end, artifact, error, done
GET  /sessions/{id}/state           latest artifacts (for a page reload)
GET  /models?plant=...              saved versions
"""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from mixpilot import models, tools
from mixpilot.agent import Agent

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
DEMO = ROOT / "demo_data"
DEFAULT_PLANT = "Riverside Ready-Mix"

app = FastAPI(title="MixPilot")
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])


@dataclass
class Live:
    session: tools.Session
    agent: Agent | None = None
    events: list[dict] = field(default_factory=list)
    artifacts: dict[str, dict] = field(default_factory=dict)
    busy: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, ev: dict) -> None:
        with self.lock:
            ev = {"seq": len(self.events), **ev}
            self.events.append(ev)
            if ev["type"] == "artifact":
                self.artifacts[ev["kind"]] = ev["data"]


LIVE: dict[str, Live] = {}


def _get(sid: str) -> Live:
    if sid not in LIVE:
        raise HTTPException(404, f"no session {sid}")
    return LIVE[sid]


class NewSession(BaseModel):
    plant: str = DEFAULT_PLANT


class Message(BaseModel):
    text: str


class Approve(BaseModel):
    inputs: list[str]


@app.post("/sessions")
def new_session(body: NewSession | None = None) -> dict:
    sid = uuid.uuid4().hex[:10]
    live = Live(session=tools.Session(plant=(body or NewSession()).plant))
    live.agent = Agent(live.session, live.emit)
    LIVE[sid] = live
    existing = tools.list_models(live.session)["models"]
    if existing:
        live.emit(dict(type="artifact", kind="models", data={"models": existing}))
    return {"id": sid, "plant": live.session.plant}


@app.post("/sessions/{sid}/files")
async def upload(sid: str, files: list[UploadFile]) -> dict:
    live = _get(sid)
    for f in files:
        if not f.filename.lower().endswith((".xlsx", ".xls", ".csv")):
            raise HTTPException(400, f"{f.filename}: only Excel or CSV files")
    d = UPLOADS / sid
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        p = d / Path(f.filename).name
        p.write_bytes(await f.read())
        paths.append(p)
    live.session.add_files(paths)
    return {"files": [p.name for p in paths]}


@app.post("/sessions/{sid}/demo-files")
def demo_files(sid: str) -> dict:
    live = _get(sid)
    d = UPLOADS / sid
    d.mkdir(parents=True, exist_ok=True)
    paths = [Path(shutil.copy(p, d / p.name)) for p in sorted((DEMO / "plant").glob("*.xlsx"))]
    live.session.add_files(paths)
    return {"files": [p.name for p in paths]}


@app.post("/sessions/{sid}/messages", status_code=202)
def message(sid: str, body: Message) -> dict:
    live = _get(sid)
    _start_turn(live, body.text, lambda: live.agent.run_turn(body.text))
    return {"ok": True}


@app.post("/sessions/{sid}/approve", status_code=202)
def approve(sid: str, body: Approve) -> dict:
    live = _get(sid)
    if live.session.data is None:
        raise HTTPException(400, "the data is not mapped yet")
    names = " and ".join(body.inputs) if body.inputs else "Meta's inputs only"
    _start_turn(live, f"Train the model with {names}.", lambda: live.agent.train(list(body.inputs)))
    return {"ok": True}


def _start_turn(live: Live, text: str, job: Callable[[], None]) -> None:
    with live.lock:
        if live.busy:
            raise HTTPException(409, "the agent is still working")
        live.busy = True
    live.emit(dict(type="message", role="user", text=text))

    def run() -> None:
        try:
            job()
        finally:
            live.busy = False

    threading.Thread(target=run, daemon=True).start()


@app.get("/sessions/{sid}/events")
async def events(sid: str, after: int = -1):
    live = _get(sid)

    async def stream():
        i = after + 1
        while True:
            while i < len(live.events):
                ev = live.events[i]
                yield {"id": str(ev["seq"]), "event": ev["type"], "data": json.dumps(ev, default=str)}
                i += 1
            await asyncio.sleep(0.15)

    return EventSourceResponse(stream())


@app.get("/sessions/{sid}/state")
def state(sid: str) -> dict:
    live = _get(sid)
    return {"plant": live.session.plant, "busy": live.busy, "artifacts": live.artifacts,
            "files": sorted({s.file for s in live.session.sheets}), "events": len(live.events)}


@app.get("/models")
def list_models(plant: str = DEFAULT_PLANT) -> dict:
    return {"models": [{k: m[k] for k in ("version", "trained", "inputs", "setup", "error_28d_psi",
                                          "vs_meta_default", "recipes", "batches", "active")}
                       for m in models.versions(plant)]}
