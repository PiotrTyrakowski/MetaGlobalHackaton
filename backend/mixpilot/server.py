"""HTTP API for the UI: sessions, uploads, chat, the Train button and an SSE event stream.

Run from backend/:  uv run uvicorn mixpilot.server:app --port 8000
"""

import asyncio
import json
import shutil
import threading

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mixpilot import tools
from mixpilot.agent import Agent, training_message, upload_message
from mixpilot.session import BACKEND, SESSIONS, Session, new_session

DEMO_FILES = [BACKEND / "data" / "demo" / f for f in ("mix_designs.xlsx", "cylinder_breaks.xlsx", "daily_QC.xlsx")]

app = FastAPI(title="MixPilot")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["*"], allow_headers=["*"])


class NewSession(BaseModel):
    model: str | None = None


class Message(BaseModel):
    text: str


def _session(sid: str) -> Session:
    if sid not in SESSIONS:
        raise HTTPException(404, "no such session")
    return SESSIONS[sid]


def _start(s: Session, kind: str, job) -> None:
    """Run a job in the background; one at a time per session."""
    if s.busy:
        raise HTTPException(409, f"busy: {s.busy}")
    s.busy = kind
    s.emit("busy", kind=kind)

    def run():
        try:
            job()
        except Exception as e:
            s.emit("error", text=f"{type(e).__name__}: {e}"[:500])
        finally:
            s.busy = None
            s.emit("idle")

    threading.Thread(target=run, daemon=True).start()


def _after_upload(s: Session, names: list[str]) -> None:
    s.emit("message", role="user", files=names)
    _start(s, "agent", lambda: Agent(s).run(upload_message(names)))


@app.post("/api/sessions")
def create_session(body: NewSession | None = None):
    s = new_session(model=body.model if body else None)
    return {"id": s.id, "model": s.model}


@app.post("/api/sessions/demo")
def create_demo_session():
    """A session preloaded with the three synthetic demo files."""
    s = new_session()
    s.uploads.mkdir(parents=True, exist_ok=True)
    for f in DEMO_FILES:
        shutil.copy(f, s.uploads / f.name)
        s.files.append(s.uploads / f.name)
    _after_upload(s, [f.name for f in DEMO_FILES])
    return {"id": s.id}


@app.post("/api/sessions/{sid}/files")
async def upload(sid: str, files: list[UploadFile]):
    s = _session(sid)
    names = []
    for f in files:
        if not f.filename.lower().endswith((".xlsx", ".xls", ".csv")):
            raise HTTPException(400, f"{f.filename}: only Excel or CSV files")
        s.add_file(f.filename, await f.read())
        names.append(f.filename)
    _after_upload(s, names)
    return {"files": names}


@app.post("/api/sessions/{sid}/messages")
def post_message(sid: str, body: Message):
    s = _session(sid)
    if not body.text.strip():
        raise HTTPException(400, "empty message")
    if s.busy:
        raise HTTPException(409, f"busy: {s.busy}")
    s.emit("message", role="user", text=body.text)
    _start(s, "agent", lambda: Agent(s).run(body.text))
    return {"ok": True}


@app.post("/api/sessions/{sid}/train")
def start_training(sid: str):
    s = _session(sid)
    if s.mapping is None:
        raise HTTPException(400, "data is not prepared yet")

    def job():
        result = tools.train(s)
        Agent(s).run(training_message(result))

    _start(s, "training", job)
    return {"ok": True}


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    s = _session(sid)
    return {"id": s.id, "files": [f.name for f in s.files], "busy": s.busy, "model": s.model,
            "artifacts": s.artifacts, "events": len(s.events)}


@app.get("/api/sessions/{sid}/events")
async def events(sid: str, request: Request, after: int = -1):
    """Server-sent events, replayed from `after` (or the Last-Event-ID header)."""
    s = _session(sid)
    cursor = int(request.headers.get("last-event-id", after))

    async def stream():
        nonlocal cursor
        quiet = 0
        while not await request.is_disconnected():
            batch = s.events_after(cursor)
            for e in batch:
                cursor = e["seq"]
                yield f"id: {e['seq']}\ndata: {json.dumps(e, default=str)}\n\n"
            if batch:
                quiet = 0
                continue
            await asyncio.sleep(0.1)
            quiet += 1
            if quiet % 150 == 0:
                yield ": ping\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/models")
def models():
    return {"models": tools.list_models_index()}


app.mount("/", StaticFiles(directory=BACKEND / "mixpilot" / "static", html=True), name="ui")
