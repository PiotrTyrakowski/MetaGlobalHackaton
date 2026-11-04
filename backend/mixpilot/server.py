"""HTTP API for the UI: sessions, uploads, chat, the Train button and an SSE event stream.

Run from backend/:  uv run uvicorn mixpilot.server:app --port 8000
"""

import asyncio
import json
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mixpilot import boxfit, tools
from mixpilot.agent import Agent, training_message, upload_message
from mixpilot.session import BACKEND, SESSIONS, Session, new_session


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=boxfit.warm_up, daemon=True).start()
    yield


app = FastAPI(title="MixPilot", lifespan=lifespan)


class NewSession(BaseModel):
    model: str | None = None


class Message(BaseModel):
    text: str


class InputName(BaseModel):
    name: str
    use: bool = True


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


@app.post("/api/sessions/{sid}/files")
async def upload(sid: str, files: list[UploadFile]):
    s = _session(sid)
    if s.busy:
        raise HTTPException(409, f"busy: {s.busy}")
    names, unreadable = [], []
    for f in files:
        if not f.filename.lower().endswith((".xlsx", ".xls", ".csv")):
            unreadable.append(f"{f.filename}: only Excel or CSV files.")
            continue
        try:
            s.add_file(f.filename, await f.read())
            names.append(f.filename)
        except ValueError as e:
            unreadable.append(str(e))
    if not names:
        raise HTTPException(400, " ".join(unreadable))
    for text in unreadable:
        s.emit("error", text=text)
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


@app.post("/api/sessions/{sid}/inputs")
def use_input(sid: str, body: InputName):
    """The card's buttons: train with a partly logged input, or leave it out."""
    s = _session(sid)
    if s.busy:
        raise HTTPException(409, f"busy: {s.busy}")
    try:
        (tools.use_input if body.use else tools.leave_out_input)(s, body.name)
    except tools.ToolError as e:
        raise HTTPException(400, str(e))
    tools.offer_training(s)
    return {"inputs": s.accepted}


@app.post("/api/sessions/{sid}/train")
def start_training(sid: str):
    s = _session(sid)
    if s.data is None:
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


class UI(StaticFiles):
    """The single-page UI, revalidated on every load so a reload picks up edits."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", UI(directory=BACKEND / "mixpilot" / "static", html=True), name="ui")
