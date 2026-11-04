"""One plant conversation: its uploaded files, data-prep state and event log.

Everything the UI shows arrives as an event (chat text, tool status, cards,
artifact updates). Events are kept in order so a client can reconnect and
replay from any point.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from mixpilot import excel
from mixpilot.dataset import Dataset
from mixpilot.excel import Sheet

BACKEND = Path(__file__).resolve().parent.parent
WORKSPACE = BACKEND / "workspace"
PLANT = "plant"  # one plant per install; its models live in models/plant/


@dataclass
class Session:
    id: str
    dir: Path
    plant: str = PLANT
    files: list[Path] = field(default_factory=list)
    sheets: list[Sheet] = field(default_factory=list)
    mapping: dict | None = None
    data: Dataset | None = None
    map_report: dict | None = None
    validation: dict | None = None
    suggestions: dict | None = None
    accepted: list[str] = field(default_factory=list)  # extra inputs used for training
    offered: dict[str, dict] = field(default_factory=dict)  # partly logged inputs the user can switch on
    ready_to_train: bool = False  # prepared data that has not been trained on yet
    training: dict | None = None
    model_meta: dict | None = None
    model: str | None = None  # model version this conversation works with
    previous_response_id: str | None = None
    pending_input: list[dict] = field(default_factory=list)  # tool results a failed call didn't deliver
    artifacts: dict[str, dict] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    busy: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def uploads(self) -> Path:
        return self.dir / "uploads"

    def emit(self, type_: str, **data) -> dict:
        with self._lock:
            event = {"seq": len(self.events), "type": type_, "time": time.time(), **data}
            self.events.append(event)
            if type_ == "artifact":
                self.artifacts[data["name"]] = data["data"]
        return event

    def events_after(self, seq: int) -> list[dict]:
        with self._lock:
            return self.events[seq + 1 :]

    def add_file(self, name: str, content: bytes) -> Path:
        self.uploads.mkdir(parents=True, exist_ok=True)
        path = self.uploads / Path(name).name
        path.write_bytes(content)
        try:
            sheets = excel.read_sheets(path)
        except Exception as e:
            path.unlink(missing_ok=True)
            raise ValueError(f"{path.name} couldn't be read. Is it a real Excel or CSV file?") from e
        if not sheets:
            path.unlink(missing_ok=True)
            raise ValueError(f"{path.name} is empty.")
        if path not in self.files:
            self.files.append(path)
        self.sheets = [s for s in self.sheets if s.upload != path.name] + sheets
        return path


SESSIONS: dict[str, Session] = {}


def new_session(model: str | None = None) -> Session:
    sid = uuid.uuid4().hex[:10]
    s = Session(id=sid, dir=WORKSPACE / "sessions" / sid, model=model)
    s.dir.mkdir(parents=True, exist_ok=True)
    SESSIONS[sid] = s
    return s
