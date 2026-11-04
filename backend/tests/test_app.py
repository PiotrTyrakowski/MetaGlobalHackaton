"""The HTTP API guards around uploads and the Train button."""

from fastapi.testclient import TestClient

from mixpilot.server import app
from mixpilot.session import SESSIONS

client = TestClient(app)  # not used as a context manager: tests don't start the worker pool


def new_session() -> str:
    return client.post("/api/sessions").json()["id"]


def test_the_page_is_served():
    assert "MixPilot" in client.get("/").text
    assert client.get("/concrete-mixer.png").status_code == 200
    assert client.get("/concrete-mixer-static.png").status_code == 200


def test_sessions_start_without_an_api_key(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    assert new_session()


def test_only_spreadsheets_can_be_uploaded():
    r = client.post(f"/api/sessions/{new_session()}/files", files={"files": ("notes.txt", b"hello")})
    assert r.status_code == 400


def test_training_needs_mapped_data():
    assert client.post(f"/api/sessions/{new_session()}/train").status_code == 400


def test_only_offered_inputs_can_be_switched_on():
    r = client.post(f"/api/sessions/{new_session()}/inputs", json={"name": "Truck"})
    assert r.status_code == 400
    r = client.post(f"/api/sessions/{new_session()}/inputs", json={"name": "Truck", "use": False})
    assert r.status_code == 400


def test_one_job_at_a_time():
    sid = new_session()
    SESSIONS[sid].data = object()
    SESSIONS[sid].busy = "agent"
    assert client.post(f"/api/sessions/{sid}/train").status_code == 409
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "hi"}).status_code == 409


def test_an_unreadable_file_is_said_plainly():
    r = client.post(f"/api/sessions/{new_session()}/files", files={"files": ("plant.xlsx", b"not a workbook")})
    assert r.status_code == 400 and "couldn't be read" in r.json()["detail"]


def test_an_empty_file_is_said_plainly():
    r = client.post(f"/api/sessions/{new_session()}/files", files={"files": ("plant.csv", b"")})
    assert r.status_code == 400 and "is empty" in r.json()["detail"]


def test_a_new_session_never_answers_from_someone_elses_model():
    from mixpilot import tools

    s = SESSIONS[new_session()]
    result = tools.call(s, "predict", {"mixes": [{"recipe": "R-1"}]})
    assert result["ok"] is False and "no model in this session" in result["errors"][0]
