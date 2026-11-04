"""The HTTP API guards around uploads and the Train model button."""

from fastapi.testclient import TestClient

from mixpilot.app import app

client = TestClient(app)


def new_session() -> str:
    return client.post("/sessions", json={"plant": "API test"}).json()["id"]


def test_sessions_start_without_an_api_key(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    assert new_session()


def test_only_spreadsheets_can_be_uploaded():
    r = client.post(f"/sessions/{new_session()}/files", files={"files": ("notes.txt", b"hello")})
    assert r.status_code == 400


def test_training_needs_mapped_data():
    assert client.post(f"/sessions/{new_session()}/approve", json={"inputs": []}).status_code == 400


def test_one_job_at_a_time():
    from mixpilot.app import LIVE

    sid = new_session()
    LIVE[sid].session.data = object()
    LIVE[sid].busy = True
    assert client.post(f"/sessions/{sid}/approve", json={"inputs": []}).status_code == 409
    assert client.post(f"/sessions/{sid}/messages", json={"text": "hi"}).status_code == 409
