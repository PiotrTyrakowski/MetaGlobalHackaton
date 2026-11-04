"""The agent loop and the prep turn on the demo files, driven by offline stand-ins for Muse Spark."""

import json
import shutil

from scripted_llm import ScriptedLLM

from mixpilot import agent as agent_module
from mixpilot import tools
from mixpilot.agent import Agent, Call, Step, upload_message
from mixpilot.session import BACKEND, new_session


class Fake:
    """Replays a list of steps; an Exception in the list is raised instead."""

    def __init__(self, steps):
        self.steps, self.seen = list(steps), []

    def step(self, items, previous_response_id, on_text):
        self.seen.append((items, previous_response_id))
        s = self.steps.pop(0)
        if isinstance(s, Exception):
            raise s
        if s.text:
            on_text(s.text)
        return s


def session():
    s = new_session()
    return s


def test_the_model_has_no_way_to_start_training():
    assert not {"train", "train_model"} & {t["name"] for t in tools.SCHEMAS}
    s = session()
    Agent(s, Fake([Step("a", "", [Call("train_model", '{"inputs": []}', "c1")]), Step("b", "ok")])).run("hello")
    assert s.events[-1]["type"] == "message_end"
    assert not s.artifacts
    shutil.rmtree(s.dir)


def test_the_agents_prompt_doesnt_know_the_demo_data():
    names = json.dumps(tools.SCHEMAS) + agent_module.SYSTEM
    for hint in ("Silica", "Mill Cert", "Air (%)", "cap failure", "Tech Notes", "Ticket"):
        assert hint not in names


def test_narration_before_tool_calls_is_marked_interim():
    s = session()
    Agent(s, Fake([Step("a", "Reading.", [Call("list_models", "{}", "c1")]), Step("b", "Done.")])).run("hello")
    ends = [(e["text"], e["interim"]) for e in s.events if e["type"] == "message_end"]
    assert ends == [("Reading.", True), ("Done.", False)]
    shutil.rmtree(s.dir)


def test_tool_results_survive_a_failed_call(monkeypatch):
    monkeypatch.setattr(agent_module, "RETRIES", 1)
    s = session()
    llm = Fake([Step("a", "", [Call("list_models", "{}", "c1")]), RuntimeError("down"), Step("b", "Back.")])
    Agent(s, llm).run("first")
    assert [e["text"] for e in s.events if e["type"] == "error"] == ["down"]
    Agent(s, llm).run("second")
    items, previous = llm.seen[-1]
    assert previous == "a"
    assert [i.get("call_id") or i["content"] for i in items] == ["c1", "second"]
    shutil.rmtree(s.dir)


def test_a_failed_call_is_retried_before_any_text_is_shown(monkeypatch):
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)
    s = session()
    Agent(s, Fake([RuntimeError("blip"), Step("a", "Fine.")])).run("hello")
    assert not [e for e in s.events if e["type"] == "error"]
    shutil.rmtree(s.dir)


def test_a_tool_mistake_goes_back_to_the_model():
    s = session()
    llm = Fake([Step("a", "", [Call("map_columns", '{"mapping": {}}', "c1"), Call("use_input", '{"name": "x"}', "c2")]),
                Step("b", "Sorry.")])
    Agent(s, llm).run("hello")
    outputs = [json.loads(i["output"]) for i in llm.seen[1][0]]
    assert [o["ok"] for o in outputs] == [False, False]
    assert [e["status"] for e in s.events if e["type"] == "tool" and e["status"] != "running"] == ["error", "error"]
    shutil.rmtree(s.dir)


def test_prep_turn_adds_two_inputs_and_offers_the_partly_logged_one():
    s = session()
    files = sorted((BACKEND / "demo_data" / "plant").glob("*.xlsx"))
    for f in files:
        s.add_file(f.name, f.read_bytes())

    Agent(s, ScriptedLLM()).run(upload_message([f.name for f in files]))

    types = [e["type"] for e in s.events]
    assert "error" not in types and "train_button" in types
    assert s.accepted == ["Air (%)", "Silica Fume (lb/cy)"]

    cards = {e["input"]: e for e in s.events if e["type"] == "card"}
    assert cards["Air (%)"]["kind"] == "input_ready" and "batches of the same mix" in cards["Air (%)"]["why"]
    assert cards["Silica Fume (lb/cy)"]["kind"] == "input_ready"
    mill = cards["Mill Cert 28d (psi)"]
    assert mill["kind"] == "input_incomplete" and mill["gain"] > 0 and mill["mixes_with_value"] < mill["mixes"]
    assert mill["recorded_from"]  # the plant started logging it later

    data = s.artifacts["data"]
    assert data["mixes"] == 24 and data["batches"] == mill["mixes"]
    assert {r["input"]: r["status"] for r in data["coverage"]}["Mill Cert 28d (psi)"] == "incomplete"
    assert "lb/yd³ → kg/m³" in data["conversions"]

    tested = {r["column"]: r for r in data["tested"]}
    assert tested["Air (%)"]["status"] == "added" and tested["Mill Cert 28d (psi)"]["status"] == "partly logged"
    assert tested["Truck"]["status"] == "not used" and "No clear link" in tested["Truck"]["why"]
    assert tested["Cement Lot"]["status"] == "not used" and "a label or date" in tested["Cement Lot"]["why"]
    assert all(len(c["points"]) == 3 for c in data["output"]["curves"])

    tools.use_input(s, "Mill Cert 28d (psi)")
    assert s.accepted[-1] == "Mill Cert 28d (psi)"
    assert any(e["type"] == "input_used" for e in s.events)
    assert s.data.tensors(s.accepted)[0].isfinite().all()  # gaps are filled for training
    shutil.rmtree(s.dir)
