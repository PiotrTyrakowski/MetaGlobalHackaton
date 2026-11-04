"""The agent loop, driven by scripted stand-ins for Muse Spark."""

import json

from mixpilot import agent as agent_module
from mixpilot import tools
from mixpilot.agent import Agent, Call, Step, tool_specs
from mixpilot.scripted_llm import ScriptedLLM


class Fake:
    """Replays a list of steps; an Exception in the list is raised instead."""

    def __init__(self, steps):
        self.steps, self.seen = list(steps), []

    def step(self, items, previous_response_id, specs, on_text):
        self.seen.append((items, previous_response_id))
        s = self.steps.pop(0)
        if isinstance(s, Exception):
            raise s
        if s.text:
            on_text(s.text)
        return s


def run(llm, text="hello", session=None):
    events = []
    Agent(session or tools.Session(), events.append, llm).run_turn(text)
    return events


def test_the_model_has_no_way_to_start_training():
    assert "train_model" not in {t["name"] for t in tool_specs()}
    events = run(Fake([Step("a", "", [Call("train_model", '{"inputs": []}', "c1")]), Step("b", "ok")]))
    assert not [e for e in events if e["type"] == "tool_start"]


def test_text_streams_and_only_the_last_message_is_final():
    events = run(Fake([Step("a", "Reading.", [Call("list_models", "{}", "c1")]), Step("b", "Done.")]))
    assert [e["text"] for e in events if e["type"] == "text_delta"] == ["Reading.", "Done."]
    assert [(e["text"], e["phase"]) for e in events if e["type"] == "message"] == [
        ("Reading.", "commentary"), ("Done.", "final")]
    assert events[-1]["type"] == "done"


def test_tool_results_survive_a_failed_call(monkeypatch):
    monkeypatch.setattr(agent_module, "RETRIES", 1)
    llm = Fake([Step("a", "", [Call("list_models", "{}", "c1")]), RuntimeError("down"), Step("b", "Back.")])
    a = Agent(tools.Session(), lambda e: None, llm)
    a.run_turn("first")
    a.run_turn("second")
    items, previous = llm.seen[-1]
    assert previous == "a"
    assert [i.get("call_id") or i["content"] for i in items] == ["c1", "second"]


def test_a_failed_call_is_retried_before_any_text_is_shown(monkeypatch):
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)
    events = run(Fake([RuntimeError("blip"), Step("a", "Fine.")]))
    assert not [e for e in events if e["type"] == "error"]
    assert [e["text"] for e in events if e["type"] == "message"] == ["Fine."]


def test_training_reports_failure_without_calling_the_model():
    llm = Fake([])
    events = []
    Agent(tools.Session(), events.append, llm).train(["Air (%)"])
    assert [e["type"] for e in events][-2:] == ["error", "done"]
    assert llm.seen == []


def test_demo_prep_turn_finds_the_inputs(tmp_path):
    s = tools.Session(plant="Test")
    from pathlib import Path
    s.add_files(sorted((Path(__file__).resolve().parents[1] / "demo_data" / "plant").glob("*.xlsx")))
    events = run(ScriptedLLM(), "Here are our lab files: mix_designs.xlsx", s)
    assert not [e for e in events if e["type"] == "error"]
    assert s.suggestions["use"] == ["Air (%)", "Silica Fume (lb/cy)"]
    assert s.suggestions["collect"] == ["Mill Cert 28d (psi)"]
    assert json.dumps(events[-2]["text"]) and events[-2]["phase"] == "final"
