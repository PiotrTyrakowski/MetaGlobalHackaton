"""The agent loop and the prep turn on the demo files, driven by offline stand-ins for Muse Spark."""

import json
import shutil

from scripted_llm import ScriptedLLM

from mixpilot import agent as agent_module
from mixpilot import automl, boxfit, inputs, models, tools
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


def test_prep_turn_adds_two_inputs_and_offers_the_partly_logged_one(monkeypatch):
    def no_model_before_training(*_):
        raise AssertionError("a model was fitted before the user pressed Train")

    monkeypatch.setattr(boxfit, "kfold_many", no_model_before_training)
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
    assert cards["Silica Fume (lb/cy)"]["kind"] == "input_ready" and "Training tests" in cards["Silica Fume (lb/cy)"]["why"]
    mill = cards["Mill Cert 28d (psi)"]
    assert mill["kind"] == "input_incomplete" and "batches of the same mix" in mill["why"] and "Air" in mill["why"]
    assert mill["mixes_with_value"] < mill["mixes"]
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

    prep = s.artifacts["prep"]
    columns = {group: {r["column"]: r for r in rows} for group, rows in prep["columns"].items()}
    assert "Air (%)" in columns["useful"] and "Air" in columns["useful"]["Air (%)"]["note"]
    assert "Explains" in columns["useful"]["Air (%)"]["why"]
    assert "No clear link" in columns["unused"]["Truck"]["why"] and columns["unused"]["Truck"]["note"]
    assert "links the files" in columns["unused"]["Ticket #"]["why"]
    assert "Strength (psi)" in columns["default"] and "Strength (psi)" not in columns["unused"]
    every = {str(c) for sheet in s.sheets for c in sheet.df.columns}
    assert every <= set(columns["default"]) | set(columns["useful"]) | set(columns["unused"])
    assert [f["file"] for f in prep["files"]] == [f.name for f in files] and prep["joins"]
    fields = {r["field"]: r for r in prep["mapped"]}
    assert fields["Cement"]["change"] == "lb/yd³ → kg/m³" and fields["Curing temperature"]["change"] == "°F → °C"
    assert prep["cylinders_read"] == 880 and prep["cylinders_used"] == data["cylinders_used"] and prep["issues"]

    days = mill["timeline"]
    assert len(days) == mill["mixes"] and days == sorted(days, key=lambda d: d["date"])
    assert sum(d["has"] for d in days) == mill["mixes_with_value"]
    assert days[0]["has"] is False and days[-1]["has"] is True  # logging started later

    tools.use_input(s, "Mill Cert 28d (psi)")
    assert s.accepted[-1] == "Mill Cert 28d (psi)"
    assert any(e["type"] == "input_used" for e in s.events)
    assert s.data.tensors(s.accepted)[0].isfinite().all()  # gaps are filled for training

    tools.leave_out_input(s, "Mill Cert 28d (psi)")  # the plant can change its mind before training
    assert "Mill Cert 28d (psi)" not in s.accepted
    assert any(e["type"] == "input_left_out" for e in s.events)
    shutil.rmtree(s.dir)


def prepared_session():
    s = session()
    files = sorted((BACKEND / "demo_data" / "plant").glob("*.xlsx"))
    for f in files:
        s.add_file(f.name, f.read_bytes())
    Agent(s, ScriptedLLM()).run(upload_message([f.name for f in files]))
    return s


def fake_training(ds, extras, n_bayes, on_round):
    row = dict(name="x", error_28d_psi=100, vs_baseline_28d=0.0, best=True, folds_won_vs_baseline="0/0",
               errors_by_age_psi={"28d": 100}, settings={}, inputs=extras, baseline=True)
    return dict(leaderboard=[row], winner=dict(name="x", vs_baseline_28d=0.0, error_28d_psi=100), meta_default={},
                tuning_helped=False, method={}, trained_on=extras, trials=[])


def test_training_tests_the_ingredient_first_and_leaves_it_out_if_it_doesnt_help(monkeypatch, tmp_path):
    s = prepared_session()
    tools.use_input(s, "Mill Cert 28d (psi)")
    monkeypatch.setattr(models, "ROOT", tmp_path)
    monkeypatch.setattr(models, "save", lambda *_: dict(version="v1", recipes=24))
    monkeypatch.setattr(automl, "run_automl", fake_training)
    monkeypatch.setattr(inputs, "_kfold28_many", lambda jobs: [400.0, 399.0, 401.0])  # without, with, shuffled

    result = tools.train(s, use_cache=False)

    assert s.training["trained_on"] == ["Air (%)", "Mill Cert 28d (psi)"]
    assert result["inputs_left_out"] == ["Silica Fume (lb/cy)"]
    assert s.accepted == ["Air (%)", "Mill Cert 28d (psi)"]
    assert s.artifacts["training"]["left_out"] == ["Silica Fume (lb/cy)"]
    assert {r["column"]: r for r in s.artifacts["data"]["tested"]}["Silica Fume (lb/cy)"]["status"] == "not used"
    useful = {r["column"]: r for r in s.artifacts["prep"]["columns"]["useful"]}  # as it read before training
    assert "Training tests" in useful["Silica Fume (lb/cy)"]["why"]
    shutil.rmtree(s.dir)


def test_an_ingredient_that_helps_stays_and_says_how_much(monkeypatch, tmp_path):
    s = prepared_session()
    monkeypatch.setattr(models, "ROOT", tmp_path)
    monkeypatch.setattr(models, "save", lambda *_: dict(version="v1", recipes=24))
    monkeypatch.setattr(automl, "run_automl", fake_training)
    monkeypatch.setattr(inputs, "_kfold28_many", lambda jobs: [400.0, 380.0, 402.0])

    result = tools.train(s, use_cache=False)

    assert s.training["trained_on"] == ["Air (%)", "Silica Fume (lb/cy)"] and result["inputs_left_out"] == []
    row = {r["column"]: r for r in s.artifacts["data"]["tested"]}["Silica Fume (lb/cy)"]
    assert row["status"] == "added" and "5% more accurate" in row["why"]
    shutil.rmtree(s.dir)


def test_the_search_shows_both_input_sets_round_by_round():
    assert tools._search_round(1, "meta default", 730.2, 297) == dict(round=1, kind="Meta's settings", base=730, plus=297)
    assert tools._search_round(2, "sobol", float("inf"), 293) == dict(round=2, kind="Exploring", base=None, plus=293)
    assert tools._search_round(1, "bayes-opt", 700, None) == dict(round=1, kind="Searching", base=700, plus=None)


def test_the_models_that_missed_the_final_test_are_told_apart_and_ranked():
    trials = [dict(setup="Meta's 10 inputs", noise=1.0, error_28d_psi=730),
              dict(setup="+ air", noise=1.0, error_28d_psi=297),
              dict(setup="Meta's 10 inputs", noise=2.0, error_28d_psi=None),
              dict(setup="+ air", noise=2.0, error_28d_psi=293),
              dict(setup="Meta's 10 inputs", noise=3.0, error_28d_psi=700),
              dict(setup="+ air", noise=3.0, error_28d_psi=290),
              dict(setup="+ air", noise=3.0, error_28d_psi=290)]
    # noise=1 with air is another model than Meta's row, and the search landing on noise=3 twice is one model
    board = [dict(inputs=[], settings=dict(noise=1.0)), dict(inputs=["air"], settings=dict(noise=2.0))]
    assert tools._others(trials, board) == [
        dict(model=6, new_inputs=True, settings=dict(noise=3.0)),
        dict(model=2, new_inputs=True, settings=dict(noise=1.0)),
        dict(model=5, new_inputs=False, settings=dict(noise=3.0))]
