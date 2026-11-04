"""The prep turn on the demo files, driven by an offline stand-in for Muse Spark."""

import shutil

from scripted_llm import ScriptedLLM

from mixpilot import tools
from mixpilot.agent import Agent, upload_message
from mixpilot.session import BACKEND, new_session

DEMO_FILES = [BACKEND / "data" / "demo" / f for f in ("mix_designs.xlsx", "cylinder_breaks.xlsx", "daily_QC.xlsx")]


def test_prep_turn_finds_one_new_and_one_incomplete_input():
    s = new_session()
    s.uploads.mkdir(parents=True, exist_ok=True)
    for f in DEMO_FILES:
        shutil.copy(f, s.uploads / f.name)
        s.files.append(s.uploads / f.name)

    Agent(s, ScriptedLLM(s)).run(upload_message([f.name for f in DEMO_FILES]))

    types = [e["type"] for e in s.events]
    assert "error" not in types
    cards = {e["kind"]: e for e in s.events if e["type"] == "card"}
    assert cards["input_ready"]["input"] == "Air (%)"
    assert cards["input_incomplete"]["mixes_with_value"] == 42
    assert cards["input_incomplete"]["rmse_28d_with"] < cards["input_incomplete"]["rmse_28d_without"]
    assert "train_button" in types
    assert s.artifacts["data"]["mixes"] == 70
    assert s.accepted == ["Air (%)"]

    tools.use_input(s, cards["input_incomplete"]["input"])
    assert s.accepted == ["Air (%)", "Site Water (gal/yd3)"]
    assert any(e["type"] == "input_used" for e in s.events)
    shutil.rmtree(s.dir)
