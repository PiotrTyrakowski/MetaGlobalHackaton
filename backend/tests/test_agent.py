"""The prep turn of the demo, driven by the scripted stand-in for Muse Spark."""

import shutil

from mixpilot.agent import Agent, upload_message
from mixpilot.scripted_llm import ScriptedLLM
from mixpilot.server import DEMO_FILES
from mixpilot.session import new_session


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
    assert cards["input_incomplete"]["mixes_with_value"] == 20
    assert cards["input_incomplete"]["rmse_28d_with"] < cards["input_incomplete"]["rmse_28d_without"]
    assert "train_button" in types
    assert s.artifacts["data"]["mixes"] == 46
    assert s.accepted == ["Air (%)"]
    shutil.rmtree(s.dir)
