"""A stand-in for Muse Spark that replays the demo flow, for work without an API key.

Enable with MIXPILOT_LLM=scripted. It makes the tool calls the real agent should
make on the demo files and fills its replies from the tool results. The demo video
uses the real model.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from mixpilot.agent import Call, Step

DEMO_MAPPING = json.loads((Path(__file__).resolve().parent.parent / "demo_data" / "mapping.json").read_text())


class ScriptedLLM:
    def __init__(self):
        self.plan: list[Call] = []
        self.results: dict[str, dict] = {}
        self.reply = None
        self._last: list[Call] = []

    def _call(self, tool: str, **args) -> Call:
        return Call(tool, json.dumps(args), "call_" + uuid.uuid4().hex[:12])

    def step(self, input_items: list[dict], previous_response_id, specs, on_text) -> Step:
        calls_by_id = {c.call_id: c.name for c in self._last}
        for item in input_items:
            if item.get("type") == "function_call_output":
                self.results[calls_by_id.get(item["call_id"], "?")] = json.loads(item["output"])
            elif item.get("role") == "user":
                self._plan_for(item["content"])
        step_id = "scripted_" + uuid.uuid4().hex[:8]
        if self.plan:
            self._last = [self.plan.pop(0)]
            return Step(step_id, "", self._last)
        text = self.reply() if self.reply else "I'm running in scripted mode, so I can only replay the demo."
        self.reply = None
        for word in text.split(" "):
            on_text(word + " ")
        return Step(step_id, text, [])

    def _plan_for(self, text: str) -> None:
        if text.startswith("[app] Training finished"):
            self.results["train"] = json.loads(text.split(" ", 3)[3])
            self.plan = [self._call("get_results")]
            self.reply = self._results_reply
        elif ".xlsx" in text:
            self.plan = [self._call("read_files"), self._call("map_columns", mapping=DEMO_MAPPING),
                         self._call("validate_data"), self._call("suggest_inputs")]
            self.reply = self._prep_reply

    def _prep_reply(self) -> str:
        m, use = self.results["map_columns"], self.results["suggest_inputs"]["use"]
        return (f"I read {m['recipes']} recipes, {m['batches']} batches and {m['cylinders']} cylinders, and I'd add "
                f"{' and '.join(use) or 'no new inputs'}. Press Train model when you're ready.")

    def _results_reply(self) -> str:
        board = self.results["train"]["leaderboard"]
        win = self.results["train"]["winner"]
        props = self.results["get_results"].get("proposals", [])
        tuned = "Re-tuning Meta's settings helped a little" if self.results["train"]["tuning_helped"] \
            else "Re-tuning Meta's settings didn't help"
        text = f"28-day error went from {board[0]['error_28d_psi']} to {win['error_28d_psi']} psi. {tuned}"
        if props:
            text += f", and the proposals need up to {max(p['co2_saving'] for p in props):.0%} less CO2"
        return text + "."
