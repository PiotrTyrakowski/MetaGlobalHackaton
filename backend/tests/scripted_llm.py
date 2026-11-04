"""A stand-in for Muse Spark in the offline agent test (no API key, no network).

It makes the tool calls the real agent should make when the demo files are
uploaded, so the test checks the tools and events, not the model. The app
itself always runs Muse Spark.
"""

import json
import uuid

from mixpilot.agent import Call, Step
from mixpilot.session import BACKEND

DEMO_MAPPING = json.loads((BACKEND / "demo_data" / "mapping.json").read_text())


class ScriptedLLM:
    def __init__(self):
        self.plan: list[Call] = []
        self.results: dict[str, dict] = {}
        self._last: list[Call] = []

    def _call(self, tool: str, **args) -> Call:
        return Call(tool, json.dumps(args), "call_" + uuid.uuid4().hex[:12])

    def step(self, input_items: list[dict], previous_response_id, on_text) -> Step:
        names = {c.call_id: c.name for c in self._last}
        for item in input_items:
            if item.get("type") == "function_call_output":
                self.results[names.get(item["call_id"], "?")] = json.loads(item["output"])
            elif item.get("role") == "user" and item["content"].startswith("[app] The user uploaded"):
                self.plan = [self._call("read_files"), self._call("map_columns", mapping=DEMO_MAPPING),
                             self._call("validate_data"), self._call("suggest_inputs")]
        if self.plan:
            self._last = [self.plan.pop(0)]
            return Step("scripted_" + uuid.uuid4().hex[:8], "", self._last)
        on_text("Done.")
        return Step("scripted_" + uuid.uuid4().hex[:8], "Done.", [])
