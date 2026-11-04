"""A stand-in for Muse Spark in the offline agent test (no API key, no network).

It makes the tool calls the real agent should make when the demo files are
uploaded, so the test checks the tools and events, not the model. The app
itself always runs Muse Spark.
"""

import json
import uuid

from mixpilot.agent import Call, Step
from mixpilot.session import BACKEND, Session

DEMO_MAPPING = json.loads((BACKEND / "tests" / "demo_mapping.json").read_text())


def demo_prepare_args() -> dict:
    m = DEMO_MAPPING
    return {
        "mix_id": [{"file": f, "column": c} for f, c in m["mix_id"].items()],
        "fields": [{"target": t, **src} for t, src in m["fields"].items()],
        "exclude": [{"column": r["column"], "contains": r["contains"]} for r in m["exclude"]],
    }


class ScriptedLLM:
    def __init__(self, session: Session):
        self.s = session
        self.plan: list[Call] = []
        self.results: dict[str, dict] = {}
        self.reply = None

    def _call(self, tool: str, **args) -> Call:
        return Call(tool, json.dumps(args), "call_" + uuid.uuid4().hex[:12])

    def step(self, input_items: list[dict], previous_response_id, on_text) -> Step:
        calls_by_id = {c.call_id: c.name for c in getattr(self, "_last", [])}
        for item in input_items:
            if item.get("type") == "function_call_output":
                self.results[calls_by_id.get(item["call_id"], "?")] = json.loads(item["output"])
            elif item.get("role") == "user":
                self._plan_for(item["content"])
        if self.plan:
            batch = [self.plan.pop(0)]
            self._last = batch
            return Step("scripted_" + uuid.uuid4().hex[:8], "", batch)
        text = self.reply() if self.reply else "Done."
        self.reply = None
        for word in text.split(" "):
            on_text(word + " ")
        return Step("scripted_" + uuid.uuid4().hex[:8], text, [])

    def _plan_for(self, text: str) -> None:
        if text.startswith("[app] The user uploaded"):
            self.plan = [
                self._call("inspect_files"),
                self._call("prepare_data", **demo_prepare_args()),
                self._call("add_input", name="Air (%)", file="daily_QC.xlsx", column="Air (%)"),
                self._call("test_input", name="Site Water (gal/yd3)", file="daily_QC.xlsx",
                           column="Water Added on Site (gal/yd³)"),
            ]
            self.reply = self._prep_reply

    def _prep_reply(self) -> str:
        rep = self.results["prepare_data"]["report"]
        t = self.results["test_input"]
        return (f"Your three files give {rep['mixes']} mixes with {rep['cylinders_used']} usable cylinders. "
                f"Air content is recorded for every mix, so I added it as a new input. "
                f"Water added on site is logged for only {t['mixes_with_value']} of {t['mixes']} mixes; on those it cuts the "
                f"28-day error from {t['rmse_28d_without']} to {t['rmse_28d_with']} psi. Use it in training?")

