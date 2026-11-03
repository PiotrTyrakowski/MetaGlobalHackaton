"""A stand-in for Muse Spark that replays the demo flow, for UI work without an API key.

Enable with MIXPILOT_LLM=scripted. It makes the same tool calls the real
agent should make on the demo files and fills its replies from the tool
results. The demo video uses the real model.
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
        text = self.reply() if self.reply else "I'm running in scripted mode, so I can only replay the demo."
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
        elif text.startswith("[app] Training finished"):
            self.results["train"] = json.loads(text.split(" ", 3)[3])
            self.plan = [self._call("propose_mixes", fixed={"Air (%)": 6})]
            self.reply = self._results_reply

    def _prep_reply(self) -> str:
        rep = self.results["prepare_data"]["report"]
        t = self.results["test_input"]
        return (f"Your three files give {rep['mixes']} mixes with {rep['cylinders_used']} usable cylinders. "
                f"Air content is recorded for every mix, so I added it as a new input. "
                f"Water added on site is logged for only {t['mixes_with_value']} of {t['mixes']} mixes; on those it cuts the "
                f"28-day error from {t['rmse_28d_without']} to {t['rmse_28d_with']} psi, so it's worth logging for every truck.")

    def _results_reply(self) -> str:
        tr = self.results["train"]
        board = tr["leaderboard"]
        best = next(r for r in board if r["chosen"])
        props = self.results["propose_mixes"]["proposals"]
        top = max(props, key=lambda p: p["co2_saving"]) if props else None
        text = (f"The new setup cuts the 28-day error from {board[0]['rmse_28d']} to {best['rmse_28d']} psi "
                f"({tr['gain_vs_meta_default']:.0%} better than Meta's default) and is saved as {tr['saved_model']['version']}.")
        if top:
            text += (f" At 6% air it finds a {top['target_psi']} psi mix with {top['co2_saving']:.0%} less CO2 "
                     f"than {top['compare_to']}.")
        return text
