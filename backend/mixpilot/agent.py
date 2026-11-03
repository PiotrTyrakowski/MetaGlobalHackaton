"""The MixPilot agent: Muse Spark on Meta's Responses API, calling our tools.

The agent decides what to call and explains results; every number comes from a
tool. Tool results also go out as UI events ("artifact"), so the panels never
depend on the agent's text.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections.abc import Callable

from dotenv import load_dotenv
from openai import OpenAI

from mixpilot import tools
from mixpilot.tools import Session

load_dotenv()

MODEL = os.environ.get("MIXPILOT_MODEL", "muse-spark-1.3")
EFFORT = os.environ.get("MIXPILOT_EFFORT", "low")
MAX_STEPS = 14

SYSTEM = """You are MixPilot, an assistant for a concrete plant's QC team. You retrain Meta's \
open-source BOxCrete concrete-strength model on the plant's own data, so it predicts their mixes \
better, and you look for extra columns in their files that make it better still.

When the user uploads files, work through the tools in this order without stopping:
1. read_files.
2. map_columns. Map every BOxCrete field you can find (cement, fly_ash, slag, water, hrwr, fine, \
coarse, cure_temp, age, strength) with its unit. List silica fume, metakaolin or other \
cementitious materials BOxCrete has no field for under recipes.other_cementitious (they stay \
candidate inputs). Do not map any other column: unmapped columns become candidate inputs. If it \
returns errors, fix the mapping and call it again.
3. validate_data.
4. suggest_inputs.
Then write ONE message of at most 2 short sentences (under 40 words). The screen shows a card \
per finding with all the numbers, so don't repeat them: say what you read (recipes, batches, \
cylinders) and which inputs you'll add, then invite them to press Train model. Never train before \
they approve.

After train_model, call get_results right away (no message in between). Then write at most 2 \
short sentences (under 45 words): the 28-day error of Meta's default setup vs the winner in psi, \
honestly whether re-tuning Meta's settings helped, and how much less CO2 the proposals need. The \
panels show the rest.

Later questions: use predict for "what will this mix reach", list_models for saved versions.

Rules:
- Only quote numbers that a tool returned. Round sensibly. Strength in psi; relative errors as percentages (use the *_pct fields).
- Before a tool call, at most one short sentence on what you're doing. Never repeat in your final message what you already said earlier in the same turn.
- Short, plain sentences for plant engineers. No tables and no markdown headings: the screen \
shows panels with the details.
- Recipe IDs and column names exactly as the tools give them.
- Answers to questions: one or two sentences."""


def _schema(props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or []}


def tool_specs() -> list[dict]:
    help_ = tools.mapping_help()
    return [
        dict(type="function", name="read_files",
             description="Profile every uploaded sheet: header row, columns, kinds, coverage, samples, "
                         "and likely join keys between files.",
             parameters=_schema()),
        dict(type="function", name="map_columns",
             description="Map the plant's columns to BOxCrete's inputs and join the files into one table. "
                         "Sections: recipes (one row per mix design; key), batches (one row per batch; key, "
                         "recipe_key; optional), tests (one row per cylinder; batch_key or recipe_key). "
                         f"Fields and allowed units: {json.dumps(help_['fields'])}. Required: "
                         f"{help_['required']}. Example: {json.dumps(help_['example'])}",
             parameters=_schema({"mapping": {"type": "object"}}, ["mapping"])),
        dict(type="function", name="validate_data",
             description="Check the mapped data: coverage, ranges vs Meta's training data, cleaned cells, "
                         "assumptions.", parameters=_schema()),
        dict(type="function", name="suggest_inputs",
             description="Test every candidate column: batch measurements by whether they explain the "
                         "scatter between batches of the same recipe (permutation test); recipe columns by "
                         "refitting BOxCrete with and without them. Verdicts: use, collect (useful but "
                         "incomplete), proxy (matters but is a label/date standing for another column), "
                         "reject, not_tested. Takes ~25 s.", parameters=_schema()),
        dict(type="function", name="train_model",
             description="AutoML: retrain BOxCrete on the plant's data with and without the approved "
                         "inputs, re-tune Meta's hard-coded settings with BoTorch, re-check on new fold "
                         "splits, save the winner as the next model version. Only after the user approved "
                         "(they press Train model). Takes ~2 min.",
             parameters=_schema({"inputs": {"type": "array", "items": {"type": "string"}}}, ["inputs"])),
        dict(type="function", name="get_results",
             description="CO2 vs strength of the plant's recipes, and lower-CO2 mixes to test next, from "
                         "the active saved model.",
             parameters=_schema({"version": {"type": "string"}})),
        dict(type="function", name="predict",
             description="Predict strength of existing recipes, optionally changed. mixes: [{recipe: id, "
                         "changes: {field: {value, unit}}, extras: {column: value}}]. Fields: cement, "
                         "fly_ash, slag, water, hrwr, fine, coarse. Extras default to the plant's median.",
             parameters=_schema({"mixes": {"type": "array", "items": {"type": "object"}},
                                 "ages": {"type": "array", "items": {"type": "number"}},
                                 "version": {"type": "string"}}, ["mixes"])),
        dict(type="function", name="list_models", description="Saved model versions for this plant.",
             parameters=_schema()),
    ]


LABELS = {"read_files": "Reading your files", "map_columns": "Mapping columns to BOxCrete",
          "validate_data": "Checking the data", "suggest_inputs": "Testing extra columns",
          "train_model": "Training on your data", "get_results": "Looking for lower-CO₂ mixes",
          "predict": "Predicting", "list_models": "Listing saved models"}


class Agent:
    def __init__(self, session: Session, emit: Callable[[dict], None]):
        self.session = session
        self.emit = emit
        self.items: list[dict] = [{"role": "system", "content": SYSTEM}]
        self.client = OpenAI(api_key=os.environ["META_API_KEY"], base_url="https://api.meta.ai/v1")
        self.specs = tool_specs()
        self.approved_inputs: list[str] | None = None

    def run_turn(self, text: str) -> None:
        self.items.append({"role": "user", "content": text})
        try:
            for _ in range(MAX_STEPS):
                r = self._call()
                calls = []
                for o in r.output:
                    self.items.append(o.model_dump(exclude_none=True))
                    if o.type == "message":
                        msg = "".join(c.text for c in o.content if getattr(c, "type", "") == "output_text")
                        if msg.strip():
                            self.emit(dict(type="message", role="assistant", text=msg.strip(),
                                           phase=getattr(o, "phase", None) or "final"))
                    elif o.type == "function_call":
                        calls.append(o)
                if not calls:
                    break
                for call in calls:
                    result = self._run_tool(call.name, call.arguments)
                    self.items.append({"type": "function_call_output", "call_id": call.call_id,
                                       "output": json.dumps(_for_llm(call.name, result), default=str)})
        except Exception as e:  # keep the session alive; show the error
            traceback.print_exc()
            self.emit(dict(type="error", text=f"{type(e).__name__}: {e}"))
        finally:
            self.emit(dict(type="done"))

    def _call(self):
        for attempt in range(3):
            try:
                return self.client.responses.create(model=MODEL, input=self.items, tools=self.specs,
                                                    reasoning={"effort": EFFORT})
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    def _run_tool(self, name: str, arguments: str) -> dict:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as e:
            return dict(ok=False, errors=[f"arguments are not valid JSON: {e}"])
        self.emit(dict(type="tool_start", tool=name, label=LABELS.get(name, name)))
        t0 = time.perf_counter()

        def progress(m: str) -> None:
            self.emit(dict(type="progress", tool=name, text=m))

        s = self.session
        try:
            if name == "read_files":
                result = tools.read_files(s)
            elif name == "map_columns":
                result = tools.map_columns(s, args.get("mapping", {}))
            elif name == "validate_data":
                result = tools.validate_data(s)
            elif name == "suggest_inputs":
                result = tools.suggest_inputs(s, progress=progress)
            elif name == "train_model":
                inputs = args.get("inputs", [])
                if self.approved_inputs is None or set(inputs) != set(self.approved_inputs):
                    result = dict(ok=False, errors=["not approved: ask the user to press Train model"
                                                    + (f" (approved: {self.approved_inputs})"
                                                       if self.approved_inputs is not None else "")])
                else:
                    result = tools.train_automl(s, inputs, progress=progress)
            elif name == "get_results":
                result = tools.results(s, args.get("version"))
            elif name == "predict":
                result = tools.predict(s, args.get("mixes", []), args.get("ages") or [7, 28],
                                       args.get("version"))
            elif name == "list_models":
                result = tools.list_models(s)
            else:
                result = dict(ok=False, errors=[f"unknown tool {name}"])
        except Exception as e:
            traceback.print_exc()
            result = dict(ok=False, errors=[f"{type(e).__name__}: {e}"])
        self.emit(dict(type="tool_end", tool=name, ok=result.get("ok", True),
                       seconds=round(time.perf_counter() - t0, 1)))
        for artifact in _artifacts(name, result, s):
            self.emit(artifact)
        return result


def _for_llm(name: str, result: dict) -> dict:
    """Trim bulky fields the agent doesn't need."""
    if name == "train_model" and result.get("ok"):
        out = {k: v for k, v in result.items() if k not in ("method",)}
        out["leaderboard"] = [{k: v for k, v in r.items() if k != "error_all_ages"} for r in out["leaderboard"]]
        out["winner"] = {k: v for k, v in out["winner"].items() if k != "error_all_ages"}
        return out
    if name == "get_results":
        return {**result, "recipes": f"{len(result.get('recipes', []))} recipes (shown in the chart)"}
    return result


def _artifacts(name: str, result: dict, s: Session) -> list[dict]:
    if result.get("ok") is False:
        return []
    if name in ("validate_data", "suggest_inputs") and s.validation is not None:
        return [dict(type="artifact", kind="data",
                     data=dict(validation=s.validation, suggestions=s.suggestions, mapping=s.map_report))]
    if name == "train_model":
        return [dict(type="artifact", kind="training", data=s.training),
                dict(type="artifact", kind="models", data=tools.list_models(s))]
    if name == "get_results":
        return [dict(type="artifact", kind="results", data=result)]
    if name == "list_models":
        return [dict(type="artifact", kind="models", data=result)]
    return []
