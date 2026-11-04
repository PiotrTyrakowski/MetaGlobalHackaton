"""The MixPilot agent: Muse Spark on Meta's Responses API, calling our tools.

The agent decides what to call and explains results; every number comes from a
tool. Tool results also go out as UI events ("artifact"), so the panels never
depend on the agent's text. Training is not one of its tools: only the Train
model button starts it (Agent.train), and the agent hears about it afterwards.

Turns are chained with previous_response_id, so reasoning carries across tool
calls. Instructions and tools are resent on every call because the Model API
does not carry them over.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from mixpilot import tools
from mixpilot.tools import Session

MAX_STEPS = 14
RETRIES = 3

SYSTEM = """You are MixPilot, an assistant for a concrete plant's QC team. You retrain Meta's \
open-source BOxCrete concrete-strength model on the plant's own data, so it predicts their mixes \
better, and you look for extra columns in their files that make it better still.

When the user uploads files, work through the tools in this order without stopping:
1. read_files.
2. map_columns. Map every BOxCrete field you can find (cement, fly_ash, slag, water, hrwr, fine, \
coarse, cure_temp, age, strength) with its unit. List silica fume, metakaolin or other \
cementitious materials BOxCrete has no field for under recipes.other_cementitious (they stay \
candidate inputs). Do not map any other column: unmapped columns become candidate inputs. Put \
cylinders a notes column marks as invalid (cap failure, for example) under tests.exclude. If it \
returns errors, fix the mapping and call it again.
3. validate_data.
4. suggest_inputs.
Then write ONE message of at most 2 short sentences (under 40 words). The screen shows a card \
per finding with all the numbers, so don't repeat them: say what you read (recipes, batches, \
cylinders) and which inputs you'll add, then invite them to press Train model. You cannot start \
training yourself; if they ask you to, point them to the button. Columns with verdict "collect" \
are not used: say recording them from now on would help.

A message starting with [app] comes from the application, not the user. When it says training \
finished, call get_results right away (no message in between). Then write at most 2 short \
sentences (under 45 words): the 28-day error of Meta's default setup vs the winner in psi, \
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
                         "recipe_key; optional), tests (one row per cylinder; batch_key if cylinders name a "
                         "batch, else recipe_key; optional exclude: rows to drop by text in a column). IDs "
                         "match ignoring case and punctuation. Breaks after 28 days are dropped. "
                         f"Fields and allowed units: {json.dumps(help_['fields'])}. Required: "
                         f"{help_['required']}. Example: {json.dumps(help_['example'])}",
             parameters=_schema({"mapping": {"type": "object"}}, ["mapping"])),
        dict(type="function", name="validate_data",
             description="Check the mapped data: coverage, ranges vs Meta's training data, cleaned cells, "
                         "rows left out, assumptions.", parameters=_schema()),
        dict(type="function", name="suggest_inputs",
             description="Test every candidate column: batch measurements by whether they explain the "
                         "scatter between batches of the same recipe (permutation test); recipe columns by "
                         "refitting BOxCrete with and without them. Verdicts: use, collect (useful but "
                         "incomplete), proxy (matters but is a label/date standing for another column), "
                         "reject, not_tested. Takes ~25 s.", parameters=_schema()),
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


@dataclass
class Call:
    name: str
    arguments: str
    call_id: str


@dataclass
class Step:
    id: str
    text: str
    calls: list[Call] = field(default_factory=list)


class MetaLLM:
    """Muse Spark through the OpenAI SDK pointed at the Meta Model API."""

    def __init__(self):
        from openai import OpenAI

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        key = os.environ.get("META_API_KEY")
        if not key:
            raise RuntimeError("META_API_KEY is not set (put it in backend/.env, or run with MIXPILOT_LLM=scripted)")
        self.client = OpenAI(api_key=key, base_url=os.environ.get("META_BASE_URL", "https://api.meta.ai/v1"))
        self.model = os.environ.get("MIXPILOT_MODEL", "muse-spark-1.3")
        self.effort = os.environ.get("MIXPILOT_EFFORT", "low")

    def step(self, input_items: list[dict], previous_response_id: str | None, specs: list[dict],
             on_text: Callable[[str], None]) -> Step:
        stream = self.client.responses.create(model=self.model, instructions=SYSTEM, tools=specs, input=input_items,
                                              previous_response_id=previous_response_id,
                                              reasoning={"effort": self.effort}, stream=True)
        final = None
        for ev in stream:
            if ev.type == "response.output_text.delta":
                on_text(ev.delta)
            elif ev.type in ("response.completed", "response.incomplete"):
                final = ev.response
            elif ev.type == "response.failed":
                raise RuntimeError(f"model call failed: {ev.response.error}")
            elif ev.type == "error":
                raise RuntimeError(f"model error: {getattr(ev, 'message', ev)}")
        if final is None:
            raise RuntimeError("model stream ended without a response")
        calls = [Call(o.name, o.arguments, o.call_id) for o in final.output if o.type == "function_call"]
        return Step(final.id, final.output_text or "", calls)


def make_llm():
    if os.environ.get("MIXPILOT_LLM") == "scripted":
        from mixpilot.scripted_llm import ScriptedLLM

        return ScriptedLLM()
    return MetaLLM()


class Agent:
    def __init__(self, session: Session, emit: Callable[[dict], None], llm=None):
        self.session = session
        self.emit = emit
        self.llm = llm
        self.specs = tool_specs()
        self.previous_response_id: str | None = None
        self.pending: list[dict] = []  # tool results the last failed call didn't deliver
        self._streams = 0

    def run_turn(self, text: str) -> None:
        try:
            self._loop(text)
        except Exception as e:  # keep the session alive; show the error
            traceback.print_exc()
            self.emit(dict(type="error", text=f"{type(e).__name__}: {e}"))
        finally:
            self.emit(dict(type="done"))

    def train(self, inputs: list[str]) -> None:
        """The Train model button: run AutoML, then let the agent report on it."""
        s = self.session
        try:
            result = self._tracked("train_model", lambda progress: tools.train_automl(s, inputs, progress=progress))
            if result.get("ok") is False:
                self.emit(dict(type="error", text="; ".join(result.get("errors", ["training failed"]))))
                return
            self._loop("[app] Training finished. " + json.dumps(_for_llm("train_model", result), default=str))
        except Exception as e:
            traceback.print_exc()
            self.emit(dict(type="error", text=f"{type(e).__name__}: {e}"))
        finally:
            self.emit(dict(type="done"))

    def _loop(self, text: str) -> None:
        if self.llm is None:
            self.llm = make_llm()
        items, self.pending = self.pending + [{"role": "user", "content": text}], []
        for _ in range(MAX_STEPS):
            try:
                step, stream = self._call(items)
            except Exception as e:  # the model never saw these results; deliver them next turn
                self.pending = [i for i in items if i.get("type") == "function_call_output"]
                traceback.print_exc()
                self.emit(dict(type="error", text=f"{type(e).__name__}: {str(e)[:300]}"))
                return
            self.previous_response_id = step.id
            if step.text.strip():
                self.emit(dict(type="message", role="assistant", text=step.text.strip(), stream=stream,
                               phase="commentary" if step.calls else "final"))
            if not step.calls:
                return
            items = []
            for c in step.calls:
                result = self._run_tool(c.name, c.arguments)
                items.append({"type": "function_call_output", "call_id": c.call_id,
                              "output": json.dumps(_for_llm(c.name, result), default=str)})
        self.pending = items
        self.emit(dict(type="error", text=f"stopped after {MAX_STEPS} steps"))

    def _call(self, items: list[dict]) -> tuple[Step, int]:
        """One model call, streamed. Retried only while no text has reached the screen."""
        self._streams += 1
        stream, sent = self._streams, False

        def on_text(delta: str) -> None:
            nonlocal sent
            sent = True
            self.emit(dict(type="text_delta", stream=stream, text=delta))

        for attempt in range(RETRIES):
            try:
                return self.llm.step(items, self.previous_response_id, self.specs, on_text), stream
            except Exception:
                if sent or attempt == RETRIES - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError("unreachable")

    def _run_tool(self, name: str, arguments: str) -> dict:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as e:
            return dict(ok=False, errors=[f"arguments are not valid JSON: {e}"])
        s = self.session
        run: dict[str, Callable[[Callable[[str], None]], dict]] = {
            "read_files": lambda _: tools.read_files(s),
            "map_columns": lambda _: tools.map_columns(s, args.get("mapping", {})),
            "validate_data": lambda _: tools.validate_data(s),
            "suggest_inputs": lambda progress: tools.suggest_inputs(s, progress=progress),
            "get_results": lambda _: tools.results(s, args.get("version")),
            "predict": lambda _: tools.predict(s, args.get("mixes", []), args.get("ages") or [7, 28],
                                               args.get("version")),
            "list_models": lambda _: tools.list_models(s),
        }
        if name not in run:
            return dict(ok=False, errors=[f"unknown tool {name}"])
        return self._tracked(name, run[name])

    def _tracked(self, name: str, fn: Callable[[Callable[[str], None]], dict]) -> dict:
        """Run a tool with its step events and the UI artifacts it produces."""
        self.emit(dict(type="tool_start", tool=name, label=LABELS.get(name, name)))
        t0 = time.perf_counter()

        def progress(m: str) -> None:
            self.emit(dict(type="progress", tool=name, text=m))

        try:
            result = fn(progress)
        except Exception as e:
            traceback.print_exc()
            result = dict(ok=False, errors=[f"{type(e).__name__}: {e}"])
        self.emit(dict(type="tool_end", tool=name, ok=result.get("ok", True),
                       seconds=round(time.perf_counter() - t0, 1)))
        for artifact in _artifacts(name, result, self.session):
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
