"""Agent loop on Meta Muse Spark (Responses API, tool calling, streamed text).

The model plans and talks; tools in mixpilot.tools do every computation. Training
is not one of its tools: only the Train model button starts it, and the agent
hears about it afterwards.

Turns are chained with previous_response_id, so reasoning carries across tool
calls. Instructions and tools are resent on every call because the Model API
does not carry them over.
"""

import json
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from mixpilot import tools
from mixpilot.session import BACKEND, Session

MAX_STEPS = 15
RETRIES = 3

SYSTEM = """You are MixPilot, an assistant for a ready-mix concrete plant's QC team. You retrain Meta's BOxCrete concrete-strength model on the plant's own data, and you find columns in their files that make it more accurate.

How a session goes:
1. When files arrive, call read_files, map_columns, validate_data and suggest_inputs, in that order, without stopping.
   - map_columns: map every BOxCrete field you can find, with its unit. A cementitious material BOxCrete has no field for goes under recipes.other_cementitious. Put in ignore only a column that holds the tested concrete's own strength result or its target (a specified or design strength). A test of a raw material is an input, not a result. Leave every other column unmapped: unmapped columns are the candidates for new inputs. Read the values of any notes column and use tests.exclude to drop cylinders it marks as invalid tests. If the tool returns errors, fix the mapping and call it again. Its result lists notes_still_counted: text on cylinders that still count as valid. If a value there says the test failed, the specimen was damaged or the test has to be repeated, call map_columns again with it in tests.exclude. A note about something else stays.
   - map_columns also takes column_notes: for each column you did not map, one short plain sentence on what it holds and why it could or couldn't matter for concrete strength. Nothing has been trained yet, so never say a column improves predictions.
   - If the files don't hold what BOxCrete needs (each mix's ingredients and strength tests by age), or the errors can't be fixed, stop and say in plain words what is missing. Never make data up.
2. Then write one message. Say what you read and which inputs you added ('added' in the tool's result). If 'offered' lists an input, it is useful but recorded on only part of the batches: say in a few plain words what it is and how much of the data has it (if recorded_from is set, say the plant started logging it then), and ask whether to use it in training. The card has a button for that; if the user agrees in chat, call use_input. Otherwise invite them to press Train model. You cannot start training yourself; if asked, point them to the button.
3. When a message says training finished, call get_results right away. Then reply: how much more accurate the model is than Meta's default setup retrained on their data (more_accurate_than_baseline_percent), honestly whether re-tuning Meta's settings helped (tuning_helped), and how much less CO2 the proposed mixes need (co2_saving_percent). If inputs_left_out lists any, say that training tested them and they didn't help.
Later the user may ask about the model or a mix: use predict or list_models.

Style:
- Work silently: write no text before or between tool calls, since the screen shows each step as it runs. Write one message when the turn is done.
- Keep that message to 1-3 short sentences for plant engineers. No headings, tables or bullet lists: the screen already shows cards and panels with the details.
- Quote only numbers a tool returned. Use recipe IDs and column names exactly as the tools give them.
- Say an improvement as a percentage ("predictions 40% more accurate"), not as before and after errors: the panels show those. A field called scatter_explained is a fraction (0.26 means 26%): how much of the strength differences between batches of one mix a column explains. If the column has conditioned_on, it means the differences those columns don't explain. It's not the same as predictions getting more accurate.
- Don't explain tools or mapping details unless asked.
- Messages starting with [app] come from the application, not the user."""


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

        load_dotenv(BACKEND / ".env")
        key = os.environ.get("META_API_KEY")
        if not key:
            raise RuntimeError("META_API_KEY is not set (put it in backend/.env)")
        self.client = OpenAI(base_url=os.environ.get("META_BASE_URL", "https://api.meta.ai/v1"), api_key=key)
        self.model = os.environ.get("MIXPILOT_MODEL", "muse-spark-1.3")
        self.effort = os.environ.get("MIXPILOT_REASONING", "low")

    def step(self, input_items: list[dict], previous_response_id: str | None, on_text) -> Step:
        stream = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM,
            tools=tools.SCHEMAS,
            input=input_items,
            previous_response_id=previous_response_id,
            reasoning={"effort": self.effort},
            stream=True,
        )
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
        calls = [Call(i.name, i.arguments, i.call_id) for i in final.output if i.type == "function_call"]
        return Step(final.id, final.output_text or "", calls)


def make_llm(session: Session):
    return MetaLLM()


class Agent:
    def __init__(self, session: Session, llm=None):
        self.s = session
        self.llm = llm or make_llm(session)

    def run(self, text: str) -> None:
        """One user turn: model calls tools until it answers in text."""
        s = self.s
        items, s.pending_input = s.pending_input + [{"role": "user", "content": text}], []
        for _ in range(MAX_STEPS):
            msg = None

            def on_text(delta: str):
                nonlocal msg
                if msg is None:
                    msg = s.emit("message_start", role="assistant")["seq"]
                s.emit("text_delta", message=msg, text=delta)

            try:
                step = self._call(items, on_text, lambda: msg is not None)
            except Exception as e:  # keep tool results so the next turn can deliver them
                s.pending_input = [i for i in items if i.get("type") == "function_call_output"]
                s.emit("error", text=str(e)[:500])
                return
            s.previous_response_id = step.id
            if msg is not None:  # text before tool calls is narration; the UI shows the steps instead
                s.emit("message_end", message=msg, text=step.text, interim=bool(step.calls))
            if not step.calls:
                tools.offer_training(s)
                return
            items = []
            for c in step.calls:
                s.emit("tool", name=c.name, status="running", label=tools.LABELS.get(c.name, c.name))
                try:
                    args = json.loads(c.arguments or "{}")
                    result = tools.call(s, c.name, args if isinstance(args, dict) else {})
                except json.JSONDecodeError as e:
                    result = dict(ok=False, errors=[f"arguments are not valid JSON: {e}"])
                except Exception as e:  # a bug on our side; tell the model instead of crashing the turn
                    result = dict(ok=False, errors=[f"{type(e).__name__}: {e}"])
                s.emit("tool", name=c.name, status="error" if result.get("ok") is False else "done",
                       label=tools.LABELS.get(c.name, c.name))
                items.append({"type": "function_call_output", "call_id": c.call_id,
                              "output": json.dumps(result, default=str)})
        s.pending_input = items
        s.emit("error", text=f"stopped after {MAX_STEPS} steps")

    def _call(self, items: list[dict], on_text, text_shown) -> Step:
        """One model call. Retried only while no text has reached the screen."""
        for attempt in range(RETRIES):
            try:
                return self.llm.step(items, self.s.previous_response_id, on_text)
            except Exception:
                if text_shown() or attempt == RETRIES - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError("unreachable")


def training_message(result: dict) -> str:
    return "[app] Training finished. " + json.dumps(result, default=str)


def upload_message(names: list[str]) -> str:
    return "[app] The user uploaded: " + ", ".join(names)
