"""Agent loop on Meta Muse Spark (Responses API, tool calling, streamed text).

The model plans and talks; tools in mixpilot.tools do every computation.
Turns are chained with previous_response_id, so reasoning carries across
tool calls. Instructions and tools are resent on every call because the
Model API does not carry them over.
"""

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from mixpilot import tools
from mixpilot.automl import BACKEND
from mixpilot.session import Session

MAX_STEPS = 15

SYSTEM = """You are MixPilot, an assistant for a ready-mix concrete plant. You retrain Meta's BOxCrete concrete-strength model on the plant's own data and help the plant use it.

How a session goes:
1. When files arrive, call inspect_files, then prepare_data with a mapping from the plant's columns to BOxCrete's inputs. If a tool returns an error, fix the call and try again. Exclude strength tests that a notes column marks as invalid.
2. Look at candidate_inputs. Pick real material or process properties that BOxCrete does not use. Skip IDs, dates, outcomes (results of the mix, not inputs to it) and anything that duplicates a mapped input.
   - The single most useful one recorded for at least 90% of mixes: add_input.
   - The single most useful one recorded for fewer mixes: test_input. Then tell the user how much more accurate it made the predictions on the mixes that have it, and ask whether to use it in training. The card has a button for that; if the user agrees in chat, call use_input.
3. End your turn with a short summary. The app then shows a Train button; the user starts training with it.
4. When a message says training finished, call propose_mixes (extra inputs stay at the plant's typical value unless the user asks for another). Then reply.
Later the user may ask about the model or a mix: use predict_strength or list_models.

Style:
- Work silently: write no text before or between tool calls, since the screen shows each step as it runs. Write one message when the turn is done.
- Keep that message to 1-3 short sentences. No headings, tables or bullet lists: the screen already shows cards and panels with the details.
- Quote only numbers a tool returned. Use the plant's units (psi, lb/yd3).
- Say an improvement as a percentage ("predictions 40% more accurate", from the tool's gain), not as before and after errors: the panels show those.
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
                step = self.llm.step(items, s.previous_response_id, on_text)
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
                    result = {"error": f"arguments are not valid JSON: {e}"}
                except Exception as e:  # a bug on our side; tell the model instead of crashing the turn
                    result = {"error": f"{type(e).__name__}: {e}"}
                s.emit("tool", name=c.name, status="error" if "error" in result else "done",
                       label=tools.LABELS.get(c.name, c.name))
                items.append({"type": "function_call_output", "call_id": c.call_id,
                              "output": json.dumps(result, default=str)})
        s.emit("error", text=f"stopped after {MAX_STEPS} steps")


def training_message(result: dict) -> str:
    return "[app] Training finished. " + json.dumps(result, default=str)


def upload_message(names: list[str]) -> str:
    return "[app] The user uploaded: " + ", ".join(names)
