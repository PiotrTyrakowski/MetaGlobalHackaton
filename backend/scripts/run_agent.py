"""Drive the agent from the terminal on the demo files (no server, no UI)."""

import json
import sys
import time
from pathlib import Path

from mixpilot import tools
from mixpilot.agent import Agent

T0 = time.perf_counter()


def show(ev: dict) -> None:
    t = f"[{time.perf_counter() - T0:5.0f}s]"
    if ev["type"] == "text_delta":
        return
    if ev["type"] == "message":
        print(f"{t} {ev['role'].upper()} ({ev.get('phase', '')}): {ev['text']}\n", flush=True)
    elif ev["type"] == "tool_start":
        print(f"{t}   -> {ev['tool']}", flush=True)
    elif ev["type"] == "tool_end":
        print(f"{t}   <- {ev['tool']} ok={ev['ok']} ({ev['seconds']}s)", flush=True)
    elif ev["type"] == "progress":
        print(f"{t}      {ev['text']}", flush=True)
    elif ev["type"] == "artifact":
        print(f"{t}   [artifact: {ev['kind']}]", flush=True)
    elif ev["type"] == "error":
        print(f"{t} ERROR {ev['text']}", flush=True)


def main() -> None:
    s = tools.Session(plant="Riverside Ready-Mix")
    s.add_files(sorted(Path("demo_data/plant").glob("*.xlsx")))
    agent = Agent(s, show)
    agent.run_turn("Here are our lab files: mix_designs.xlsx, cylinder_breaks.xlsx, daily_QC.xlsx.")
    if "--train" in sys.argv and s.suggestions:
        agent.train(s.suggestions["use"])
        agent.run_turn("What will R4K-PC-AE-1 reach at 7 days if air comes in at 7.5%?")
    print(f"mapping used: {json.dumps(s.mapping)}")


if __name__ == "__main__":
    main()
