"""Run one demo conversation in-process and print the event stream compactly.

MIXPILOT_LLM=scripted for the stand-in, unset for Muse Spark. --train also runs AutoML (~3 min).
"""

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixpilot import tools  # noqa: E402
from mixpilot.agent import Agent, training_message, upload_message  # noqa: E402
from mixpilot.server import DEMO_FILES  # noqa: E402
from mixpilot.session import new_session  # noqa: E402

s = new_session()
s.uploads.mkdir(parents=True, exist_ok=True)
for f in DEMO_FILES:
    shutil.copy(f, s.uploads / f.name)
    s.files.append(s.uploads / f.name)

seen = -1


def show():
    global seen
    text = ""
    for e in s.events_after(seen):
        seen = e["seq"]
        t = e["type"]
        if t == "text_delta":
            text += e["text"]
            continue
        if text:
            print("   assistant:", text.strip())
            text = ""
        if t == "tool":
            if e["status"] != "running":
                print(f"   tool {e['name']}: {e['status']}")
        elif t in ("card", "train_button", "error", "tool_progress"):
            print("  ", t, json.dumps({k: v for k, v in e.items() if k not in ("seq", "type", "time")}))
        elif t == "artifact":
            d = e["data"]
            print(f"   artifact {e['name']}: {d.get('status', '')} {list(d)[:6]}")
    if text:
        print("   assistant:", text.strip())


t0 = time.time()
Agent(s).run(upload_message([f.name for f in DEMO_FILES]))
show()
print(f"prep turn {time.time() - t0:.0f}s, accepted={s.accepted}, tested={list(s.tested)}")

if "--train" in sys.argv:
    t0 = time.time()
    result = tools.train(s)
    Agent(s).run(training_message(result))
    show()
    print(f"train turn {time.time() - t0:.0f}s, model={s.model}")

for q in sys.argv[1:]:
    if not q.startswith("--"):
        print("user:", q)
        Agent(s).run(q)
        show()
