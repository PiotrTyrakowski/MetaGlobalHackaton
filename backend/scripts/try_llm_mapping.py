"""Can Muse Spark write the column mapping from read_files output alone?"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from mixpilot import tools


def main() -> None:
    load_dotenv()
    client = OpenAI(api_key=os.environ["META_API_KEY"], base_url="https://api.meta.ai/v1")
    s = tools.Session()
    s.add_files(sorted(Path("demo_data/plant").glob("*.xlsx")))
    help_ = tools.mapping_help()
    tool = dict(type="function", name="map_columns",
                description="Map the plant's columns to BOxCrete's inputs. Units must be one of: "
                            + json.dumps(help_["fields"]) + ". Example: " + json.dumps(help_["example"]),
                parameters={"type": "object", "properties": {"mapping": {"type": "object"}},
                            "required": ["mapping"]})
    messages = [{"role": "user", "content":
                 "These are a concrete plant's uploaded files, profiled. Call map_columns with the mapping. "
                 "Map every BOxCrete field you can find (cement, fly_ash, slag, water, hrwr, fine, coarse, "
                 "cure_temp, age, strength). Do not map other columns; they become candidate inputs.\n\n"
                 + json.dumps(tools.read_files(s), default=str)}]
    for attempt in range(3):
        t0 = time.perf_counter()
        r = client.responses.create(model="muse-spark-1.3", input=messages, tools=[tool])
        calls = [o for o in r.output if o.type == "function_call"]
        print(f"attempt {attempt + 1}: {time.perf_counter() - t0:.1f}s, "
              f"{r.usage.input_tokens} in / {r.usage.output_tokens} out tokens")
        if not calls:
            print("no tool call:", r.output_text[:500])
            return
        mapping = json.loads(calls[0].arguments)["mapping"]
        result = tools.map_columns(s, mapping)
        if result["ok"]:
            print(json.dumps(mapping, indent=1))
            print("OK:", {k: result[k] for k in ("recipes", "batches", "rows", "tests_matched", "assumptions")})
            print("candidates:", [c["column"] for c in result["candidate_columns"]])
            return
        print("errors:", result["errors"])
        messages += [o.model_dump() for o in r.output] + [
            {"type": "function_call_output", "call_id": calls[0].call_id, "output": json.dumps(result)}]


if __name__ == "__main__":
    main()
