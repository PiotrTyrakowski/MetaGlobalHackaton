"""Run the tools end to end on the demo files, with a hand-written mapping."""

import json
import sys
import time
from pathlib import Path

from mixpilot import dataset, excel

PLANT = Path("demo_data/plant")
MAPPING = json.loads(Path("demo_data/mapping.json").read_text())


def main() -> None:
    sheets = [s for f in sorted(PLANT.glob("*.xlsx")) for s in excel.read_sheets(f)]
    print("== read_files")
    for s in sheets:
        p = excel.profile(s)
        print(f"{p['file']} [{p['sheet']}] header row {p['header_row']}, {p['rows']} rows")
        for c in p["columns"]:
            print(f"   {c['name']:24s} {c['kind']:11s} {c['coverage']:.0%}  {c['samples'][:3]}")
    print("join keys:", json.dumps(excel.join_keys(sheets)[:6], indent=None))

    print("== map_columns")
    ds, report = dataset.build(sheets, MAPPING)
    print(json.dumps(report, indent=1, default=str))
    print({k: (v.level, v.kind) for k, v in ds.candidates.items()})
    print(ds.rows.head(3).to_string())

    # conversions vs the generator's hidden truth
    truth = {b["ticket"]: b for b in json.load(open("demo_data/truth.json"))["batches"]}
    r = ds.rows.drop_duplicates("batch").set_index("batch")
    for f, tf in [("cement", "cement"), ("fly_ash", "fly_ash"), ("slag", "slag"), ("water", "water"),
                  ("hrwr", "hrwr_kg_m3"), ("fine", "fine"), ("coarse", "coarse")]:
        err = max(abs(r.loc[str(t), f] - b[tf]) / max(b[tf], 1) for t, b in truth.items())
        print(f"  {f:8s} max relative error vs truth {err:.2%}")
    air = ds.feature("Air (%)")
    print("  air max abs error", max(abs(air[str(t)] - b["air"]) for t, b in truth.items()))

    if "--suggest" in sys.argv or "--validate" in sys.argv:
        from mixpilot import validate
        print("== validate")
        v = validate.validate(ds, MAPPING, report)
        print(json.dumps(v["summary"]))
        for i in v["boxcrete_inputs"]:
            print(f"   {i['input']:20s} {str(i['source']):45s} plant {i['plant_range']}  meta {i['meta_range']}")
        for i in v["issues"]:
            print(f"   [{i['level']}] {i['message']}")

    if "--suggest" in sys.argv:
        from mixpilot import inputs
        print("== suggest_inputs")
        t0 = time.perf_counter()
        s = inputs.suggest_inputs(ds, progress=lambda m: print("   ...", m))
        print(f"({time.perf_counter() - t0:.0f}s) use={s['use']} collect={s['collect']}")
        for r in s["columns"]:
            extra = {k: v for k, v in r.items() if k not in ("column", "verdict", "level", "kind")}
            print(f"   {r['verdict']:10s} {r['column']:22s} {json.dumps(extra, default=str)}")


if __name__ == "__main__":
    main()
