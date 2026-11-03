"""Run the tools end to end on the demo files, with a hand-written mapping."""

import json
import sys
import time
from pathlib import Path

from mixpilot import dataset, excel

PLANT = Path("demo_data/plant")
MAPPING = {
    "recipes": {
        "file": "mix_designs.xlsx", "key": "Mix ID",
        "fields": {
            "cement": {"column": "Type IL Cement (lb/cy)", "unit": "lb/yd3"},
            "fly_ash": {"column": "Class F Fly Ash (lb/cy)", "unit": "lb/yd3"},
            "slag": {"column": "Slag Cement (lb/cy)", "unit": "lb/yd3"},
            "water": {"column": "Water (gal/cy)", "unit": "gal/yd3"},
            "hrwr": {"column": "HRWR (oz/cwt)", "unit": "fl_oz/cwt"},
            "fine": {"column": "Sand SSD (lb/cy)", "unit": "lb/yd3"},
            "coarse": {"column": "#57 Stone SSD (lb/cy)", "unit": "lb/yd3"},
        },
        "other_cementitious": [{"column": "Silica Fume (lb/cy)", "unit": "lb/yd3"}],
    },
    "batches": {"file": "daily_QC.xlsx", "key": "Ticket No.", "recipe_key": "Mix Code",
                "fields": {"cure_temp": {"column": "Cure Tank (F)", "unit": "F"}}},
    "tests": {"file": "cylinder_breaks.xlsx", "batch_key": "Ticket #",
              "fields": {"age": {"column": "Age", "unit": "days"},
                         "strength": {"column": "Strength (psi)", "unit": "psi"}}},
}


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
