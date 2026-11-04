"""Run the whole non-LLM pipeline on the demo Excel files and print what the agent would see."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixpilot import automl  # noqa: E402
from mixpilot.checks import candidate_inputs, issues  # noqa: E402
from mixpilot.prepare import build_dataset  # noqa: E402

DEMO = ROOT / "data" / "demo"
files = [DEMO / f for f in ("mix_designs.xlsx", "cylinder_breaks.xlsx", "daily_QC.xlsx")]
mapping = json.loads((ROOT / "tests" / "demo_mapping.json").read_text())
work = ROOT / "workspace" / "demo"
work.mkdir(parents=True, exist_ok=True)

df, report = build_dataset(files, mapping)
data = work / "dataset.csv"
df.to_csv(data, index=False)
print("report:", json.dumps(report))
print("issues:", *issues(df, list(mapping["extra_inputs"])), sep="\n  ")
base_mapping = {k: v for k, v in mapping.items() if k != "extra_inputs"}
print("candidates:", *candidate_inputs(files, base_mapping), sep="\n  ")

t = time.time()
print("site water:", automl.evaluate_input(data, "Site Water (gal/yd3)", ["Air (%)"]), f"{time.time() - t:.0f}s")

t = time.time()
res = automl.search(data, ["Air (%)", "Site Water (gal/yd3)"], on_result=lambda r: print(f"  done {r['label']:<40} {r['settings']} 28d={r['rmse_28d']}"))
print(f"search {time.time() - t:.0f}s")
for r in res["summary"]:
    print(f"  {r['row']:<45} 28d={r['rmse_28d']}  {r['settings']}")
print("best:", res["best"]["label"], res["best"]["rmse_28d"], "gain", res["gain"])

entry = automl.save_model(data, res["best"], work / "models", res["best"])
print("saved:", entry)
row = df[df["Time"] == 28].iloc[0].to_dict()
print("predict", row["Mix Name"], "measured", row["Strength (Mean)"], "->", automl.predict(work / "models" / entry["version"], [row]))
for fixed in ({}, {"Air (%)": 6.0}, {"Air (%)": 2.0}):
    r = automl.propose(work / "models" / entry["version"], fixed)
    print(f"proposals at {fixed or 'typical values'} (vs {r['compared_with']} similar mixes):")
    for p in r["proposals"]:
        print(f"  {p['target_psi']} psi class: {p['pred_psi']} ± {p['sd_psi']} psi, GWP {p['gwp']} vs {p['compare_to']} {p['compare_gwp']} -> {p['co2_saving']:.0%} less CO2")
