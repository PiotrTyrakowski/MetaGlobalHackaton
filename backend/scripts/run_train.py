"""Train + save + results + predict on the demo data (uses a short search by default)."""

import json
import sys
import time

from run_pipeline import MAPPING, PLANT
from mixpilot import tools


def main() -> None:
    s = tools.Session(plant="Riverside Ready-Mix")
    s.add_files(sorted(PLANT.glob("*.xlsx")))
    assert tools.map_columns(s, MAPPING)["ok"]
    if "--results-only" in sys.argv:
        return show(s)
    t0 = time.perf_counter()
    n_bayes = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    r = tools.train_automl(s, ["Air (%)", "Silica Fume (lb/cy)"], n_bayes=n_bayes,
                           progress=lambda m: print(f"  [{time.perf_counter() - t0:4.0f}s] {m}", flush=True))
    print("winner:", r["winner"], "tuning helped:", r["tuning_helped"], "saved as", r["saved_as"])
    show(s)


def show(s) -> None:
    t0 = time.perf_counter()
    res = tools.results(s)
    print(f"results ({time.perf_counter() - t0:.0f}s, {res['method']['candidates']} candidates):")
    for p in res["proposals"]:
        print("  ", json.dumps({k: p[k] for k in ("id", "replaces", "co2_saving", "gwp_kg_m3", "recipe_gwp_kg_m3",
                                                   "target_28d_psi", "predicted_28d_psi", "lower90_28d_psi",
                                                   "composition", "extras")}))
    rid = res["recipes"][0]["recipe"]
    print("predict:", json.dumps(tools.predict(s, [{"recipe": rid}, {"recipe": rid, "extras": {"Air (%)": 7.5}},
                                                   {"recipe": rid, "changes": {"cement": {"value": 600, "unit": "lb/yd3"}}}])))
    print("models:", json.dumps(tools.list_models(s)))


if __name__ == "__main__":
    main()
