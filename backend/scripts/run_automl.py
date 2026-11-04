"""Run AutoML on the demo data with the inputs suggest_inputs approves."""

import json
import sys
import time

from mixpilot import automl
from run_pipeline import MAPPING, PLANT
from mixpilot import tools


def main() -> None:
    s = tools.Session()
    s.add_files(sorted(PLANT.glob("*.xlsx")))
    assert tools.map_columns(s, MAPPING)["ok"]
    n_bayes = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    inputs = ["Air (%)", "Silica Fume (lb/cy)"]
    t0 = time.perf_counter()
    r = automl.run_automl(s.data, inputs, n_bayes=n_bayes,
                          progress=lambda m: print(f"  [{time.perf_counter() - t0:5.0f}s] {m}", flush=True))
    for row in r["leaderboard"]:
        print(f"  {'*' if row['best'] else ' '} {row['name']:42s} all ages {row['error_all_ages']:.2%} "
              f"({row['vs_baseline']:+.1%})  28d {row['error_28d_psi']:4d} psi ({row['vs_baseline_28d']:+.0%})  "
              f"won {row['folds_won_vs_baseline']:5s} {row.get('folds_won_vs_meta_settings', ''):5s} "
              f"{row['errors_by_age_psi']}  {row['settings'] if row['tuned'] else ''}")
    print(f"winner: {r['winner']['name']}  tuning helped: {r['tuning_helped']}  ({r['seconds']}s)")
    json.dump(r, open("demo_data/automl_last.json", "w"), indent=1)


if __name__ == "__main__":
    main()
