"""Run a small AutoML grid over BOxCrete setups in parallel subprocesses."""

import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

data = sys.argv[1] if len(sys.argv) > 1 else "data/plant_truth.csv"
configs = []
for extra, tau, mll in itertools.product(["", "Air (%)"], [0.05, 0.1, 0.2], [0.3, 0.5, 0.8]):
    configs.append(dict(extra=extra, tau=tau, mll=mll, sigma=0.001))
for extra in ["", "Air (%)"]:
    configs.append(dict(extra=extra, tau=0.1, mll=0.5, sigma=0.1))


def run(c):
    env = dict(os.environ, PYTHONPATH=".", OMP_NUM_THREADS="2", MIXPILOT_EXTRA_X=c["extra"],
               MIXPILOT_GATE_TAU=str(c["tau"]), MIXPILOT_PRIOR_SIGMA=str(c["sigma"]))
    name = f"{'air' if c['extra'] else 'base'} tau={c['tau']} mll={c['mll']} sigma={c['sigma']}"
    out = subprocess.run([sys.executable, "-W", "ignore", "spike/evaluate.py", "--name", name, "--data", data,
                          "--mll-weight", str(c["mll"])], env=env, capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {"name": name, "error": out.stderr[-400:]}


with ThreadPoolExecutor(5) as ex:
    results = list(ex.map(run, configs))
results.sort(key=lambda r: r.get("rmse_28d", 1e9))
for r in results:
    print(f"{r['name']:<40} 28d={r.get('rmse_28d')} all={r.get('rmse_all')} {r.get('error', '')}")
json.dump(results, open(data.replace(".csv", "_grid.json"), "w"), indent=1)
