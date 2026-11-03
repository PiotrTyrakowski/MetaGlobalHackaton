"""Agentic AutoML over BOxCrete setups: which inputs and settings fit this plant best.

Every setup is scored by 5-fold cross-validation grouped by mix: the model
never sees any test of a held-out mix. Meta's default setup is always the
baseline; another setup wins only if it beats it clearly.
"""

import itertools
import json
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
DEFAULT = {"gate_tau": 0.1, "prior_sigma": 0.001, "mll_weight": 0.5}
GRID = {"gate_tau": [0.05, 0.1, 0.2], "mll_weight": [0.3, 0.5, 0.8]}
MIN_GAIN = 0.05  # a setup must beat Meta's default by 5% to replace it
PARALLEL = 5


def run_worker(job: str, *, data: str | Path | None = None, extra=(), gate_tau=0.1, prior_sigma=0.001,
               mll_weight=0.5, **kw) -> dict:
    cmd = [sys.executable, "-m", "mixpilot.worker", job]
    if data:
        cmd += ["--data", str(data)]
    for e in extra:
        cmd += ["--extra", e]
    cmd += ["--tau", str(gate_tau), "--sigma", str(prior_sigma), "--mll", str(mll_weight)]
    for k, v in kw.items():
        if v is not None:
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONPATH=str(BACKEND))
    out = subprocess.run(cmd, cwd=BACKEND, env=env, capture_output=True, text=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if out.returncode or not lines:
        raise RuntimeError(f"worker {job} failed: {out.stderr[-800:]}")
    return json.loads(lines[-1])


def label(setup: dict) -> str:
    parts = ["Meta's default setup" if setup["settings"] == DEFAULT else "tuned settings"]
    parts += [f"+ {e.split(' (')[0].lower()}" for e in setup["extra"]]
    return " ".join(parts)


def search(data: str | Path, extra_inputs: list[str], on_result: Callable[[dict], None] | None = None) -> dict:
    """Grid over input sets (base, base + all accepted extras) x tuning settings."""
    input_sets = [[], list(extra_inputs)] if extra_inputs else [[]]
    setups = []
    for extra in input_sets:
        for tau, mll in itertools.product(GRID["gate_tau"], GRID["mll_weight"]):
            setups.append({"extra": extra, "settings": {**DEFAULT, "gate_tau": tau, "mll_weight": mll}})

    results = []
    with ThreadPoolExecutor(PARALLEL) as ex:
        futures = {ex.submit(run_worker, "cv", data=data, extra=s["extra"], **s["settings"]): s for s in setups}
        for f in as_completed(futures):
            s = futures[f]
            r = {**s, "label": label(s), **{k: f.result()[k] for k in ("rmse_28d", "rmse_all", "mixes")}}
            results.append(r)
            if on_result:
                on_result(r)

    results.sort(key=lambda r: r["rmse_28d"])
    baseline = next(r for r in results if not r["extra"] and r["settings"] == DEFAULT)
    best = results[0] if results[0]["rmse_28d"] < (1 - MIN_GAIN) * baseline["rmse_28d"] else baseline

    def best_of(extra: list[str], tuned: bool):
        pool = [r for r in results if r["extra"] == extra and (tuned or r["settings"] == DEFAULT)]
        return min(pool, key=lambda r: r["rmse_28d"]) if pool else None

    summary = [baseline, best_of([], True)]
    if extra_inputs:
        summary += [best_of(list(extra_inputs), False), best_of(list(extra_inputs), True)]
    summary = [dict(r, row=name) for r, name in zip(summary, [
        "Meta's default setup", "+ tuned settings",
        f"+ {', '.join(e.split(' (')[0].lower() for e in extra_inputs)}",
        f"+ tuned settings + {', '.join(e.split(' (')[0].lower() for e in extra_inputs)}",
    ])]
    return {
        "baseline": baseline,
        "best": best,
        "gain": round(1 - best["rmse_28d"] / baseline["rmse_28d"], 3),
        "summary": summary,
        "all": results,
    }


def evaluate_input(data: str | Path, column: str, base_extra: list[str] = ()) -> dict:
    """Error with and without `column`, on only the mixes that record it."""
    with ThreadPoolExecutor(2) as ex:
        without = ex.submit(run_worker, "cv", data=data, extra=list(base_extra), only_with=column)
        with_ = ex.submit(run_worker, "cv", data=data, extra=[*base_extra, column], only_with=column)
        a, b = without.result(), with_.result()
    return {
        "column": column,
        "mixes": a["mixes"],
        "rmse_28d_without": a["rmse_28d"],
        "rmse_28d_with": b["rmse_28d"],
        "gain": round(1 - b["rmse_28d"] / a["rmse_28d"], 3),
    }


def save_model(data: str | Path, setup: dict, models_dir: str | Path, cv: dict, **info) -> dict:
    """Fit the chosen setup on all data and register it as the next version."""
    models_dir = Path(models_dir)
    index_path = models_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    version = f"v{len(index) + 1}"
    run_worker("fit", data=data, extra=setup["extra"], **setup["settings"], out=models_dir / version)
    entry = {
        "version": version,
        "trained": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "extra_inputs": setup["extra"],
        "settings": setup["settings"],
        "mixes": cv["mixes"],
        "rmse_28d": cv["rmse_28d"],
        **info,
    }
    index.append(entry)
    index_path.write_text(json.dumps(index, indent=1))
    return entry


def predict(model_dir: str | Path, rows: list[dict]) -> list[dict]:
    rows_path = Path(model_dir) / "_rows.json"
    rows_path.write_text(json.dumps(rows))
    return run_worker("predict", model=model_dir, rows=rows_path)["predictions"]


def propose(model_dir: str | Path, fixed: dict | None = None) -> dict:
    return run_worker("propose", model=model_dir, fixed=json.dumps(fixed or {}))
