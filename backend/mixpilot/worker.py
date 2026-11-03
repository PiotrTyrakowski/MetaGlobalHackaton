"""One BOxCrete job in its own process: cross-validate, fit, or predict.

Patched BOxCrete reads its input schema and settings from the environment
at import, so every setup runs in a fresh process. This module sets those
variables from its arguments before importing boxcrete.

  python -m mixpilot.worker cv      --data D.csv [--extra "Air (%)"] [--tau 0.1] [--sigma 0.001] [--mll 0.5]
  python -m mixpilot.worker fit     --data D.csv --out models/v1 [same settings]
  python -m mixpilot.worker predict --model models/v1 --rows rows.json
  python -m mixpilot.worker propose --model models/v1 [--fixed '{"Air (%)": 6}']

Prints one JSON object on the last line of stdout.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("job", choices=["cv", "fit", "predict", "propose"])
ap.add_argument("--data")
ap.add_argument("--extra", action="append", default=[])
ap.add_argument("--tau", type=float, default=0.1)
ap.add_argument("--sigma", type=float, default=0.001)
ap.add_argument("--mll", type=float, default=0.5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--only-with", default=None)
ap.add_argument("--out")
ap.add_argument("--model")
ap.add_argument("--rows")
ap.add_argument("--fixed", default="{}", help='JSON of inputs held fixed, e.g. {"Air (%)": 6}')
ap.add_argument("--n", type=int, default=4000)
args = ap.parse_args()

if args.job in ("predict", "propose"):
    cfg = json.loads((Path(args.model) / "config.json").read_text())
    args.extra, args.tau, args.sigma, args.mll = cfg["extra_inputs"], cfg["gate_tau"], cfg["prior_sigma"], cfg["mll_weight"]
    args.data = str(Path(args.model) / "data.csv")

os.environ["MIXPILOT_EXTRA_X"] = "|".join(args.extra)
os.environ["MIXPILOT_GATE_TAU"] = str(args.tau)
os.environ["MIXPILOT_PRIOR_SIGMA"] = str(args.sigma)

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from boxcrete import fit_strength_gp, load_concrete_strength  # noqa: E402
from boxcrete.block_loo import train_block_loo  # noqa: E402
from boxcrete.utils import DEFAULT_BOUNDS_DICT, DEFAULT_X_COLUMNS  # noqa: E402


def bounds_dict(df: pd.DataFrame) -> dict:
    """Meta's bounds for the base inputs; padded data range for extra inputs."""
    b = dict(DEFAULT_BOUNDS_DICT)
    for col in args.extra:
        lo, hi = float(df[col].min()), float(df[col].max())
        pad = 0.1 * max(hi - lo, 1e-3)
        b[col] = (lo - pad, hi + pad)
    return b


def fit(df: pd.DataFrame, bounds: dict, max_iter: int = 150):
    data = load_concrete_strength(df.reset_index(drop=True), bounds_dict=bounds)
    X, Y, Yvar, X_bounds = data.strength_data
    model = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=X_bounds, seed=0, max_optimizer_iter=0)
    if max_iter:
        train_block_loo(model, X.shape[0], mll_weight=args.mll, max_iter=max_iter, lr=0.1)
    model.eval()
    return model


def predict(model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = torch.tensor(df[DEFAULT_X_COLUMNS].to_numpy(dtype=float), dtype=torch.float64)
    scale = float(model._study_y_std)
    with torch.no_grad():
        post = model.posterior(X)
    return post.mean.squeeze(-1).numpy() * scale, post.variance.squeeze(-1).sqrt().numpy() * scale


t0 = time.time()
settings = {"extra_inputs": args.extra, "gate_tau": args.tau, "prior_sigma": args.sigma, "mll_weight": args.mll}

if args.job == "cv":
    df = pd.read_csv(args.data)
    if args.only_with:
        df = df[df[args.only_with].notna()]
    bounds = bounds_dict(df)
    mixes = df["Mix Name"].unique()
    fold_of = dict(zip(np.random.default_rng(0).permutation(mixes), np.arange(len(mixes)) % args.folds))
    fold = df["Mix Name"].map(fold_of)
    pred = np.zeros(len(df))
    for k in range(args.folds):
        model = fit(df[fold != k], bounds)
        pred[(fold == k).to_numpy()] = predict(model, df[fold == k])[0]
    err = pred - df["Strength (Mean)"].to_numpy()
    age = df["Time"].to_numpy()
    result = {
        **settings,
        "mixes": int(len(mixes)),
        "rmse_28d": round(float(np.sqrt(np.mean(err[age == 28] ** 2)))),
        "rmse_all": round(float(np.sqrt(np.mean(err**2)))),
    }

elif args.job == "fit":
    df = pd.read_csv(args.data)
    bounds = bounds_dict(df)
    model = fit(df, bounds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    df.to_csv(out / "data.csv", index=False)
    (out / "config.json").write_text(json.dumps({**settings, "bounds": bounds, "inputs": DEFAULT_X_COLUMNS}, indent=1))
    result = {**settings, "path": str(out), "mixes": int(df["Mix Name"].nunique())}

else:
    df = pd.read_csv(args.data)
    model = fit(df, json.loads((Path(args.model) / "config.json").read_text())["bounds"], max_iter=0)
    model.load_state_dict(torch.load(Path(args.model) / "model.pt", weights_only=True))
    model.eval()
    if args.job == "predict":
        rows = pd.DataFrame(json.loads(Path(args.rows).read_text()))
        mean, sd = predict(model, rows)
        result = {"predictions": [{"mean_psi": round(float(m)), "sd_psi": round(float(s))} for m, s in zip(mean, sd)]}
    else:
        from mixpilot.propose import propose

        result = propose(df, lambda rows: predict(model, rows), json.loads(args.fixed), n=args.n)

result["seconds"] = round(time.time() - t0, 1)
print(json.dumps(result), flush=True)
sys.exit(0)
