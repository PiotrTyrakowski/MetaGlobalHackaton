"""Score one BOxCrete setup on plant data with grouped K-fold CV (refit per fold).

Setup comes from env (read by the patched boxcrete at import):
  MIXPILOT_EXTRA_X, MIXPILOT_GATE_TAU, MIXPILOT_PRIOR_SIGMA
and from --mll-weight. Prints one JSON line.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from boxcrete import fit_strength_gp, load_concrete_strength
from boxcrete.block_loo import train_block_loo
from boxcrete.utils import DEFAULT_X_COLUMNS

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--data", default="data/plant_truth.csv")
ap.add_argument("--mll-weight", type=float, default=0.5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--only-with", default=None, help="keep only mixes where this column is recorded")
args = ap.parse_args()

df = pd.read_csv(args.data)
if args.only_with:
    df = df[df[args.only_with].notna()]
mixes = df["Mix Name"].unique()
rng = np.random.default_rng(0)
fold_of = dict(zip(rng.permutation(mixes), np.arange(len(mixes)) % args.folds))

t0 = time.time()
pred, true, age = [], [], []
for k in range(args.folds):
    test = df["Mix Name"].map(fold_of) == k
    train = load_concrete_strength(df[~test].reset_index(drop=True))
    X, Y, Yvar, bounds = train.strength_data
    model = fit_strength_gp(X=X, Y=Y, Yvar=Yvar, X_bounds=bounds, seed=0, max_optimizer_iter=0)
    train_block_loo(model, X.shape[0], mll_weight=args.mll_weight, max_iter=150, lr=0.1)
    model.eval()
    Xt = torch.tensor(df[test][DEFAULT_X_COLUMNS].to_numpy(), dtype=torch.float64)
    with torch.no_grad():
        mu = model.posterior(Xt).mean.squeeze(-1).numpy() * float(model._study_y_std)
    pred += list(mu)
    true += list(df[test]["Strength (Mean)"])
    age += list(df[test]["Time"])

pred, true, age = map(np.asarray, (pred, true, age))
rmse = lambda m: float(np.sqrt(np.mean((pred[m] - true[m]) ** 2)))
print(json.dumps({
    "name": args.name,
    "inputs": DEFAULT_X_COLUMNS,
    "gate_tau": float(os.environ.get("MIXPILOT_GATE_TAU", 0.1)),
    "prior_sigma": float(os.environ.get("MIXPILOT_PRIOR_SIGMA", 0.001)),
    "mll_weight": args.mll_weight,
    "mixes": len(mixes),
    "rmse_all": round(rmse(age >= 0)),
    "rmse_28d": round(rmse(age == 28)),
    "seconds": round(time.time() - t0, 1),
}))
