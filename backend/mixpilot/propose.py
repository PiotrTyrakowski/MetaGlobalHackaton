"""Screen candidate mixes with a fitted model: lower CO2 at the same 28-day strength.

Candidates are sampled inside the plant's own experience (binder content,
w/b and SCM share it already uses), with aggregates, curing temperature and
any extra inputs held at the plant's typical values. The model predicts
28-day strength; GWP comes from per-material factors. We keep the
Pareto front of CO2 vs the strength the model is confident of (prediction
minus one SD) and pick mixes that match the strength of the plant's
existing mixes.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from mixpilot.schema import COMPOSITION, GWP_FACTORS
from mixpilot.units import LB_PER_YD3_TO_KG_PER_M3

BINDER = ["Cement (kg/m3)", "Fly Ash (kg/m3)", "Slag (kg/m3)"]
MAX_SCM = 0.5  # common spec limit on cement replacement


def gwp(df: pd.DataFrame) -> np.ndarray:
    return sum(df[c].to_numpy() * GWP_FACTORS[c] for c in COMPOSITION)


def pareto(gwp_: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Indices on the front (lower GWP, higher strength), sorted by GWP."""
    order = np.argsort(gwp_)
    front, best = [], -np.inf
    for i in order:
        if strength[i] > best:
            front.append(i)
            best = strength[i]
    return np.array(front)


def propose(
    data: pd.DataFrame,
    predict: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]],
    fixed: dict,
    n: int = 4000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    mixes = data.drop_duplicates("Mix Name")
    binder = mixes[BINDER].sum(axis=1)
    wb = mixes["Water (kg/m3)"] / binder
    hrwr_per_binder = mixes["HRWR (kg/m3)"] / binder

    b = rng.uniform(binder.min(), binder.max(), n)
    scm = rng.uniform(0, MAX_SCM, n)
    slag_share = rng.uniform(0, 1, n)
    w = rng.uniform(wb.min(), wb.max(), n)
    cand = pd.DataFrame({
        "Cement (kg/m3)": b * (1 - scm),
        "Fly Ash (kg/m3)": b * scm * (1 - slag_share),
        "Slag (kg/m3)": b * scm * slag_share,
        "Water (kg/m3)": b * w,
        "HRWR (kg/m3)": b * rng.uniform(hrwr_per_binder.min(), hrwr_per_binder.max(), n),
    })
    for col in data.columns:
        if col not in cand and col not in ("Mix Name", "Time", "GWP", "Strength (Mean)", "Strength (Std)", "# of measurements"):
            cand[col] = fixed.get(col, float(mixes[col].median()))
    cand["Time"] = 28.0

    mean, sd = predict(cand)
    cand["pred"], cand["sd"], cand["gwp"] = mean, sd, gwp(cand)
    # Rank by prediction minus one SD, so an optimistic but uncertain prediction
    # can't push out a cleaner mix the model is sure of.
    front = cand.iloc[pareto(cand["gwp"].to_numpy(), (cand["pred"] - cand["sd"]).to_numpy())]

    existing = data[data["Time"] == 28].copy()
    existing["gwp"] = gwp(existing)
    # Compare like with like: only existing mixes near the held-fixed values
    # (an air-entrained proposal is compared with air-entrained mixes).
    for col, val in fixed.items():
        tol = max(1.0, 0.15 * float(existing[col].max() - existing[col].min()))
        existing = existing[(existing[col] - val).abs() <= tol]

    # One proposal per strength level the plant actually sells.
    proposals = []
    for target in np.quantile(existing["Strength (Mean)"], [0.25, 0.5, 0.75]):
        ok = front[front["pred"] - front["sd"] >= target]
        if not len(ok):
            continue
        p = ok.iloc[0]  # lowest GWP that still reaches the target
        current = existing[existing["Strength (Mean)"] >= target]
        cur = current.loc[current["gwp"].idxmin()]
        if p["gwp"] >= cur["gwp"]:
            continue  # the plant already has a mix at least this clean
        if proposals and abs(proposals[-1]["gwp"] - float(p["gwp"])) < 1.0:
            continue  # same front point as the previous target
        proposals.append({
            "target_psi": int(round(target, -2)),
            "mix_kg_m3": {c: round(float(p[c]), 1) for c in COMPOSITION},
            "mix_lb_yd3": {c.split(" (")[0]: round(float(p[c]) / LB_PER_YD3_TO_KG_PER_M3) for c in COMPOSITION},
            "fixed": {k: v for k, v in fixed.items()},
            "pred_psi": round(float(p["pred"])),
            "sd_psi": round(float(p["sd"])),
            "gwp": round(float(p["gwp"]), 1),
            "compare_to": cur["Mix Name"],
            "compare_gwp": round(float(cur["gwp"]), 1),
            "compare_psi": int(cur["Strength (Mean)"]),
            "co2_saving": round(1 - float(p["gwp"]) / float(cur["gwp"]), 3),
        })

    return {
        "compared_with": int(len(existing)),
        "existing": [{"mix": r["Mix Name"], "gwp": round(float(r["gwp"]), 1), "psi": int(r["Strength (Mean)"])} for _, r in existing.iterrows()],
        "proposals": proposals,
    }
