"""What a trained model gives the plant: predictions, and lower-CO2 mixes to test next."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from boxcrete.utils import DEFAULT_GWP_COEFFICIENTS
from botorch.models import SingleTaskGP

from mixpilot import boxfit, units
from mixpilot.dataset import MATERIAL_SOURCE, RECIPE_FIELDS

DEFAULT_TEMP_C = 23.0
Z90 = 1.2816  # one-sided 90%
_GWP_COLS = {"cement": "Cement (kg/m3)", "fly_ash": "Fly Ash (kg/m3)", "slag": "Slag (kg/m3)",
             "water": "Water (kg/m3)", "hrwr": "HRWR (kg/m3)", "fine": "Fine Aggregate (kg/m3)",
             "coarse": "Coarse Aggregates (kg/m3)"}
GWP = {f: DEFAULT_GWP_COEFFICIENTS[int(MATERIAL_SOURCE)][c][0] for f, c in _GWP_COLS.items()}
SILICA_FUME_GWP = 0.028  # kg CO2e/kg; industrial by-product, similar to fly ash (assumption)
LABELS = {"cement": "Cement", "fly_ash": "Fly ash", "slag": "Slag", "water": "Water", "hrwr": "HRWR",
          "fine": "Fine agg.", "coarse": "Coarse agg."}


def recipe_table(rows: pd.DataFrame, extras: list[str], batch_values: pd.DataFrame | None) -> pd.DataFrame:
    """One row per recipe: composition (kg/m3), mean extras, observed 28-day mean."""
    rec = rows.drop_duplicates("recipe").set_index("recipe")[RECIPE_FIELDS].copy()
    if batch_values is not None:
        for e in extras:
            rec[e] = batch_values.groupby("recipe")[e].mean().reindex(rec.index)
    s28 = rows[rows["age"] == 28].groupby("recipe")["strength"].mean()
    rec["observed_28d"] = s28.reindex(rec.index)
    return rec


def gwp(comp: pd.DataFrame, extras: list[str]) -> np.ndarray:
    """kg CO2e per m3, with BOxCrete's emission factors (material source 2)."""
    total = sum(comp[f].to_numpy() * k for f, k in GWP.items())
    for e in extras:
        if "silica" in e.lower():
            total = total + units.convert("cement", _unit_of(e), comp[e].to_numpy()) * SILICA_FUME_GWP
    return total


def predict_comp(model: SingleTaskGP, comp: pd.DataFrame, extras: list[str], ages: list[float],
                 temp_c: float = DEFAULT_TEMP_C) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for a in ages:
        cols = [comp[f].to_numpy() for f in RECIPE_FIELDS]
        cols += [np.full(len(comp), MATERIAL_SOURCE), np.full(len(comp), temp_c), np.full(len(comp), a)]
        cols += [comp[e].to_numpy() for e in extras]
        X = torch.tensor(np.column_stack(cols).astype(float), dtype=torch.float64)
        out[a] = boxfit.predict(model, X)
    return out


def propose(model: SingleTaskGP, rec: pd.DataFrame, extras: list[str], batch_extras: list[str] = (),
            per_recipe: int = 400, n_proposals: int = 3, seed: int = 0) -> dict:
    """Lower-CO2 mixes that the model expects to match an existing recipe's 28-day strength.

    For each recipe, candidates stay close to what the plant already makes: a blend
    with a similar recipe (at least half the original), optionally moving up to 30%
    of the cement into slag or fly ash. Batch conditions such as air content are an
    exposure requirement, so they are kept at the recipe's own value. A candidate
    only counts if its cautious estimate (90% lower bound) reaches the recipe's own
    predicted strength.
    """
    rng = np.random.default_rng(seed)
    rec = rec.dropna(subset=extras)
    ids = rec.index.to_numpy()
    cols = RECIPE_FIELDS + list(extras)
    comp = rec[cols].to_numpy(dtype=float)
    span = comp.max(0) - comp.min(0) + 1e-9
    fixed = [cols.index(e) for e in batch_extras]
    lo, hi = rec[RECIPE_FIELDS].min().to_numpy(), rec[RECIPE_FIELDS].max().to_numpy()

    rp = predict_comp(model, rec, extras, [28.0])[28.0]
    rec = rec.assign(pred_28d=rp[0], gwp=gwp(rec, extras))

    cands, target = [], []
    for t in range(len(ids)):
        similar = [j for j in range(len(ids))
                   if not fixed or np.all(np.abs(comp[j, fixed] - comp[t, fixed]) <= 0.25 * span[fixed])]
        for _ in range(per_recipe):
            j = similar[rng.integers(len(similar))]
            lam = rng.uniform(0.5, 1.0)
            c = lam * comp[t] + (1 - lam) * comp[j]
            if rng.uniform() < 0.8:
                move = rng.uniform(0, 0.3) * c[0]
                c[0] -= move
                c[2 if rng.uniform() < 0.6 else 1] += move  # slag or fly ash
            c[fixed] = comp[t, fixed]
            if np.all(c[:len(RECIPE_FIELDS)] >= lo * 0.98) and np.all(c[:len(RECIPE_FIELDS)] <= hi * 1.02):
                cands.append(c)
                target.append(ids[t])
    C = pd.DataFrame(np.array(cands), columns=cols)
    C["target"] = target

    pred = predict_comp(model, C, extras, [7.0, 28.0])
    m28, s28 = pred[28.0]
    C["pred_28d"], C["sd_28d"], C["lower_28d"] = m28, s28, m28 - Z90 * s28
    C["pred_7d"] = pred[7.0][0]
    C["gwp"] = gwp(C, extras)

    picks = []
    for rid, r in rec.iterrows():
        ok = C[(C["target"] == rid) & (C["lower_28d"] >= r["pred_28d"]) & (C["gwp"] < r["gwp"])]
        if ok.empty:
            continue
        best = ok.loc[ok["gwp"].idxmin()]
        picks.append((1 - best["gwp"] / r["gwp"], rid, best))
    picks.sort(key=lambda p: -p[0])

    proposals, used = [], set()
    for saving, rid, best in picks:
        if rid in used or saving < 0.05:
            continue
        used.add(rid)
        proposals.append(dict(
            id=f"P-{len(proposals) + 1}", replaces=rid, co2_saving=round(float(saving), 3),
            gwp_kg_m3=round(float(best["gwp"])), recipe_gwp_kg_m3=round(float(rec.loc[rid, "gwp"])),
            target_28d_psi=round(float(rec.loc[rid, "pred_28d"])),
            predicted_28d_psi=round(float(best["pred_28d"])), sd_28d_psi=round(float(best["sd_28d"])),
            lower90_28d_psi=round(float(best["lower_28d"])), predicted_7d_psi=round(float(best["pred_7d"])),
            composition_kg_m3={f: round(float(best[f]), 1) for f in RECIPE_FIELDS},
            extras={e: round(float(best[e]), 2) for e in extras},
        ))
        if len(proposals) == n_proposals:
            break

    return dict(
        recipes=[dict(recipe=rid, gwp_kg_m3=round(float(r["gwp"])), observed_28d_psi=_round(r["observed_28d"]),
                      predicted_28d_psi=round(float(r["pred_28d"]))) for rid, r in rec.iterrows()],
        proposals=proposals,
        method=dict(candidates=int(len(C)), rule="cautious estimate (90% lower bound) must reach the recipe's "
                                                 "predicted 28-day strength; candidates stay close to the recipe "
                                                 "(blend with a similar recipe, swap up to 30% of cement for slag "
                                                 "or fly ash); batch conditions kept: " + (", ".join(batch_extras) or "none"),
                    co2="BOxCrete emission factors (material source 2); silica fume "
                        f"{SILICA_FUME_GWP} kg CO2e/kg (our assumption)"),
    )


def _unit_of(column: str) -> str:
    return "lb/yd3" if "lb" in column.lower() else "kg/m3"


def _round(x):
    return None if pd.isna(x) else round(float(x))
