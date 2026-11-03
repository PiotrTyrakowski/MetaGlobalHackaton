"""Check the mapped data before training (the validate tool)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from boxcrete.utils import DATA_PATH

from mixpilot.dataset import RECIPE_FIELDS, Dataset
from mixpilot.excel import to_number

META_COLUMNS = {
    "cement": "Cement (kg/m3)", "fly_ash": "Fly Ash (kg/m3)", "slag": "Slag (kg/m3)",
    "water": "Water (kg/m3)", "hrwr": "HRWR (kg/m3)", "fine": "Fine Aggregate (kg/m3)",
    "coarse": "Coarse Aggregates (kg/m3)", "temp_c": "Temp (C)", "age": "Time",
}
LABELS = {"cement": "Cement", "fly_ash": "Fly ash", "slag": "Slag", "water": "Water",
          "hrwr": "HRWR", "fine": "Fine aggregate", "coarse": "Coarse aggregate",
          "temp_c": "Curing temperature", "age": "Age at test"}
CYLINDER_OUTLIER = 0.15  # a cylinder this far from its set's mean is flagged


@lru_cache
def meta_ranges() -> dict[str, tuple[float, float]]:
    """Range of each input in Meta's training data (concrete mixes)."""
    d = pd.read_csv(DATA_PATH)
    d = d[d["Coarse Aggregates (kg/m3)"] > 0]
    return {f: (float(d[c].min()), float(d[c].max())) for f, c in META_COLUMNS.items()}


def validate(ds: Dataset, mapping: dict, report: dict) -> dict:
    issues: list[dict] = []
    rows = ds.rows
    recipes = rows.drop_duplicates("recipe")

    # inputs BOxCrete already has: coverage and range vs Meta's data
    inputs = []
    for f in RECIPE_FIELDS + ["temp_c", "age"]:
        src = _source(mapping, f)
        lo, hi = meta_ranges()[f]
        frame = recipes if f in RECIPE_FIELDS else rows
        vals = frame[f]
        outside = int(((vals < lo * 0.9) | (vals > hi * 1.1)).sum()) if hi > 0 else 0
        inputs.append(dict(input=LABELS[f], source=src, coverage=1.0 if src else 0.0,
                           plant_range=[_r(vals.min()), _r(vals.max())], meta_range=[_r(lo), _r(hi)],
                           outside_meta_range=outside))
        if outside and f != "age":
            issues.append(dict(level="warning", message=f"{LABELS[f]}: {outside} of {len(frame)} "
                               f"{'recipes' if f in RECIPE_FIELDS else 'rows'} outside the range Meta's "
                               f"model was trained on ({_r(lo)}–{_r(hi)})"))
    if (rows["age"] > 28).any():
        n = int((rows["age"] > 28).sum())
        issues.append(dict(level="warning", message=f"{n} results older than 28 days; Meta's data stops "
                                                    f"at 28, so later ages are extrapolation"))

    # candidate columns: coverage and cells that needed cleanup
    candidates = []
    for c, k in ds.candidates.items():
        col = ds.batches[c]
        cand = dict(column=c, level=k.level, kind=k.kind, file=k.file,
                    coverage=round(float(col.notna().mean()), 3))
        if k.kind == "numeric":
            _, cleaned, failed = to_number(col)
            if cleaned:
                cand["cleaned_cells"] = cleaned
                issues.append(dict(level="info", message=f"{c}: {cleaned} cells had text around the "
                                                         f"number (e.g. '5.8%'); read as numbers"))
            if failed:
                issues.append(dict(level="warning", message=f"{c}: {failed} cells are not numbers"))
        candidates.append(cand)

    # cylinders far from their set
    sets = rows[rows["cylinders"] >= 2]
    cv = (sets["strength_sd"] / sets["strength"]).fillna(0)
    flagged = int((cv > CYLINDER_OUTLIER / 2).sum())
    if flagged:
        issues.append(dict(level="warning", message=f"{flagged} cylinder sets have a spread above "
                                                    f"{CYLINDER_OUTLIER / 2:.0%}; kept, but worth a look"))

    for side, val in (("tests matched to batches", report.get("tests_matched")),
                      ("batches matched to recipes", report.get("batches_matched"))):
        if val and val.split("/")[0] != val.split("/")[1]:
            issues.append(dict(level="warning", message=f"only {val} {side}; the rest are dropped"))

    for a in report.get("assumptions", []):
        issues.append(dict(level="info", message=f"assumption: {a}"))

    per_recipe = rows.groupby("recipe")["batch"].nunique()
    return dict(
        summary=dict(recipes=int(rows["recipe"].nunique()), batches=int(rows["batch"].nunique()),
                     strength_results=int(len(rows)), cylinders=ds.cylinders,
                     ages_days=sorted(int(a) if float(a).is_integer() else float(a) for a in rows["age"].unique()),
                     batches_per_recipe=[int(per_recipe.min()), int(per_recipe.max())],
                     strength_28d_psi=_range28(rows)),
        boxcrete_inputs=inputs, candidate_columns=candidates, issues=issues,
        ok=not any(i["level"] == "error" for i in issues),
    )


def _source(mapping: dict, f: str) -> str | None:
    f = {"temp_c": "cure_temp"}.get(f, f)
    for section in ("recipes", "batches", "tests"):
        spec = mapping.get(section, {}).get("fields", {}).get(f)
        if spec:
            return f"{mapping[section]['file']}: {spec['column']}"
    return None


def _range28(rows: pd.DataFrame) -> list[float] | None:
    r = rows[rows["age"] == 28]["strength"]
    return [_r(r.min()), _r(r.max())] if len(r) else None


def _r(x: float) -> float:
    return float(f"{x:.4g}") if np.isfinite(x) else x
