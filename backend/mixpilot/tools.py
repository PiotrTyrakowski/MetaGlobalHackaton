"""The agent's data tools: read -> map -> validate -> suggest inputs.

Each tool takes the session plus the agent's arguments and returns JSON-ready
data. The same data feeds the UI panels, so the agent only explains numbers
that this code computed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from mixpilot import automl, dataset, design, excel, inputs, models, units, validate
from mixpilot.dataset import RECIPE_FIELDS, Dataset, MappingError
from mixpilot.excel import Sheet


@dataclass
class Session:
    plant: str = "Plant"
    sheets: list[Sheet] = field(default_factory=list)
    mapping: dict | None = None
    data: Dataset | None = None
    map_report: dict | None = None
    validation: dict | None = None
    suggestions: dict | None = None
    training: dict | None = None
    model_meta: dict | None = None

    def add_files(self, paths: list[Path]) -> None:
        names = {p.name for p in paths}
        self.sheets = [s for s in self.sheets if s.file not in names]
        for p in paths:
            self.sheets += excel.read_sheets(Path(p))


def read_files(s: Session) -> dict:
    """Profile every uploaded sheet and suggest join keys."""
    return dict(files=[excel.profile(sh) for sh in s.sheets], join_keys=excel.join_keys(s.sheets))


def map_columns(s: Session, mapping: dict) -> dict:
    """Apply the agent's mapping; on failure return the errors so it can fix them."""
    try:
        ds, report = dataset.build(s.sheets, mapping)
    except MappingError as e:
        return dict(ok=False, errors=e.errors)
    s.mapping, s.data, s.map_report = mapping, ds, report
    s.validation = s.suggestions = None
    return dict(ok=True, **report,
                candidate_columns=[dict(column=c.name, level=c.level, kind=c.kind, file=c.file)
                                   for c in ds.candidates.values()])


def validate_data(s: Session) -> dict:
    if s.data is None:
        return dict(ok=False, errors=["map the columns first (map_columns)"])
    s.validation = validate.validate(s.data, s.mapping, s.map_report)
    return s.validation


def suggest_inputs(s: Session, progress: Callable[[str], None] = lambda _: None) -> dict:
    if s.data is None:
        return dict(ok=False, errors=["map the columns first (map_columns)"])
    s.suggestions = inputs.suggest_inputs(s.data, progress=progress)
    return s.suggestions


def mapping_help() -> dict:
    """What map_columns accepts: fields, units and an example (goes in the tool description)."""
    return dict(
        fields={f: sorted(u) for f, u in units.UNITS.items()},
        recipe_fields=dataset.RECIPE_FIELDS,
        required=sorted(dataset.REQUIRED),
        example=dataset.MAPPING_EXAMPLE,
    )


def train_automl(s: Session, inputs: list[str], n_bayes: int = 10,
                 progress: Callable[[str], None] = lambda _: None, use_cache: bool = True) -> dict:
    """AutoML on the approved inputs, then save the winner as the next model version.

    A finished run is cached by (data, inputs, budget), so a live demo can't stall;
    a cache hit is reported as such.
    """
    if s.data is None:
        return dict(ok=False, errors=["map the columns first (map_columns)"])
    unknown = [c for c in inputs if c not in s.data.candidates]
    if unknown:
        return dict(ok=False, errors=[f"not candidate columns: {unknown}; have {list(s.data.candidates)}"])
    key = _run_key(s.data, inputs, n_bayes)
    cache = models.ROOT / "_cache" / f"{key}.json"
    if use_cache and cache.exists():
        s.training = json.loads(cache.read_text())
        progress(f"same data and inputs as the run finished at {s.training['finished'][:16]}: reusing it")
    else:
        s.training = automl.run_automl(s.data, inputs, n_bayes=n_bayes, progress=progress)
        s.training["finished"] = datetime.now().isoformat(timespec="seconds")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(s.training, default=str))
    progress("fitting the winning setup on all data and saving it")
    s.model_meta = models.save(s.plant, s.data, s.training, s.mapping)
    return dict(ok=True, **{k: v for k, v in s.training.items() if k != "trials"},
                n_trials=len(s.training["trials"]), saved_as=s.model_meta["version"])


def results(s: Session, version: str | None = None) -> dict:
    """CO2 vs strength of the plant's recipes, and lower-CO2 mixes to test next."""
    model, meta = models.load(s.plant, version)
    rows = pd.read_csv(models.ROOT / models.slug(s.plant) / meta["version"] / "rows.csv",
                       dtype={"recipe": str, "batch": str})
    extras = meta["inputs"]
    batch_values = None
    if s.data is not None and extras:
        batch_values = pd.DataFrame({"recipe": s.data.batches["recipe"].to_numpy(),
                                     **{e: s.data.feature(e).to_numpy() for e in extras}})
    rec = design.recipe_table(rows, extras, batch_values)
    levels = meta.get("input_levels", {})
    out = design.propose(model, rec, extras, batch_extras=[e for e in extras if levels.get(e) == "batch"])
    unit = _display_unit(meta.get("mapping") or s.mapping)
    for p in out["proposals"]:
        p["composition"] = {design.LABELS[f]: round(units.convert(f, "kg/m3", v) / _factor(unit))
                            for f, v in p["composition_kg_m3"].items()}
        p["composition_unit"] = unit
    return dict(model=meta["version"], **out)


def predict(s: Session, mixes: list[dict], ages: list[float] = (7, 28), version: str | None = None) -> dict:
    """Predict strength for existing recipes, optionally changed (changes in any supported unit)."""
    model, meta = models.load(s.plant, version)
    rows = pd.read_csv(models.ROOT / models.slug(s.plant) / meta["version"] / "rows.csv",
                       dtype={"recipe": str, "batch": str})
    base = rows.drop_duplicates("recipe").set_index("recipe")
    extras = meta["inputs"]
    out, errors = [], []
    for i, m in enumerate(mixes):
        rid = m.get("recipe")
        if rid is not None and rid not in base.index:
            errors.append(f"mix {i + 1}: unknown recipe '{rid}'")
            continue
        comp = {f: float(base.loc[rid, f]) if rid else 0.0 for f in RECIPE_FIELDS}
        for f, ch in (m.get("changes") or {}).items():
            if f not in RECIPE_FIELDS:
                errors.append(f"mix {i + 1}: can't change '{f}'; fields are {RECIPE_FIELDS}")
                continue
            if (msg := units.check_unit(f, ch.get("unit", "kg/m3"))) is not None:
                errors.append(f"mix {i + 1}: {msg}")
                continue
            comp[f] = float(units.convert(f, ch.get("unit", "kg/m3"), ch["value"]))
        used = {}
        for e in extras:
            v = (m.get("extras") or {}).get(e)
            used[e] = float(v) if v is not None else meta["typical_extra_values"][e]["median"]
        df = pd.DataFrame([{**comp, **used}])
        pred = design.predict_comp(model, df, extras, [float(a) for a in ages])
        outside = [f for f in RECIPE_FIELDS if not (meta["ranges"][f][0] * 0.9 <= comp[f]
                                                   <= meta["ranges"][f][1] * 1.1 + 1e-9)]
        out.append(dict(mix=rid or f"mix {i + 1}", extras_used=used,
                        predictions=[dict(age_days=a, strength_psi=round(float(pred[float(a)][0][0])),
                                          plus_minus_psi=round(float(1.645 * pred[float(a)][1][0])))
                                     for a in ages],
                        outside_training_range=outside, gwp_kg_m3=round(float(design.gwp(df, extras)[0]))))
    return dict(model=meta["version"], mixes=out, errors=errors,
                note="± is the model's 90% uncertainty for the mix's mean strength")


def list_models(s: Session) -> dict:
    return dict(models=[{k: m[k] for k in ("version", "trained", "inputs", "setup", "error_28d_psi",
                                           "vs_meta_default", "recipes", "batches", "active")}
                        for m in models.versions(s.plant)])


def _display_unit(mapping: dict | None) -> str:
    spec = (mapping or {}).get("recipes", {}).get("fields", {}).get("cement", {})
    return "lb/yd3" if units.normalize_unit(spec.get("unit", "kg/m3")) == "lb/yd3" else "kg/m3"


def _factor(unit: str) -> float:
    return units.LB_YD3_TO_KG_M3 if unit == "lb/yd3" else 1.0


def _run_key(ds: Dataset, inputs: list[str], n_bayes: int) -> str:
    h = hashlib.sha256()
    h.update(ds.rows.to_csv(index=False).encode())
    for c in sorted(inputs):
        h.update(c.encode())
        h.update(ds.feature(c).to_csv().encode())
    h.update(f"{n_bayes}".encode())
    return h.hexdigest()[:16]
