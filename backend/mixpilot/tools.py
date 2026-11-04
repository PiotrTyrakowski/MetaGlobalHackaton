"""The agent's tools and the Train button: read -> map -> validate -> suggest inputs -> train -> results.

The engine (dataset, inputs, automl, design, models) computes everything. Each
tool here runs one engine step, sends the page what it needs as events, and
returns JSON-ready data for the model, so the agent only explains numbers this
code computed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pandas as pd

from mixpilot import automl, boxfit, dataset, design, excel, inputs, models, units, validate
from mixpilot.dataset import RECIPE_FIELDS, Dataset, MappingError
from mixpilot.session import PLANT, Session


class ToolError(Exception):
    """Bad arguments or state; the message goes back to the model so it can fix the call."""


# --------------------------------------------------------------------------- data prep

def read_files(s: Session) -> dict:
    """Profile every uploaded sheet and suggest join keys."""
    return dict(files=[excel.profile(sh) for sh in s.sheets], join_keys=excel.join_keys(s.sheets))


def map_columns(s: Session, mapping: dict) -> dict:
    """Apply the agent's mapping; on failure return the errors so it can fix them."""
    try:
        sheets, mapping = dataset.prepare(s.sheets, mapping)
        ds, report = dataset.build(sheets, mapping)
        dataset.check_enough(ds)
    except MappingError as e:
        return dict(ok=False, errors=e.errors)
    s.mapping, s.data, s.map_report = mapping, ds, report
    s.validation = s.suggestions = None
    s.accepted, s.offered, s.ready_to_train = [], {}, False
    return dict(ok=True, **report, notes_still_counted=_notes_still_counted(sheets, mapping),
                candidate_columns=[dict(column=c.name, level=c.level, kind=c.kind, file=c.file)
                                   for c in ds.candidates.values()])


def _notes_still_counted(sheets: list, mapping: dict) -> dict[str, list[str]]:
    """Text on cylinders that still count as valid tests, for the agent to read once more."""
    tests = mapping["tests"]
    table = next(sh.df for sh in sheets if sh.file == tests["file"])
    rules = tests.get("exclude", [])
    skip = {tests.get("batch_key"), tests.get("recipe_key"), *(f["column"] for f in tests["fields"].values())}
    out = {}
    for c in table.columns:
        if c in skip or excel.column_kind(table[c]) not in ("category", "text"):
            continue
        counts = table[c].dropna().astype(str).str.strip().value_counts()
        counts = counts[[bool(v) and not any(r["column"] == c and r["contains"].lower() in v.lower() for r in rules)
                         for v in counts.index]]
        is_id = counts.sum() > len(table) / 2 and len(counts) > counts.sum() / 5  # filled in, and rarely repeated
        if len(counts) and not is_id:
            out[c] = counts.index[:20].tolist()
    return out


def validate_data(s: Session) -> dict:
    _need_data(s)
    s.validation = validate.validate(s.data, s.mapping, s.map_report)
    _data_artifact(s)
    return s.validation


def suggest_inputs(s: Session) -> dict:
    """Test every candidate column. Proven complete ones are added; proven partial ones are offered."""
    _need_data(s)

    def progress(text: str) -> None:
        s.emit("tool_progress", text=text[0].upper() + text[1:])

    found = inputs.suggest_inputs(s.data, progress=progress)
    s.suggestions = found
    s.accepted = list(found["use"])
    s.offered = {r["column"]: r for r in found["columns"]
                 if r["verdict"] == "collect" and r.get("gain", 0) >= inputs.MIN_GAIN}
    s.ready_to_train = True

    by_name = {r["column"]: r for r in found["columns"]}
    for c in s.accepted:
        s.emit("card", kind="input_ready", input=c, file=s.data.candidates[c].file, **_counts(s, c),
               why=_why(by_name[c]))
    for c, r in s.offered.items():
        s.emit("card", kind="input_incomplete", input=c, file=s.data.candidates[c].file, **_counts(s, c),
               gain=r["gain"], recorded_from=r.get("recorded_from"))
    _data_artifact(s)
    return dict(added=s.accepted, offered=list(s.offered), columns=found["columns"], method=found["method"],
                note="'added' inputs will be used in training. 'offered' inputs are recorded on only part of "
                     "the batches and stay out unless the user agrees; their gain was measured on the batches "
                     "that have them, and batches without a value would get the plant's median.")


def use_input(s: Session, name: str) -> dict:
    """Train with an offered partly logged input (the card's button, or the user agreeing in chat)."""
    if name not in s.offered:
        raise ToolError(f"{name!r} is not on offer; offered inputs are {sorted(s.offered)}")
    if name not in s.accepted:
        s.accepted.append(name)
    s.ready_to_train = True
    s.emit("input_used", input=name)
    _data_artifact(s)
    return dict(input=name, used_in_training=True, inputs=s.accepted,
                note="batches without a value get the plant's median")


def mapping_help() -> dict:
    """What map_columns accepts: fields, units and an example (goes in the tool description)."""
    return dict(fields={f: sorted(u) for f, u in units.UNITS.items()}, recipe_fields=RECIPE_FIELDS,
                required=sorted(dataset.REQUIRED), example=dataset.MAPPING_EXAMPLE)


def _need_data(s: Session) -> None:
    if s.data is None:
        raise ToolError("map the columns first (map_columns)")


def _counts(s: Session, c: str) -> dict:
    return dict(mixes_with_value=int(s.data.feature(c).notna().sum()), mixes=len(s.data.batches))


def _why(r: dict) -> str:
    """The evidence for an added input, in plain words."""
    if r["level"] == "batch":
        return f"Explains {round(100 * r['scatter_explained'])}% of the strength differences between batches of the same mix."
    return f"Makes 28-day strength predictions {round(100 * r['gain'])}% more accurate on mixes the model never saw."


def _tested(s: Session) -> list[dict] | None:
    """Every extra column and what the tests said, in plain words, for the Data panel."""
    if not s.suggestions:
        return None
    out = []
    for r in s.suggestions["columns"]:
        c, verdict = r["column"], r["verdict"]
        if verdict == "use":
            status, why = "added", _why(r)
        elif verdict == "collect" and c in s.offered:
            status = "added" if c in s.accepted else "partly logged"
            why = f"On the batches that have it, 28-day predictions are {round(100 * r['gain'])}% more accurate."
        elif verdict == "collect":
            status, why = "not used", "Recorded on part of the batches, and it didn't improve predictions on them."
        elif verdict == "proxy":
            status, why = "not used", "Linked to strength, but it's a label or date, not a measurement."
            if r.get("associated_with"):
                why += f" It stands in for {r['associated_with'][0]['column']}."
        elif r.get("restates_result"):
            status, why = "not used", "It repeats the strength test results, so it's an outcome, not an input."
        elif verdict == "reject" and r["level"] == "batch":
            status, why = "not used", "No clear link to strength between batches of the same mix."
        elif verdict == "reject":
            status, why = "not used", "Didn't improve predictions on mixes the model never saw."
        else:
            status, why = "not used", "A label, not a quantity, so it can't go in the model."
        out.append(dict(column=c, file=s.data.candidates[c].file, status=status, why=why))
    return out


UNIT_NAMES = {"lb/yd3": "lb/yd³", "gal/yd3": "gal/yd³", "fl_oz/cwt": "fl oz/cwt", "F": "°F", "MPa": "MPa", "ksi": "ksi",
              "hours": "hours", "L/m3": "L/m³", "mL/100kg": "mL/100 kg", "mL/kg": "mL/kg", "%": "% of binder"}
UNIT_TARGETS = {"kg/m3": "kg/m³", "°C": "°C", "psi": "psi", "days": "days"}


def _conversions(report: dict) -> list[str]:
    """The unit changes the data needed, once each: 'lb/yd³ → kg/m³'."""
    out = []
    for entry in report["conversions"]:
        unit, target = entry.rsplit(": ", 1)[1].split(" -> ")
        unit = units.normalize_unit(unit)
        if unit in UNIT_NAMES and f"{UNIT_NAMES[unit]} → {UNIT_TARGETS[target]}" not in out:
            out.append(f"{UNIT_NAMES[unit]} → {UNIT_TARGETS[target]}")
    return out


def _data_artifact(s: Session) -> None:
    ds, report = s.data, s.map_report
    if s.validation is None:
        s.validation = validate.validate(ds, s.mapping, report)
    total = len(ds.batches)

    coverage = []
    for row in s.validation["boxcrete_inputs"]:
        file, _, column = (row["source"] or "").partition(": ")
        coverage.append(dict(input=row["input"], status=None, source=dict(file=file, column=column) if column else None,
                             mixes_with_value=total if column else 0, mixes=total))
    for c in [*s.accepted, *(c for c in s.offered if c not in s.accepted)]:
        coverage.append(dict(input=c, status="new" if c in s.accepted else "incomplete",
                             source=dict(file=ds.candidates[c].file, column=c), **_counts(s, c)))

    tests = s.mapping["tests"]
    means = ds.rows.groupby(["recipe", "age"], sort=True)["strength"].mean().reset_index()
    curves = [dict(mix=mix, points=[[float(a), round(float(v))] for a, v in zip(g["age"], g["strength"])])
              for mix, g in means.groupby("recipe", sort=False)]
    s.emit("artifact", name="data", data=dict(
        mixes=report["recipes"], batches=report["batches"], ages=report["ages"],
        cylinders_used=ds.cylinders, cylinders_read=int(report["tests_matched"].split("/")[1]),
        dropped=report["dropped"], mixes_left_out=report.get("recipes_left_out", []),
        conversions=_conversions(report), coverage=coverage, tested=_tested(s),
        output=dict(source=dict(file=tests["file"], column=tests["fields"]["strength"]["column"]),
                    age=dict(file=tests["file"], column=tests["fields"]["age"]["column"]), curves=curves),
    ))


# --------------------------------------------------------------------------- training (Train button)

def offer_training(s: Session) -> None:
    """Show the Train button after a turn that left prepared, untrained data."""
    if s.data is not None and s.ready_to_train:
        s.emit("train_button", inputs=s.accepted)


def train(s: Session, n_bayes: int = 10, use_cache: bool = True) -> dict:
    """AutoML on the accepted inputs, then save the winner as the next model version.

    A finished run is cached by (data, inputs, budget), so a repeat run can't stall.
    """
    _need_data(s)
    ds, extras = s.data, list(s.accepted)
    total = automl.n_rounds(n_bayes=n_bayes)

    def running(done: int, best: float | None) -> None:
        s.emit("artifact", name="training", data=dict(
            status="running", done=done, total=total, extra_inputs=extras,
            best_so_far=dict(rmse_28d=round(best)) if best is not None and best < float("inf") else None))

    running(0, None)
    cache = models.ROOT / "_cache" / f"{_run_key(ds, extras, n_bayes)}.json"
    reused = use_cache and cache.exists()
    if reused:
        s.training = json.loads(cache.read_text())
    else:
        s.training = automl.run_automl(ds, extras, n_bayes=n_bayes, on_round=running)
        s.training["finished"] = datetime.now().isoformat(timespec="seconds")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(s.training, default=str))
    s.emit("tool_progress", text="Fitting the best setup on all your mixes")
    s.model_meta = models.save(s.plant, ds, s.training, s.mapping)
    s.model = s.model_meta["version"]
    s.ready_to_train = False

    board, win = s.training["leaderboard"], s.training["winner"]
    at_28 = ds.rows.loc[ds.rows["age"] == 28, "strength"]
    gain = round(max(0.0, -win["vs_baseline_28d"]), 3)
    s.emit("artifact", name="training", data=dict(
        status="done", total=len(board), gain=gain, best_rmse_28d=win["error_28d_psi"],
        baseline_rmse_28d=board[0]["error_28d_psi"], tuning_helped=s.training["tuning_helped"],
        meta_default=s.training["meta_default"], ages=[r for r in board[0]["errors_by_age_psi"]], reused=reused,
        typical_28d_psi=round(float(at_28.median())) if len(at_28) else None,
        rows=[dict(row=r["name"], rmse_28d=r["error_28d_psi"], settings=r["settings"], extra_inputs=r["inputs"],
                   chosen=r["best"]) for r in board],
        model=dict(version=s.model, mixes=s.model_meta["recipes"]),
    ))
    s.emit("models_updated", model=s.model)
    return dict(ok=True, more_accurate_than_baseline_percent=round(100 * gain), tuning_helped=s.training["tuning_helped"],
                saved_as=s.model, winner=win["name"], reused_identical_earlier_run=reused,
                leaderboard=[{k: r[k] for k in ("name", "error_28d_psi", "vs_baseline_28d", "best",
                                                "folds_won_vs_baseline")} for r in board],
                method=s.training["method"])


def _run_key(ds: Dataset, extras: list[str], n_bayes: int) -> str:
    h = hashlib.sha256()
    h.update(ds.rows[sorted(ds.rows.columns)].to_csv(index=False).encode())  # same data, whatever the mapping's order
    for c in sorted(extras):
        h.update(c.encode())
        h.update(ds.feature(c).to_csv().encode())
    h.update(f"{n_bayes}".encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- using a saved model

def _saved(s: Session, version: str | None) -> tuple:
    version = version or s.model
    if version is None:  # never answer from a model trained on someone else's upload by accident
        raise ToolError("no model in this session yet: the user can press Train model, or pick a saved "
                        "version from list_models and pass it as version")
    model, meta = models.load(s.plant, version)
    rows = pd.read_csv(models.ROOT / models.slug(s.plant) / meta["version"] / "rows.csv",
                       dtype={"recipe": str, "batch": str})
    return model, meta, rows


def get_results(s: Session, version: str | None = None) -> dict:
    """CO2 vs strength of the plant's mixes, and lower-CO2 mixes to test next."""
    model, meta, rows = _saved(s, version)
    extras = meta["inputs"]
    levels = meta.get("input_levels", {})
    kept = [e for e in extras if levels.get(e) == "batch"]
    per_batch = rows.drop_duplicates("batch")[["recipe", *extras]] if extras else None
    rec = design.recipe_table(rows, extras, per_batch)
    out = design.propose(model, rec, extras, batch_extras=kept)

    unit = _display_unit(meta.get("mapping"))
    factor = units.LB_YD3_TO_KG_M3 if unit == "lb/yd3" else 1.0
    for p in out["proposals"]:
        p["composition"] = {design.LABELS[f]: round(v / factor, 1 if f == "hrwr" else None)
                            for f, v in p["composition_kg_m3"].items()}
        p["composition_unit"] = unit
        p["plus_minus_28d_psi"] = round(1.645 * p.pop("sd_28d_psi"))  # 90%, the same ± the chat's predict gives
    s.emit("artifact", name="results", data=dict(
        model=meta["version"], compared_with=len(out["recipes"]), unit=unit, kept=kept,
        recipe_extras=[e for e in extras if e not in kept],
        existing=[dict(mix=r["recipe"], gwp=r["gwp_kg_m3"],
                       psi=r["observed_28d_psi"] if r["observed_28d_psi"] is not None else r["predicted_28d_psi"])
                  for r in out["recipes"]],
        proposals=[dict(id=p["id"], gwp=p["gwp_kg_m3"], pred_psi=p["predicted_28d_psi"],
                        plus_minus_psi=p["plus_minus_28d_psi"], pred_7d_psi=p["predicted_7d_psi"],
                        target_psi=p["target_28d_psi"], composition=p["composition"], co2_saving=p["co2_saving"],
                        replaces=p["replaces"], extras=p["extras"]) for p in out["proposals"]],
    ))
    for p in out["proposals"]:
        p["co2_saving_percent"] = round(100 * p.pop("co2_saving"))
    return dict(model=meta["version"], recipes=f"{len(out['recipes'])} recipes (shown in the chart)",
                proposals=out["proposals"], method=out["method"],
                note="± is the model's 90% uncertainty for the mix's mean strength")


def predict(s: Session, mixes: list[dict], ages: list[float] | None = None, version: str | None = None) -> dict:
    """Predict strength for existing recipes, optionally changed (changes in any supported unit)."""
    model, meta, rows = _saved(s, version)
    ages = ages or [7, 28]
    base = rows.drop_duplicates("recipe").set_index("recipe")
    extras = meta["inputs"]
    own = rows.drop_duplicates("batch").groupby("recipe")[extras].mean() if extras else None
    out, errors = [], []
    for i, m in enumerate(mixes):
        rid = m.get("recipe")
        if rid is not None and rid not in base.index:
            errors.append(f"mix {i + 1}: unknown recipe '{rid}'; recipes are {list(base.index)}")
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
            if v is None:  # the mix's own average, or the plant's median for a mix that isn't one of theirs
                v = own.loc[rid, e] if rid else meta["typical_extra_values"][e]["median"]
            used[e] = round(float(v), 2)
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
    return dict(models=list_models_index(), active=s.model)


def list_models_index() -> list[dict]:
    return [dict(version=m["version"], trained=m["trained"], extra_inputs=m["inputs"], mixes=m["recipes"],
                 batches=m["batches"], rmse_28d=m["error_28d_psi"], gain=max(0.0, -m["vs_meta_default_28d"]))
            for m in models.versions(PLANT)]


def _display_unit(mapping: dict | None) -> str:
    spec = (mapping or {}).get("recipes", {}).get("fields", {}).get("cement", {})
    return "lb/yd3" if units.normalize_unit(spec.get("unit", "kg/m3")) == "lb/yd3" else "kg/m3"


# --------------------------------------------------------------------------- what the model sees

LABELS = {"read_files": "Reading your files", "map_columns": "Mapping columns to BOxCrete",
          "validate_data": "Checking the data", "suggest_inputs": "Testing the extra columns",
          "use_input": "Adding the input", "get_results": "Looking for lower-CO₂ mixes",
          "predict": "Predicting", "list_models": "Listing saved models"}


def _schema(props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or []}


_HELP = mapping_help()

SCHEMAS = [
    dict(type="function", name="read_files",
         description="Profile every uploaded sheet: header row, columns, kinds, coverage, sample values, and "
                     "likely join keys between files.",
         parameters=_schema()),
    dict(type="function", name="map_columns",
         description="Map the plant's columns to BOxCrete's inputs and join the files into one table. Sections: "
                     "recipes (one row per mix design; key), batches (one row per batch; key, recipe_key; "
                     "optional), tests (one row per cylinder; batch_key if cylinders name a batch, else "
                     "recipe_key; optional exclude: rows to drop by text in a column). IDs match ignoring case "
                     "and punctuation. Breaks after 28 days are dropped. Unmapped columns become candidates for "
                     "new inputs. A workbook with several sheets names each as 'file / sheet', as read_files shows. "
                     "One column per age: instead of the age and strength fields give tests.age_columns "
                     "{column: age in days} and tests.fields.strength.unit. Sections may share one table; if it "
                     "has no mix ID, leave out recipes.key and rows with the same ingredients count as one mix. "
                     f"Fields and allowed units: {json.dumps(_HELP['fields'])}. Required: "
                     f"{_HELP['required']}. Shape (names made up): {json.dumps(_HELP['example'])}",
         parameters=_schema({"mapping": {"type": "object"}}, ["mapping"])),
    dict(type="function", name="validate_data",
         description="Check the mapped data: coverage, ranges vs Meta's training data, cleaned cells, rows left "
                     "out, assumptions.",
         parameters=_schema()),
    dict(type="function", name="suggest_inputs",
         description="Test every candidate column on the plant's data: batch measurements by whether they "
                     "explain the scatter between batches of the same recipe (permutation test); recipe columns "
                     "by refitting BOxCrete with and without them. Verdicts: use, collect (useful but on part of "
                     "the batches), proxy (matters but is a label or date standing for another column), reject, "
                     "not_tested. Returns 'added' (now used in training) and 'offered' (the user decides).",
         parameters=_schema()),
    dict(type="function", name="use_input",
         description="Use an offered partly logged input in training, once the user has agreed.",
         parameters=_schema({"name": {"type": "string", "description": "the column, exactly as 'offered' gives it"}},
                            ["name"])),
    dict(type="function", name="get_results",
         description="CO2 vs strength of the plant's mixes, and lower-CO2 mixes to test next, from the saved model.",
         parameters=_schema({"version": {"type": "string"}})),
    dict(type="function", name="predict",
         description="Predict strength of existing recipes, optionally changed. mixes: [{recipe: id, changes: "
                     "{field: {value, unit}}, extras: {column: value}}]. Fields: cement, fly_ash, slag, water, "
                     "hrwr, fine, coarse. Extras default to the recipe's own average.",
         parameters=_schema({"mixes": {"type": "array", "items": {"type": "object"}},
                             "ages": {"type": "array", "items": {"type": "number"}},
                             "version": {"type": "string"}}, ["mixes"])),
    dict(type="function", name="list_models", description="Saved model versions for this plant.",
         parameters=_schema()),
]

TOOLS = {"read_files": read_files, "map_columns": map_columns, "validate_data": validate_data,
         "suggest_inputs": suggest_inputs, "use_input": use_input, "get_results": get_results,
         "predict": predict, "list_models": list_models}


def call(s: Session, name: str, args: dict) -> dict:
    """Run a tool for the model. Mistakes come back as data so it can correct itself."""
    if name not in TOOLS:
        return dict(ok=False, errors=[f"unknown tool {name}"])
    try:
        return TOOLS[name](s, **args)
    except (ToolError, TypeError, FileNotFoundError) as e:
        return dict(ok=False, errors=[str(e)])
