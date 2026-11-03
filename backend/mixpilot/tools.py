"""The agent's tools: thin wrappers over the pipeline that also update the UI.

Each tool takes the session and the model's arguments and returns a small
JSON-able dict for the model. What the user sees (cards, artifacts) is
emitted here, from computed values, never from the model's text.
"""

import json
from pathlib import Path

import pandas as pd

from mixpilot import automl, units
from mixpilot.checks import candidate_inputs, coverage, issues
from mixpilot.ingest import profile_file, read_file
from mixpilot.prepare import build_dataset, normalize_mix_id
from mixpilot.schema import COMPOSITION
from mixpilot.session import MODELS_DIR, Session

REQUIRED = [*COMPOSITION, "Temp (C)", "Time", "Strength"]
UNITS = {c: units.MASS_UNITS for c in COMPOSITION} | {
    "HRWR (kg/m3)": units.HRWR_UNITS,
    "Temp (C)": units.TEMP_UNITS,
    "Time": {"days"},
    "Strength": units.STRENGTH_UNITS,
}
ALL_UNITS = sorted(set().union(*UNITS.values()))
COMPLETE = 0.9  # share of mixes an extra input needs to be used for training
MIN_TEST_MIXES = 10

# What the plant types, in its own units -> BOxCrete column
US_FIELDS = {c.replace("(kg/m3)", "(lb/yd3)"): c for c in COMPOSITION if c != "HRWR (kg/m3)"}
US_FIELDS |= {"HRWR (oz/cwt)": "HRWR (kg/m3)", "Temp (F)": "Temp (C)"}


class ToolError(Exception):
    """A problem the model can fix by calling the tool differently."""


# --------------------------------------------------------------------------- helpers

def _columns(s: Session) -> dict[str, list[str]]:
    return {f.name: list(next(iter(read_file(f).values())).columns) for f in s.files}


def _check_source(cols: dict[str, list[str]], file: str, column: str) -> None:
    if file not in cols:
        raise ToolError(f"no file named {file!r}; files are {sorted(cols)}")
    if column not in cols[file]:
        raise ToolError(f"{file} has no column {column!r}; columns are {cols[file]}")


def _rebuild(s: Session) -> tuple[pd.DataFrame, dict]:
    df, report = build_dataset(s.files, s.mapping)
    df.to_csv(s.dataset, index=False)
    return df, report


def _data_artifact(s: Session, df: pd.DataFrame, report: dict) -> None:
    extras = list(s.mapping.get("extra_inputs", {}))
    rows = coverage(df, extras)
    sources = s.mapping["fields"] | s.mapping.get("extra_inputs", {})
    for r in rows:
        r["status"] = "new" if r["input"] in s.accepted else "incomplete" if r["input"] in s.tested else None
        src = sources.get(r["input"])
        r["source"] = src and {"file": src["file"], "column": src["column"]}
    strength, age = s.mapping["fields"]["Strength"], s.mapping["fields"]["Time"]
    curves = [{"mix": mix, "points": [[float(a), round(float(v))] for a, v in zip(g["Time"], g["Strength (Mean)"])]}
              for mix, g in df.sort_values("Time").groupby("Mix Name", sort=False)]
    s.emit("artifact", name="data", data={
        "mixes": report["mixes"],
        "rows": report["rows"],
        "ages": report["ages"],
        "cylinders_read": report["cylinders_read"],
        "cylinders_used": report["cylinders_used"],
        "dropped": {k: v for k, v in report["dropped"].items() if v},
        "conversions": report["conversions"],
        "coverage": rows,
        "output": {"source": {"file": strength["file"], "column": strength["column"]},
                   "age": {"file": age["file"], "column": age["column"]}, "curves": curves},
    })


def _extra_coverage(s: Session, name: str, file: str, column: str) -> tuple[pd.DataFrame, dict, dict]:
    """Add an extra input to the mapping, rebuild, and measure its coverage."""
    if s.mapping is None:
        raise ToolError("call prepare_data first")
    if "|" in name or name in REQUIRED or "(" not in name:
        raise ToolError("name must look like 'Air (%)': a short label with its unit in brackets")
    _check_source(_columns(s), file, column)
    if file not in s.mapping["mix_id"]:
        raise ToolError(f"{file} has no mix ID column in the mapping")
    s.mapping.setdefault("extra_inputs", {})[name] = {"file": file, "column": column, "unit": ""}
    df, report = _rebuild(s)
    mixes = df.drop_duplicates("Mix Name")
    have = int(mixes[name].notna().sum())
    info = {
        "input": name,
        "mixes_with_value": have,
        "mixes": len(mixes),
        "share": round(have / len(mixes), 2),
        "min": round(float(mixes[name].min()), 2) if have else None,
        "max": round(float(mixes[name].max()), 2) if have else None,
    }
    return df, report, info


def _model_dir(version: str | None) -> tuple[Path, dict]:
    index = list_models_index()
    if not index:
        raise ToolError("no trained model yet")
    entry = next((m for m in index if m["version"] == version), None) if version else index[-1]
    if entry is None:
        raise ToolError(f"no model {version!r}; have {[m['version'] for m in index]}")
    return MODELS_DIR / entry["version"], entry


def list_models_index() -> list[dict]:
    path = MODELS_DIR / "index.json"
    return json.loads(path.read_text()) if path.exists() else []


# --------------------------------------------------------------------------- tools

def inspect_files(s: Session) -> dict:
    if not s.files:
        raise ToolError("no files uploaded yet")
    return {"files": [profile_file(f) for f in s.files]}


def prepare_data(s: Session, mix_id: list[dict], fields: list[dict], exclude: list[dict] | None = None,
                 hrwr_density: float | None = None) -> dict:
    cols = _columns(s)
    for m in mix_id:
        _check_source(cols, m["file"], m["column"])
    by_target = {}
    for f in fields:
        if f["target"] not in UNITS:
            raise ToolError(f"unknown target {f['target']!r}; targets are {REQUIRED}")
        _check_source(cols, f["file"], f["column"])
        if f["unit"] not in UNITS[f["target"]]:
            raise ToolError(f"{f['target']} unit must be one of {sorted(UNITS[f['target']])}, got {f['unit']!r}")
        by_target[f["target"]] = {"file": f["file"], "column": f["column"], "unit": f["unit"]}
    missing = [t for t in REQUIRED if t not in by_target]
    if missing:
        raise ToolError(f"missing targets: {missing}")
    id_files = {m["file"]: m["column"] for m in mix_id}
    for t, src in by_target.items():
        if src["file"] not in id_files:
            raise ToolError(f"{t} comes from {src['file']}, which has no mix ID column in mix_id")
    strength_file = by_target["Strength"]["file"]
    rules = []
    for r in exclude or []:
        _check_source(cols, strength_file, r["column"])
        rules.append({"file": strength_file, "column": r["column"], "contains": r["contains"]})

    s.mapping = {"mix_id": id_files, "fields": by_target, "extra_inputs": {}, "exclude": rules}
    if hrwr_density:
        s.mapping["hrwr_density"] = hrwr_density
    s.accepted, s.tested = [], {}
    s.ready_to_train = True
    df, report = _rebuild(s)
    _data_artifact(s, df, report)
    return {
        "report": report,
        "issues": issues(df),
        "candidate_inputs": candidate_inputs(s.files, s.mapping),
    }


def add_input(s: Session, name: str, file: str, column: str) -> dict:
    df, report, info = _extra_coverage(s, name, file, column)
    if info["share"] < COMPLETE:
        del s.mapping["extra_inputs"][name]
        _rebuild(s)
        raise ToolError(f"{name} is recorded for only {info['mixes_with_value']} of {info['mixes']} mixes; "
                        "use test_input for inputs that are not complete")
    if name not in s.accepted:
        s.accepted.append(name)
    s.ready_to_train = True
    s.emit("card", kind="input_ready", **info, file=file, column=column)
    _data_artifact(s, df, report)
    return {**info, "used_in_training": True}


def test_input(s: Session, name: str, file: str, column: str) -> dict:
    df, report, info = _extra_coverage(s, name, file, column)
    if info["share"] >= COMPLETE:
        raise ToolError(f"{name} is complete ({info['share']:.0%} of mixes); use add_input")
    if info["mixes_with_value"] < MIN_TEST_MIXES:
        del s.mapping["extra_inputs"][name]
        _rebuild(s)
        raise ToolError(f"only {info['mixes_with_value']} mixes record {name}; too few to measure its value")
    s.emit("tool_progress", text=f"Testing {name.split(' (')[0].lower()} on the {info['mixes_with_value']} mixes that record it")
    result = automl.evaluate_input(s.dataset, name, s.accepted)
    s.tested[name] = result
    card = {**info, "rmse_28d_without": result["rmse_28d_without"], "rmse_28d_with": result["rmse_28d_with"],
            "gain": result["gain"], "file": file, "column": column}
    s.emit("card", kind="input_incomplete", **card)
    _data_artifact(s, df, report)
    return {**card, "used_in_training": False,
            "note": "errors are 28-day RMSE in psi, measured only on the mixes that record this input"}


def propose_mixes(s: Session, fixed: dict | None = None, model: str | None = None) -> dict:
    model_dir, entry = _model_dir(model or s.model)
    fixed = {k: float(v) for k, v in (fixed or {}).items() if k in entry["extra_inputs"]}
    r = automl.propose(model_dir, fixed)
    s.emit("artifact", name="results", data={"model": entry["version"], **r})
    keep = ("target_psi", "pred_psi", "sd_psi", "gwp", "compare_to", "compare_psi", "compare_gwp", "co2_saving",
            "mix_lb_yd3", "fixed")
    return {
        "model": entry["version"],
        "compared_with_mixes": r["compared_with"],
        "proposals": [{k: p[k] for k in keep if k in p} for p in r["proposals"]],
        "units": "strength psi, GWP kg CO2e/m3, mixes lb/yd3",
    }


def _us_mix(base: dict, hrwr_oz: float, extras: list[str]) -> dict:
    """A mix row from BOxCrete's units in the plant's units."""
    us = {k: base[v] / units.LB_PER_YD3_TO_KG_PER_M3 for k, v in US_FIELDS.items()
          if k not in ("HRWR (oz/cwt)", "Temp (F)")}
    return us | {"HRWR (oz/cwt)": hrwr_oz, "Temp (F)": base["Temp (C)"] * 9 / 5 + 32} | {e: base[e] for e in extras}


def predict_strength(s: Session, ages: list[float], mix_name: str | None = None, mix: dict | None = None,
                     change: dict | None = None, model: str | None = None) -> dict:
    model_dir, entry = _model_dir(model or s.model)
    data = pd.read_csv(model_dir / "data.csv")
    extras = entry["extra_inputs"]
    mix = dict(mix or {})
    if mix_name:
        rows = data[data["Mix Name"] == normalize_mix_id(mix_name)]
        if rows.empty:
            raise ToolError(f"no mix {mix_name!r} in the model's data")
        base = rows.iloc[0].to_dict()
    else:
        base = data.median(numeric_only=True).to_dict()  # anything not given: the plant's typical value
        needed = [k for k in US_FIELDS if k not in mix]
        if needed:
            raise ToolError(f"without mix_name, give every field of mix: missing {needed}")

    hrwr_oz = units.hrwr_kg_m3_to_oz_cwt(base["HRWR (kg/m3)"], sum(base[c] for c in ("Cement (kg/m3)", "Fly Ash (kg/m3)", "Slag (kg/m3)")))
    now = _us_mix(base, hrwr_oz, extras)
    for key, delta in (change or {}).items():
        if key not in now:
            raise ToolError(f"unknown field {key!r}; use {list(now)}")
        mix[key] = mix.get(key, now[key]) + float(delta)
    for key, value in mix.items():
        if float(value) < 0 and key != "Temp (F)":
            raise ToolError(f"{key} would be {float(value):.0f}: mix takes amounts, not differences; "
                            "use change for differences")
        if key in US_FIELDS and key not in ("HRWR (oz/cwt)", "Temp (F)"):
            base[US_FIELDS[key]] = units.mass_to_kg_m3(float(value), "lb/yd3")
        elif key == "Temp (F)":
            base["Temp (C)"] = units.temp_to_c(float(value), "F")
        elif key == "HRWR (oz/cwt)":
            hrwr_oz = float(value)
        elif key in extras:
            base[key] = float(value)
        else:
            raise ToolError(f"unknown field {key!r}; use {list(US_FIELDS) + extras}")
    binder = sum(base[c] for c in ("Cement (kg/m3)", "Fly Ash (kg/m3)", "Slag (kg/m3)"))
    base["HRWR (kg/m3)"] = units.hrwr_oz_cwt_to_kg_m3(hrwr_oz, binder)

    rows = [{**base, "Time": float(a)} for a in ages]
    preds = automl.predict(model_dir, rows)
    measured = {}
    if mix_name and not mix:
        m = data[data["Mix Name"] == normalize_mix_id(mix_name)]
        measured = dict(zip(m["Time"].astype(float), m["Strength (Mean)"]))
    return {
        "model": entry["version"],
        "mix": {k: round(v, 1 if k == "HRWR (oz/cwt)" else 2 if k in extras else None)
                for k, v in _us_mix(base, hrwr_oz, extras).items()},
        "predictions": [{"age_days": a, "psi": p["mean_psi"], "plus_minus": p["sd_psi"],
                         **({"measured_psi": measured[float(a)]} if float(a) in measured else {})}
                        for a, p in zip(ages, preds)],
    }


def list_models(s: Session) -> dict:
    return {"models": list_models_index(), "active": s.model}


# --------------------------------------------------------------------------- training (Train button)

def offer_training(s: Session) -> None:
    """Show the Train button after a turn that left prepared, untrained data."""
    if s.mapping is not None and s.ready_to_train:
        s.emit("train_button", inputs=s.accepted, setups=18 if s.accepted else 9)


def train(s: Session) -> dict:
    """AutoML over setups, then fit and save the winner. Runs when the user clicks Train."""
    total = 18 if s.accepted else 9
    done = []

    def progress():
        best = min(done, key=lambda r: r["rmse_28d"]) if done else None
        s.emit("artifact", name="training", data={
            "status": "running", "done": len(done), "total": total, "extra_inputs": s.accepted,
            "best_so_far": best and {"label": best["label"], "rmse_28d": best["rmse_28d"]},
        })

    progress()
    res = automl.search(s.dataset, s.accepted, on_result=lambda r: (done.append(r), progress()))
    s.emit("tool_progress", text="Fitting the winning setup on all mixes")
    entry = automl.save_model(s.dataset, res["best"], MODELS_DIR, res["best"],
                              baseline_rmse_28d=res["baseline"]["rmse_28d"], gain=res["gain"])
    s.model = entry["version"]
    s.ready_to_train = False

    def is_best(r):
        return r["extra"] == res["best"]["extra"] and r["settings"] == res["best"]["settings"]

    rows = [{"row": r["row"], "rmse_28d": r["rmse_28d"], "settings": r["settings"], "extra_inputs": r["extra"],
             "chosen": is_best(r)} for r in res["summary"]]
    s.emit("artifact", name="training", data={
        "status": "done", "total": total, "rows": rows, "gain": res["gain"],
        "baseline_rmse_28d": res["baseline"]["rmse_28d"], "best_rmse_28d": res["best"]["rmse_28d"],
        "model": entry,
    })
    s.emit("models_updated", model=entry)
    return {"leaderboard": rows, "gain_vs_meta_default": res["gain"], "saved_model": entry,
            "units": "28-day RMSE in psi, 5-fold cross-validation by mix"}


TOOLS = {
    "inspect_files": inspect_files,
    "prepare_data": prepare_data,
    "add_input": add_input,
    "test_input": test_input,
    "propose_mixes": propose_mixes,
    "predict_strength": predict_strength,
    "list_models": list_models,
}

LABELS = {
    "inspect_files": "Reading your files",
    "prepare_data": "Building the dataset",
    "add_input": "Adding input",
    "test_input": "Testing input",
    "propose_mixes": "Screening new mixes",
    "predict_strength": "Predicting strength",
    "list_models": "Looking up models",
}

_SOURCE = {"file": {"type": "string", "description": "file name, e.g. mix_designs.xlsx"},
           "column": {"type": "string", "description": "exact column name as shown by inspect_files"}}
_INPUT_ARGS = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": ("short plain-language name with unit in brackets, "
                                                    "e.g. 'Air (%)' or 'Site water (gal/yd3)', not the column header")},
        **_SOURCE,
    },
    "required": ["name", "file", "column"],
}

SCHEMAS = [
    {"type": "function", "name": "inspect_files",
     "description": "Describe the uploaded files: sheets, row counts, and for each column its type, fill rate, range and examples.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "prepare_data",
     "description": (
         "Map the plant's columns to BOxCrete's inputs and build the training dataset. Converts units, matches "
         "mix IDs across files (case and punctuation are ignored), averages several batches of the same mix, "
         "groups cylinders into mean/std/count per mix and age, and drops breaks after 28 days. Returns a "
         "report, data issues, and unused numeric columns that could become extra inputs. Every target must "
         "be mapped: " + ", ".join(REQUIRED) + ". Temp (C) is the curing temperature."),
     "parameters": {
         "type": "object",
         "properties": {
             "mix_id": {"type": "array", "description": "For every file used: the column holding the mix ID.",
                        "items": {"type": "object", "properties": _SOURCE, "required": ["file", "column"]}},
             "fields": {"type": "array", "description": "Where each BOxCrete input comes from, with the unit the plant used.",
                        "items": {"type": "object", "properties": {
                            "target": {"type": "string", "enum": REQUIRED}, **_SOURCE,
                            "unit": {"type": "string", "enum": ALL_UNITS}},
                            "required": ["target", "file", "column", "unit"]}},
             "exclude": {"type": "array", "description": "Strength-table rows to drop, e.g. tests a notes column marks as invalid.",
                         "items": {"type": "object", "properties": {
                             "column": {"type": "string"},
                             "contains": {"type": "string", "description": "case-insensitive text to match"}},
                             "required": ["column", "contains"]}},
             "hrwr_density": {"type": "number", "description": "admixture density in kg/L for oz/cwt; default 1.06"},
         },
         "required": ["mix_id", "fields"],
     }},
    {"type": "function", "name": "add_input",
     "description": "Use an extra column as a new model input. Only for inputs recorded for at least 90% of mixes. Shows the user a card.",
     "parameters": _INPUT_ARGS},
    {"type": "function", "name": "test_input",
     "description": ("For an input recorded for only part of the mixes: measure how much it improves the model on "
                     "the mixes that have it (two cross-validated fits, about 20 s). It is not used for training. "
                     "Shows the user a card."),
     "parameters": _INPUT_ARGS},
    {"type": "function", "name": "propose_mixes",
     "description": ("With a trained model: screen candidate mixes and return ones predicted to reach the plant's "
                     "usual strength classes with less CO2 than its existing mixes. Updates the Results panel."),
     "parameters": {"type": "object", "properties": {
         "fixed": {"type": "object", "description": "values held fixed for extra inputs, e.g. {\"Air (%)\": 6}",
                   "additionalProperties": {"type": "number"}},
         "model": {"type": "string", "description": "model version; default the latest"}}}},
    {"type": "function", "name": "predict_strength",
     "description": ("Predict strength with a trained model at given ages. Start from an existing mix "
                     "(mix_name) and set fields (mix) or shift them (change), or give a whole mix. "
                     "Fields use the plant's units: "
                     + ", ".join(US_FIELDS) + ", plus the model's extra inputs (e.g. 'Air (%)')."),
     "parameters": {"type": "object", "properties": {
         "ages": {"type": "array", "items": {"type": "number"}, "description": "ages in days, at most 28"},
         "mix_name": {"type": "string", "description": "existing mix to start from, e.g. R-101"},
         "mix": {"type": "object", "description": "field -> new amount, e.g. {\"Cement (lb/yd3)\": 400}",
                 "additionalProperties": {"type": "number"}},
         "change": {"type": "object", "description": "field -> amount to add, negative to remove, "
                    "e.g. {\"Cement (lb/yd3)\": -50}", "additionalProperties": {"type": "number"}},
         "model": {"type": "string"}},
         "required": ["ages"]}},
    {"type": "function", "name": "list_models",
     "description": "List the plant's saved models with date, inputs and error.",
     "parameters": {"type": "object", "properties": {}}},
]


def call(s: Session, name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name!r}"}
    try:
        return TOOLS[name](s, **args)
    except ToolError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"bad arguments: {e}"}
    except (KeyError, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
