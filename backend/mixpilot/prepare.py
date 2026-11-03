"""Turn a plant's tables into one BOxCrete-format dataset, following a mapping.

The agent writes the mapping; this module applies it deterministically, so
every number the agent later quotes was computed here.

Mapping shape::

    {
      "mix_id": {"mix_designs.xlsx": "Mix #", "cylinder_breaks.xlsx": "Mix", ...},
      "fields": {                       # BOxCrete column -> where it comes from
        "Cement (kg/m3)": {"file": "...", "column": "...", "unit": "lb/yd3"},
        "HRWR (kg/m3)":   {"file": "...", "column": "...", "unit": "oz/cwt"},
        "Temp (C)":       {"file": "...", "column": "...", "unit": "F"},
        "Time":           {"file": "...", "column": "...", "unit": "days"},
        "Strength":       {"file": "...", "column": "...", "unit": "psi"}
      },
      "extra_inputs": {"Air (%)": {"file": "...", "column": "...", "unit": "%"}},
      "exclude": [{"file": "...", "column": "Notes", "contains": "cap failure"}],
      "hrwr_density": 1.06
    }

Per-mix values (composition, temperature, extra inputs) may come from any
table keyed by mix; several rows per mix (e.g. truck batches) are averaged.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from mixpilot import units
from mixpilot.ingest import read_file
from mixpilot.schema import COMPOSITION, GWP_FACTORS, MAX_AGE_DAYS

BINDER = ["Cement (kg/m3)", "Fly Ash (kg/m3)", "Slag (kg/m3)"]


def normalize_mix_id(v) -> str:
    return re.sub(r"[^0-9A-Z]", "", str(v).upper())


def _table(tables: dict[str, pd.DataFrame], file: str) -> pd.DataFrame:
    if file not in tables:
        raise KeyError(f"no file named {file!r}; have {sorted(tables)}")
    return tables[file]


def _convert(name: str, values: pd.Series, unit: str) -> pd.Series:
    if name in COMPOSITION and name != "HRWR (kg/m3)":
        return units.mass_to_kg_m3(values, unit)
    if name == "Temp (C)":
        return units.temp_to_c(values, unit)
    return values  # extra inputs keep the plant's unit (named in the column)


def build_dataset(files: list[str | Path], mapping: dict, material_source: int = 2) -> tuple[pd.DataFrame, dict]:
    """Returns (BOxCrete-format dataframe, report of what was done)."""
    tables = {}
    for f in files:
        sheets = read_file(f)
        tables[Path(f).name] = next(iter(sheets.values()))  # first non-empty sheet

    report = {"conversions": [], "dropped": {}}
    ids = {file: _table(tables, file)[col].map(normalize_mix_id) for file, col in mapping["mix_id"].items()}

    # --- per-mix values, averaged over rows of the same mix ---
    per_mix = {}
    fields = mapping["fields"]
    per_mix_fields = {k: v for k, v in fields.items() if k not in ("Time", "Strength")}
    per_mix_fields |= mapping.get("extra_inputs", {})
    for name, src in per_mix_fields.items():
        if name == "HRWR (kg/m3)":
            continue
        t = _table(tables, src["file"])
        raw = pd.to_numeric(t[src["column"]], errors="coerce")
        per_mix[name] = _convert(name, raw, src["unit"]).groupby(ids[src["file"]]).mean()
        if name in fields and src["unit"] not in ("kg/m3", "C"):
            report["conversions"].append(f"{src['column']} [{src['unit']}] -> {name}")
    mixes = pd.DataFrame(per_mix)

    if "HRWR (kg/m3)" in fields:
        src = fields["HRWR (kg/m3)"]
        t = _table(tables, src["file"])
        raw = pd.to_numeric(t[src["column"]], errors="coerce").groupby(ids[src["file"]]).mean()
        if src["unit"] == "oz/cwt":
            density = mapping.get("hrwr_density", units.DEFAULT_HRWR_DENSITY)
            binder = mixes[BINDER].sum(axis=1)
            mixes["HRWR (kg/m3)"] = units.hrwr_oz_cwt_to_kg_m3(raw.reindex(mixes.index), binder, density)
            report["conversions"].append(f"{src['column']} [oz/cwt, density {density} kg/L] -> HRWR (kg/m3)")
        else:
            mixes["HRWR (kg/m3)"] = units.mass_to_kg_m3(raw, src["unit"]).reindex(mixes.index)

    # --- strength tests: one row per cylinder (or per test) ---
    s_src, t_src = fields["Strength"], fields["Time"]
    if s_src["file"] != t_src["file"]:
        raise ValueError("strength and age must come from the same table")
    tests = _table(tables, s_src["file"]).copy()
    tests["_mix"] = ids[s_src["file"]]
    tests["_age"] = pd.to_numeric(tests[t_src["column"]], errors="coerce")
    tests["_psi"] = units.strength_to_psi(pd.to_numeric(tests[s_src["column"]], errors="coerce"), s_src["unit"])
    n0 = len(tests)
    for rule in mapping.get("exclude", []):
        if rule["file"] != s_src["file"]:
            continue
        hit = tests[rule["column"]].astype(str).str.contains(rule["contains"], case=False, na=False)
        report["dropped"][f"{rule['column']} contains '{rule['contains']}'"] = int(hit.sum())
        tests = tests[~hit]
    late = tests["_age"] > MAX_AGE_DAYS
    report["dropped"][f"older than {MAX_AGE_DAYS} days"] = int(late.sum())
    tests = tests[~late & tests["_psi"].notna() & tests["_age"].notna()]
    no_design = ~tests["_mix"].isin(mixes.index)
    report["dropped"]["no matching mix design"] = int(no_design.sum())
    tests = tests[~no_design]

    agg = tests.groupby(["_mix", "_age"])["_psi"].agg(["mean", "std", "count"]).reset_index()
    out = agg.join(mixes, on="_mix")
    out.insert(0, "Mix Name", out.pop("_mix"))
    out.insert(1, "Material Source", material_source)
    out["Time"] = out.pop("_age")
    out["GWP"] = sum(out[c] * GWP_FACTORS[c] for c in COMPOSITION if c in out)
    out["Strength (Mean)"] = out.pop("mean").round(0)
    out["Strength (Std)"] = out.pop("std").fillna(0).round(0)
    out["# of measurements"] = out.pop("count")

    extras = list(mapping.get("extra_inputs", {}))
    order = ["Mix Name", "Material Source", *COMPOSITION, "Temp (C)", *extras, "Time", "GWP",
             "Strength (Mean)", "Strength (Std)", "# of measurements"]
    out = out[[c for c in order if c in out]].sort_values(["Mix Name", "Time"]).reset_index(drop=True)

    report |= {
        "mixes": int(out["Mix Name"].nunique()),
        "cylinders_used": int(len(tests)),
        "cylinders_read": int(n0),
        "rows": int(len(out)),
        "ages": sorted(int(a) for a in out["Time"].unique()),
        "mix_designs_without_tests": int((~mixes.index.isin(out["Mix Name"])).sum()),
    }
    return out, report
