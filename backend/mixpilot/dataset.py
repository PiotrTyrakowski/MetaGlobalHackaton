"""Turn the plant's sheets plus a column mapping into one BOxCrete-ready table.

The mapping is what the agent proposes (see MAPPING_EXAMPLE). Three levels:
recipes (one row per mix design), batches (one row per batch, optional) and tests
(one row per cylinder). Every column that is not mapped and not a key becomes a
candidate input: recipe-level or batch-level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from mixpilot import units
from mixpilot.excel import Sheet, as_feature, column_kind, to_number

RECIPE_FIELDS = ["cement", "fly_ash", "slag", "water", "hrwr", "fine", "coarse"]
REQUIRED = {"cement", "water", "age", "strength"}
DEFAULT_CURE_TEMP_C = 23.0  # ASTM C511 moist room / tank, 73.5 °F
MATERIAL_SOURCE = 2.0  # Meta's ready-mix-like concrete class; constant within one plant

MAPPING_EXAMPLE = {
    "recipes": {
        "file": "mix_designs.xlsx", "key": "Mix ID",
        "fields": {"cement": {"column": "Type IL Cement (lb/cy)", "unit": "lb/yd3"},
                   "water": {"column": "Water (gal/cy)", "unit": "gal/yd3"},
                   "hrwr": {"column": "HRWR (oz/cwt)", "unit": "fl_oz/cwt"}},
        "other_cementitious": [{"column": "Silica Fume (lb/cy)", "unit": "lb/yd3"}],
    },
    "batches": {"file": "daily_QC.xlsx", "key": "Ticket No.", "recipe_key": "Mix Code",
                "fields": {"cure_temp": {"column": "Cure Tank (F)", "unit": "F"}}},
    "tests": {"file": "cylinder_breaks.xlsx", "batch_key": "Ticket #",
              "fields": {"age": {"column": "Age", "unit": "days"},
                         "strength": {"column": "Strength (psi)", "unit": "psi"}}},
    "ignore": ["Notes"],
}


@dataclass
class Candidate:
    name: str
    level: str  # "recipe" | "batch"
    kind: str  # numeric | date | time_of_day | category
    file: str


@dataclass
class Dataset:
    rows: pd.DataFrame  # one row per (batch, age): BOxCrete inputs + mean strength
    batches: pd.DataFrame  # one row per batch: recipe id + every candidate column
    cylinders: int
    candidates: dict[str, Candidate]
    notes: list[str] = field(default_factory=list)

    def tensors(self, extras: list[str] = (), rows: pd.DataFrame | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """X in BOxCrete's DEFAULT_X_COLUMNS order, then the extras; Y in psi."""
        r = self.rows if rows is None else rows
        cols = [r[f] for f in RECIPE_FIELDS] + [pd.Series(MATERIAL_SOURCE, index=r.index),
                                                r["temp_c"], r["age"]]
        cols += [self.feature(e).reindex(r["batch"]).to_numpy() for e in extras]
        X = torch.tensor(np.column_stack(cols).astype(float), dtype=torch.float64)
        return X, torch.tensor(r["strength"].to_numpy(dtype=float), dtype=torch.float64)

    def feature(self, name: str) -> pd.Series:
        """Numeric values of a candidate per batch (index = batch id)."""
        c = self.candidates[name]
        return as_feature(self.batches.set_index("batch")[name], c.kind)

    @property
    def groups(self) -> np.ndarray:
        """Mix design of every row, for holding out whole designs."""
        return self.rows["recipe"].to_numpy()


class MappingError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def build(sheets: list[Sheet], mapping: dict) -> tuple[Dataset, dict]:
    """Apply the mapping. Returns the dataset and a report; raises MappingError."""
    errors: list[str] = []
    report: dict = dict(conversions=[], assumptions=[], cleaned_cells={}, failed_cells={})
    by_file = {s.file: s for s in sheets}
    ignore = set(mapping.get("ignore", []))

    def sheet_for(section: str) -> pd.DataFrame | None:
        spec = mapping.get(section)
        if not spec:
            return None
        s = by_file.get(spec.get("file"))
        if s is None:
            errors.append(f"{section}: file '{spec.get('file')}' not uploaded; have {sorted(by_file)}")
            return None
        for col in _columns_of(spec):
            if col not in s.df.columns:
                errors.append(f"{section}: column '{col}' not in {s.file}; columns are {list(s.df.columns)}")
        return s.df

    R, B, T = sheet_for("recipes"), sheet_for("batches"), sheet_for("tests")
    if R is None or T is None:
        errors.append("need at least 'recipes' and 'tests' sections")
    for section in ("recipes", "batches", "tests"):
        for f, spec in mapping.get(section, {}).get("fields", {}).items():
            if (msg := units.check_unit(f, spec.get("unit", ""))) is not None:
                errors.append(f"{section}.{f}: {msg}")
    mapped = {f for sec in ("recipes", "batches", "tests") for f in mapping.get(sec, {}).get("fields", {})}
    if missing := REQUIRED - mapped:
        errors.append(f"required fields not mapped: {sorted(missing)}")
    if errors:
        raise MappingError(errors)

    def numeric(df: pd.DataFrame, section: str, f: str) -> np.ndarray:
        spec = mapping[section]["fields"][f]
        x, cleaned, failed = to_number(df[spec["column"]])
        if cleaned:
            report["cleaned_cells"][spec["column"]] = cleaned
        if failed:
            report["failed_cells"][spec["column"]] = failed
        return x.to_numpy()

    def field_values(df: pd.DataFrame, section: str, cementitious=None) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for f, spec in mapping[section].get("fields", {}).items():
            if f == "hrwr":
                continue
            out[f] = units.convert(f, spec["unit"], numeric(df, section, f))
            _note_conversion(report, f, spec)
        if "hrwr" in mapping[section].get("fields", {}):
            spec = mapping[section]["fields"]["hrwr"]
            out["hrwr"] = units.convert("hrwr", spec["unit"], numeric(df, section, "hrwr"), cementitious)
            _note_conversion(report, "hrwr", spec)
        return out

    # --- recipes ---
    rs = mapping["recipes"]
    recipes = pd.DataFrame({"recipe": _key(R[rs["key"]])})
    binder = np.zeros(len(R))
    for f in ("cement", "fly_ash", "slag"):
        if f in rs["fields"]:
            spec = rs["fields"][f]
            binder = binder + np.nan_to_num(units.convert(f, spec["unit"], numeric(R, "recipes", f)))
    for extra in rs.get("other_cementitious", []):
        x, _, _ = to_number(R[extra["column"]])
        binder = binder + np.nan_to_num(units.convert("cement", extra["unit"], x.to_numpy()))
    recipes = recipes.join(field_values(R, "recipes", cementitious=binder))
    for f in RECIPE_FIELDS:
        if f not in recipes:
            recipes[f] = 0.0
            report["assumptions"].append(f"no {f.replace('_', ' ')} column: assumed 0")

    # --- batches ---
    if B is not None:
        bs = mapping["batches"]
        batches = pd.DataFrame({"batch": _key(B[bs["key"]]), "recipe": _key(B[bs["recipe_key"]])})
        batches = batches.join(field_values(B, "batches"))
        used_b = {bs["key"], bs["recipe_key"]} | _columns_of(bs)
        level = {c: "batch" for c in B.columns if c not in used_b and c not in ignore}
        for c in level:
            batches[c] = B[c].to_numpy()
    else:
        batches = pd.DataFrame({"batch": recipes["recipe"], "recipe": recipes["recipe"]})
        level = {}

    # recipe-level candidates are carried onto each batch (other cementitious
    # materials such as silica fume stay candidates: BOxCrete has no slot for them)
    used_r = {rs["key"]} | (_columns_of(rs) - {e["column"] for e in rs.get("other_cementitious", [])})
    rc = pd.DataFrame({"recipe": recipes["recipe"]})
    for c in R.columns:
        if c not in used_r and c not in ignore:
            name = c if c not in level else f"{c} (recipe)"
            rc[name] = R[c].to_numpy()
            level[name] = "recipe"
    batches = batches.merge(rc, on="recipe", how="left")

    # --- tests ---
    ts = mapping["tests"]
    tests = field_values(T, "tests")
    if "batch_key" in ts:
        tests["batch"] = _key(T[ts["batch_key"]])
    else:
        tests["batch"] = _key(T[ts["recipe_key"]])

    matched = tests["batch"].isin(batches["batch"])
    report["tests_matched"] = f"{int(matched.sum())}/{len(tests)}"
    b_matched = batches["recipe"].isin(recipes["recipe"])
    report["batches_matched"] = f"{int(b_matched.sum())}/{len(batches)}"
    if matched.mean() < 0.5 or b_matched.mean() < 0.5:
        raise MappingError([f"joins look wrong: tests matched to batches {report['tests_matched']}, "
                            f"batches matched to recipes {report['batches_matched']}"])
    tests = tests[matched & tests["strength"].notna() & tests["age"].notna()]
    batches = batches[b_matched].reset_index(drop=True)

    # --- one row per (batch, age) ---
    agg = tests.groupby(["batch", "age"])["strength"].agg(["mean", "std", "size"]).reset_index()
    agg.columns = ["batch", "age", "strength", "strength_sd", "cylinders"]
    rows = agg.merge(batches[["batch", "recipe"]], on="batch").merge(recipes, on="recipe")
    if "cure_temp" in batches:
        rows = rows.merge(batches[["batch", "cure_temp"]], on="batch")
    elif "cure_temp" in tests:  # recorded per cylinder
        rows = rows.join(tests.groupby("batch")["cure_temp"].mean(), on="batch")
    if "cure_temp" not in rows:
        rows["temp_c"] = DEFAULT_CURE_TEMP_C
        report["assumptions"].append(f"no curing temperature: assumed {DEFAULT_CURE_TEMP_C} °C (73.5 °F)")
    else:
        rows["temp_c"] = rows["cure_temp"].fillna(DEFAULT_CURE_TEMP_C)
    report["assumptions"].append("all mixes treated as BOxCrete material source 2 (ready-mix concrete)")

    # --- candidates ---
    candidates = {}
    for c, lvl in level.items():
        kind = column_kind(batches[c])
        if kind in ("text", "empty"):
            report.setdefault("skipped_columns", []).append(f"{c} ({kind})")
            continue
        candidates[c] = Candidate(name=c, level=lvl, kind=kind,
                                  file=mapping["recipes" if lvl == "recipe" else "batches"]["file"])

    ds = Dataset(rows=rows.sort_values(["recipe", "batch", "age"]).reset_index(drop=True),
                 batches=batches, cylinders=int(len(tests)), candidates=candidates)
    report.update(recipes=int(rows["recipe"].nunique()), batches=int(rows["batch"].nunique()),
                  rows=int(len(rows)), cylinders=ds.cylinders,
                  ages=sorted(float(a) for a in rows["age"].unique()))
    return ds, report


def _columns_of(spec: dict) -> set[str]:
    cols = {v["column"] for v in spec.get("fields", {}).values()}
    cols |= {e["column"] for e in spec.get("other_cementitious", [])}
    cols |= {spec[k] for k in ("key", "recipe_key", "batch_key") if k in spec}
    return cols


def _key(s: pd.Series) -> pd.Series:
    """Join keys as clean strings (7310002, 7310002.0 and ' 7310002' all match)."""
    def clean(v):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v).strip()
    return s.map(clean).reset_index(drop=True)


def _note_conversion(report: dict, f: str, spec: dict) -> None:
    unit = units.normalize_unit(spec["unit"])
    target = {"cure_temp": "°C", "strength": "psi", "age": "days"}.get(f, "kg/m3")
    entry = f"{spec['column']}: {spec['unit']} -> {target}"
    if entry not in report["conversions"]:
        report["conversions"].append(entry)
        if unit in units.ASSUMPTIONS:
            report["assumptions"].append(f"{spec['column']}: {units.ASSUMPTIONS[unit]}")
