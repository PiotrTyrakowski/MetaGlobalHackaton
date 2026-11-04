"""Turn the plant's sheets plus a column mapping into one BOxCrete-ready table.

The mapping is what the agent proposes (see MAPPING_EXAMPLE). Three levels:
recipes (one row per mix design), batches (one row per batch, optional) and tests
(one row per cylinder). Every column that is not mapped and not a key becomes a
candidate input: recipe-level or batch-level.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import torch

from mixpilot import units
from mixpilot.excel import Sheet, as_feature, column_kind, to_number

RECIPE_FIELDS = ["cement", "fly_ash", "slag", "water", "hrwr", "fine", "coarse"]
REQUIRED = {"cement", "water", "age", "strength"}
MAX_AGE_DAYS = 28  # BOxCrete models strength from day 0 to day 28
DEFAULT_CURE_TEMP_C = 23.0  # ASTM C511 moist room / tank, 73.5 °F
MATERIAL_SOURCE = 2.0  # Meta's ready-mix-like concrete class; constant within one plant
MIN_MIXES = 5  # every score holds out whole mixes in 5 folds
MIN_LAST_AGE = 21.0  # the model is judged on 28-day strength
MIX_ID = "Mix (same ingredients)"
AGE_FROM_HEADERS, STRENGTH_FROM_HEADERS = "Age (from the column names)", "Strength (from the column names)"

# Shows the shape only: the names and units here are made up, not any plant's.
MAPPING_EXAMPLE = {
    "recipes": {
        "file": "recipes.xlsx", "key": "Recipe",
        "fields": {"cement": {"column": "OPC kg", "unit": "kg/m3"},
                   "water": {"column": "Free water L", "unit": "L/m3"},
                   "hrwr": {"column": "SP mL/100kg", "unit": "mL/100kg"}},
        "other_cementitious": [{"column": "Pozzolan kg", "unit": "kg/m3"}],
    },
    "batches": {"file": "production.xlsx", "key": "Batch", "recipe_key": "Recipe code",
                "fields": {"cure_temp": {"column": "Curing C", "unit": "C"}}},
    "tests": {"file": "lab.xlsx", "batch_key": "Batch no",
              "fields": {"age": {"column": "Hours", "unit": "hours"},
                         "strength": {"column": "MPa", "unit": "MPa"}},
              "exclude": [{"column": "Remarks", "contains": "void"}]},
    "ignore": ["Comment"],
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
        cols += [self.filled(e).reindex(r["batch"]).to_numpy() for e in extras]
        X = torch.tensor(np.column_stack(cols).astype(float), dtype=torch.float64)
        return X, torch.tensor(r["strength"].to_numpy(dtype=float), dtype=torch.float64)

    def feature(self, name: str) -> pd.Series:
        """Numeric values of a candidate per batch (index = batch id)."""
        c = self.candidates[name]
        return as_feature(self.batches.set_index("batch")[name], c.kind)

    def filled(self, name: str) -> pd.Series:
        """The feature for model inputs: batches without a value get the plant's median."""
        v = self.feature(name)
        return v.fillna(v.median())

    @property
    def groups(self) -> np.ndarray:
        """Mix design of every row, for holding out whole designs."""
        return self.rows["recipe"].to_numpy()


class MappingError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def prepare(sheets: list[Sheet], mapping: dict) -> tuple[list[Sheet], dict]:
    """Two other layouts plants keep, reshaped into what build() reads:

    - one table with everything and no mix ID (no recipes.key): rows with the same
      ingredients are the same mix;
    - one column per age (tests.age_columns): one row per result.
    """
    mapping = copy.deepcopy(mapping)
    tables = {s.file: s.df for s in sheets}
    recipes, tests = mapping.get("recipes") or {}, mapping.get("tests") or {}

    if recipes and "key" not in recipes:
        if recipes.get("file") != tests.get("file"):
            raise MappingError(["recipes.key is required unless one table holds both the recipes and the tests"])
        if (df := tables.get(recipes.get("file"))) is not None:
            used = sorted(c for c in _columns_of(recipes) if c in df.columns)
            ids = df[used].apply(lambda row: "|".join(map(str, row)), axis=1)
            tables[recipes["file"]] = df.assign(**{MIX_ID: [f"Mix {i + 1}" for i in pd.factorize(ids)[0]]})
        recipes["key"] = MIX_ID
        if "batch_key" not in tests:
            tests["recipe_key"] = MIX_ID

    if "age_columns" in tests:
        ages, file = tests.pop("age_columns"), tests.get("file")
        if (df := tables.get(file)) is not None:
            if missing := [c for c in ages if c not in df.columns]:
                raise MappingError([f"tests.age_columns: {missing} not in {file}; columns are {list(df.columns)}"])
            long = df.melt(id_vars=[c for c in df.columns if c not in ages], value_vars=list(ages),
                           var_name="_column", value_name=STRENGTH_FROM_HEADERS)
            long[AGE_FROM_HEADERS] = long.pop("_column").map(lambda c: float(ages[c]))
            tables[file] = long.dropna(subset=[STRENGTH_FROM_HEADERS]).reset_index(drop=True)
        unit = tests.get("fields", {}).get("strength", {}).get("unit", "")
        tests.setdefault("fields", {}).update(age=dict(column=AGE_FROM_HEADERS, unit="days"),
                                              strength=dict(column=STRENGTH_FROM_HEADERS, unit=unit))
    return [replace(s, df=tables[s.file]) for s in sheets], mapping


def build(sheets: list[Sheet], mapping: dict) -> tuple[Dataset, dict]:
    """Apply the mapping. Returns the dataset and a report; raises MappingError."""
    errors: list[str] = []
    report: dict = dict(conversions=[], assumptions=[], cleaned_cells={}, failed_cells={}, dropped={})
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
    if T is not None:
        for rule in mapping["tests"].get("exclude", []):
            if rule.get("column") not in T.columns:
                errors.append(f"tests.exclude: column '{rule.get('column')}' not in {mapping['tests']['file']}")
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

    # a table that also holds another section's columns must not offer them as new inputs
    taken: dict[str, set[str]] = {}
    for section in ("recipes", "batches", "tests"):
        if spec := mapping.get(section):
            taken.setdefault(spec["file"], set()).update(
                _columns_of(spec), (r["column"] for r in spec.get("exclude", [])))

    # --- recipes ---
    rs = mapping["recipes"]
    R = _one_row_each(R, rs["key"], "mix", report, repeats_expected=rs["file"] == mapping["tests"]["file"])
    recipes = pd.DataFrame({"recipe": _key(R[rs["key"]])})
    if recipes["recipe"].map(_norm).nunique() < recipes["recipe"].nunique():
        raise MappingError(["recipes: different mix IDs become the same when case and punctuation are ignored"])
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
    for f in ("fly_ash", "slag", "hrwr"):  # a blank here means the mix doesn't use it
        if blank := int(recipes[f].isna().sum()):
            recipes[f] = recipes[f].fillna(0.0)
            report["assumptions"].append(f"{blank} blank {f.replace('_', ' ')} cells read as 0")
    complete = recipes[["cement", "water", "fine", "coarse"]].notna().all(axis=1).to_numpy()
    if not complete.all():
        report["recipes_left_out"] = recipes.loc[~complete, "recipe"].tolist()
        report["assumptions"].append(f"left out {int((~complete).sum())} mixes with a blank main ingredient: "
                                     f"{report['recipes_left_out'][:8]}")
        recipes, R = recipes[complete].reset_index(drop=True), R[complete].reset_index(drop=True)

    # --- batches ---
    if B is not None:
        bs = mapping["batches"]
        B = _one_row_each(B, bs["key"], "batch", report, repeats_expected=bs["file"] == mapping["tests"]["file"])
        batches = pd.DataFrame({"batch": _key(B[bs["key"]]),
                                "recipe": _match(_key(B[bs["recipe_key"]]), recipes["recipe"], report)})
        batches = batches.join(field_values(B, "batches"))
        used_b = {bs["key"], bs["recipe_key"]} | taken[bs["file"]]
        level = {c: "batch" for c in B.columns if c not in used_b and c not in ignore}
        for c in level:
            batches[c] = B[c].to_numpy()
        sources = {c: bs["file"] for c in level}
        if "batch_key" not in mapping["tests"]:  # cylinders only name the mix: no batch to attach them to
            batches = _collapse_to_recipes(batches)
            level = {c: "recipe" for c in level}
            report["assumptions"].append("cylinders name only the mix, so QC rows are averaged per mix")
    else:
        sources = {}
        batches = pd.DataFrame({"batch": recipes["recipe"], "recipe": recipes["recipe"]})
        level = {}

    # recipe-level candidates are carried onto each batch (other cementitious
    # materials such as silica fume stay candidates: BOxCrete has no slot for them)
    other_cm = {e["column"] for e in rs.get("other_cementitious", [])}
    used_r = {rs["key"]} | (taken[rs["file"]] - other_cm)
    rc = pd.DataFrame({"recipe": recipes["recipe"]})
    for c in R.columns:
        if c not in used_r and c not in ignore:
            name = c if c not in level else f"{c} (recipe)"
            rc[name] = to_number(R[c])[0].fillna(0.0).to_numpy() if c in other_cm else R[c].to_numpy()
            level[name] = "recipe"
            sources[name] = rs["file"]
    batches = batches.merge(rc, on="recipe", how="left")

    # --- tests ---
    ts = mapping["tests"]
    tests = field_values(T, "tests")
    raw = T[ts["batch_key"] if "batch_key" in ts else ts["recipe_key"]]
    tests["batch"] = _match(_key(raw), batches["batch"], report)

    matched = tests["batch"].isin(batches["batch"])
    report["tests_matched"] = f"{int(matched.sum())}/{len(tests)}"
    b_matched = batches["recipe"].isin(recipes["recipe"])
    report["batches_matched"] = f"{int(b_matched.sum())}/{len(batches)}"
    if matched.mean() < 0.5 or b_matched.mean() < 0.5:
        raise MappingError([f"joins look wrong: tests matched to batches {report['tests_matched']}, "
                            f"batches matched to recipes {report['batches_matched']}"])
    keep = matched & tests["strength"].notna() & tests["age"].notna()
    _note_dropped(report, "no readable age or strength", int((matched & ~keep).sum()))
    for rule in ts.get("exclude", []):
        hit = T[rule["column"]].fillna("").astype(str).str.contains(rule["contains"], case=False, regex=False)
        _note_dropped(report, f"{rule['column']} contains '{rule['contains']}'", int((keep & hit).sum()))
        keep &= ~hit
    late = tests["age"] > MAX_AGE_DAYS
    _note_dropped(report, f"older than {MAX_AGE_DAYS} days", int((keep & late).sum()))
    tests = tests[keep & ~late]
    if tests.empty:
        raise MappingError([f"no usable tests left after cleaning: {report['dropped']}"])
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
        candidates[c] = Candidate(name=c, level=lvl, kind=kind, file=sources[c])

    ds = Dataset(rows=rows.sort_values(["recipe", "batch", "age"]).reset_index(drop=True),
                 batches=batches, cylinders=int(len(tests)), candidates=candidates)
    report.update(recipes=int(rows["recipe"].nunique()), batches=int(rows["batch"].nunique()),
                  rows=int(len(rows)), cylinders=ds.cylinders,
                  ages=sorted(float(a) for a in rows["age"].unique()))
    return ds, report


def check_enough(ds: Dataset) -> None:
    """Can this data be trained and scored at all? Raises MappingError if not."""
    if (n := int(ds.rows["recipe"].nunique())) < MIN_MIXES:
        raise MappingError([f"only {n} mix designs have usable tests; training needs at least {MIN_MIXES}, "
                            "because each mix is predicted by a model that never saw it"])
    if ds.rows["age"].max() < MIN_LAST_AGE:
        raise MappingError([f"the latest usable break is at {ds.rows['age'].max():g} days; the model is judged on "
                            f"28-day strength, so it needs breaks at {MIN_LAST_AGE:g} to {MAX_AGE_DAYS} days"])


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


def _one_row_each(df: pd.DataFrame, key: str, what: str, report: dict, repeats_expected: bool) -> pd.DataFrame:
    """One row per ID, the latest one. A table that holds the tests too repeats it on every row."""
    repeated = _key(df[key]).duplicated(keep="last").to_numpy()
    if repeated.any() and not repeats_expected:
        report["assumptions"].append(f"{int(repeated.sum())} rows repeat a {what} ID in {key}: kept the last of each")
    return df[~repeated].reset_index(drop=True)


def _norm(v) -> str:
    return re.sub(r"[^0-9A-Z]", "", str(v).upper())


def _match(keys: pd.Series, known: pd.Series, report: dict) -> pd.Series:
    """Spell IDs the way `known` does: "r 101", "R-101" and "R101" are the same mix."""
    canon = {_norm(k): k for k in known}
    out = keys.map(lambda k: canon.get(_norm(k), k))
    report["ids_matched_loosely"] = report.get("ids_matched_loosely", 0) + int((out != keys).sum())
    return out


def _collapse_to_recipes(batches: pd.DataFrame) -> pd.DataFrame:
    """One row per recipe: numbers averaged, everything else the first value."""
    out = {}
    for c in batches.columns.drop(["batch", "recipe"]):
        x = to_number(batches[c])[0] if column_kind(batches[c]) == "numeric" else batches[c]
        grouped = x.groupby(batches["recipe"])
        out[c] = grouped.mean() if column_kind(batches[c]) == "numeric" else grouped.first()
    collapsed = pd.DataFrame(out).reset_index()
    return collapsed.assign(batch=collapsed["recipe"])


def _note_dropped(report: dict, why: str, n: int) -> None:
    if n:
        report["dropped"][why] = n


def _note_conversion(report: dict, f: str, spec: dict) -> None:
    unit = units.normalize_unit(spec["unit"])
    target = {"cure_temp": "°C", "strength": "psi", "age": "days"}.get(f, "kg/m3")
    entry = f"{spec['column']}: {spec['unit']} -> {target}"
    if entry not in report["conversions"]:
        report["conversions"].append(entry)
        if unit in units.ASSUMPTIONS:
            report["assumptions"].append(f"{spec['column']}: {units.ASSUMPTIONS[unit]}")
