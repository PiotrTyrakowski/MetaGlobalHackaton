"""Read messy plant spreadsheets: find the header row, profile every column."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

MIN_KEY_VALUES = 5
_NUMBER = re.compile(r"[-+]?\d*\.?\d+")
_HHMM = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*$")
# a number, optionally followed by a short unit: "5.8%", "7 day", "1,250 psi" (not "T-24", "7310001-A")
_NUMBER_WITH_UNIT = re.compile(r"^\s*[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|[A-Za-z°/³]{1,6}\.?)?\s*$")


@dataclass
class Sheet:
    file: str
    sheet: str
    header_row: int  # 0-based row index in the spreadsheet
    df: pd.DataFrame


def read_sheets(path: Path) -> list[Sheet]:
    """Every non-empty sheet of a workbook (or a CSV), with its header row detected."""
    if path.suffix.lower() == ".csv":
        raw = {"csv": pd.read_csv(path, header=None, dtype=object)}
    else:
        raw = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    sheets = []
    for name, grid in raw.items():
        grid = grid.dropna(how="all").dropna(axis=1, how="all")
        if grid.empty:
            continue
        h = _header_row(grid)
        header = [_clean_header(v, i) for i, v in enumerate(grid.loc[h])]
        body = grid.loc[grid.index > h].copy()
        body.columns = header
        body = body.dropna(how="all").reset_index(drop=True)
        body = body.loc[:, [c for c in body.columns if body[c].notna().any() or not c.startswith("col_")]]
        sheets.append(Sheet(file=path.name, sheet=str(name), header_row=int(h), df=body))
    return sheets


def _header_row(grid: pd.DataFrame) -> int:
    """First row (of the top 15) that is wide and all text: title lines above it are skipped."""
    top = grid.head(15)
    counts = top.notna().sum(axis=1)
    widest = counts.max()
    for idx, row in top.iterrows():
        vals = row.dropna()
        if len(vals) >= max(2, 0.5 * widest) and all(isinstance(v, str) for v in vals):
            return idx
    return int(top.index[0])


def _clean_header(v, i: int) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"col_{i + 1}"
    return re.sub(r"\s+", " ", str(v)).strip()


# --- Value coercion ---------------------------------------------------------------

def to_number(s: pd.Series) -> tuple[pd.Series, int, int]:
    """Parse numbers out of messy cells ("5.8%", "7 day", "1,250").

    Returns (numbers, cells that needed text cleanup, cells that failed).
    """
    out = pd.Series(np.nan, index=s.index, dtype=float)
    cleaned = failed = 0
    for i, v in s.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[i] = float(v)
            continue
        m = _NUMBER.search(str(v).replace(",", ""))
        if m:
            out[i] = float(m.group())
            cleaned += 1
        else:
            failed += 1
    return out, cleaned, failed


def column_kind(s: pd.Series) -> str:
    """numeric | date | time_of_day | category | text | empty."""
    vals = s.dropna()
    if vals.empty:
        return "empty"
    if all(isinstance(v, (datetime, date, pd.Timestamp)) for v in vals):
        return "date"
    if all(isinstance(v, time) or (isinstance(v, str) and _HHMM.match(v)) for v in vals):
        return "time_of_day"
    nums, _, failed = to_number(vals)
    if failed == 0 and _mostly_numeric_text(vals):
        return "numeric"
    n_unique = vals.astype(str).nunique()
    if n_unique <= max(20, len(vals) // 2):
        return "category"
    return "text"


def _mostly_numeric_text(vals: pd.Series) -> bool:
    """Numbers, or strings that are a number plus a short unit ("5.8%", "7 day")."""
    return all(isinstance(v, (int, float, np.integer, np.floating)) or _NUMBER_WITH_UNIT.match(str(v))
               for v in vals)


def as_feature(s: pd.Series, kind: str) -> pd.Series:
    """Numeric version of a column for statistics (dates -> days, HH:MM -> hours)."""
    if kind == "numeric":
        return to_number(s)[0]
    if kind == "date":
        d = pd.to_datetime(s, errors="coerce")
        return (d - d.min()).dt.days.astype(float)
    if kind == "time_of_day":
        def hours(v):
            if isinstance(v, time):
                return v.hour + v.minute / 60
            m = _HHMM.match(str(v)) if isinstance(v, str) else None
            return int(m.group(1)) + int(m.group(2)) / 60 if m else np.nan
        return s.map(hours).astype(float)
    return s.astype("string")


def profile(sheet: Sheet, n_samples: int = 4) -> dict:
    """What the agent sees of a sheet: columns, kinds, coverage, sample values."""
    cols = []
    n = len(sheet.df)
    for c in sheet.df.columns:
        s = sheet.df[c]
        vals = s.dropna()
        kind = column_kind(s)
        samples = []
        for v in vals.drop_duplicates().head(n_samples):
            samples.append(v.isoformat()[:10] if isinstance(v, (datetime, date, pd.Timestamp)) else
                           v.strftime("%H:%M") if isinstance(v, time) else
                           v.item() if isinstance(v, np.generic) else v)
        col = dict(name=c, kind=kind, filled=int(len(vals)), coverage=round(len(vals) / n, 3) if n else 0,
                   unique=int(vals.astype(str).nunique()), samples=samples)
        if kind == "numeric":
            nums = to_number(vals)[0]
            col.update(min=_r(nums.min()), max=_r(nums.max()))
        cols.append(col)
    return dict(file=sheet.file, sheet=sheet.sheet, header_row=sheet.header_row + 1, rows=n, columns=cols)


def _r(x: float) -> float:
    return float(f"{x:.4g}")


def _loose_id(v) -> str:
    """Compare IDs ignoring case and punctuation ("r 101" = "R-101"), and 7310002.0 = 7310002."""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return re.sub(r"[^0-9A-Z]", "", str(v).upper())


def join_keys(sheets: list[Sheet], min_overlap: float = 0.6) -> list[dict]:
    """Column pairs in different files whose values overlap: likely join keys."""
    keys = []
    for i, a in enumerate(sheets):
        for b in sheets[i + 1:]:
            if a.file == b.file:
                continue
            for ca in a.df.columns:
                if column_kind(a.df[ca]) in ("date", "time_of_day"):
                    continue
                va = set(a.df[ca].dropna().map(_loose_id))
                if len(va) < MIN_KEY_VALUES:
                    continue
                for cb in b.df.columns:
                    if column_kind(b.df[cb]) in ("date", "time_of_day"):
                        continue
                    vb = set(b.df[cb].dropna().map(_loose_id))
                    if len(vb) < MIN_KEY_VALUES:
                        continue
                    shared = len(va & vb)
                    overlap = shared / min(len(va), len(vb))
                    if overlap >= min_overlap and shared >= 3:
                        keys.append(dict(a=f"{a.file}:{ca}", b=f"{b.file}:{cb}",
                                         overlap=round(overlap, 3), shared_values=shared))
    return sorted(keys, key=lambda k: -k["overlap"])
