"""Read a plant's Excel files and describe them compactly for the agent."""

from pathlib import Path

import pandas as pd


def _header_row(raw: pd.DataFrame, scan: int = 15) -> int:
    """First row that looks like column names: mostly text, as wide as the table."""
    width = raw.notna().sum(axis=1).max()
    for i in range(min(scan, len(raw))):
        cells = raw.iloc[i].dropna()
        if len(cells) >= max(2, 0.6 * width) and all(isinstance(c, str) for c in cells):
            return i
    return 0


def read_sheet(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    h = _header_row(raw)
    df = raw.iloc[h + 1 :].copy()
    df.columns = [str(c).strip() for c in raw.iloc[h]]
    df = df.dropna(how="all").reset_index(drop=True)
    return df.infer_objects()


def read_file(path: str | Path) -> dict[str, pd.DataFrame]:
    """All non-empty sheets of a workbook (or the single table of a CSV)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return {"csv": pd.read_csv(path)}
    sheets = pd.ExcelFile(path).sheet_names
    out = {s: read_sheet(path, s) for s in sheets}
    return {s: df for s, df in out.items() if len(df)}


def profile_column(s: pd.Series) -> dict:
    filled = float(s.notna().mean())
    values = s.dropna()
    if pd.api.types.is_numeric_dtype(s):
        kind = "number"
    elif pd.api.types.is_datetime64_any_dtype(s):
        kind = "date"
    else:
        kind = "text"
    info = {"name": s.name, "type": kind, "filled": round(filled, 2), "distinct": int(values.nunique())}
    if kind == "number" and len(values):
        info |= {"min": round(float(values.min()), 3), "max": round(float(values.max()), 3)}
    info["examples"] = [str(v)[:40] for v in values.drop_duplicates().head(3)]
    return info


def profile_file(path: str | Path) -> dict:
    """What the agent sees about a file: sheets, row counts, column summaries."""
    path = Path(path)
    return {
        "file": path.name,
        "sheets": [
            {"sheet": name, "rows": len(df), "columns": [profile_column(df[c]) for c in df.columns]}
            for name, df in read_file(path).items()
        ],
    }
