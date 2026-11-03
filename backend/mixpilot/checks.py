"""Data checks and candidate extra inputs, computed for the agent to explain."""

from pathlib import Path

import pandas as pd

from mixpilot.ingest import read_file
from mixpilot.prepare import normalize_mix_id
from mixpilot.schema import BASE_INPUTS, COMPOSITION, META_RANGES


def coverage(df: pd.DataFrame, extra_inputs: list[str] = ()) -> list[dict]:
    """For every model input: how many mixes have a value."""
    mixes = df.drop_duplicates("Mix Name")
    n = len(mixes)
    out = []
    for col in [c for c in BASE_INPUTS if c != "Time"] + list(extra_inputs):
        have = int(mixes[col].notna().sum()) if col in mixes else 0
        out.append({"input": col, "mixes_with_value": have, "mixes": n, "extra": col in extra_inputs})
    return out


def issues(df: pd.DataFrame, extra_inputs: list[str] = ()) -> list[dict]:
    """Problems worth telling the plant about, most severe first."""
    found = []
    mixes = df.drop_duplicates("Mix Name")
    n = len(mixes)

    for c in coverage(df, extra_inputs):
        missing = c["mixes"] - c["mixes_with_value"]
        if missing and not c["extra"]:
            found.append({"severity": "blocking", "what": f"{c['input']} missing for {missing} of {n} mixes"})
        elif missing:
            found.append({"severity": "partial", "what": f"{c['input']} recorded for only {c['mixes_with_value']} of {n} mixes"})

    for col, (lo, hi) in META_RANGES.items():
        out = mixes[(mixes[col] < lo * 0.9) | (mixes[col] > hi * 1.1)]
        if len(out):
            found.append({"severity": "info", "what": f"{col} outside Meta's training range ({lo}-{hi}) for {len(out)} mixes"})

    ages = df.groupby("Mix Name")["Time"].nunique()
    if (ages < 2).any():
        found.append({"severity": "warning", "what": f"{int((ages < 2).sum())} mixes tested at only one age"})
    early = df[df["Time"] <= 3]["Mix Name"].nunique()
    if early < n * 0.5:
        found.append({"severity": "warning", "what": f"only {early} of {n} mixes have a break at 3 days or earlier"})

    scatter = df[(df["Strength (Std)"] / df["Strength (Mean)"]) > 0.10]
    if len(scatter):
        found.append({"severity": "warning", "what": f"{len(scatter)} test sets with cylinder scatter above 10%"})
    order = {"blocking": 0, "warning": 1, "partial": 2, "info": 3}
    return sorted(found, key=lambda i: order[i["severity"]])


def candidate_inputs(files: list[str | Path], mapping: dict) -> list[dict]:
    """Numeric columns the mapping doesn't use, summarised per mix.

    The agent decides which are real material/process properties (air
    content, cement mill-cert strength) and which are not inputs at all
    (ticket numbers, slump, which is an outcome, dates).
    """
    used = {(s["file"], s["column"]) for s in mapping["fields"].values()}
    used |= {(s["file"], s["column"]) for s in mapping.get("extra_inputs", {}).values()}
    used |= set(mapping["mix_id"].items())
    out = []
    for f in files:
        name = Path(f).name
        if name not in mapping["mix_id"]:
            continue
        t = next(iter(read_file(f).values()))
        ids = t[mapping["mix_id"][name]].map(normalize_mix_id)
        n_mixes = ids.nunique()
        for col in t.columns:
            if (name, col) in used or not pd.api.types.is_numeric_dtype(t[col]):
                continue
            per_mix = pd.to_numeric(t[col], errors="coerce").groupby(ids).mean()
            out.append({
                "file": name,
                "column": col,
                "mixes_with_value": int(per_mix.notna().sum()),
                "mixes": int(n_mixes),
                "min": round(float(per_mix.min()), 2),
                "max": round(float(per_mix.max()), 2),
                "looks_like_id": bool(t[col].nunique() == t[col].notna().sum() and len(t) > 10),
            })
    return out


__all__ = ["coverage", "issues", "candidate_inputs", "COMPOSITION"]
