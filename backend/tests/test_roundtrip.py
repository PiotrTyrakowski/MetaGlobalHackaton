"""The demo Excel files, prepared with the expected mapping, must give back the synthetic truth."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mixpilot.prepare import build_dataset

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
FILES = [DEMO / f for f in ("mix_designs.xlsx", "cylinder_breaks.xlsx", "daily_QC.xlsx")]


def build():
    mapping = json.loads((ROOT / "tests" / "demo_mapping.json").read_text())
    return build_dataset(FILES, mapping)


def test_roundtrip_matches_truth():
    df, report = build()
    truth = pd.read_csv(ROOT / "data" / "plant_truth.csv")
    m = df.merge(truth, on=["Mix Name", "Time"], suffixes=("", "_true"))
    assert len(m) == len(truth) == len(df)
    tol = {
        "Cement (kg/m3)": 0.5, "Fly Ash (kg/m3)": 0.5, "Slag (kg/m3)": 0.5, "Water (kg/m3)": 0.5,
        "HRWR (kg/m3)": 0.05, "Fine Aggregate (kg/m3)": 0.5, "Coarse Aggregates (kg/m3)": 0.5,
        "Temp (C)": 0.05, "Air (%)": 0.05, "Site Water (gal/yd3)": 0.06, "Strength (Mean)": 1, "Strength (Std)": 1,
    }
    for col, t in tol.items():
        err = (m[col] - m[f"{col}_true"]).abs()
        assert np.nanmax(err) <= t, (col, float(np.nanmax(err)))
    assert report["dropped"]["Notes contains 'cap failure'"] == 4
    assert report["dropped"]["older than 28 days"] == 18
    assert report["ages"] == [3, 7, 28]
