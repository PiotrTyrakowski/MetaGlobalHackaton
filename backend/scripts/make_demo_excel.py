"""Turn the synthetic plant (data/plant_truth.csv) into three messy Excel files.

Each file looks like something a US ready-mix plant keeps: imperial units,
its own column names, a title row, inconsistent mix IDs, one row per
cylinder, a few failed cylinders, breaks past 28 days, and a QC log where
water added on site is only filled in since the plant started logging it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixpilot.units import (  # noqa: E402
    LB_PER_YD3_TO_KG_PER_M3,
    hrwr_kg_m3_to_oz_cwt,
)

ROOT = Path(__file__).resolve().parents[1]
truth = pd.read_csv(ROOT / "data" / "plant_truth.csv")
out_dir = ROOT / "data" / "demo"
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

mixes = truth.drop_duplicates("Mix Name").reset_index(drop=True)
lb = lambda kg: kg / LB_PER_YD3_TO_KG_PER_M3  # noqa: E731


def describe(r) -> str:
    binder = r["Cement (kg/m3)"] + r["Slag (kg/m3)"] + r["Fly Ash (kg/m3)"]
    scm = (r["Slag (kg/m3)"] + r["Fly Ash (kg/m3)"]) / binder
    kind = "air-entrained exterior" if r["Air (%)"] > 4 else "interior"
    return f"{kind}, {scm:.0%} SCM" if scm > 0.01 else f"{kind}, straight cement"


# --- mix_designs.xlsx: one row per mix, lb/yd3, HRWR in oz/cwt, title row ---
binder = mixes["Cement (kg/m3)"] + mixes["Slag (kg/m3)"] + mixes["Fly Ash (kg/m3)"]
designs = pd.DataFrame(
    {
        "Mix #": "R-" + mixes["Mix Name"].str[1:],
        "Description": mixes.apply(describe, axis=1),
        "Cement (lb/yd³)": lb(mixes["Cement (kg/m3)"]).round(0),
        "Slag (lb/yd³)": lb(mixes["Slag (kg/m3)"]).round(0),
        "Fly Ash F (lb/yd³)": lb(mixes["Fly Ash (kg/m3)"]).round(0),
        "Water (lb/yd³)": lb(mixes["Water (kg/m3)"]).round(0),
        "HRWR (oz/cwt)": hrwr_kg_m3_to_oz_cwt(mixes["HRWR (kg/m3)"], binder).round(1),
        "Sand SSD (lb/yd³)": lb(mixes["Fine Aggregate (kg/m3)"]).round(0),
        "#57 Stone (lb/yd³)": lb(mixes["Coarse Aggregates (kg/m3)"]).round(0),
    }
)
with pd.ExcelWriter(out_dir / "mix_designs.xlsx") as xw:
    pd.DataFrame([["Approved mix designs (2025)"], [""]]).to_excel(
        xw, sheet_name="Mix Designs", header=False, index=False
    )
    designs.to_excel(xw, sheet_name="Mix Designs", startrow=2, index=False)

# --- cylinder_breaks.xlsx: one row per cylinder ---
rows = []
cast = pd.Timestamp("2025-01-06")
step = 600 // len(mixes)  # days between mixes: about 20 months of history
for i, r in truth.iterrows():
    cast_date = cast + pd.Timedelta(days=(int(r["Mix Name"][1:]) - 101) * step)
    for j, col in enumerate(["Strength1 (psi)", "Strength2 (psi)", "Strength3 (psi)"]):
        rows.append(
            {
                "Sample ID": f"{r['Mix Name']}-{int(r['Time'])}-{'ABC'[j]}",
                "Mix": r["Mix Name"],
                "Cast Date": cast_date.date(),
                "Age (d)": int(r["Time"]),
                "Strength (psi)": int(r[col]),
                "Break Type": int(rng.choice([1, 2, 3])),
                "Notes": "",
            }
        )
breaks = pd.DataFrame(rows)
# Failed cylinders: an extra low break flagged in the notes (must be excluded).
for idx in rng.choice(len(mixes), 4, replace=False):
    m = mixes.loc[idx, "Mix Name"]
    good = breaks[(breaks["Mix"] == m) & (breaks["Age (d)"] == 28)].iloc[0]
    breaks.loc[len(breaks)] = {
        **good.to_dict(),
        "Sample ID": good["Sample ID"][:-1] + "D",
        "Strength (psi)": int(good["Strength (psi)"] * 0.55),
        "Break Type": 6,
        "Notes": "cap failure, retest",
    }
# A few 56-day breaks (beyond BOxCrete's 0-28 day range).
for m in mixes["Mix Name"].iloc[:9]:
    s28 = truth[(truth["Mix Name"] == m) & (truth["Time"] == 28)]["Strength (Mean)"].iloc[0]
    for j in "AB":
        breaks.loc[len(breaks)] = {
            "Sample ID": f"{m}-56-{j}", "Mix": m, "Cast Date": breaks[breaks["Mix"] == m]["Cast Date"].iloc[0],
            "Age (d)": 56, "Strength (psi)": int(s28 * rng.uniform(1.06, 1.12)), "Break Type": 2, "Notes": "",
        }
breaks = breaks.sort_values(["Cast Date", "Mix", "Age (d)", "Sample ID"]).reset_index(drop=True)
breaks.to_excel(out_dir / "cylinder_breaks.xlsx", sheet_name="Breaks", index=False)

# --- daily_QC.xlsx: one row per truck batch; per-mix values average to the truth ---
qc = []
ticket = 50211
for _, r in mixes.iterrows():
    n = int(rng.integers(1, 4))
    air_dev = rng.normal(0, 0.15, n)
    air_dev -= air_dev.mean()
    temp_dev = rng.normal(0, 0.5, n)
    temp_dev -= temp_dev.mean()
    date = cast + pd.Timedelta(days=(int(r["Mix Name"][1:]) - 101) * step)
    raw_id = r["Mix Name"][1:]
    mix_id = rng.choice([f"R{raw_id}", f"r-{raw_id}", f"R {raw_id}"])
    has_water = not pd.isna(r["Site Water (gal/yd3)"])
    water_dev = rng.normal(0, 0.2, n)
    water_dev -= water_dev.mean()
    for k in range(n):
        qc.append(
            {
                "Date": date.date(),
                "Ticket #": ticket,
                "Mix ID": mix_id,
                "Slump (in)": round(float(rng.uniform(3.5, 6.0)), 1),
                "Air (%)": round(float(r["Air (%)"] + air_dev[k]), 2),
                "Conc Temp (°F)": round(float(rng.uniform(62, 78))),
                "Cure Temp (°F)": round(float(r["Temp (C)"] * 9 / 5 + 32 + temp_dev[k]), 2),
                "Water Added on Site (gal/yd³)": round(float(r["Site Water (gal/yd3)"] * (1 + water_dev[k])), 2) if has_water else None,
            }
        )
        ticket += int(rng.integers(3, 40))
pd.DataFrame(qc).to_excel(out_dir / "daily_QC.xlsx", sheet_name="QC Log", index=False)

print(f"mix_designs: {len(designs)} rows | breaks: {len(breaks)} cylinders | QC: {len(qc)} batches -> {out_dir}")
