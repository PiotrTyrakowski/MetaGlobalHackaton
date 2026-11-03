"""Generate a synthetic ready-mix plant's lab history in BOxCrete's CSV format.

"True" strength = Meta's pretrained BOxCrete curve for the composition
x air penalty x site-water penalty, then three noisy cylinders per test
age. Air content is recorded for every mix; water added on site (by the
truck driver) for ~40% of them.
"""

import sys

import numpy as np
import pandas as pd
import torch

from boxcrete import load_pretrained_strength_gp

N_MIXES = 46
AGES = [3, 7, 28]
SOURCE = 2  # Meta's ready-mix-like material set
AIR_PENALTY = 0.05  # ~5% strength lost per 1% air above 2%
WATER_PENALTY = 0.06  # ~6% strength (~300 psi) lost per gal/yd3 of water added on site
CYL_CV = 0.035  # cylinder-to-cylinder scatter

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)

rows = []
for i in range(N_MIXES):
    binder = rng.uniform(300, 430)
    scm = rng.choice([0.0, rng.uniform(0.15, 0.5)], p=[0.3, 0.7])
    slag_share = rng.uniform(0.3, 1.0)
    cement = binder * (1 - scm)
    slag = binder * scm * slag_share
    fly_ash = binder * scm * (1 - slag_share)
    wb = rng.uniform(0.38, 0.52)
    hrwr = max(0.0, rng.normal(2.0 * (0.52 - wb) / 0.14, 0.3))
    air = rng.uniform(4.5, 7.0) if rng.random() < 0.45 else rng.uniform(1.2, 2.8)
    rows.append(
        dict(
            mix=f"R{101 + i}",
            cement=cement, fly_ash=fly_ash, slag=slag, water=wb * binder, hrwr=hrwr,
            fine=rng.uniform(740, 900), coarse=rng.uniform(1050, 1150),
            temp=rng.normal(22.5, 0.8), air=air, site_water=max(0.0, rng.uniform(-0.5, 3.0)),
            has_water=rng.random() < 0.4,
        )
    )

model = load_pretrained_strength_gp("model/strength_model.pt")
model.eval()

out = []
for r in rows:
    X = torch.tensor(
        [[r["cement"], r["fly_ash"], r["slag"], r["water"], r["hrwr"], r["fine"], r["coarse"], SOURCE, r["temp"], t] for t in AGES],
        dtype=torch.float64,
    )
    with torch.no_grad():
        meta = model.posterior(X).mean.squeeze(-1).numpy() * float(model._study_y_std)
    factor = (1 - AIR_PENALTY * (r["air"] - 2.0)) * (1 - WATER_PENALTY * r["site_water"])
    for t, s in zip(AGES, meta * factor):
        cyl = s * (1 + CYL_CV * rng.standard_normal(3))
        out.append(
            {
                "Mix Name": r["mix"],
                "Material Source": SOURCE,
                "Cement (kg/m3)": round(r["cement"], 1),
                "Fly Ash (kg/m3)": round(r["fly_ash"], 1),
                "Slag (kg/m3)": round(r["slag"], 1),
                "Water (kg/m3)": round(r["water"], 1),
                "HRWR (kg/m3)": round(r["hrwr"], 2),
                "Fine Aggregate (kg/m3)": round(r["fine"], 1),
                "Coarse Aggregates (kg/m3)": round(r["coarse"], 1),
                "Temp (C)": round(r["temp"], 1),
                "Air (%)": round(r["air"], 1),
                "Site Water (gal/yd3)": round(r["site_water"], 1) if r["has_water"] else np.nan,
                "Time": t,
                "GWP": round(0.9 * r["cement"] + 0.08 * r["slag"] + 0.03 * r["fly_ash"] + 20, 1),
                "Strength1 (psi)": round(cyl[0]),
                "Strength2 (psi)": round(cyl[1]),
                "Strength3 (psi)": round(cyl[2]),
                "Strength (Mean)": round(cyl.mean()),
                "Strength (Std)": round(cyl.std(ddof=1)),
                "# of measurements": 3,
            }
        )

df = pd.DataFrame(out)
df.to_csv(sys.argv[2] if len(sys.argv) > 2 else "data/plant_truth.csv", index=False)
d28 = df[df.Time == 28]
print(f"rows={len(df)} mixes={df['Mix Name'].nunique()} 28d psi: min={d28['Strength (Mean)'].min()} median={d28['Strength (Mean)'].median()} max={d28['Strength (Mean)'].max()}")
print(f"air: {df.drop_duplicates('Mix Name')['Air (%)'].describe().round(1).to_dict()}")
print(f"site water recorded: {df.drop_duplicates('Mix Name')['Site Water (gal/yd3)'].notna().mean():.0%}")
