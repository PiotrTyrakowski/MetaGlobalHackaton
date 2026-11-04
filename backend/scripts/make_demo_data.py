"""Generate the synthetic Riverside Ready-Mix demo files.

Writes three messy Excel files with US units to demo_data/plant/ and the hidden
ground truth to demo_data/truth.json (never shown to the agent).

The plant has 24 mix designs, each batched about 5 times. Batches of the same
recipe differ in measured air, cement lot and noise, which is the scatter
BOxCrete's 10 inputs can't explain.

"True" strength of a cylinder:
    Meta's pretrained BOxCrete prediction (composition, cure temp, age)
    x plant material factor
    x slower early strength gain than Meta's lab mixes (Type IL cement, slag):
          1 - (0.20 + 0.25 x slag share of binder) x exp(-age / 4 days)
          e.g. 3 days: -9% (no slag) to -14% (40% slag); 28 days: ~0%
    x air effect          (-5% per 1% air above 2%)
    x silica fume effect  (grows with age)
    x cement strength effect (mill certificate, per cement lot)
    x batch noise x cylinder noise

Air and silica fume are recorded for every batch. Mill certificates come per cement
lot, and 4 of the 13 lots have none (~70% of batches covered).
Truck, operator, ticket, batch time, slump and concrete temperature have no effect.

Mess the cleanup has to handle, drawn from its own random stream so the clean data
stays the same: mix codes retyped in the QC log, a few "cap failure, retest" cylinders
and some 56-day breaks (BOxCrete stops at 28 days).
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import date, datetime, time, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from boxcrete import load_pretrained_strength_gp
from boxcrete.utils import DEFAULT_X_COLUMNS
from openpyxl import Workbook
from openpyxl.styles import Font

SEED = 7
N_DESIGNS = 24
BATCHES_PER_DESIGN = (4, 6)  # inclusive range
SOURCE = 2  # Meta's ready-mix-like concrete class
AGES = {3: 2, 7: 2, 28: 3}  # age (days) -> cylinders broken per batch
OUT = Path(__file__).resolve().parent.parent / "demo_data"

PLANT_FACTOR = 0.95
AIR_REF = 2.0
AIR_LOSS_PER_PCT = 0.05
SF_GAIN_PER_KG = 0.006  # at late age, per kg/m3 of silica fume
SF_RAMP_DAYS = 5.0
CEMENT_REF_PSI = 6600.0
CEMENT_GAIN_PER_PSI = 1.2e-4
SLOW_GAIN_BASE = 0.20
SLOW_GAIN_PER_SLAG_SHARE = 0.25
SLOW_GAIN_DAYS = 4.0
BATCH_CV = 0.03
CYLINDER_CV = 0.04
N_CEMENT_LOTS = 13
MISSING_CERT_LOTS = 4  # ~70% of batches keep a certificate

KG_M3_TO_LB_YD3 = 1.68555
LB_PER_GAL_WATER = 8.34
HRWR_KG_PER_L = 1.08
L_PER_FL_OZ = 0.0295735
YD3_PER_M3 = 1.30795
CYL_AREA_IN2 = math.pi * 2.0**2  # 4 x 8 in cylinders

rng = np.random.default_rng(SEED)
mess_rng = np.random.default_rng(SEED + 1)

RETYPED_CODE_SHARE = 0.15
CAP_FAILURES = 6
LATE_BREAK_BATCHES = 9
LATE_AGE = 56
LATE_GAIN = 1.08  # strength at 56 days vs 28 days


def f_to_c(f):
    return (f - 32.0) * 5.0 / 9.0


def make_designs() -> list[dict]:
    """24 plausible ready-mix designs in kg/m3."""
    classes = [3000, 4000, 5000, 6000]
    binder_by_class = {3000: 270, 4000: 310, 5000: 350, 6000: 395}
    wb_by_class = {3000: 0.54, 4000: 0.47, 5000: 0.42, 6000: 0.37}
    scm_types = ["none", "fa", "slag", "fa+slag"]
    mixes = []
    counters: dict[str, int] = {}
    for i in range(N_DESIGNS):
        fc = classes[i % 4]
        scm = scm_types[(i // 4) % 4] if i < 16 else str(rng.choice(scm_types))
        air_entrained = i % 2 == 1 if i < 16 else bool(rng.random() < 0.5)
        uses_sf = fc >= 5000 and rng.random() < 0.7 or fc == 4000 and rng.random() < 0.25

        binder = binder_by_class[fc] * rng.uniform(0.94, 1.06)
        fa_frac = {"none": 0, "fa": rng.uniform(0.15, 0.25), "slag": 0,
                   "fa+slag": rng.uniform(0.10, 0.15)}[scm]
        sl_frac = {"none": 0, "fa": 0, "slag": rng.uniform(0.25, 0.45),
                   "fa+slag": rng.uniform(0.20, 0.30)}[scm]
        sf = rng.uniform(15, 35) if uses_sf else 0.0
        fly_ash = binder * fa_frac
        slag = binder * sl_frac
        cement = binder - fly_ash - slag
        water = (binder + sf) * (wb_by_class[fc] + rng.uniform(-0.02, 0.02))
        wb = water / (binder + sf)
        hrwr_oz_cwt = 0.0 if wb > 0.5 else float(np.clip((0.52 - wb) * 60 + rng.normal(0, 1), 3, 14))

        design_air = 6.0 if air_entrained else 1.5
        paste_vol = (cement / 3150 + fly_ash / 2300 + slag / 2900 + sf / 2200
                     + water / 1000 + design_air / 100)
        agg_vol = 1.0 - paste_vol
        coarse_frac = rng.uniform(0.56, 0.60)
        coarse = agg_vol * coarse_frac * 2700
        fine = agg_vol * (1 - coarse_frac) * 2650

        cm_lb = (cement + fly_ash + slag + sf) * KG_M3_TO_LB_YD3
        hrwr_kg_m3 = hrwr_oz_cwt * cm_lb / 100 * L_PER_FL_OZ * HRWR_KG_PER_L * YD3_PER_M3

        tag = {"none": "PC", "fa": "FA", "slag": "SL", "fa+slag": "TR"}[scm]
        tag += "SF" if uses_sf else ""
        tag += "-AE" if air_entrained else ""
        base = f"{fc // 1000}K-{tag}"
        counters[base] = counters.get(base, 0) + 1
        code = f"R{base}-{counters[base]}"

        mixes.append(dict(
            code=code, fc=fc, air_entrained=air_entrained,
            cement=cement, fly_ash=fly_ash, slag=slag, sf=sf, water=water,
            hrwr_kg_m3=hrwr_kg_m3, hrwr_oz_cwt=hrwr_oz_cwt, fine=fine, coarse=coarse,
        ))
    return mixes


def make_batches(designs: list[dict]) -> tuple[list[dict], list[dict]]:
    """QC batches of each design: date, ticket, air, cement lot, distractors."""
    start = date(2026, 5, 4)
    lots = [dict(lot=f"CL-26{k:03d}", psi=float(rng.normal(CEMENT_REF_PSI, 300)))
            for k in range(1, N_CEMENT_LOTS + 1)]
    missing = set(rng.choice(len(lots), MISSING_CERT_LOTS, replace=False).tolist())
    for k, lot in enumerate(lots):
        lot["recorded"] = k not in missing
    operators = ["J. Alvarez", "K. Brooks", "M. Chen", "D. Okafor"]

    design_of_batch = [d for d in range(len(designs))
                       for _ in range(int(rng.integers(BATCHES_PER_DESIGN[0], BATCHES_PER_DESIGN[1] + 1)))]
    design_of_batch = [design_of_batch[i] for i in rng.permutation(len(design_of_batch))]
    n = len(design_of_batch)
    batches = []
    for rank, d in enumerate(design_of_batch):
        m = dict(designs[d])
        day = start + timedelta(days=int(rank * 1.0 + rng.integers(0, 2)))
        lot = lots[min(rank * len(lots) // n, len(lots) - 1)]
        if m["air_entrained"]:
            air = float(np.clip(rng.normal(6.0, 1.1), 3.5, 9.0))
        else:
            air = float(np.clip(rng.normal(2.2, 0.6), 1.0, 3.5))
        m.update(
            air=round(air, 1),
            batch_date=day,
            batch_time=time(int(rng.integers(5, 15)), int(rng.integers(0, 60))),
            ticket=7310000 + rank * 7 + int(rng.integers(0, 7)),
            truck=f"T-{int(rng.integers(3, 29)):02d}",
            operator=str(rng.choice(operators)),
            slump_in=round(float(np.clip(
                (2.5 if m["hrwr_oz_cwt"] == 0 else 5.5) + rng.normal(0, 0.8), 1, 9)) * 4) / 4,
            conc_temp_f=round(float(rng.normal(74, 5))),
            cure_temp_f=round(float(rng.normal(73.5, 1.2)), 1),
            cement_lot=lot["lot"],
            mill_cert_psi=lot["psi"],
            mill_cert_recorded=lot["recorded"],
            batch_noise=float(rng.normal(1.0, BATCH_CV)),
        )
        batches.append(m)
    return batches, lots


def meta_prediction(mixes: list[dict]) -> np.ndarray:
    """Meta's pretrained BOxCrete mean strength (psi), shape [mixes, ages]."""
    model = load_pretrained_strength_gp()
    rows = []
    for m in mixes:
        for age in AGES:
            row = {
                "Cement (kg/m3)": m["cement"], "Fly Ash (kg/m3)": m["fly_ash"],
                "Slag (kg/m3)": m["slag"], "Water (kg/m3)": m["water"],
                "HRWR (kg/m3)": m["hrwr_kg_m3"], "Fine Aggregate (kg/m3)": m["fine"],
                "Coarse Aggregates (kg/m3)": m["coarse"], "Material Source": SOURCE,
                "Temp (C)": f_to_c(m["cure_temp_f"]), "Time": age,
            }
            rows.append([row[c] for c in DEFAULT_X_COLUMNS])
    X = torch.tensor(rows, dtype=torch.float64)
    with torch.no_grad():
        mean = model.posterior(X).mean.squeeze(-1) * model._study_y_std
    return mean.numpy().reshape(len(mixes), len(AGES))


def true_factor(m: dict, age: float) -> dict:
    air = 1.0 - AIR_LOSS_PER_PCT * (m["air"] - AIR_REF)
    sf = 1.0 + SF_GAIN_PER_KG * m["sf"] * (1.0 - math.exp(-age / SF_RAMP_DAYS))
    cement = 1.0 + CEMENT_GAIN_PER_PSI * (m["mill_cert_psi"] - CEMENT_REF_PSI)
    slag_share = m["slag"] / (m["cement"] + m["fly_ash"] + m["slag"] + m["sf"])
    slow = 1.0 - (SLOW_GAIN_BASE + SLOW_GAIN_PER_SLAG_SHARE * slag_share) * math.exp(-age / SLOW_GAIN_DAYS)
    return dict(air=air, sf=sf, cement=cement, slow_gain=slow,
                total=PLANT_FACTOR * slow * air * sf * cement * m["batch_noise"])


def retype_code(code: str) -> str:
    """How a code gets typed by hand: R4K-SL-1 -> r4k-sl-1 or R4K SL 1."""
    return code.lower() if mess_rng.random() < 0.5 else code.replace("-", " ")


def messy_cylinders(cyl_rows: list[dict]) -> list[dict]:
    """Failed cylinders and late breaks; both must be left out of training."""
    extra = []
    for i in mess_rng.choice(len(cyl_rows), CAP_FAILURES, replace=False):
        psi = round(float(cyl_rows[i]["Strength (psi)"]) * mess_rng.uniform(0.45, 0.7), -1)
        extra.append(cyl_rows[i] | {"Set": f"{cyl_rows[i]['Ticket #']}-X{len(extra)}", "Strength (psi)": psi,
                                    "Max Load (lbf)": round(psi * CYL_AREA_IN2, -1),
                                    "Tech Notes": "cap failure, retest"})
    last = [r for r in cyl_rows if r["Age"] == "28 day"]
    tickets = sorted({r["Ticket #"] for r in last})
    for t in mess_rng.choice(tickets, LATE_BREAK_BATCHES, replace=False):
        for k, r in enumerate(r for r in last if r["Ticket #"] == t):
            psi = round(float(r["Strength (psi)"]) * LATE_GAIN, -1)
            extra.append(r | {"Set": f"{t}-L{k}", "Tested": r["Cast"] + timedelta(days=LATE_AGE),
                              "Age": f"{LATE_AGE} day", "Strength (psi)": psi,
                              "Max Load (lbf)": round(psi * CYL_AREA_IN2, -1), "Tech Notes": ""})
    return extra


def write_sheet(path: Path, sheet: str, title: list[str] | None, df: pd.DataFrame) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    if title:
        for line in title:
            ws.append([line])
        ws["A1"].font = Font(bold=True, size=13)
        ws.append([])
    ws.append(list(df.columns))
    for cell in ws[ws.max_row]:
        cell.font = Font(bold=True)
    for row in df.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and math.isnan(v)) else v for v in row])
    for col in ws.columns:
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 40)
    wb.save(path)


def main() -> None:
    designs = make_designs()
    mixes, lots = make_batches(designs)
    base = meta_prediction(mixes)

    lb = KG_M3_TO_LB_YD3
    design_sheet = pd.DataFrame([{
        "Mix ID": m["code"],
        "Description": f"{m['fc']} psi{' air-entrained' if m['air_entrained'] else ''}",
        "Type IL Cement (lb/cy)": round(m["cement"] * lb),
        "Class F Fly Ash (lb/cy)": round(m["fly_ash"] * lb),
        "Slag Cement (lb/cy)": round(m["slag"] * lb),
        "Silica Fume (lb/cy)": round(m["sf"] * lb),
        "Water (gal/cy)": round(m["water"] * lb / LB_PER_GAL_WATER, 1),
        "HRWR (oz/cwt)": round(m["hrwr_oz_cwt"], 1),
        "Sand SSD (lb/cy)": round(m["fine"] * lb),
        "#57 Stone SSD (lb/cy)": round(m["coarse"] * lb),
        "Notes": "",
    } for m in designs])

    qc_rows, cyl_rows, truth_rows = [], [], []
    for i, m in enumerate(mixes):
        air_val: float | str = m["air"]
        if rng.random() < 0.12:
            air_val = f"{m['air']}%"
        qc_rows.append({
            "Date": m["batch_date"], "Ticket No.": m["ticket"], "Truck": m["truck"],
            "Mix Code": m["code"], "Batch Time": m["batch_time"].strftime("%H:%M"),
            "Operator": m["operator"], "Slump (in)": m["slump_in"],
            "Air (%)": air_val, "Conc Temp (F)": m["conc_temp_f"],
            "Cure Tank (F)": m["cure_temp_f"], "Cement Lot": m["cement_lot"],
            "Mill Cert 28d (psi)": round(m["mill_cert_psi"]) if m["mill_cert_recorded"] else float("nan"),
        })
        for j, (age, n_cyl) in enumerate(AGES.items()):
            f = true_factor(m, age)
            mean = base[i, j] * f["total"]
            truth_rows.append(dict(ticket=m["ticket"], mix=m["code"], age=age, meta_psi=round(float(base[i, j])),
                                   true_mean_psi=round(float(mean)),
                                   **{k: round(v, 4) for k, v in f.items()}))
            for k in range(n_cyl):
                psi = mean * rng.normal(1.0, CYLINDER_CV)
                note = ""
                if rng.random() < 0.03:
                    note = str(rng.choice(["re-capped", "chipped edge", "type 3 break"]))
                cyl_rows.append({
                    "Ticket #": m["ticket"],
                    "Set": f"{m['ticket']}-{chr(65 + k)}",
                    "Cast": m["batch_date"],
                    "Tested": m["batch_date"] + timedelta(days=age),
                    "Age": f"{age} day",
                    "Max Load (lbf)": round(psi * CYL_AREA_IN2, -1),
                    "Strength (psi)": round(psi, -1),
                    "Tech Notes": note,
                })

    clean_cylinders = len(cyl_rows)
    cyl_rows += messy_cylinders(cyl_rows)
    for row in qc_rows:
        if mess_rng.random() < RETYPED_CODE_SHARE:
            row["Mix Code"] = retype_code(row["Mix Code"])

    qc = pd.DataFrame(qc_rows).sort_values("Date")
    cyl = pd.DataFrame(cyl_rows).sort_values(["Tested", "Ticket #", "Set"])

    plant = OUT / "plant"
    plant.mkdir(parents=True, exist_ok=True)
    write_sheet(plant / "mix_designs.xlsx", "Mix Designs",
                ["Riverside Ready-Mix: Mix Design Register", "Rev. 2026-09, approved by QC"], design_sheet)
    write_sheet(plant / "cylinder_breaks.xlsx", "Breaks", None, cyl)
    write_sheet(plant / "daily_QC.xlsx", "QC Log", ["Riverside Ready-Mix: Daily QC Log"], qc)

    missing_lots = [l["lot"] for l in lots if not l["recorded"]]

    truth = dict(
        seed=SEED, source=SOURCE, ages=list(AGES),
        effects=dict(plant_factor=PLANT_FACTOR, air_ref_pct=AIR_REF,
                     air_loss_per_pct=AIR_LOSS_PER_PCT, sf_gain_per_kg_m3=SF_GAIN_PER_KG,
                     sf_ramp_days=SF_RAMP_DAYS, cement_ref_psi=CEMENT_REF_PSI,
                     cement_gain_per_psi=CEMENT_GAIN_PER_PSI, slow_gain_base=SLOW_GAIN_BASE,
                     slow_gain_per_slag_share=SLOW_GAIN_PER_SLAG_SHARE, slow_gain_days=SLOW_GAIN_DAYS,
                     batch_cv=BATCH_CV,
                     cylinder_cv=CYLINDER_CV),
        useful=["Air (%)", "Silica Fume (lb/cy)"],
        incomplete=["Mill Cert 28d (psi)"],
        no_effect=["Ticket No.", "Truck", "Batch Time", "Operator", "Slump (in)", "Conc Temp (F)"],
        designs=designs,
        batches=[{k: (v.isoformat() if isinstance(v, (date, datetime, time)) else v)
                  for k, v in m.items()} for m in mixes],
        strengths=truth_rows,
        mess=dict(clean_cylinders=clean_cylinders, cap_failures=CAP_FAILURES,
                  late_breaks=len(cyl) - clean_cylinders - CAP_FAILURES,
                  retyped_codes=int((qc["Mix Code"] != qc["Ticket No."].map(
                      {m["ticket"]: m["code"] for m in mixes})).sum()),
                  missing_cert_lots=missing_lots),
    )
    (OUT / "truth.json").write_text(json.dumps(truth, indent=1, default=float))
    print(f"{len(designs)} designs, {len(mixes)} batches, {len(cyl)} cylinders ({clean_cylinders} clean), "
          f"{int(qc['Mill Cert 28d (psi)'].notna().sum())} with mill cert, missing lots "
          f"{', '.join(missing_lots)} -> {plant}")


if __name__ == "__main__":
    main()
