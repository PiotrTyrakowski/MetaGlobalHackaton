"""Generate a second synthetic plant to show the agent isn't fitted to the first one.

A new seed gives different mixes, batches and strengths. Every column then gets another
name and metric units, the notes are reworded, and the files have other names. Nothing
in the app knows about any of this: upload demo_data/plant_metric/*.xlsx like any plant's.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

SEED = 23
BACKEND = Path(__file__).resolve().parent.parent
OUT = BACKEND / "demo_data" / "plant_metric"
LB = 1.68555  # kg/m3 to lb/yd3
PSI_PER_MPA = 145.038


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(BACKEND / "scripts" / "make_demo_data.py"), str(SEED), tmp], check=True)
        src = Path(tmp) / "plant"
        mix = pd.read_excel(src / "mix_designs.xlsx", header=3)
        qc = pd.read_excel(src / "daily_QC.xlsx", header=2)
        cy = pd.read_excel(src / "cylinder_breaks.xlsx")
        shutil.copy(Path(tmp) / "truth.json", BACKEND / "demo_data" / "plant_metric_truth.json")

    recipes = pd.DataFrame({
        "Design Code": mix["Mix ID"], "Class": mix["Description"],
        "CEM II/A-LL (kg/m3)": (mix["Type IL Cement (lb/cy)"] / LB).round(1),
        "PFA (kg/m3)": (mix["Class F Fly Ash (lb/cy)"] / LB).round(1),
        "GGBS (kg/m3)": (mix["Slag Cement (lb/cy)"] / LB).round(1),
        "Microsilica (kg/m3)": (mix["Silica Fume (lb/cy)"] / LB).round(1),
        "Free water (L/m3)": (mix["Water (gal/cy)"] * 3.78541 * 1.30795).round(1),
        "PCE (mL/100kg)": (mix["HRWR (oz/cwt)"] * 29.5735).round(0),
        "Fine agg (kg/m3)": (mix["Sand SSD (lb/cy)"] / LB).round(1),
        "20mm agg (kg/m3)": (mix["#57 Stone SSD (lb/cy)"] / LB).round(1),
    })
    production = pd.DataFrame({
        "Pour date": qc["Date"], "Docket": qc["Ticket No."], "Vehicle": qc["Truck"], "Design": qc["Mix Code"],
        "Dispatch": qc["Batch Time"], "Batcher": qc["Operator"], "Slump (mm)": (qc["Slump (in)"] * 25.4).round(0),
        "Entrained gas (%)": qc["Air (%)"].astype(str).str.replace("%", "").astype(float),
        "Fresh temp (C)": ((qc["Conc Temp (F)"] - 32) * 5 / 9).round(1),
        "Water bath (C)": ((qc["Cure Tank (F)"] - 32) * 5 / 9).round(1), "Silo batch": qc["Cement Lot"],
        "Supplier 28d (MPa)": (qc["Mill Cert 28d (psi)"] / PSI_PER_MPA).round(1),
    })
    notes = cy["Tech Notes"].fillna("").replace({
        "cap failure, retest": "capping defect, discard", "chipped edge": "edge damage",
        "re-capped": "recapped", "type 3 break": "shear cone"})
    results = pd.DataFrame({
        "Docket no": cy["Ticket #"], "Specimen": cy["Set"], "Made": cy["Cast"], "Crushed": cy["Tested"],
        "Age (d)": cy["Age"].str.extract(r"(\d+)")[0].astype(int),
        "fc (MPa)": (cy["Strength (psi)"] / PSI_PER_MPA).round(2), "Lab comment": notes,
    })

    OUT.mkdir(parents=True, exist_ok=True)
    recipes.to_excel(OUT / "recipes_register.xlsx", index=False)
    production.to_excel(OUT / "production_log.xlsx", index=False)
    results.to_excel(OUT / "lab_results.xlsx", index=False)
    print(f"{len(recipes)} mixes, {len(production)} batches, {len(results)} cylinders -> {OUT}")


if __name__ == "__main__":
    main()
