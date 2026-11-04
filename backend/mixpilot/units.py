"""Unit conversion into BOxCrete's units (kg/m3, °C, psi, days)."""

from __future__ import annotations

import numpy as np

LB_YD3_TO_KG_M3 = 0.453592 / 0.764555
GAL_WATER_YD3_TO_KG_M3 = 3.78541 / 0.764555  # 1 US gal of water = 3.785 kg
FL_OZ_TO_L = 0.0295735
HRWR_KG_PER_L = 1.08  # typical polycarboxylate HRWR density (assumption)
MPA_TO_PSI = 145.038

# BOxCrete field -> {unit: factor to BOxCrete's unit}. Dose and temperature units are special.
MASS = {"kg/m3": 1.0, "lb/yd3": LB_YD3_TO_KG_M3}
UNITS = {
    "cement": MASS, "fly_ash": MASS, "slag": MASS, "fine": MASS, "coarse": MASS,
    "water": {**MASS, "gal/yd3": GAL_WATER_YD3_TO_KG_M3, "L/m3": 1.0},
    "hrwr": {**MASS, "fl_oz/cwt": None, "mL/100kg": None, "mL/kg": None, "%": None, "L/m3": HRWR_KG_PER_L},
    "cure_temp": {"C": None, "F": None},
    "strength": {"psi": 1.0, "MPa": MPA_TO_PSI, "ksi": 1000.0},
    "age": {"days": 1.0, "hours": 1 / 24},
}
ALIASES = {"lb/cy": "lb/yd3", "lb/yd³": "lb/yd3", "kg/m³": "kg/m3", "gal/cy": "gal/yd3",
           "oz/cwt": "fl_oz/cwt", "°F": "F", "°C": "C", "day": "days", "d": "days", "N/mm2": "MPa", "N/mm²": "MPa",
           "L/m³": "L/m3", "l/m3": "L/m3", "l/m³": "L/m3", "kg/m^3": "kg/m3", "lb/yd^3": "lb/yd3", "lbs/yd3": "lb/yd3",
           "pcy": "lb/yd3", "ml/100kg": "mL/100kg", "ml/kg": "mL/kg"}

ASSUMPTIONS = {
    "fl_oz/cwt": f"HRWR density {HRWR_KG_PER_L} kg/L; dose per 100 lb of all cementitious materials",
    "mL/100kg": f"HRWR density {HRWR_KG_PER_L} kg/L; dose per 100 kg of all cementitious materials",
    "mL/kg": f"HRWR density {HRWR_KG_PER_L} kg/L; dose per kg of all cementitious materials",
    "%": "HRWR dose as % of the mass of all cementitious materials",
    "gal/yd3": "1 US gallon of water = 3.785 kg",
}


def normalize_unit(unit: str) -> str:
    u = unit.strip()
    return ALIASES.get(u, u)


def check_unit(field: str, unit: str) -> str | None:
    """None if fine, else an error message for the agent."""
    if field not in UNITS:
        return f"unknown field '{field}'"
    if normalize_unit(unit) not in UNITS[field]:
        return f"unit '{unit}' not supported for {field}; use one of {sorted(UNITS[field])}"
    return None


def convert(field: str, unit: str, x: np.ndarray, cementitious_kg_m3: np.ndarray | None = None) -> np.ndarray:
    """Convert values of a BOxCrete field into BOxCrete's unit."""
    unit = normalize_unit(unit)
    if field == "cure_temp":
        return (x - 32.0) * 5.0 / 9.0 if unit == "F" else x
    if field == "hrwr" and UNITS["hrwr"].get(unit, 0) is None:
        if cementitious_kg_m3 is None:
            raise ValueError("HRWR dose needs the cementitious content")
        if unit == "fl_oz/cwt":
            cm_lb_yd3 = cementitious_kg_m3 / LB_YD3_TO_KG_M3
            kg_per_yd3 = x * cm_lb_yd3 / 100 * FL_OZ_TO_L * HRWR_KG_PER_L
            return kg_per_yd3 / 0.764555
        if unit == "%":
            return x / 100 * cementitious_kg_m3
        per_kg = x / 100 if unit == "mL/100kg" else x
        return per_kg * cementitious_kg_m3 / 1000 * HRWR_KG_PER_L
    return x * UNITS[field][unit]
