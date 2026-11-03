"""BOxCrete's input schema and the ranges Meta's training data covers."""

COMPOSITION = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
]
BASE_INPUTS = COMPOSITION + ["Temp (C)", "Time"]
MAX_AGE_DAYS = 28  # BOxCrete models strength from day 0 to day 28

# Range of Meta's ready-mix-like training set (Material Source 2), kg/m3.
META_RANGES = {
    "Cement (kg/m3)": (63, 305),
    "Fly Ash (kg/m3)": (0, 235),
    "Slag (kg/m3)": (0, 259),
    "Water (kg/m3)": (120, 186),
    "HRWR (kg/m3)": (0, 2),
    "Fine Aggregate (kg/m3)": (719, 992),
    "Coarse Aggregates (kg/m3)": (1051, 1184),
}

# Default cradle-to-gate GWP factors, kg CO2e per kg of material. Replaced
# by EPD values when the plant has them.
GWP_FACTORS = {
    "Cement (kg/m3)": 0.90,
    "Fly Ash (kg/m3)": 0.03,
    "Slag (kg/m3)": 0.08,
    "Water (kg/m3)": 0.0003,
    "HRWR (kg/m3)": 1.9,
    "Fine Aggregate (kg/m3)": 0.005,
    "Coarse Aggregates (kg/m3)": 0.005,
}
