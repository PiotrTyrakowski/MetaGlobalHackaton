"""Unit conversions from what US plants record to what BOxCrete expects."""

LB_PER_YD3_TO_KG_PER_M3 = 0.45359237 / 0.764554858
GAL_WATER_PER_YD3_TO_KG_PER_M3 = 3.785411784 / 0.764554858
FL_OZ_TO_L = 0.0295735296
MPA_TO_PSI = 145.0377377
DEFAULT_HRWR_DENSITY = 1.06  # kg/L, typical polycarboxylate superplasticizer

MASS_UNITS = {"kg/m3", "lb/yd3", "gal/yd3"}
HRWR_UNITS = {"kg/m3", "lb/yd3", "oz/cwt"}
STRENGTH_UNITS = {"psi", "MPa"}
TEMP_UNITS = {"C", "F"}


def mass_to_kg_m3(value, unit: str):
    if unit == "kg/m3":
        return value
    if unit == "lb/yd3":
        return value * LB_PER_YD3_TO_KG_PER_M3
    if unit == "gal/yd3":  # water only
        return value * GAL_WATER_PER_YD3_TO_KG_PER_M3
    raise ValueError(f"unsupported mass unit {unit!r}")


def hrwr_oz_cwt_to_kg_m3(oz_cwt, binder_kg_m3, density_kg_l: float = DEFAULT_HRWR_DENSITY):
    """Fluid ounces per 100 lb of binder -> kg per m3 of concrete."""
    binder_lb_yd3 = binder_kg_m3 / LB_PER_YD3_TO_KG_PER_M3
    kg_per_yd3 = oz_cwt * binder_lb_yd3 / 100 * FL_OZ_TO_L * density_kg_l
    return kg_per_yd3 / 0.764554858


def hrwr_kg_m3_to_oz_cwt(kg_m3, binder_kg_m3, density_kg_l: float = DEFAULT_HRWR_DENSITY):
    return kg_m3 / hrwr_oz_cwt_to_kg_m3(1.0, binder_kg_m3, density_kg_l)


def strength_to_psi(value, unit: str):
    if unit == "psi":
        return value
    if unit == "MPa":
        return value * MPA_TO_PSI
    raise ValueError(f"unsupported strength unit {unit!r}")


def temp_to_c(value, unit: str):
    if unit == "C":
        return value
    if unit == "F":
        return (value - 32) * 5 / 9
    raise ValueError(f"unsupported temperature unit {unit!r}")
