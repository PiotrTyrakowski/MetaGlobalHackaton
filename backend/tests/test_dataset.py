"""The data layer: the demo files must give back the generator's truth, and bad input must fail loudly."""

import json
from pathlib import Path

import pandas as pd
import pytest

from mixpilot import dataset, excel
from mixpilot.dataset import MappingError

DEMO = Path(__file__).resolve().parents[1] / "demo_data"
MAPPING = json.loads((DEMO / "mapping.json").read_text())
TRUTH = json.loads((DEMO / "truth.json").read_text())


def demo_sheets():
    return [s for f in sorted((DEMO / "plant").glob("*.xlsx")) for s in excel.read_sheets(f)]


def sheet(file, **columns):
    return excel.Sheet(file=file, sheet="s", header_row=0, df=pd.DataFrame(columns))


def small_mapping(tests=None):
    return {"recipes": {"file": "r.xlsx", "key": "Mix", "fields": {"cement": {"column": "C", "unit": "kg/m3"},
                                                                   "water": {"column": "W", "unit": "kg/m3"}}},
            "tests": {"file": "t.xlsx", "recipe_key": "Mix",
                      "fields": {"age": {"column": "Age", "unit": "days"},
                                 "strength": {"column": "Psi", "unit": "psi"}}, **(tests or {})}}


def small_sheets(mix=("R 101", "r-102", "R103"), age=("7 day", "28 day", 56), notes=("", "cap failure", "")):
    return [sheet("r.xlsx", Mix=["R-101", "R-102", "R-103"], C=[300, 310, 320], W=[150, 150, 150]),
            sheet("t.xlsx", Mix=list(mix), Age=list(age), Psi=[3000, 4000, 5000], Notes=list(notes))]


def test_demo_files_give_back_the_truth():
    ds, report = dataset.build(demo_sheets(), MAPPING)
    mess = TRUTH["mess"]
    assert report["recipes"] == len(TRUTH["designs"]) and report["batches"] == len(TRUTH["batches"])
    assert report["cylinders"] == mess["clean_cylinders"]
    assert report["dropped"] == {"Tech Notes contains 'cap failure'": mess["cap_failures"],
                                 "older than 28 days": mess["late_breaks"]}
    assert report["ids_matched_loosely"] == mess["retyped_codes"] > 0
    assert report["ages"] == [3.0, 7.0, 28.0]

    rows = ds.rows.drop_duplicates("batch").set_index("batch")
    air = ds.feature("Air (%)")
    for b in TRUTH["batches"]:
        r = rows.loc[str(b["ticket"])]
        assert r["recipe"] == b["code"]
        assert r["cement"] == pytest.approx(b["cement"], abs=1)
        assert r["hrwr"] == pytest.approx(b["hrwr_kg_m3"], abs=0.1)
        assert air[str(b["ticket"])] == pytest.approx(b["air"], abs=0.01)


def test_ids_match_ignoring_case_and_punctuation():
    ds, report = dataset.build(small_sheets(age=(7, 28, 28), notes=("", "", "")), small_mapping())
    assert sorted(ds.rows["recipe"].unique()) == ["R-101", "R-102", "R-103"]
    assert report["ids_matched_loosely"] == 3


def test_rows_marked_invalid_and_late_breaks_are_left_out():
    exclude = {"exclude": [{"column": "Notes", "contains": "CAP FAILURE"}]}
    ds, report = dataset.build(small_sheets(), small_mapping(exclude))
    assert ds.rows["recipe"].tolist() == ["R-101"]
    assert ds.rows["age"].tolist() == [7.0]  # "7 day" read as a number
    assert report["dropped"] == {"Notes contains 'CAP FAILURE'": 1, "older than 28 days": 1}


@pytest.mark.parametrize("sheets, mapping, message", [
    (small_sheets(mix=("X", "Y", "Z")), small_mapping(), "joins look wrong"),
    (small_sheets(age=("n/a", "n/a", "n/a")), small_mapping(), "no usable tests"),
    (small_sheets(), small_mapping({"exclude": [{"column": "Nope", "contains": "x"}]}), "tests.exclude"),
])
def test_bad_input_fails_loudly(sheets, mapping, message):
    with pytest.raises(MappingError, match=message):
        dataset.build(sheets, mapping)


def test_cylinders_that_name_only_the_mix_average_the_qc_rows():
    qc = sheet("q.xlsx", Ticket=[1, 2, 3], Mix=["r 101", "R-101", "R-102"], Air=["2%", 4, 6], Cure=[70.0, 72.0, 74.0])
    mapping = small_mapping()
    mapping["batches"] = {"file": "q.xlsx", "key": "Ticket", "recipe_key": "Mix",
                          "fields": {"cure_temp": {"column": "Cure", "unit": "F"}}}
    sheets = small_sheets(mix=("R101", "R102", "R102"), age=(7, 7, 28), notes=("", "", "")) + [qc]
    ds, _ = dataset.build(sheets, mapping)
    assert ds.candidates["Air"].level == "recipe" and ds.candidates["Air"].file == "q.xlsx"
    assert ds.feature("Air").to_dict() == {"R-101": 3.0, "R-102": 6.0}
    assert sorted(ds.rows["batch"].unique()) == ["R-101", "R-102"]


def test_join_keys_see_through_id_spelling():
    keys = excel.join_keys([sheet("a.xlsx", Mix=[f"R-{i}" for i in range(6)], Day=list(range(6))),
                            sheet("b.xlsx", Code=[f"r {i}" for i in range(6)])])
    assert any(k["a"] == "a.xlsx:Mix" and k["b"] == "b.xlsx:Code" for k in keys)


def test_the_automl_cache_ignores_the_order_of_the_mapping():
    from mixpilot import tools

    ds, _ = dataset.build(demo_sheets(), MAPPING)
    shuffled = json.loads(json.dumps(MAPPING))
    shuffled["recipes"]["fields"] = dict(reversed(list(shuffled["recipes"]["fields"].items())))
    other, _ = dataset.build(demo_sheets(), shuffled)
    assert tools._run_key(ds, ["Air (%)"], 10) == tools._run_key(other, ["Air (%)"], 10)


# --- other layouts plants keep ------------------------------------------------------

def mixes(n=6, **extra):
    return dict(Mix=[f"M{i}" for i in range(n)], C=[300 + 10 * i for i in range(n)], W=[150] * n,
                F=[800] * n, G=[1000] * n, **extra)


def full_recipes(file="r.xlsx", key=True):
    fields = {"cement": ("C", "kg/m3"), "water": ("W", "kg/m3"), "fine": ("F", "kg/m3"), "coarse": ("G", "kg/m3")}
    spec = {"file": file, "fields": {f: {"column": c, "unit": u} for f, (c, u) in fields.items()}}
    return {**spec, "key": "Mix"} if key else spec


def test_a_workbook_with_several_sheets_keeps_them_apart(tmp_path):
    path = tmp_path / "plant.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame(dict(Mix=["A"], C=[300])).to_excel(w, sheet_name="Recipes", index=False)
        pd.DataFrame(dict(Mix=["A"], Psi=[4000])).to_excel(w, sheet_name="Lab", index=False)
        pd.DataFrame().to_excel(w, sheet_name="Empty")
    assert [s.file for s in excel.read_sheets(path)] == ["plant.xlsx / Recipes", "plant.xlsx / Lab"]


def test_a_single_sheet_workbook_is_just_the_file(tmp_path):
    path = tmp_path / "plant.xlsx"
    pd.DataFrame(dict(Mix=["A"], C=[300])).to_excel(path, index=False)
    assert [s.file for s in excel.read_sheets(path)] == ["plant.xlsx"]


def test_a_european_csv_with_a_title_line_reads_as_numbers_and_dates(tmp_path):
    path = tmp_path / "lab.csv"
    path.write_bytes("Lab export, Plant Süd\n\nMix;Cast;fc (N/mm²)\nA;2026-06-12;31,5\nB;2026-06-13;28\n".encode("latin-1"))
    s = excel.read_sheets(path)[0]
    assert list(s.df.columns) == ["Mix", "Cast", "fc (N/mm²)"]
    assert s.df["fc (N/mm²)"].tolist() == [31.5, 28.0]
    assert excel.column_kind(s.df["Cast"]) == "date"
    assert excel.as_feature(s.df["Cast"], "date").tolist() == [0.0, 1.0]


def test_repeated_headers_stay_separate_columns(tmp_path):
    path = tmp_path / "lab.xlsx"
    pd.DataFrame([["Mix", "7 day", "7 day"], ["A", 1, 2]]).to_excel(path, index=False, header=False)
    assert list(excel.read_sheets(path)[0].df.columns) == ["Mix", "7 day", "7 day (2)"]


def test_one_column_per_age_becomes_one_row_per_result():
    sheets = [sheet("r.xlsx", **mixes()),
              sheet("t.xlsx", Mix=[f"M{i}" for i in range(6)], D7=[3000.0] * 5 + [None], D28=[4000.0] * 6)]
    mapping = {"recipes": full_recipes(), "tests": {"file": "t.xlsx", "recipe_key": "Mix", "age_columns": {"D7": 7, "D28": 28},
                                                    "fields": {"strength": {"unit": "psi"}}}}
    prepared, mapping = dataset.prepare(sheets, mapping)
    ds, _ = dataset.build(prepared, mapping)
    assert ds.cylinders == 11  # the blank 7-day cell is no test at all
    assert sorted(ds.rows["age"].unique()) == [7.0, 28.0]
    assert not {"D7", "D28"} & set(ds.candidates)


def test_one_table_without_mix_ids_groups_rows_by_ingredients():
    table = mixes(Age=[7] * 6, Psi=[3000] * 6, Note=[""] * 6, Air=[2.0] * 6)
    table = {k: v * 2 for k, v in table.items()}
    table["Age"] = [7] * 6 + [28] * 6
    del table["Mix"]
    sheets = [sheet("all.xlsx", **table)]
    mapping = {"recipes": full_recipes("all.xlsx", key=False),
               "tests": {"file": "all.xlsx", "fields": {"age": {"column": "Age", "unit": "days"},
                                                        "strength": {"column": "Psi", "unit": "psi"}},
                         "exclude": [{"column": "Note", "contains": "bad"}]}}
    prepared, mapping = dataset.prepare(sheets, mapping)
    ds, report = dataset.build(prepared, mapping)
    assert ds.rows["recipe"].nunique() == 6 and len(ds.rows) == 12
    assert set(ds.candidates) == {"Air"}  # never the test's own columns
    assert not any("repeat" in a for a in report["assumptions"])


def test_a_mix_listed_twice_keeps_its_last_row():
    table = mixes()
    table = {k: [*v, v[0]] for k, v in table.items()}
    table["C"][-1] = 999
    sheets = [sheet("r.xlsx", **table), sheet("t.xlsx", Mix=["M0"], Age=[28], Psi=[4000])]
    ds, report = dataset.build(sheets, {"recipes": full_recipes(), "tests": small_mapping()["tests"]})
    assert ds.rows["cement"].tolist() == [999.0] and len(ds.rows) == 1
    assert any("repeat a mix ID" in a for a in report["assumptions"])


def test_a_blank_optional_ingredient_is_zero_and_a_blank_main_one_drops_the_mix():
    table = mixes(FA=[None, 50.0, None, 50.0, None, 50.0], SF=[None, 10.0, None, 10.0, None, 10.0])
    table["C"][5] = None
    recipes = full_recipes()
    recipes["fields"]["fly_ash"] = {"column": "FA", "unit": "kg/m3"}
    recipes["other_cementitious"] = [{"column": "SF", "unit": "kg/m3"}]
    sheets = [sheet("r.xlsx", **table),
              sheet("t.xlsx", Mix=[f"M{i}" for i in range(6)], Age=[28] * 6, Psi=[4000] * 6)]
    ds, report = dataset.build(sheets, {"recipes": recipes, "tests": small_mapping()["tests"]})
    assert ds.rows["fly_ash"].notna().all() and ds.rows["fly_ash"].iloc[0] == 0.0
    assert report["recipes_left_out"] == ["M5"]
    assert ds.feature("SF").notna().all()  # blank silica fume means none, not "partly logged"


@pytest.mark.parametrize("n, ages, message", [(4, [28] * 4, "only 4 mix designs"), (6, [7] * 6, "latest usable break")])
def test_too_little_data_to_score_a_model_is_said_plainly(n, ages, message):
    sheets = [sheet("r.xlsx", **mixes(n)), sheet("t.xlsx", Mix=[f"M{i}" for i in range(n)], Age=ages, Psi=[4000] * n)]
    ds, _ = dataset.build(sheets, {"recipes": full_recipes(), "tests": small_mapping()["tests"]})
    with pytest.raises(MappingError, match=message):
        dataset.check_enough(ds)


@pytest.mark.parametrize("unit, value, kg_m3", [("%", 1.0, 4.0), ("mL/kg", 10.0, 4.32), ("mL/100kg", 1000.0, 4.32), ("L/m3", 4.0, 4.32)])
def test_hrwr_doses(unit, value, kg_m3):
    from mixpilot import units
    assert float(units.convert("hrwr", unit, value, 400.0)) == pytest.approx(kg_m3)


def test_the_agent_sees_the_notes_on_cylinders_that_still_count():
    from mixpilot import tools

    sheets = [sheet("t.xlsx", Mix=["M"] * 6, Cylinder=[f"C{i}" for i in range(6)], Age=[28] * 6, Psi=[4000] * 6,
                    Notes=["", "", "cap failure, retest", "re-capped", "re-capped", None])]
    assert tools._notes_still_counted(sheets, small_mapping()) == {"Notes": ["re-capped", "cap failure, retest"]}
    rule = {"exclude": [{"column": "Notes", "contains": "CAP FAILURE"}]}
    assert tools._notes_still_counted(sheets, small_mapping(rule)) == {"Notes": ["re-capped"]}


def test_a_column_that_repeats_a_strength_result_is_caught():
    from mixpilot import inputs

    ds, _ = dataset.build(demo_sheets(), MAPPING)
    at_28 = ds.rows[ds.rows["age"] == 28].set_index("batch")["strength"]
    ds.batches["28d result"] = ds.batches["batch"].map(at_28).round(-1)
    ds.candidates["28d result"] = dataset.Candidate("28d result", "batch", "numeric", "q.xlsx")
    assert inputs._restates_result(ds, "28d result")
    assert not inputs._restates_result(ds, "Air (%)")
    assert not inputs._restates_result(ds, "Mill Cert 28d (psi)")
