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
