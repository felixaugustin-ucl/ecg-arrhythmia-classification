import pandas as pd

from ecg.data.labels import (
    attach_conditions,
    label_sets,
    normalize_code,
    parse_code_list,
    to_title_case,
)


def test_normalize_code_strips_float_artifact():
    assert normalize_code("426783006.0") == "426783006"
    assert normalize_code("426783006") == "426783006"


def test_normalize_code_handles_missing():
    assert normalize_code("") == ""
    assert normalize_code("nan") == ""
    assert normalize_code(float("nan")) == ""


def test_normalize_code_leaves_genuine_decimals_alone():
    # Only a trailing '.0' is an artifact; '.5' is not, so it must survive.
    assert normalize_code("123.5") == "123.5"


def test_to_title_case_strips_wrappers():
    assert to_title_case("electrocardiogram: sinus rhythm") == "Sinus Rhythm"
    assert to_title_case("atrial fibrillation (disorder)") == "Atrial Fibrillation"


def test_parse_code_list_drops_blanks():
    assert parse_code_list("164889003, ,426783006.0") == ["164889003", "426783006"]


def test_attach_conditions_marks_unknown_codes():
    metadata = pd.DataFrame({"dx_codes": ["164889003,999999"]})
    out = attach_conditions(metadata, {"164889003": "Atrial Fibrillation"})
    assert out.loc[0, "dx_condition_list"] == ["Atrial Fibrillation", "Unknown Condition (999999)"]


def test_label_sets_deduplicates_and_sorts():
    metadata = pd.DataFrame({"dx_codes": ["b,a,b"]})
    metadata = attach_conditions(metadata, {})
    assert label_sets(metadata).iloc[0] == ["a", "b"]
