import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ml import build_samples, feature_columns
from validation import ISSUE_COLUMNS, validate_input


def metadata(task="final"):
    return {"task": task, "features": feature_columns(task)}


def messages():
    return pd.DataFrame({
        "event_id": ["001", "002"], "time_to_tca": [3.0, 2.5],
        "miss_distance": [6000.0, 7000.0], "risk": [-7.0, -5.0],
        "relative_speed": [11000.0, 12000.0],
        "relative_position_r": [1.0, 2.0], "relative_position_t": [2.0, 3.0],
        "relative_position_n": [3.0, 4.0], "c_object_type": ["DEBRIS", "TBA"],
        "c_sigma_r": [3.0, 4.0], "t_sigma_r": [4.0, 3.0],
        "max_risk_estimate": [-4.0, -3.0], "F10": [80.0, 90.0], "AP": [12.0, 13.0],
    }, index=[7, 11])


def test_conversion_keeps_source_ids_extra_columns_and_does_not_mutate_input():
    source = messages().astype({"miss_distance": str, "time_to_tca": str})
    source["extra"] = "untouched"
    before = source.copy(deep=True)
    valid, issues = validate_input(source, metadata())
    assert_frame_equal(source, before)
    assert issues.empty
    assert issues.columns.tolist() == ISSUE_COLUMNS
    assert valid.index.tolist() == [7, 11]
    assert valid.input_record_id.tolist() == [7, 11]
    assert valid.event_id.tolist() == ["001", "002"]
    samples = build_samples(valid, "final", 2, labeled=False)
    assert samples.input_record_id.tolist() == [7, 11]
    assert samples.miss_distance.tolist() == [6000, 7000]
    assert valid.extra.tolist() == ["untouched", "untouched"]


@pytest.mark.parametrize("column,value,code", [
    ("miss_distance", -1, "NEGATIVE_VALUE"),
    ("c_sigma_r", -1, "NEGATIVE_VALUE"),
    ("risk", 0.1, "INVALID_LOG10_PC"),
    ("relative_speed", "bad", "INVALID_NUMERIC"),
    ("miss_distance", np.inf, "NON_FINITE"),
    ("F10", -np.inf, "NON_FINITE"),
    ("risk", "inf", "NON_FINITE"),
    ("risk", "NaN", "INVALID_NUMERIC"),
    ("event_id", np.inf, "NON_FINITE"),
    ("event_id", " ", "MISSING_VALUE"),
    ("time_to_tca", None, "MISSING_VALUE"),
    ("risk", None, "MISSING_VALUE"),
])
def test_bad_rows_are_excluded_with_traceable_issues(column, value, code):
    source = messages().astype(object)
    source.loc[11, column] = value
    valid, issues = validate_input(source, metadata())
    assert valid.index.tolist() == [7]
    issue = issues[issues.code == code].iloc[0]
    assert issue.input_record_id == 11
    assert issue.row_position == 1
    assert issue.column == column
    assert issue.severity == "ERROR"


@pytest.mark.parametrize("column", ["c_sigma_r", "t_sigma_r", "F10", "AP", "miss_distance"])
@pytest.mark.parametrize("value", [None, "  ", pd.NA])
def test_allowed_missing_numeric_values_are_kept_and_reported(column, value):
    source = messages().astype(object)
    source.loc[11, column] = value
    valid, issues = validate_input(source, metadata())
    assert len(valid) == 2
    assert pd.isna(valid.loc[11, column])
    assert issues.iloc[0].severity == "WARNING"
    assert issues.iloc[0].code == "MISSING_VALUE"
    assert issues.iloc[0].event_id == "002"


def test_zero_distance_and_negative_time_are_not_filtered():
    source = messages()
    source.loc[7, "miss_distance"] = 0
    source.loc[11, "time_to_tca"] = -0.1
    valid, issues = validate_input(source, metadata())
    assert len(valid) == 2
    assert issues.empty


def test_missing_category_is_compatible_with_ml_unknown():
    source = messages()
    source.loc[11, "c_object_type"] = " "
    valid, issues = validate_input(source, metadata())
    assert issues.iloc[0].severity == "WARNING"
    samples = build_samples(valid, "final", 2, labeled=False).set_index("event_id")
    assert samples.loc["002", "c_object_type"] == "UNKNOWN"


def test_current_does_not_require_label_risk_but_final_does():
    source = messages().drop(columns="risk")
    valid, issues = validate_input(source, metadata("current"))
    assert len(valid) == 2 and issues.empty
    assert len(build_samples(valid, "current", 2, labeled=False)) == 2
    valid, issues = validate_input(source, metadata())
    assert valid.empty
    assert issues.iloc[0].code == "MISSING_COLUMN"
    assert issues.iloc[0].column == "risk"
    assert pd.isna(issues.iloc[0].row_position)


def test_metadata_features_require_raw_fields_but_not_derived_fields():
    config = metadata()
    config["features"] = config["features"] + ["new_sensor"]
    valid, issues = validate_input(messages(), config)
    assert valid.empty
    assert issues.column.tolist() == ["new_sensor"]


def test_duplicate_columns_are_reported_before_accessing_values():
    source = pd.concat([messages(), messages()[["risk"]]], axis=1)
    valid, issues = validate_input(source, metadata())
    assert valid.empty
    assert issues.code.tolist() == ["DUPLICATE_COLUMN"]


def test_duplicate_indices_still_have_unambiguous_issue_positions():
    source = messages().astype(object)
    source.index = [7, 7]
    source.iloc[1, source.columns.get_loc("miss_distance")] = "bad"
    valid, issues = validate_input(source, metadata())
    assert len(valid) == 1
    assert issues.iloc[0].input_record_id == 7
    assert issues.iloc[0].row_position == 1


def test_empty_input_returns_stable_schemas():
    valid, issues = validate_input(messages().iloc[:0], metadata())
    assert valid.empty and "input_record_id" in valid
    assert issues.code.tolist() == ["EMPTY_INPUT"]


def test_bad_configuration_is_not_misreported_as_bad_data():
    with pytest.raises(ValueError, match="task"):
        validate_input(messages(), {})
    with pytest.raises(ValueError, match="target risk"):
        validate_input(messages(), {"task": "current", "features": ["risk"]})
    with pytest.raises(TypeError, match="DataFrame"):
        validate_input([], metadata())


def test_shipped_metadata_needs_only_raw_csv_columns():
    config = json.loads((Path(__file__).parents[1] / "models/metadata.json").read_text(encoding="utf-8"))
    valid, issues = validate_input(messages(), config)
    assert len(valid) == 2 and issues.empty
