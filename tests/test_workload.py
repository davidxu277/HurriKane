import json
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from evaluate_workload import evaluate_workload, main


def predictions():
    return pd.DataFrame({
        "event_id": [3, 1, 2, 4], "final_tier": ["Critical", "Critical", "Safe", "Elevated"],
        "has_future": [True] * 4,
        "baseline_score": [0.0, 1.0, 3.0, 2.0], "alert_score": [0.9, 0.8, 0.1, 0.2],
    })


def test_workload_counts_events_at_same_recall_and_curves_include_origin():
    frame = predictions()
    before = frame.copy(deep=True)
    report, curves = evaluate_workload(frame, target_recall=0.5, bootstrap_iterations=50)
    assert_frame_equal(frame, before)
    assert report["positive_events"] == 2
    assert report["target_positive_events"] == 1
    assert report["baseline"]["reviews"] == 3
    assert report["model"]["reviews"] == 1
    assert report["baseline"]["recall"] == report["model"]["recall"] == 0.5
    assert report["review_reduction_fraction"] == pytest.approx(2 / 3)
    curves = curves[curves.scope.eq("future")]
    assert len(curves) == 5
    assert curves.iloc[0].baseline_recall == curves.iloc[0].model_recall == 0
    assert curves.iloc[-1].baseline_recall == curves.iloc[-1].model_recall == 1


def test_target_rounds_up_in_small_samples_and_increase_is_reported_as_negative():
    frame = predictions().rename(columns={"baseline_score": "alert_score", "alert_score": "baseline_score"})
    report, _ = evaluate_workload(frame, target_recall=0.9, bootstrap_iterations=50)
    assert report["target_positive_events"] == 2
    assert report["baseline"]["reviews"] == 2
    assert report["model"]["reviews"] == 4
    assert report["review_reduction_fraction"] == -1
    assert report["reference_only"]


def test_float_roundoff_does_not_increase_integer_positive_target():
    frame = pd.DataFrame({
        "event_id": np.arange(100), "final_tier": "Critical", "has_future": True,
        "baseline_score": 0.5, "alert_score": 0.5,
    })
    report, _ = evaluate_workload(frame, target_recall=0.07, bootstrap_iterations=10)
    assert report["target_positive_events"] == 7
    assert report["baseline"]["reviews"] == 7


def test_equal_scores_use_event_id_and_results_ignore_input_row_order():
    frame = predictions().assign(baseline_score=0.5, alert_score=0.5)
    first, first_curve = evaluate_workload(frame, bootstrap_iterations=100)
    second, second_curve = evaluate_workload(frame.iloc[::-1], bootstrap_iterations=100)
    assert first == second
    assert_frame_equal(first_curve, second_curve)
    assert first["baseline"]["reviews"] == first["model"]["reviews"] == 3
    assert first["bootstrap"]["ci95_lower"] == first["bootstrap"]["ci95_upper"] == 0


def test_bootstrap_is_paired_reproducible_and_accounts_for_zero_positive_draws():
    first, _ = evaluate_workload(predictions(), bootstrap_iterations=200, seed=17)
    again, _ = evaluate_workload(predictions(), bootstrap_iterations=200, seed=17)
    assert first == again
    boot = first["bootstrap"]
    assert boot["valid_samples"] + boot["zero_positive_samples"] == 200
    assert boot["zero_positive_samples"] > 0
    assert boot["ci95_lower"] <= boot["ci95_upper"]


def test_no_positives_is_undefined_and_json_contains_no_nan():
    frame = predictions().assign(final_tier="Safe")
    report, curves = evaluate_workload(frame, bootstrap_iterations=20)
    assert report["target_positive_events"] is None
    assert report["baseline"]["reviews"] is None
    assert report["model"]["recall"] is None
    assert report["review_reduction_fraction"] is None
    assert report["bootstrap"]["ci95_lower"] is None
    assert report["bootstrap"]["valid_samples"] == 0
    assert curves.baseline_recall.isna().all()
    json.dumps(report, allow_nan=False)


def test_official_tier_is_truth_even_when_pc_and_current_distance_disagree():
    frame = predictions().assign(y_true=-30.0, miss_distance_m=99999)
    report, _ = evaluate_workload(frame, bootstrap_iterations=20)
    assert report["positive_events"] == 2
    assert report["positive_definition"] == "final_tier == 'Critical'"
    assert report["target"] == "final_tier"
    assert "positive_threshold" not in report


@pytest.mark.parametrize("column", ["baseline_score", "alert_score"])
@pytest.mark.parametrize("value", [np.inf, np.nan, "bad"])
def test_invalid_values_are_rejected_without_silent_filtering(column, value):
    frame = predictions().astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="finite numeric"):
        evaluate_workload(frame, bootstrap_iterations=10)


def test_duplicate_or_missing_event_ids_are_rejected():
    for values in [[1, 1, 2, 3], [1, None, 2, 3], [1, " ", 2, 3]]:
        frame = predictions().assign(event_id=values)
        with pytest.raises(ValueError, match="event_id"):
            evaluate_workload(frame, bootstrap_iterations=10)


@pytest.mark.parametrize("recall", [0, 1.1, np.nan])
def test_invalid_target_recall(recall):
    with pytest.raises(ValueError, match="target_recall"):
        evaluate_workload(predictions(), target_recall=recall)


def test_cli_refuses_to_label_validation_as_holdout(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_workload.py", "--predictions",
                                     "reports/validation_predictions.csv", "--split", "holdout"])
    with pytest.raises(SystemExit):
        main()


def test_primary_excludes_known_labels_and_all_events_are_separate_supplement():
    frame = predictions().assign(has_future=[False, True, True, False])
    report, curves = evaluate_workload(frame, bootstrap_iterations=50)
    assert report["schema_version"] == 2
    assert report["scope"] == "future" and report["role"] == "primary"
    assert report["events"] == 2 and report["positive_events"] == 1
    assert report["input_events"] == 4 and report["excluded_no_future_events"] == 2
    supplementary = report["supplementary_all"]
    assert supplementary["scope"] == "all" and supplementary["role"] == "supplementary"
    assert supplementary["events"] == 4 and supplementary["positive_events"] == 2
    assert supplementary["no_future_events"] == 2
    assert curves.groupby("scope").size().to_dict() == {"all": 5, "future": 3}


def test_no_future_events_is_undefined_primary_not_promoted_to_all_events():
    report, curves = evaluate_workload(predictions().assign(has_future=False), bootstrap_iterations=20)
    assert report["events"] == 0 and report["baseline"]["reviews"] is None
    assert report["supplementary_all"]["events"] == 4
    assert report["supplementary_all"]["model"]["reviews"] == 2
    assert curves[curves.scope.eq("future")].baseline_recall.isna().all()
    json.dumps(report, allow_nan=False)


def test_empty_primary_keeps_zero_workload_origin_without_invented_recall():
    report, curves = evaluate_workload(predictions().assign(has_future=False), bootstrap_iterations=20)
    origin = curves[curves.scope.eq("future")]
    assert len(origin) == 1
    assert origin.iloc[0].reviewed_events == 0
    assert pd.isna(origin.iloc[0].model_recall)


def test_string_false_is_not_cast_to_true():
    frame = predictions().assign(has_future=["False", "true", "0", "1"])
    report, _ = evaluate_workload(frame, bootstrap_iterations=20)
    assert report["events"] == 2 and report["positive_events"] == 1


@pytest.mark.parametrize("value", [None, np.nan, pd.NA, "unknown", "", 2, 0.5])
def test_invalid_future_flags_are_rejected(value):
    frame = predictions().astype(object)
    frame.loc[0, "has_future"] = value
    with pytest.raises(ValueError, match="has_future"):
        evaluate_workload(frame, bootstrap_iterations=10)


@pytest.mark.parametrize("value", [None, "UNKNOWN", "critical", "", np.nan])
def test_unknown_final_labels_are_not_interpreted_as_negative(value):
    frame = predictions().astype(object)
    frame.loc[0, "final_tier"] = value
    with pytest.raises(ValueError, match="final_tier"):
        evaluate_workload(frame, bootstrap_iterations=10)


@pytest.mark.parametrize("column", ["has_future", "final_tier"])
def test_missing_official_columns_do_not_fall_back_to_old_pc_target(column):
    frame = predictions().drop(columns=column).assign(y_true=-5.0)
    with pytest.raises(ValueError, match="Missing prediction columns"):
        evaluate_workload(frame, bootstrap_iterations=10)


def test_ranking_scores_above_one_are_used_without_clipping():
    frame = predictions().assign(alert_score=[1.0000001, 1.0000002, 0.0, 0.5])
    report, _ = evaluate_workload(frame, target_recall=0.5, bootstrap_iterations=20)
    assert report["model"]["reviews"] == 1


def test_cli_generates_separate_official_scopes_and_preserves_input(tmp_path, monkeypatch):
    source = tmp_path / "validation_predictions.csv"
    predictions().assign(has_future=[False, True, True, False]).to_csv(source, index=False)
    original = source.read_bytes()
    output = tmp_path / "workload.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_workload.py", "--predictions", str(source),
                                     "--output", str(output), "--bootstrap", "20"])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scope"] == "future" and report["events"] == 2
    assert report["supplementary_all"]["events"] == 4
    assert report["purpose"] == "development_only"
    assert report["split"] == "validation"
    text = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "主结果" in text and "补充结果" in text
    assert "--positive-threshold" not in text
    assert (tmp_path / "workload_curve.png").is_file()
    assert set(pd.read_csv(tmp_path / "workload_curve.csv").scope) == {"future", "all"}
    assert source.read_bytes() == original
