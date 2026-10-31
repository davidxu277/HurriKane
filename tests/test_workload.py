import json
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from evaluate_workload import evaluate_workload, main


def predictions():
    return pd.DataFrame({
        "event_id": [3, 1, 2, 4], "y_true": [-5.0, -5.0, -10.0, -10.0],
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
    assert len(curves) == 5
    assert curves.iloc[0].baseline_recall == curves.iloc[0].model_recall == 0
    assert curves.iloc[-1].baseline_recall == curves.iloc[-1].model_recall == 1


def test_target_rounds_up_in_small_samples_and_increase_is_reported_as_negative():
    frame = predictions().rename(columns={"baseline_score": "alert_score", "alert_score": "baseline_score"})
    # Exchanged baseline scores need a valid classifier probability range.
    frame["alert_score"] /= 3
    report, _ = evaluate_workload(frame, target_recall=0.9, bootstrap_iterations=50)
    assert report["target_positive_events"] == 2
    assert report["baseline"]["reviews"] == 2
    assert report["model"]["reviews"] == 4
    assert report["review_reduction_fraction"] == -1
    assert report["reference_only"]


def test_float_roundoff_does_not_increase_integer_positive_target():
    frame = pd.DataFrame({
        "event_id": np.arange(100), "y_true": -5.0,
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
    frame = predictions().assign(y_true=-30.0)
    report, curves = evaluate_workload(frame, bootstrap_iterations=20)
    assert report["target_positive_events"] is None
    assert report["baseline"]["reviews"] is None
    assert report["model"]["recall"] is None
    assert report["review_reduction_fraction"] is None
    assert report["bootstrap"]["ci95_lower"] is None
    assert report["bootstrap"]["valid_samples"] == 0
    assert curves.baseline_recall.isna().all()
    json.dumps(report, allow_nan=False)


def test_threshold_is_inclusive_and_distance_does_not_change_labels():
    frame = predictions().assign(y_true=[-6.0, -6.001, -30.0, -30.0], miss_distance_m=0)
    report, _ = evaluate_workload(frame, bootstrap_iterations=20)
    assert report["positive_events"] == 1


@pytest.mark.parametrize("column", ["y_true", "baseline_score", "alert_score"])
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
