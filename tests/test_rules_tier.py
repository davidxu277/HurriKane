"""分级必须基于 tier_pc，并能接住 ml.py 的真实输出。"""
import math

import numpy as np
import pandas as pd
import pytest

from ml import build_samples, combine_predictions
from rules import apply_risk_rules

THRESHOLD = 0.1


def grade(tier, distance, conservative=None, predicted=1e-9, sigma_ratio=1e-3):
    if conservative is None:
        conservative = predicted if tier is None else max(tier, predicted)
    frame = pd.DataFrame([{
        'tier_pc': np.nan if tier is None else tier,
        'predicted_pc': predicted,
        'conservative_pc': conservative,
        'miss_distance_m': np.nan if distance is None else distance,
        'sigma_ratio': sigma_ratio,
    }])
    return apply_risk_rules(frame, THRESHOLD).iloc[0]


@pytest.mark.parametrize('tier, distance, level, status', [
    (1e-4, 6000, 'CRITICAL', 'OK'),
    (1e-6, 6000, 'ELEVATED', 'OK'),
    (9.9e-7, 5000, 'ELEVATED', 'OK'),
    (9.9e-7, 5001, 'SAFE', 'OK'),
    (1e-7, 999, 'CRITICAL', 'OK'),
    (1e-7, 1000, 'ELEVATED', 'OK'),
    (None, 800, 'CRITICAL', 'NEEDS_REVIEW'),
    (1e-5, None, 'ELEVATED', 'NEEDS_REVIEW'),
    (1e-7, None, 'UNKNOWN', 'NEEDS_REVIEW'),
    (None, 6000, 'UNKNOWN', 'NEEDS_REVIEW'),
    (None, None, 'UNKNOWN', 'NEEDS_REVIEW'),
])
def test_grade_table_from_plan(tier, distance, level, status):
    row = grade(tier, distance)
    assert (row['risk_level'], row['data_quality_status']) == (level, status)


@pytest.mark.parametrize('tier, conservative, level, status', [
    (1e-5, 3e-4, 'ELEVATED', 'NEEDS_REVIEW'),
    (1e-5, 5e-5, 'ELEVATED', 'OK'),
    (1e-7, 5e-6, 'SAFE', 'NEEDS_REVIEW'),
    (1e-7, 5e-7, 'SAFE', 'OK'),
])
def test_near_threshold_table_from_plan(tier, conservative, level, status):
    row = grade(tier, 6000, conservative)
    assert (row['risk_level'], row['data_quality_status']) == (level, status)


def test_grade_ignores_tiny_model_estimate():
    assert grade(2e-5, 6000, predicted=1e-9)['risk_level'] == 'ELEVATED'


def test_trigger_reason_quotes_latest_known_pc():
    assert 'Pc=2e-05' in grade(2e-5, 6000, predicted=1e-9)['trigger_reason']


def test_quality_reason_names_latest_pc_not_median():
    reason = grade(1e-5, 6000, 3e-4)['data_quality_reason']
    assert 'latest Pc=1e-05' in reason
    assert 'median' not in reason.lower()


def test_low_latest_pc_with_large_sigma_ratio_needs_review():
    row = grade(1e-7, 6000, sigma_ratio=THRESHOLD * 2)
    assert row['risk_level'] == 'SAFE'
    assert row['data_quality_status'] == 'NEEDS_REVIEW'
    assert 'sigma_ratio' in row['data_quality_reason']


def test_evidence_rule_follows_latest_pc_not_estimate():
    assert grade(1e-5, 6000, sigma_ratio=THRESHOLD * 2)['data_quality_status'] == 'OK'


def test_missing_sigma_ratio_needs_review():
    assert grade(1e-7, 6000, sigma_ratio=np.nan)['data_quality_status'] == 'NEEDS_REVIEW'


def test_infinite_sigma_ratio_at_zero_distance_needs_review():
    row = grade(1e-7, 0, sigma_ratio=np.inf)
    assert row['risk_level'] == 'CRITICAL'
    assert row['data_quality_status'] == 'NEEDS_REVIEW'


def test_infinite_values_elsewhere_are_still_rejected():
    with pytest.raises(ValueError):
        grade(1e-7, np.inf)


def test_negative_sigma_ratio_is_still_rejected():
    with pytest.raises(ValueError):
        grade(1e-7, 6000, sigma_ratio=-np.inf)


def test_tier_pc_stays_in_result():
    assert grade(2e-5, 6000)['tier_pc'] == 2e-5


def test_log_only_tier_input_is_converted_to_pc():
    frame = pd.DataFrame([{'tier_log10_pc': -5.0, 'predicted_pc': 1e-9, 'conservative_pc': 1e-9,
                           'miss_distance_m': 6000.0, 'sigma_ratio': 1e-3}])
    row = apply_risk_rules(frame, THRESHOLD).iloc[0]
    assert row['risk_level'] == 'ELEVATED'
    assert row['tier_pc'] == pytest.approx(1e-5)


def test_grades_real_ml_output_without_sigma_columns():
    messages = pd.DataFrame([{
        'event_id': 1, 'time_to_tca': 2.5, 'risk': -5.0, 'miss_distance': 6000.0,
        'relative_speed': 11000.0, 'relative_position_r': 1.0, 'relative_position_t': 2.0,
        'relative_position_n': 3.0, 'c_object_type': 'DEBRIS', 'c_sigma_r': 3.0, 't_sigma_r': 4.0,
        'max_risk_estimate': -4.0, 'F10': 80.0, 'AP': 12.0,
    }])
    samples = build_samples(messages, 'final', 2.0, labeled=False)
    probabilities = pd.DataFrame({'Safe': [0.5], 'Elevated': [0.3], 'Critical': [0.2]})
    predictions = combine_predictions(samples, [-9.0], [-8.0], probabilities, None, 0.4)

    row = apply_risk_rules(predictions, THRESHOLD).iloc[0]

    assert row['risk_level'] == 'ELEVATED'
    assert row['tier_pc'] == pytest.approx(1e-5)
    assert math.isclose(row['predicted_pc'], 1e-9, rel_tol=1e-9)
    assert row['data_quality_status'] == 'OK'
