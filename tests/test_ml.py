import math

import numpy as np
import pandas as pd
import pytest

from ml import (
    TIERS, alert_scores, build_samples, combine_predictions, feature_columns, predicted_tiers, risk_tier,
    tier_probabilities,
)


def message(event_id, time_to_tca, risk, miss_distance=8000.0, **overrides):
    row = {
        'event_id': event_id,
        'time_to_tca': time_to_tca,
        'risk': risk,
        'miss_distance': miss_distance,
        'relative_speed': 11000.0,
        'relative_position_r': 10.0,
        'relative_position_t': 20.0,
        'relative_position_n': 30.0,
        'c_object_type': 'DEBRIS',
        'c_sigma_r': 3.0,
        't_sigma_r': 4.0,
        'max_risk_estimate': -5.0,
        'F10': 80.0,
        'AP': 12.0,
    }
    row.update(overrides)
    return row


def typical_event():
    return pd.DataFrame([
        message(7, 5.0, -20.0, 9000.0, c_sigma_r=9.0),
        message(7, 3.0, -12.0, 7000.0, c_sigma_r=5.0),
        message(7, 2.5, -9.0, 6000.0, c_sigma_r=3.0),
        message(7, 1.5, -6.5, 3000.0),
        message(7, 0.5, -3.5, 500.0),
    ])


def probabilities(safe, elevated, critical):
    return pd.DataFrame({'Safe': [safe], 'Elevated': [elevated], 'Critical': [critical]})


@pytest.mark.parametrize('risk, miss, tier', [
    (-3.9, 9000, 'Critical'),
    (-4.0, 9000, 'Critical'),
    (-20.0, 999, 'Critical'),
    (-4.1, 9000, 'Elevated'),
    (-6.0, 9000, 'Elevated'),
    (-20.0, 1000, 'Elevated'),
    (-20.0, 5000, 'Elevated'),
    (-6.1, 5001, 'Safe'),
    (-30.0, 20000, 'Safe'),
])
def test_risk_tier_follows_official_rules(risk, miss, tier):
    assert risk_tier(np.array([risk]), np.array([miss]))[0] == tier


def test_final_input_is_latest_message_at_or_before_cutoff():
    sample = build_samples(typical_event(), 'final', 2.0).iloc[0]
    assert sample['time_to_tca'] == 2.5
    assert sample['risk'] == -9.0
    assert sample['miss_distance'] == 6000.0


def test_final_label_comes_from_last_message_of_event():
    sample = build_samples(typical_event(), 'final', 2.0).iloc[0]
    assert sample['y'] == -3.5
    assert sample['final_miss_distance'] == 500.0
    assert sample['final_tier'] == 'Critical'


def test_current_tier_comes_from_latest_input_message():
    assert build_samples(typical_event(), 'final', 2.0).iloc[0]['current_tier'] == 'Safe'


def test_features_ignore_messages_after_cutoff():
    changed = typical_event()
    changed.loc[changed['time_to_tca'] == 1.5, ['risk', 'miss_distance']] = [-1.0, 1.0]
    columns = feature_columns('final', history=True)
    original = build_samples(typical_event(), 'final', 2.0)[columns]
    assert build_samples(changed, 'final', 2.0)[columns].equals(original)


def test_change_features_use_last_two_input_messages():
    sample = build_samples(typical_event(), 'final', 2.0).iloc[0]
    assert sample['risk_delta'] == 3.0
    assert sample['miss_distance_delta'] == -1000.0
    assert sample['c_sigma_r_delta'] == -2.0
    assert sample['message_count'] == 3


def test_history_summaries_cover_all_input_messages_only():
    sample = build_samples(typical_event(), 'final', 2.0).iloc[0]
    assert sample['risk_max'] == -9.0
    assert sample['miss_distance_min'] == 6000.0
    assert sample['risk_std'] == pytest.approx(np.std([-20.0, -12.0, -9.0], ddof=1))


def test_history_slope_is_change_per_day_toward_tca():
    df = pd.DataFrame([message(1, 3.0, -12.0, 7000.0), message(1, 2.5, -9.0, 6000.0), message(1, 0.5, -7.0)])
    sample = build_samples(df, 'final', 2.0).iloc[0]
    assert sample['risk_slope'] == pytest.approx(6.0)
    assert sample['miss_distance_slope'] == pytest.approx(-2000.0)


def test_single_input_message_leaves_history_features_missing():
    df = pd.DataFrame([message(1, 3.0, -9.0), message(1, 0.5, -7.0)])
    sample = build_samples(df, 'final', 2.0).iloc[0]
    for column in ('risk_delta', 'miss_distance_delta', 'risk_std', 'risk_slope', 'miss_distance_slope'):
        assert math.isnan(sample[column])
    assert sample['risk_max'] == -9.0
    assert sample['message_count'] == 1


def test_handles_unsorted_messages():
    sample = build_samples(typical_event().iloc[[3, 0, 4, 2, 1]], 'final', 2.0).iloc[0]
    assert sample['time_to_tca'] == 2.5
    assert sample['y'] == -3.5


def test_event_ending_after_one_day_is_kept_with_later_messages():
    df = pd.DataFrame([message(1, 3.0, -9.0), message(1, 1.5, -7.0)])
    sample = build_samples(df, 'final', 2.0).iloc[0]
    assert sample['y'] == -7.0
    assert sample['has_future']


def test_event_whose_last_message_is_the_input_is_marked_without_future():
    df = pd.DataFrame([message(1, 5.0, -12.0), message(1, 3.57, -9.0)])
    sample = build_samples(df, 'final', 2.0).iloc[0]
    assert sample['y'] == -9.0
    assert not sample['has_future']
    assert sample['current_tier'] == sample['final_tier']


def test_negative_time_message_counts_as_final():
    df = pd.DataFrame([message(1, 3.0, -9.0), message(1, -0.1, -7.0)])
    assert build_samples(df, 'final', 2.0).iloc[0]['y'] == -7.0


def test_event_without_input_message_is_dropped():
    df = pd.DataFrame([message(1, 1.5, -9.0), message(1, 0.5, -7.0)])
    assert build_samples(df, 'final', 2.0).empty


def test_unlabeled_data_keeps_events_and_leaves_labels_missing():
    df = pd.DataFrame([message(1, 4.0, -9.0), message(1, 2.1, -8.0)])
    sample = build_samples(df, 'final', 2.0, labeled=False).iloc[0]
    assert sample['risk'] == -8.0
    assert math.isnan(sample['y'])


def test_baseline_score_keeps_tier_order_and_prefers_closer_events():
    df = pd.DataFrame([
        message(1, 3.0, -20.0, 900.0), message(2, 3.0, -20.0, 400.0),
        message(3, 3.0, -20.0, 3000.0), message(4, 3.0, -20.0, 20000.0),
    ])
    scores = build_samples(df, 'final', 2.0, labeled=False).set_index('event_id')['baseline_score']
    assert scores[2] > scores[1] > scores[3] > scores[4]


def test_current_task_keeps_each_row_without_needing_labels():
    samples = build_samples(typical_event().drop(columns='risk'), 'current', 2.0, labeled=False)
    assert len(samples) == 5
    assert 'risk' not in feature_columns('current')


def test_current_task_can_be_combined_without_current_tier_rule():
    samples = build_samples(typical_event().drop(columns='risk'), 'current', 2.0, labeled=False)
    frame = pd.concat([probabilities(0.5, 0.3, 0.2)] * 5, ignore_index=True)
    result = combine_predictions(samples, [-5.0] * 5, [-4.0] * 5, frame, None, 0.4)
    assert result['tier_log10_pc'].tolist() == [-5.0] * 5
    assert result['current_tier'].tolist() == ['Unknown'] * 5


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        build_samples(typical_event(), 'tomorrow', 2.0)


def test_keeps_source_row_of_input_message():
    assert build_samples(typical_event(), 'final', 2.0).iloc[0]['input_record_id'] == 2


def test_history_features_are_optional():
    assert 'risk_slope' in feature_columns('final', history=True)
    assert 'risk_slope' not in feature_columns('final', history=False)
    assert 'risk' in feature_columns('final', history=False)


def test_sigma_ratio_combines_both_radial_sigmas():
    assert build_samples(typical_event(), 'final', 2.0).iloc[0]['sigma_ratio'] == pytest.approx(5.0 / 6000.0)


def test_sigma_ratio_is_infinite_at_zero_distance():
    df = pd.DataFrame([message(1, 3.0, -9.0, 0.0), message(1, 0.5, -7.0)])
    assert math.isinf(build_samples(df, 'final', 2.0).iloc[0]['sigma_ratio'])


def test_missing_object_type_becomes_unknown():
    df = pd.DataFrame([message(1, 3.0, -9.0, c_object_type=None), message(1, 0.5, -7.0)])
    assert build_samples(df, 'final', 2.0).iloc[0]['c_object_type'] == 'UNKNOWN'


def test_wide_real_shaped_frame_builds_without_performance_warning(recwarn):
    wide = typical_event()
    for index in range(120):
        wide[f'extra_{index}'] = 1.0
    recwarn.clear()
    build_samples(wide, 'final', 2.0)
    assert [w for w in recwarn.list if 'fragmented' in str(w.message)] == []


def test_fallback_keeps_currently_critical_event_above_low_model_score():
    scores = alert_scores(np.array([0.1]), np.array(['Critical']), {'Critical': 0.8, 'Elevated': 0.05, 'Safe': 0.0})
    assert scores[0] >= 0.8


def test_fallback_keeps_high_model_score():
    scores = alert_scores(np.array([0.9]), np.array(['Safe']), {'Critical': 0.8, 'Elevated': 0.05, 'Safe': 0.0})
    assert scores[0] >= 0.9


def test_fallback_still_orders_events_with_same_floor_by_model():
    rates = {'Critical': 0.8, 'Elevated': 0.05, 'Safe': 0.0}
    scores = alert_scores(np.array([0.1, 0.2]), np.array(['Critical', 'Critical']), rates)
    assert scores[1] > scores[0]


def test_without_rates_alert_score_is_model_probability():
    assert alert_scores(np.array([0.3]), np.array(['Critical']), None)[0] == 0.3


def test_predicted_tier_is_critical_above_threshold():
    assert predicted_tiers(probabilities(0.7, 0.2, 0.1), np.array([0.5]), 0.4)[0] == 'Critical'


def test_predicted_tier_otherwise_picks_more_likely_lower_tier():
    assert predicted_tiers(probabilities(0.3, 0.6, 0.1), np.array([0.1]), 0.4)[0] == 'Elevated'
    assert predicted_tiers(probabilities(0.6, 0.3, 0.1), np.array([0.1]), 0.4)[0] == 'Safe'


def test_tier_order_is_safe_to_critical():
    assert TIERS == ('Safe', 'Elevated', 'Critical')


def test_combined_output_converts_units_and_keeps_all_columns():
    samples = build_samples(typical_event(), 'final', 2.0)
    result = combine_predictions(samples, [-5.0], [-4.0], probabilities(0.2, 0.3, 0.5), None, 0.4).iloc[0]
    assert result['predicted_pc'] == pytest.approx(1e-5)
    assert result['conservative_pc'] == pytest.approx(1e-4)
    assert result['time_to_tca_hours'] == 60.0
    assert result['miss_distance_m'] == 6000.0
    assert result['tier_log10_pc'] == -9.0
    assert result['tier_pc'] == pytest.approx(1e-9)
    assert result['current_tier'] == 'Safe'
    assert result['predicted_final_tier'] == 'Critical'
    assert (result['p_safe'], result['p_elevated'], result['p_critical']) == (0.2, 0.3, 0.5)
    assert result['alert_score'] == 0.5


def test_conservative_estimate_never_falls_below_point_estimate():
    samples = build_samples(typical_event(), 'final', 2.0)
    result = combine_predictions(samples, [-5.0], [-6.0], probabilities(0.8, 0.1, 0.1), None, 0.4).iloc[0]
    assert result['conservative_log10_pc'] == -5.0


def test_output_uses_fallback_score_for_currently_critical_event():
    df = pd.DataFrame([message(1, 3.0, -9.0, 500.0), message(1, 0.5, -7.0)])
    samples = build_samples(df, 'final', 2.0)
    rates = {'Critical': 0.8, 'Elevated': 0.05, 'Safe': 0.0}
    result = combine_predictions(samples, [-5.0], [-4.0], probabilities(0.8, 0.1, 0.1), rates, 0.4).iloc[0]
    assert result['alert_score'] >= 0.8
    assert result['predicted_final_tier'] == 'Critical'


class FakeClassifier:
    classes_ = np.array(['Elevated', 'Safe', 'Critical'])

    def predict_proba(self, features):
        return np.array([[0.2, 0.7, 0.1]])


def test_probabilities_follow_tier_names_not_model_column_order():
    result = tier_probabilities(FakeClassifier(), None)
    assert list(result.columns) == ['Safe', 'Elevated', 'Critical']
    assert result.iloc[0].tolist() == [0.7, 0.2, 0.1]
