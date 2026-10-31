import math

import pandas as pd
import pytest

from ml import build_samples, combine_predictions, feature_columns


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
        message(7, 5.0, -20.0, 9000.0),
        message(7, 3.0, -12.0, 7000.0),
        message(7, 2.5, -9.0, 6000.0),
        message(7, 1.5, -6.5, 3000.0),
        message(7, 0.5, -3.5, 500.0),
    ])


def test_final_input_is_latest_message_at_or_before_cutoff():
    sample = build_samples(typical_event(), 'final', 2.0).iloc[0]
    assert sample['time_to_tca'] == 2.5
    assert sample['risk'] == -9.0
    assert sample['miss_distance'] == 6000.0


def test_final_label_is_last_message_of_event():
    sample = build_samples(typical_event(), 'final', 2.0).iloc[0]
    assert sample['y'] == -3.5


def test_final_features_ignore_messages_after_cutoff():
    changed = typical_event()
    changed.loc[changed['time_to_tca'] == 1.5, 'risk'] = -1.0
    changed.loc[changed['time_to_tca'] == 1.5, 'miss_distance'] = 1.0
    columns = feature_columns('final')

    original = build_samples(typical_event(), 'final', 2.0)[columns]
    assert build_samples(changed, 'final', 2.0)[columns].equals(original)


def test_final_change_features_use_last_two_input_messages():
    sample = build_samples(typical_event(), 'final', 2.0).iloc[0]
    assert sample['risk_delta'] == 3.0
    assert sample['miss_distance_delta'] == -1000.0
    assert sample['message_count'] == 3


def test_final_single_input_message_leaves_change_features_missing():
    df = pd.DataFrame([message(1, 3.0, -9.0), message(1, 0.5, -7.0)])
    sample = build_samples(df, 'final', 2.0).iloc[0]
    assert math.isnan(sample['risk_delta'])
    assert math.isnan(sample['miss_distance_delta'])
    assert sample['message_count'] == 1


def test_final_handles_unsorted_messages():
    shuffled = typical_event().iloc[[3, 0, 4, 2, 1]]
    sample = build_samples(shuffled, 'final', 2.0).iloc[0]
    assert sample['time_to_tca'] == 2.5
    assert sample['y'] == -3.5


def test_final_drops_event_without_message_near_tca():
    df = pd.DataFrame([message(1, 3.0, -9.0), message(1, 1.5, -7.0)])
    assert build_samples(df, 'final', 2.0).empty


def test_final_drops_event_without_input_message():
    df = pd.DataFrame([message(1, 1.5, -9.0), message(1, 0.5, -7.0)])
    assert build_samples(df, 'final', 2.0).empty


def test_final_unlabeled_data_keeps_events_and_leaves_label_missing():
    df = pd.DataFrame([message(1, 4.0, -9.0), message(1, 2.1, -8.0)])
    samples = build_samples(df, 'final', 2.0, labeled=False)
    assert len(samples) == 1
    assert samples.iloc[0]['risk'] == -8.0
    assert math.isnan(samples.iloc[0]['y'])


def test_final_baseline_is_latest_known_risk():
    sample = build_samples(typical_event(), 'final', 2.0).iloc[0]
    assert sample['baseline_score'] == -9.0


def test_final_keeps_source_row_of_input_message():
    sample = build_samples(typical_event(), 'final', 2.0).iloc[0]
    assert sample['input_record_id'] == 2


def test_final_features_include_known_risk_but_current_features_do_not():
    assert 'risk' in feature_columns('final')
    assert 'risk' not in feature_columns('current')


def test_current_uses_each_row_with_its_own_label():
    samples = build_samples(typical_event(), 'current', 2.0)
    assert len(samples) == 5
    assert samples['y'].tolist() == [-20.0, -12.0, -9.0, -6.5, -3.5]
    assert samples['baseline_score'].tolist() == [-5.0] * 5


def test_sigma_ratio_combines_both_radial_sigmas():
    sample = build_samples(typical_event(), 'final', 2.0).iloc[0]
    assert sample['sigma_ratio'] == pytest.approx(5.0 / 6000.0)


def test_sigma_ratio_is_infinite_at_zero_distance():
    df = pd.DataFrame([message(1, 3.0, -9.0, 0.0), message(1, 0.5, -7.0)])
    assert math.isinf(build_samples(df, 'final', 2.0).iloc[0]['sigma_ratio'])


def test_missing_object_type_becomes_unknown():
    df = pd.DataFrame([
        message(1, 3.0, -9.0, c_object_type=None),
        message(1, 0.5, -7.0),
    ])
    assert build_samples(df, 'final', 2.0).iloc[0]['c_object_type'] == 'UNKNOWN'


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        build_samples(typical_event(), 'tomorrow', 2.0)


def test_conservative_estimate_never_falls_below_point_estimate():
    samples = build_samples(typical_event(), 'final', 2.0)
    result = combine_predictions(samples, [-5.0], [-6.0], [0.7]).iloc[0]
    assert result['predicted_log10_pc'] == -5.0
    assert result['conservative_log10_pc'] == -5.0


def test_final_tier_uses_latest_known_risk_not_model_estimate():
    samples = build_samples(typical_event(), 'final', 2.0)
    result = combine_predictions(samples, [-8.0], [-5.0], [0.7]).iloc[0]
    assert result['tier_log10_pc'] == -9.0
    assert result['tier_pc'] == pytest.approx(1e-9)


def test_current_tier_uses_model_estimate_because_risk_is_the_label():
    samples = build_samples(typical_event().head(1), 'current', 2.0)
    result = combine_predictions(samples, [-8.0], [-5.0], [0.7]).iloc[0]
    assert result['tier_log10_pc'] == -8.0


def test_wide_real_shaped_frame_builds_without_performance_warning(recwarn):
    wide = typical_event()
    for index in range(120):
        wide[f'extra_{index}'] = 1.0
    recwarn.clear()
    build_samples(wide, 'final', 2.0)
    build_samples(wide, 'current', 2.0)
    assert [w for w in recwarn.list if 'fragmented' in str(w.message)] == []


def test_predictions_convert_log_values_and_units():
    samples = build_samples(typical_event(), 'final', 2.0)
    result = combine_predictions(samples, [-5.0], [-4.0], [0.7]).iloc[0]
    assert result['predicted_pc'] == pytest.approx(1e-5)
    assert result['conservative_pc'] == pytest.approx(1e-4)
    assert result['time_to_tca_hours'] == 60.0
    assert result['miss_distance_m'] == 6000.0
    assert result['alert_score'] == 0.7
