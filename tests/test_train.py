import json

import numpy as np
import pandas as pd
import pytest

from train import (
    alert_report, choose_alpha, evaluate, load_or_create_split, rank_scores, recall_at_fpr, split_events,
)


def fake_samples(events=1000, positives=100):
    y = np.full(events, -12.0)
    y[:positives] = -5.0
    return pd.DataFrame({'event_id': np.arange(events), 'y': y})


def test_split_puts_each_event_in_exactly_one_group():
    split = split_events(fake_samples(), seed=42)
    train, val, holdout = (set(split[k]) for k in ('train', 'validation', 'holdout'))
    assert not train & val
    assert not train & holdout
    assert not val & holdout
    assert train | val | holdout == set(range(1000))


def test_split_sizes_are_70_15_15():
    split = split_events(fake_samples(), seed=42)
    assert [len(split[k]) for k in ('train', 'validation', 'holdout')] == [700, 150, 150]


def test_split_spreads_high_risk_events_across_groups():
    split = split_events(fake_samples(), seed=42)
    high_risk = set(range(100))
    assert [len(high_risk & set(split[k])) for k in ('train', 'validation', 'holdout')] == [70, 15, 15]


def test_split_is_reproducible_for_same_seed():
    first = split_events(fake_samples(), seed=42)
    second = split_events(fake_samples(), seed=42)
    assert list(first['holdout']) == list(second['holdout'])


def test_split_falls_back_when_high_risk_events_are_too_rare():
    split = split_events(fake_samples(events=100, positives=2), seed=42)
    assert sum(len(split[k]) for k in ('train', 'validation', 'holdout')) == 100


def test_recall_at_fpr_alerts_only_within_allowed_false_alarms():
    positive = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    scores = np.array([10, 7, 2, 9, 8, 6, 5, 4, 3, 1, 0, -1], dtype=float)
    result = recall_at_fpr(positive, scores, target_fpr=2 / 9)
    assert result['true_positives'] == 2
    assert result['false_positives'] == 2
    assert result['recall'] == 2 / 3


def test_recall_at_fpr_never_exceeds_target_when_scores_tie():
    positive = np.array([1, 0, 0, 0], dtype=bool)
    scores = np.array([5.0, 5.0, 5.0, 1.0])
    result = recall_at_fpr(positive, scores, target_fpr=1 / 3)
    assert result['false_positives'] <= 1


def test_recall_at_fpr_reports_none_without_positive_events():
    positive = np.zeros(4, dtype=bool)
    result = recall_at_fpr(positive, np.arange(4.0), target_fpr=0.5)
    assert result['recall'] is None


def test_choose_alpha_keeps_default_when_gain_is_within_noise():
    assert choose_alpha({0.8: 10, 0.9: 10, 0.95: 11}) == 0.9


def test_choose_alpha_switches_for_clear_gain():
    assert choose_alpha({0.8: 10, 0.9: 10, 0.95: 12}) == 0.95


def test_alert_report_counts_and_scores():
    report = alert_report([True, True, False, False], [True, False, True, False])
    assert report == {'precision': 0.5, 'recall': 0.5, 'f2': 0.5,
                      'true_positives': 1, 'false_positives': 1, 'false_negatives': 1}


def test_alert_report_without_alerts_has_no_precision():
    report = alert_report([True, False], [False, False])
    assert report['precision'] is None
    assert report['recall'] == 0.0
    assert report['f2'] is None


def test_alert_report_without_positive_events_has_no_recall():
    assert alert_report([False, False], [True, False])['recall'] is None


def sample_report():
    return evaluate(
        y_raw=[-30.0, -12.0, -5.0, -3.0],
        baseline=[-20.0, -5.0, -7.0, -3.0],
        value_scores={'model': [-10.0, -10.0, -6.0, -4.0]},
        rank_scores={'ranker': [0.1, 0.2, 0.9, 0.8]},
        constant=-8.0,
    )


def test_evaluate_clips_labels_and_baseline_before_regression_errors():
    regression = sample_report()['regression']
    assert regression['latest_risk']['rmse'] == pytest.approx(7.25 ** 0.5)
    assert regression['latest_risk']['mae'] == 1.75
    assert regression['constant']['rmse'] == pytest.approx(10.5 ** 0.5)
    assert regression['constant']['mae'] == 3.0


def test_evaluate_counts_high_risk_from_unclipped_labels():
    level = sample_report()['levels']['-6.0']
    assert level['positive_events'] == 2
    assert level['negative_events'] == 2


def test_evaluate_takes_false_alarm_rate_from_baseline():
    assert sample_report()['levels']['-6.0']['baseline_fpr'] == 0.5
    assert sample_report()['levels']['-4.0']['baseline_fpr'] == 0.0


def test_evaluate_applies_level_as_threshold_to_value_scores_only():
    thresholds = sample_report()['levels']['-6.0']['at_threshold']
    assert thresholds['model']['true_positives'] == 2
    assert thresholds['model']['false_positives'] == 0
    assert 'ranker' not in thresholds


def test_evaluate_compares_every_score_at_baseline_false_alarm_rate():
    matched = sample_report()['levels']['-6.0']['at_matched_fpr']
    assert matched['ranker']['true_positives'] == 2
    assert matched['ranker']['false_positives'] == 1
    assert matched['latest_risk']['true_positives'] == 2


def test_evaluate_reports_recall_as_unknown_without_positive_events():
    report = evaluate([-12.0, -11.0], [-9.0, -8.0], {'model': [-9.0, -9.0]}, {'ranker': [0.2, 0.1]}, -10.0)
    level = report['levels']['-4.0']
    assert level['positive_events'] == 0
    assert level['at_threshold']['model']['recall'] is None
    assert level['at_matched_fpr']['ranker']['recall'] is None


def test_rank_scores_compare_the_same_conservative_values_as_output():
    scores = rank_scores(np.array([-5.0, -8.0]), {0.9: np.array([-6.0, -7.0])}, np.array([0.7, 0.2]))
    assert scores['conservative_0.9'].tolist() == [-5.0, -7.0]
    assert scores['classifier'].tolist() == [0.7, 0.2]


def test_existing_split_list_is_reused_without_rewriting(tmp_path):
    path = tmp_path / 'split_events.json'
    saved = {'train': [3, 1], 'validation': [2], 'holdout': [0]}
    path.write_text(json.dumps(saved), encoding='utf-8')
    before = path.read_bytes()

    split = load_or_create_split(path, fake_samples(events=4, positives=1))

    assert split == saved
    assert path.read_bytes() == before


def test_missing_split_list_is_created_once(tmp_path):
    path = tmp_path / 'split_events.json'
    samples = fake_samples()
    first = load_or_create_split(path, samples)
    assert path.exists()
    assert load_or_create_split(path, samples) == {k: [int(i) for i in v] for k, v in first.items()}


@pytest.mark.parametrize('saved', [
    {'train': [0, 1], 'validation': [2], 'holdout': [3, 4]},
    {'train': [0, 1], 'validation': [2], 'holdout': []},
    {'train': [0, 1], 'validation': [2], 'holdout': [2, 3]},
    {'train': [0, 1], 'validation': [2, 3]},
])
def test_split_list_that_does_not_match_events_is_rejected(tmp_path, saved):
    path = tmp_path / 'split_events.json'
    path.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError):
        load_or_create_split(path, fake_samples(events=4, positives=1))
