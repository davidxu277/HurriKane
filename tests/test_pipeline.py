"""用合成数据跑完整流程，不读取真实数据。"""
import json
import math
import sys

import numpy as np
import pandas as pd
import pytest

import train
from ml import TIERS, build_samples, feature_columns, load_bundle, predict_batch
from validation import validate_input

ITERATIONS = '20'
PROFILES = (
    ((-14.0, -12.0, -11.0), -30.0, 9000.0),   # 全程安全
    ((-14.0, -12.0, -7.0), -5.0, 3000.0),     # 最后升到 Elevated
    ((-12.0, -9.0, -7.0), -3.5, 500.0),       # 最后升到 Critical
)


def synthetic_messages(events=300, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for event_id in range(events):
        risks, final_risk, final_miss = PROFILES[event_id % 3]
        stops_early = event_id % 5 == 4
        times = (5.0, 3.0, 2.2) if stops_early else (5.0, 3.0, 2.2, 0.5)
        values = risks if stops_early else (*risks, final_risk)
        for days, risk in zip(times, values):
            is_last = days == 0.5
            rows.append({
                'event_id': event_id,
                'time_to_tca': days,
                'risk': risk,
                'miss_distance': final_miss if is_last else float(rng.uniform(5500, 9000)),
                'relative_speed': float(rng.uniform(100, 14000)),
                'relative_position_r': float(rng.normal()),
                'relative_position_t': float(rng.normal()),
                'relative_position_n': float(rng.normal()),
                'c_object_type': 'DEBRIS' if event_id % 2 else 'UNKNOWN',
                'c_sigma_r': float(rng.uniform(1, 50)),
                't_sigma_r': float(rng.uniform(1, 50)),
                'max_risk_estimate': risk + 1,
                'F10': 80.0,
                'AP': 12.0,
            })
    return pd.DataFrame(rows)


def run_training(monkeypatch, csv_path, models, reports, *extra):
    argv = ['train.py', '--train', str(csv_path), '--iterations', ITERATIONS,
            '--models', str(models), '--reports', str(reports), *extra]
    monkeypatch.setattr(sys, 'argv', argv)
    train.main()


@pytest.fixture
def trained(tmp_path, monkeypatch):
    csv_path = tmp_path / 'train.csv'
    synthetic_messages().to_csv(csv_path, index=False)
    run_training(monkeypatch, csv_path, tmp_path / 'models', tmp_path / 'reports')
    return tmp_path


def unlabeled_events():
    frame = synthetic_messages(events=4, seed=11)
    return frame[frame['time_to_tca'] >= 2].reset_index(drop=True)


def test_predict_batch_keeps_every_event_without_near_tca_message(trained):
    result = predict_batch(unlabeled_events(), load_bundle(trained / 'models'))
    assert sorted(result['event_id']) == [0, 1, 2, 3]


def test_predict_batch_tiers_on_latest_known_risk(trained):
    result = predict_batch(unlabeled_events(), load_bundle(trained / 'models')).set_index('event_id')
    assert result.loc[0, 'tier_log10_pc'] == -11.0
    assert result.loc[1, 'tier_log10_pc'] == -7.0


def test_predict_batch_output_is_complete_and_ordered(trained):
    result = predict_batch(unlabeled_events(), load_bundle(trained / 'models'))
    columns = ['predicted_pc', 'conservative_pc', 'alert_score', 'tier_pc', 'p_critical']
    assert not result[columns].isna().any().any()
    assert (result['conservative_log10_pc'] >= result['predicted_log10_pc']).all()
    assert result['alert_score'].between(0, 1.01).all()
    assert set(result['predicted_final_tier']) <= set(TIERS)


def test_tier_probabilities_sum_to_one(trained):
    result = predict_batch(unlabeled_events(), load_bundle(trained / 'models'))
    totals = result[['p_safe', 'p_elevated', 'p_critical']].sum(axis=1)
    assert np.allclose(totals, 1.0)


def test_predict_batch_leaves_ratio_unknown_when_sigma_is_missing(trained):
    frame = unlabeled_events()
    frame.loc[frame['event_id'] == 2, 'c_sigma_r'] = np.nan
    result = predict_batch(frame, load_bundle(trained / 'models')).set_index('event_id')
    assert math.isnan(result.loc[2, 'sigma_ratio'])
    assert not math.isnan(result.loc[2, 'alert_score'])


def test_metrics_compare_all_variants_on_events_with_later_messages(trained):
    metrics = json.loads((trained / 'reports' / 'metrics.json').read_text(encoding='utf-8'))
    assert set(metrics['validation']['variants']) == {'baseline', *train.VARIANTS}
    assert metrics['validation']['tiers']['future']['events'] < metrics['validation']['tiers']['all']['events']
    assert metrics['metadata']['variant'] in metrics['validation']['variants']


def test_validation_accepts_saved_metadata_with_history_features(trained):
    metadata = load_bundle(trained / 'models')['metadata']
    frame = unlabeled_events()
    valid, issues = validate_input(frame, metadata)
    assert len(valid) == len(frame)
    assert issues.empty


def test_training_without_flag_never_writes_holdout_results(trained):
    assert not (trained / 'reports' / 'holdout_predictions.csv').exists()
    assert 'holdout' not in json.loads((trained / 'reports' / 'metrics.json').read_text(encoding='utf-8'))


def test_holdout_flag_leaves_models_and_validation_results_unchanged(trained, monkeypatch):
    run_training(monkeypatch, trained / 'train.csv', trained / 'models2', trained / 'reports2',
                 '--evaluate-holdout')
    # 文件字节不能比，CatBoost 每次都会写入随机的 model_guid
    first_bundle, second_bundle = load_bundle(trained / 'models'), load_bundle(trained / 'models2')
    for name in ('estimate', 'conservative', 'classifier'):
        assert first_bundle[name] == second_bundle[name]
    first = (trained / 'reports' / 'validation_predictions.csv').read_bytes()
    assert first == (trained / 'reports2' / 'validation_predictions.csv').read_bytes()
    assert (trained / 'reports2' / 'holdout_predictions.csv').exists()


def test_training_reuses_saved_split_list(trained, monkeypatch):
    split_path = trained / 'reports' / 'split_events.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    split['holdout'][0], split['train'][0] = split['train'][0], split['holdout'][0]
    split_path.write_text(json.dumps(split), encoding='utf-8')
    before = split_path.read_bytes()

    run_training(monkeypatch, trained / 'train.csv', trained / 'models', trained / 'reports')

    assert split_path.read_bytes() == before
    validation = pd.read_csv(trained / 'reports' / 'validation_predictions.csv')
    assert sorted(validation['event_id']) == sorted(split['validation'])


def read_metrics(root):
    return json.loads((root / 'reports' / 'metrics.json').read_text(encoding='utf-8'))


def test_future_only_variant_excludes_events_without_later_messages(tmp_path, monkeypatch):
    csv_path = tmp_path / 'train.csv'
    synthetic_messages().to_csv(csv_path, index=False)
    monkeypatch.setattr(train, 'choose_variant', lambda results: 'D')
    run_training(monkeypatch, csv_path, tmp_path / 'models', tmp_path / 'reports')
    metrics = read_metrics(tmp_path)
    split = json.loads((tmp_path / 'reports' / 'split_events.json').read_text(encoding='utf-8'))
    # 合成数据里 event_id % 5 == 4 的事件没有后续消息
    assert metrics['metadata']['training_events'] == sum(1 for e in split['train'] if e % 5 != 4)


def test_all_events_variant_trains_on_events_without_later_messages_too(tmp_path, monkeypatch):
    csv_path = tmp_path / 'train.csv'
    synthetic_messages().to_csv(csv_path, index=False)
    monkeypatch.setattr(train, 'choose_variant', lambda results: 'D_all_events')
    run_training(monkeypatch, csv_path, tmp_path / 'models', tmp_path / 'reports')
    metrics = read_metrics(tmp_path)
    split = json.loads((tmp_path / 'reports' / 'split_events.json').read_text(encoding='utf-8'))
    assert metrics['metadata']['training_events'] == len(split['train'])


def test_variants_are_compared_only_on_events_with_later_messages(trained):
    validation = read_metrics(trained)['validation']
    baseline = validation['variants']['baseline']
    assert baseline['true_positives'] + baseline['false_negatives'] == \
        validation['tiers']['future']['critical']['positive_events']


def test_fallback_variants_use_rates_and_plain_variant_does_not(tmp_path, monkeypatch):
    samples = build_samples(synthetic_messages(), 'final', 2.0)
    future = samples[samples['has_future']]
    seen = []
    real = train.alert_scores

    def spy(p_critical, current_tier, rates):
        seen.append(rates)
        return real(p_critical, current_tier, rates)

    monkeypatch.setattr(train, 'alert_scores', spy)
    rates = train.critical_rates(future)
    train.compare_variants(samples, future, samples, rates, 5)
    assert seen == [None] + [rates] * (len(train.VARIANTS) - 1)


def test_chosen_fallback_is_saved_and_applied_at_inference(tmp_path, monkeypatch):
    csv_path = tmp_path / 'train.csv'
    synthetic_messages().to_csv(csv_path, index=False)
    monkeypatch.setattr(train, 'choose_variant', lambda results: 'C')
    run_training(monkeypatch, csv_path, tmp_path / 'models', tmp_path / 'reports')
    bundle = load_bundle(tmp_path / 'models')
    rates = bundle['metadata']['critical_rates']
    assert set(rates) == set(TIERS)

    frame = unlabeled_events()
    frame.loc[frame['event_id'] == 0, 'miss_distance'] = 500.0
    event = predict_batch(frame, bundle).set_index('event_id').loc[0]
    assert event['current_tier'] == 'Critical'
    assert event['alert_score'] >= rates['Critical']


def test_validation_accepts_every_history_feature():
    metadata = {'task': 'final', 'features': feature_columns('final', history=True)}
    frame = unlabeled_events()
    valid, issues = validate_input(frame, metadata)
    assert len(valid) == len(frame)
    assert issues.empty


def test_main_compares_variants_only_on_events_with_later_messages(tmp_path, monkeypatch):
    csv_path = tmp_path / 'train.csv'
    synthetic_messages().to_csv(csv_path, index=False)
    seen = {}
    real = train.compare_variants

    def spy(train_all, train_future, validation, rates, iterations):
        seen.update(train_all=train_all, train_future=train_future, validation=validation)
        return real(train_all, train_future, validation, rates, iterations)

    monkeypatch.setattr(train, 'compare_variants', spy)
    run_training(monkeypatch, csv_path, tmp_path / 'models', tmp_path / 'reports')

    assert not seen['validation']['has_future'].all()
    assert seen['train_future']['has_future'].all()
    assert not seen['train_all']['has_future'].all()


def test_variants_report_accuracy_on_both_event_groups(trained):
    variants = read_metrics(trained)['validation']['variants']
    for result in variants.values():
        assert 0 <= result['accuracy_future'] <= 1
        assert 0 <= result['accuracy_all'] <= 1
        assert 'critical_true_positives' in result


def test_chosen_stale_rule_is_saved_and_applied_at_inference(tmp_path, monkeypatch):
    csv_path = tmp_path / 'train.csv'
    synthetic_messages().to_csv(csv_path, index=False)
    monkeypatch.setattr(train, 'choose_variant', lambda results: 'D_stale')
    run_training(monkeypatch, csv_path, tmp_path / 'models', tmp_path / 'reports')
    bundle = load_bundle(tmp_path / 'models')
    assert bundle['metadata']['stale_days'] == train.STALE_DAYS

    frame = unlabeled_events()
    frame = frame[frame['time_to_tca'] >= 3.0]
    result = predict_batch(frame, bundle)
    assert (result['predicted_final_tier'] == result['current_tier']).all()


def test_training_writes_input_issues(trained):
    assert (trained / 'reports' / 'train_input_issues.csv').exists()


def test_training_stops_on_invalid_input(tmp_path, monkeypatch):
    csv_path = tmp_path / 'train.csv'
    frame = synthetic_messages()
    frame.loc[0, 'miss_distance'] = -1.0
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        run_training(monkeypatch, csv_path, tmp_path / 'models', tmp_path / 'reports')
    assert (tmp_path / 'reports' / 'train_input_issues.csv').exists()


class AlwaysSafe:
    classes_ = np.array(['Critical', 'Elevated', 'Safe'])

    def predict_proba(self, features):
        return np.tile([0.0, 0.0, 1.0], (len(features), 1))


def compare_with_always_safe(monkeypatch):
    samples = build_samples(synthetic_messages(), 'final', 2.0)
    future = samples[samples['has_future']]
    fitted_on = []

    def fit(train_frame, validation, features, iterations):
        fitted_on.append(len(train_frame))
        return AlwaysSafe()

    monkeypatch.setattr(train, 'fit_classifier', fit)
    _, _, report = train.compare_variants(samples, future, samples, None, 5)
    return samples, future, fitted_on, report


def test_compare_measures_accuracy_separately_on_each_event_group(monkeypatch):
    samples, future, _, report = compare_with_always_safe(monkeypatch)
    assert report['B']['accuracy_future'] == pytest.approx((future['final_tier'] == 'Safe').mean())
    assert report['B']['accuracy_all'] == pytest.approx((samples['final_tier'] == 'Safe').mean())
    assert report['B']['accuracy_future'] != report['B']['accuracy_all']


def test_compare_counts_critical_only_on_events_with_later_messages(monkeypatch):
    _, future, _, report = compare_with_always_safe(monkeypatch)
    baseline = report['baseline']
    assert baseline['true_positives'] + baseline['false_negatives'] == (future['final_tier'] == 'Critical').sum()


def test_compare_fits_each_training_set_it_claims(monkeypatch):
    samples, future, fitted_on, _ = compare_with_always_safe(monkeypatch)
    assert set(fitted_on) == {len(future), len(samples)}
