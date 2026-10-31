"""用合成数据跑完整流程，不读取真实数据。"""
import json
import math
import sys

import numpy as np
import pandas as pd
import pytest

import train
from ml import load_bundle, predict_batch

ITERATIONS = '20'


def synthetic_messages(events=240, high_risk=40, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for event_id in range(events):
        high = event_id < high_risk
        final_risk = -5.0 if high else -30.0
        for days, risk in ((5.0, -14.0), (3.0, -12.0), (2.2, -7.0 if high else -11.0), (0.5, final_risk)):
            rows.append({
                'event_id': event_id,
                'time_to_tca': days,
                'risk': risk,
                'miss_distance': float(rng.uniform(500, 9000)),
                'relative_speed': float(rng.uniform(100, 14000)),
                'relative_position_r': float(rng.normal()),
                'relative_position_t': float(rng.normal()),
                'relative_position_n': float(rng.normal()),
                'c_object_type': 'DEBRIS' if event_id % 3 else 'UNKNOWN',
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
    frame = synthetic_messages(events=4, high_risk=1, seed=11)
    return frame[frame['time_to_tca'] >= 2].reset_index(drop=True)


def test_predict_batch_keeps_every_event_without_near_tca_message(trained):
    result = predict_batch(unlabeled_events(), load_bundle(trained / 'models'))
    assert sorted(result['event_id']) == [0, 1, 2, 3]


def test_predict_batch_tiers_on_latest_known_risk(trained):
    result = predict_batch(unlabeled_events(), load_bundle(trained / 'models')).set_index('event_id')
    assert result.loc[0, 'tier_log10_pc'] == -7.0
    assert result.loc[1, 'tier_log10_pc'] == -11.0


def test_predict_batch_output_is_complete_and_ordered(trained):
    result = predict_batch(unlabeled_events(), load_bundle(trained / 'models'))
    columns = ['predicted_pc', 'conservative_pc', 'alert_score', 'tier_pc']
    assert not result[columns].isna().any().any()
    assert (result['conservative_log10_pc'] >= result['predicted_log10_pc']).all()
    assert result['alert_score'].between(0, 1).all()


def test_predict_batch_leaves_ratio_unknown_when_sigma_is_missing(trained):
    frame = unlabeled_events()
    frame.loc[frame['event_id'] == 2, 'c_sigma_r'] = np.nan
    result = predict_batch(frame, load_bundle(trained / 'models')).set_index('event_id')
    assert math.isnan(result.loc[2, 'sigma_ratio'])
    assert not math.isnan(result.loc[2, 'alert_score'])


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
