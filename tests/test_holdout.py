"""留出评估只用合成数据测试，不读取真实留出集。"""
import json
import shutil
import sys

import pandas as pd
import pytest

import evaluate_holdout
import train
from test_pipeline import run_training, synthetic_messages


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    csv_path = tmp_path / 'train.csv'
    frame = synthetic_messages()
    # 一部分事件的最新消息停在 3 天，让时间规则真正触发
    stale = (frame['event_id'] % 7 == 0) & (frame['time_to_tca'] < 3)
    frame[~stale].to_csv(csv_path, index=False)
    # 这次运行带 --evaluate-holdout，用它的输出和新脚本交叉核对
    run_training(monkeypatch, csv_path, tmp_path / 'models', tmp_path / 'trained_reports',
                 '--variant', 'D_stale', '--evaluate-holdout')
    return tmp_path


def run_holdout(root, monkeypatch, reports='holdout_reports'):
    (root / reports).mkdir(exist_ok=True)
    argv = ['evaluate_holdout.py', '--train', str(root / 'train.csv'), '--models', str(root / 'models'),
            '--split', str(root / 'trained_reports' / 'split_events.json'), '--reports', str(root / reports)]
    monkeypatch.setattr(sys, 'argv', argv)
    evaluate_holdout.main()


def test_predictions_match_the_training_pipeline_exactly(fixed, monkeypatch):
    run_holdout(fixed, monkeypatch)
    ours = pd.read_csv(fixed / 'holdout_reports' / 'holdout_predictions.csv')
    pipeline = pd.read_csv(fixed / 'trained_reports' / 'holdout_predictions.csv')
    pd.testing.assert_frame_equal(ours, pipeline)


def test_covers_only_holdout_events(fixed, monkeypatch):
    run_holdout(fixed, monkeypatch)
    split = json.loads((fixed / 'trained_reports' / 'split_events.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(fixed / 'holdout_reports' / 'holdout_predictions.csv')
    assert sorted(predictions['event_id']) == sorted(split['holdout'])


def test_metrics_match_the_training_pipeline(fixed, monkeypatch):
    run_holdout(fixed, monkeypatch)
    ours = json.loads((fixed / 'holdout_reports' / 'holdout_metrics.json').read_text(encoding='utf-8'))
    pipeline = json.loads((fixed / 'trained_reports' / 'metrics.json').read_text(encoding='utf-8'))
    assert ours['tiers'] == pipeline['holdout']['tiers']
    assert ours['variant'] == 'D_stale'


def test_does_not_touch_saved_models(fixed, monkeypatch):
    before = {p.name: p.read_bytes() for p in (fixed / 'models').iterdir()}
    run_holdout(fixed, monkeypatch)
    assert {p.name: p.read_bytes() for p in (fixed / 'models').iterdir()} == before


def test_never_trains(fixed, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('holdout evaluation must not train')

    monkeypatch.setattr(train, 'fit_classifier', fail)
    monkeypatch.setattr(train, 'fit_regressor', fail)
    run_holdout(fixed, monkeypatch)


def test_refuses_to_run_twice(fixed, monkeypatch):
    run_holdout(fixed, monkeypatch)
    before = (fixed / 'holdout_reports' / 'holdout_predictions.csv').read_bytes()
    with pytest.raises(SystemExit):
        run_holdout(fixed, monkeypatch)
    assert (fixed / 'holdout_reports' / 'holdout_predictions.csv').read_bytes() == before


def test_stops_when_holdout_events_are_missing_from_data(fixed, monkeypatch):
    split_path = fixed / 'trained_reports' / 'split_events.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    split['holdout'].append(999999)
    split_path.write_text(json.dumps(split), encoding='utf-8')
    with pytest.raises(ValueError):
        run_holdout(fixed, monkeypatch)
    assert not (fixed / 'holdout_reports' / 'holdout_predictions.csv').exists()


def test_stops_on_invalid_input(fixed, monkeypatch):
    frame = pd.read_csv(fixed / 'train.csv')
    frame.loc[0, 'miss_distance'] = -1.0
    frame.to_csv(fixed / 'train.csv', index=False)
    with pytest.raises(ValueError):
        run_holdout(fixed, monkeypatch)
    assert not (fixed / 'holdout_reports' / 'holdout_predictions.csv').exists()


def test_stale_rule_is_applied_to_old_latest_messages(fixed, monkeypatch):
    run_holdout(fixed, monkeypatch)
    predictions = pd.read_csv(fixed / 'holdout_reports' / 'holdout_predictions.csv')
    stale = predictions[predictions['event_id'] % 7 == 0]
    assert len(stale) > 0
    assert (stale['predicted_final_tier'] == stale['current_tier']).all()
