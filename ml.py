"""样本构造与模型推理。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

BASE_FEATURES = [
    'time_to_tca', 'miss_distance', 'relative_speed',
    'relative_position_r', 'relative_position_t', 'relative_position_n',
    'c_object_type', 'c_sigma_r', 't_sigma_r',
    'max_risk_estimate', 'F10', 'AP',
]
# 仅 final 模式增加，全部只用截止时刻前的消息计算
FINAL_EXTRA_FEATURES = ['risk', 'risk_delta', 'miss_distance_delta', 'message_count']
CATEGORICAL_FEATURES = ['c_object_type']


def feature_columns(task):
    if task == 'final':
        return BASE_FEATURES + FINAL_EXTRA_FEATURES
    if task == 'current':
        return list(BASE_FEATURES)
    raise ValueError(f'未知任务口径：{task}')


def build_samples(df, task, horizon_days, labeled=True, label_max_days=1.0):
    """把原始消息转成模型样本：current 一行一样本，final 一事件一样本。

    labeled=False 用于没有最终标签的数据，此时 y 为空。
    """
    df = df.copy()
    if task == 'current':
        samples = _current_samples(df, labeled)
    elif task == 'final':
        samples = _final_samples(df, horizon_days, labeled, label_max_days)
    else:
        raise ValueError(f'未知任务口径：{task}')

    samples['c_object_type'] = samples['c_object_type'].fillna('UNKNOWN').astype(str)
    samples['sigma_ratio'] = _sigma_ratio(samples)
    return samples.reset_index(drop=True)


def _current_samples(df, labeled):
    samples = df.assign(input_record_id=df.index)
    samples['y'] = samples['risk'] if labeled else np.nan
    samples['baseline_score'] = samples['max_risk_estimate']
    samples['known_log10_pc'] = np.nan
    return samples


def _final_samples(df, horizon_days, labeled, label_max_days):
    # time_to_tca 越小消息越新，排序后每个事件的最后一行就是最新消息
    ordered = df.assign(input_record_id=df.index).sort_values(
        ['event_id', 'time_to_tca'], ascending=[True, False], kind='stable'
    )
    inputs = ordered[ordered['time_to_tca'] >= horizon_days].copy()
    grouped = inputs.groupby('event_id')
    inputs['risk_delta'] = grouped['risk'].diff()
    inputs['miss_distance_delta'] = grouped['miss_distance'].diff()
    inputs['message_count'] = grouped.cumcount() + 1

    samples = inputs.groupby('event_id').tail(1).copy()
    samples['baseline_score'] = samples['risk']
    samples['known_log10_pc'] = samples['risk']
    if not labeled:
        samples['y'] = np.nan
        return samples

    last_message = ordered.groupby('event_id').tail(1).set_index('event_id')
    samples['y'] = samples['event_id'].map(last_message['risk'])
    near_tca = samples['event_id'].map(last_message['time_to_tca']) <= label_max_days
    return samples[near_tca].copy()


def _sigma_ratio(samples):
    sigma = np.sqrt(samples['c_sigma_r'] ** 2 + samples['t_sigma_r'] ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = sigma / samples['miss_distance']
    return ratio.where(samples['miss_distance'] != 0, np.inf)


def guard_conservative(estimate, conservative):
    # 两个模型独立训练，结果可能交叉
    return np.maximum(np.asarray(estimate, dtype=float), np.asarray(conservative, dtype=float))


def combine_predictions(samples, estimate, conservative, alert_score):
    predicted = np.asarray(estimate, dtype=float)
    guarded = guard_conservative(predicted, conservative)
    # 分级用已知的最新 Pc；current 模式没有已知值，只能用模型估计
    known = samples['known_log10_pc'].to_numpy(dtype=float)
    tier = np.where(np.isnan(known), predicted, known)
    return pd.DataFrame({
        'input_record_id': samples['input_record_id'].to_numpy(),
        'event_id': samples['event_id'].to_numpy(),
        'time_to_tca_hours': samples['time_to_tca'].to_numpy() * 24,
        'miss_distance_m': samples['miss_distance'].to_numpy(),
        'tier_log10_pc': tier,
        'tier_pc': 10.0 ** tier,
        'predicted_log10_pc': predicted,
        'predicted_pc': 10.0 ** predicted,
        'conservative_log10_pc': guarded,
        'conservative_pc': 10.0 ** guarded,
        'alert_score': np.asarray(alert_score, dtype=float),
        'baseline_score': samples['baseline_score'].to_numpy(),
        'sigma_ratio': samples['sigma_ratio'].to_numpy(),
    })


def load_bundle(model_dir):
    model_dir = Path(model_dir)
    return {
        'estimate': CatBoostRegressor().load_model(str(model_dir / 'estimate.cbm')),
        'conservative': CatBoostRegressor().load_model(str(model_dir / 'conservative.cbm')),
        'classifier': CatBoostClassifier().load_model(str(model_dir / 'classifier.cbm')),
        'metadata': json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8')),
    }


def predict_batch(df, bundle):
    metadata = bundle['metadata']
    samples = build_samples(df, metadata['task'], metadata['horizon_days'], labeled=False)
    features = samples[metadata['features']]
    return combine_predictions(
        samples,
        bundle['estimate'].predict(features),
        bundle['conservative'].predict(features),
        bundle['classifier'].predict_proba(features)[:, 1],
    )
