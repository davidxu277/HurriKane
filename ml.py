"""样本构造与模型推理。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

TIERS = ('Safe', 'Elevated', 'Critical')
BASE_FEATURES = [
    'time_to_tca', 'miss_distance', 'relative_speed',
    'relative_position_r', 'relative_position_t', 'relative_position_n',
    'c_object_type', 'c_sigma_r', 't_sigma_r',
    'max_risk_estimate', 'F10', 'AP',
]
# 仅 final 模式增加，全部只用截止时刻前的消息计算
FINAL_EXTRA_FEATURES = ['risk', 'risk_delta', 'miss_distance_delta', 'message_count', 'current_tier']
HISTORY_FEATURES = [
    'risk_max', 'risk_std', 'risk_slope', 'miss_distance_min', 'miss_distance_slope', 'c_sigma_r_delta',
]
CATEGORICAL_FEATURES = ['c_object_type', 'current_tier']


def feature_columns(task, history=False):
    if task == 'final':
        return BASE_FEATURES + FINAL_EXTRA_FEATURES + (HISTORY_FEATURES if history else [])
    if task == 'current':
        return list(BASE_FEATURES)
    raise ValueError(f'未知任务口径：{task}')


def risk_tier(risk, miss_distance):
    """官方分级规则，Critical 优先，其次 Elevated。"""
    risk = np.asarray(risk, dtype=float)
    miss_distance = np.asarray(miss_distance, dtype=float)
    return np.select(
        [(risk >= -4) | (miss_distance < 1000), (risk >= -6) | (miss_distance <= 5000)],
        ['Critical', 'Elevated'], 'Safe',
    )


def build_samples(df, task, horizon_days, labeled=True):
    """把原始消息转成模型样本：current 一行一样本，final 一事件一样本。

    final 的标签取同事件最后一条消息，与官方 challenge1_event_labels 一致；
    has_future 为 False 的事件，最后一条消息本身就是输入。
    labeled=False 用于没有最终标签的数据，此时标签列为空。
    """
    df = df.copy()
    if task == 'current':
        samples = _current_samples(df, labeled)
    elif task == 'final':
        samples = _final_samples(df, horizon_days, labeled)
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
    samples['current_tier'] = 'Unknown'
    return samples


def _final_samples(df, horizon_days, labeled):
    # time_to_tca 越小消息越新，排序后每个事件的最后一行就是最新消息
    ordered = df.assign(input_record_id=df.index).sort_values(
        ['event_id', 'time_to_tca'], ascending=[True, False], kind='stable'
    )
    inputs = ordered[ordered['time_to_tca'] >= horizon_days].copy()
    grouped = inputs.groupby('event_id')
    inputs['risk_delta'] = grouped['risk'].diff()
    inputs['miss_distance_delta'] = grouped['miss_distance'].diff()
    inputs['c_sigma_r_delta'] = grouped['c_sigma_r'].diff()
    inputs['message_count'] = grouped.cumcount() + 1
    inputs['risk_max'] = grouped['risk'].transform('max')
    inputs['risk_std'] = grouped['risk'].transform('std')
    inputs['miss_distance_min'] = grouped['miss_distance'].transform('min')
    inputs['risk_slope'] = _slope(inputs, 'risk')
    inputs['miss_distance_slope'] = _slope(inputs, 'miss_distance')

    samples = inputs.groupby('event_id').tail(1).copy()
    samples['current_tier'] = risk_tier(samples['risk'], samples['miss_distance'])
    samples['baseline_score'] = _baseline_score(samples)
    samples['known_log10_pc'] = samples['risk']
    if not labeled:
        samples['y'] = np.nan
        return samples

    last_message = ordered.groupby('event_id').tail(1).set_index('event_id')
    samples['y'] = samples['event_id'].map(last_message['risk'])
    samples['final_miss_distance'] = samples['event_id'].map(last_message['miss_distance'])
    samples['final_tier'] = risk_tier(samples['y'], samples['final_miss_distance'])
    samples['has_future'] = samples['event_id'].map(last_message['time_to_tca']) < horizon_days
    return samples


def _slope(inputs, column):
    """每个事件的线性趋势：朝 TCA 每过一天，column 变化多少。"""
    events = inputs['event_id']
    days = -inputs['time_to_tca']
    x = days - days.groupby(events).transform('mean')
    y = inputs[column] - inputs.groupby('event_id')[column].transform('mean')
    covariance = (x * y).groupby(events).transform('sum')
    variance = (x * x).groupby(events).transform('sum')
    return (covariance / variance).where(variance > 0)


def _baseline_score(samples):
    # 基线排序：先按当前等级，同级距离越近越靠前
    rank = samples['current_tier'].map({tier: index for index, tier in enumerate(TIERS)})
    return rank + 1 / (1 + samples['miss_distance'] / 1000)


def _sigma_ratio(samples):
    sigma = np.sqrt(samples['c_sigma_r'] ** 2 + samples['t_sigma_r'] ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = sigma / samples['miss_distance']
    return ratio.where(samples['miss_distance'] != 0, np.inf)


def guard_conservative(estimate, conservative):
    # 两个模型独立训练，结果可能交叉
    return np.maximum(np.asarray(estimate, dtype=float), np.asarray(conservative, dtype=float))


def alert_scores(p_critical, current_tier, critical_rates):
    """当前等级兜底：分数不低于该等级历史上最终变成 Critical 的比例。"""
    p_critical = np.asarray(p_critical, dtype=float)
    if critical_rates is None:
        return p_critical
    floor = np.array([critical_rates[tier] for tier in current_tier], dtype=float)
    # 加一点模型分数，让同一兜底线上的事件仍按模型排序
    return np.maximum(p_critical, floor) + 1e-6 * p_critical


def predicted_tiers(probabilities, alert_score, critical_threshold):
    lower = np.where(probabilities['Elevated'] > probabilities['Safe'], 'Elevated', 'Safe')
    return np.where(np.asarray(alert_score) > critical_threshold, 'Critical', lower)


def tier_probabilities(model, features):
    frame = pd.DataFrame(model.predict_proba(features), columns=[str(c) for c in model.classes_])
    return frame.reindex(columns=list(TIERS), fill_value=0.0)


def combine_predictions(samples, estimate, conservative, probabilities, critical_rates, critical_threshold):
    predicted = np.asarray(estimate, dtype=float)
    guarded = guard_conservative(predicted, conservative)
    # 分级用已知的最新 Pc；current 模式没有已知值，只能用模型估计
    known = samples['known_log10_pc'].to_numpy(dtype=float)
    tier = np.where(np.isnan(known), predicted, known)
    alert = alert_scores(probabilities['Critical'], samples['current_tier'], critical_rates)
    return pd.DataFrame({
        'input_record_id': samples['input_record_id'].to_numpy(),
        'event_id': samples['event_id'].to_numpy(),
        'time_to_tca_hours': samples['time_to_tca'].to_numpy() * 24,
        'miss_distance_m': samples['miss_distance'].to_numpy(),
        'current_tier': samples['current_tier'].to_numpy(),
        'predicted_final_tier': predicted_tiers(probabilities, alert, critical_threshold),
        'p_safe': probabilities['Safe'].to_numpy(),
        'p_elevated': probabilities['Elevated'].to_numpy(),
        'p_critical': probabilities['Critical'].to_numpy(),
        'tier_log10_pc': tier,
        'tier_pc': 10.0 ** tier,
        'predicted_log10_pc': predicted,
        'predicted_pc': 10.0 ** predicted,
        'conservative_log10_pc': guarded,
        'conservative_pc': 10.0 ** guarded,
        'alert_score': alert,
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
        tier_probabilities(bundle['classifier'], features),
        metadata['critical_rates'],
        metadata['critical_threshold'],
    )
