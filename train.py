"""划分事件、跑基线、训练模型并输出评估结果。

留出集只在加上 --evaluate-holdout 时才会使用，模型固定后只跑一次。
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from ml import (
    CATEGORICAL_FEATURES, build_samples, combine_predictions, feature_columns, guard_conservative,
    load_bundle, predict_batch,
)

ALPHAS = (0.8, 0.9, 0.95)
DEFAULT_ALPHA = 0.9
MIN_ALPHA_GAIN = 2           # 至少多抓这么多事件才换 alpha，否则视为样本波动
HIGH_RISK_LEVELS = (-4.0, -6.0)
CLASSIFIER_LEVEL = -6.0
TARGET_FLOOR = -10.0         # 低于这个值都属于 Safe，差别没有操作意义
MIN_STRATIFY_EVENTS = 10
MIN_LEVEL_POSITIVES = 20
SEED = 42

PREDICTION_COLUMNS = [
    'event_id', 'y_true', 'baseline_score', 'alert_score', 'tier_log10_pc', 'predicted_log10_pc',
    'conservative_log10_pc', 'miss_distance_m', 'sigma_ratio',
]


def split_events(samples, seed=SEED):
    """按事件划分 70/15/15，高风险事件足够多时分层。"""
    events = samples.groupby('event_id')['y'].max()
    high_risk = (events >= CLASSIFIER_LEVEL).to_numpy()
    counts = np.bincount(high_risk.astype(int), minlength=2)
    stratify = high_risk if counts.min() >= MIN_STRATIFY_EVENTS else None

    train, rest, _, rest_high_risk = train_test_split(
        events.index.to_numpy(), high_risk, test_size=0.3, random_state=seed, stratify=stratify
    )
    validation, holdout = train_test_split(
        rest, test_size=0.5, random_state=seed,
        stratify=rest_high_risk if stratify is not None else None,
    )
    return {'train': train, 'validation': validation, 'holdout': holdout}


def load_or_create_split(path, samples, seed=SEED):
    """已有清单就原样使用并核对；没有才切分一次。"""
    path = Path(path)
    if not path.exists():
        split = {name: [int(i) for i in ids] for name, ids in split_events(samples, seed).items()}
        path.write_text(json.dumps(split), encoding='utf-8')
        return split

    split = json.loads(path.read_text(encoding='utf-8'))
    listed = [event for ids in split.values() for event in ids]
    if set(split) != {'train', 'validation', 'holdout'}:
        raise ValueError(f'切分清单分组不对：{sorted(split)}')
    if len(listed) != len(set(listed)) or set(listed) != set(samples['event_id']):
        raise ValueError('切分清单和当前样本的事件不一致，请检查数据和任务口径')
    return split


def recall_at_fpr(positive, scores, target_fpr):
    """把告警阈值调到误报率不超过 target_fpr，返回此时的召回。"""
    positive = np.asarray(positive, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    negatives = np.sort(scores[~positive])[::-1]
    allowed = int(np.floor(target_fpr * len(negatives) + 1e-9))
    threshold = negatives[allowed] if allowed < len(negatives) else -np.inf
    alert = scores > threshold

    true_positives = int((alert & positive).sum())
    false_positives = int((alert & ~positive).sum())
    return {
        'threshold': float(threshold),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': int(positive.sum()) - true_positives,
        'recall': true_positives / int(positive.sum()) if positive.any() else None,
    }


def choose_alpha(true_positives_by_alpha, default=DEFAULT_ALPHA, min_gain=MIN_ALPHA_GAIN):
    best = max(true_positives_by_alpha, key=true_positives_by_alpha.get)
    if true_positives_by_alpha[best] - true_positives_by_alpha[default] >= min_gain:
        return best
    return default


def alert_report(positive, alert):
    positive = np.asarray(positive, dtype=bool)
    alert = np.asarray(alert, dtype=bool)
    tp = int((alert & positive).sum())
    fp = int((alert & ~positive).sum())
    fn = int((~alert & positive).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f2 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f2 = 5 * precision * recall / (4 * precision + recall)
    return {'precision': precision, 'recall': recall, 'f2': f2,
            'true_positives': tp, 'false_positives': fp, 'false_negatives': fn}


def error_report(y, prediction):
    error = np.asarray(prediction, dtype=float) - y
    return {'rmse': float(np.sqrt(np.mean(error ** 2))), 'mae': float(np.mean(np.abs(error)))}


def clip_target(values):
    return np.maximum(np.asarray(values, dtype=float), TARGET_FLOOR)


def evaluate(y_raw, baseline, value_scores, rank_scores, constant):
    """value_scores 是 log10(Pc) 估计；rank_scores 只用于排序比较。"""
    y_raw = np.asarray(y_raw, dtype=float)
    y = clip_target(y_raw)
    regression = {
        'constant': error_report(y, np.full(len(y), constant)),
        'latest_risk': error_report(y, clip_target(baseline)),
    }
    regression.update({name: error_report(y, values) for name, values in value_scores.items()})

    levels = {}
    for level in HIGH_RISK_LEVELS:
        positive = y_raw >= level
        baseline_alert = np.asarray(baseline, dtype=float) >= level
        negatives = int((~positive).sum())
        baseline_fpr = float((baseline_alert & ~positive).sum() / negatives) if negatives else 0.0
        scores = {'latest_risk': baseline, **value_scores, **rank_scores}
        levels[str(level)] = {
            'positive_events': int(positive.sum()),
            'negative_events': negatives,
            'baseline_fpr': baseline_fpr,
            'at_threshold': {
                'latest_risk': alert_report(positive, baseline_alert),
                **{name: alert_report(positive, np.asarray(values) >= level)
                   for name, values in value_scores.items()},
            },
            'at_matched_fpr': {name: recall_at_fpr(positive, values, baseline_fpr)
                               for name, values in scores.items()},
        }
    return {'events': len(y), 'regression': regression, 'levels': levels}


def rank_scores(estimate, conservative_by_alpha, classifier_scores):
    scores = {'classifier': classifier_scores}
    scores.update({f'conservative_{alpha}': guard_conservative(estimate, values)
                   for alpha, values in conservative_by_alpha.items()})
    return scores


def selection_level(report):
    strict = str(HIGH_RISK_LEVELS[0])
    if report['levels'][strict]['positive_events'] >= MIN_LEVEL_POSITIVES:
        return strict
    return str(HIGH_RISK_LEVELS[1])


def fit_regressor(loss, pool, eval_pool, iterations):
    model = CatBoostRegressor(
        iterations=iterations, depth=6, learning_rate=0.05, loss_function=loss,
        random_seed=SEED, verbose=False, allow_writing_files=False,
    )
    return model.fit(pool, eval_set=eval_pool, early_stopping_rounds=50)


def fit_classifier(features, labels, eval_features, eval_labels, iterations):
    model = CatBoostClassifier(
        iterations=iterations, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
        random_seed=SEED, verbose=False, allow_writing_files=False,
    )
    pool = Pool(features, labels, cat_features=CATEGORICAL_FEATURES)
    eval_pool = Pool(eval_features, eval_labels, cat_features=CATEGORICAL_FEATURES)
    return model.fit(pool, eval_set=eval_pool, early_stopping_rounds=50)


def write_predictions(path, samples, estimate, conservative, alert_score):
    predictions = combine_predictions(samples, estimate, conservative, alert_score)
    predictions['y_true'] = samples['y'].to_numpy()
    predictions[PREDICTION_COLUMNS].to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--task', choices=['final', 'current'], default='final')
    parser.add_argument('--horizon-days', type=float, default=2.0)
    parser.add_argument('--iterations', type=int, default=400)
    parser.add_argument('--models', default='models')
    parser.add_argument('--reports', default='reports')
    parser.add_argument('--evaluate-holdout', action='store_true')
    parser.add_argument('--test', help='主办方测试文件，用保存的模型生成预测')
    args = parser.parse_args()

    models_dir, reports_dir = Path(args.models), Path(args.reports)
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    raw = pd.read_csv(args.train, low_memory=False)
    samples = build_samples(raw, args.task, args.horizon_days)
    print(f'原始 {len(raw)} 行、{raw["event_id"].nunique()} 个事件 -> 样本 {len(samples)} 个')

    split = load_or_create_split(reports_dir / 'split_events.json', samples)
    parts = {name: samples[samples['event_id'].isin(ids)] for name, ids in split.items()}

    features = feature_columns(args.task)
    train, validation = parts['train'], parts['validation']
    train_y, validation_y = clip_target(train['y']), clip_target(validation['y'])
    pool = Pool(train[features], train_y, cat_features=CATEGORICAL_FEATURES)
    eval_pool = Pool(validation[features], validation_y, cat_features=CATEGORICAL_FEATURES)

    # 最终 risk 大量堆在下限，中位数模型只会输出下限，所以最佳估计用 RMSE 模型
    estimate_model = fit_regressor('RMSE', pool, eval_pool, args.iterations)
    candidates = {alpha: fit_regressor(f'Quantile:alpha={alpha}', pool, eval_pool, args.iterations)
                  for alpha in ALPHAS}
    classifier = fit_classifier(
        train[features], (train['y'] >= CLASSIFIER_LEVEL).astype(int),
        validation[features], (validation['y'] >= CLASSIFIER_LEVEL).astype(int),
        args.iterations,
    )

    constant = float(train_y.mean())

    def report_for(part, conservative_models):
        estimate = estimate_model.predict(part[features])
        rank = rank_scores(
            estimate,
            {alpha: model.predict(part[features]) for alpha, model in conservative_models.items()},
            classifier.predict_proba(part[features])[:, 1],
        )
        return evaluate(part['y'], part['baseline_score'], {'estimate_model': estimate}, rank, constant)

    validation_report = report_for(validation, candidates)
    level = selection_level(validation_report)
    matched = validation_report['levels'][level]['at_matched_fpr']
    alpha = choose_alpha({a: matched[f'conservative_{a}']['true_positives'] for a in ALPHAS})
    conservative_model = candidates[alpha]
    print(f'选定 alpha={alpha}（比较标签 y >= {level}）')

    safe_looking = train.loc[train['y'] < -6, 'sigma_ratio'].replace(np.inf, np.nan).dropna()
    metadata = {
        'task': args.task,
        'features': features,
        'categorical_features': CATEGORICAL_FEATURES,
        'horizon_days': args.horizon_days,
        'label': 'final 为同事件最后一条消息的 risk；current 为当前行 risk',
        'units': {'time_to_tca': 'days', 'miss_distance': 'm', 'relative_speed': 'm/s'},
        'target_floor': TARGET_FLOOR,
        'classifier_level': CLASSIFIER_LEVEL,
        'split_seed': SEED,
        'alpha': alpha,
        'selection_level': float(level),
        'sigma_ratio_threshold': float(safe_looking.quantile(0.9)),
        'train_file': args.train,
        'events': {name: len(part) for name, part in parts.items()},
        'model_version': datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'),
    }
    estimate_model.save_model(str(models_dir / 'estimate.cbm'))
    conservative_model.save_model(str(models_dir / 'conservative.cbm'))
    classifier.save_model(str(models_dir / 'classifier.cbm'))
    (models_dir / 'metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    def predict_part(part):
        return (estimate_model.predict(part[features]), conservative_model.predict(part[features]),
                classifier.predict_proba(part[features])[:, 1])

    write_predictions(reports_dir / 'validation_predictions.csv', validation, *predict_part(validation))
    metrics = {'metadata': metadata, 'validation': validation_report}

    if args.evaluate_holdout:
        holdout = parts['holdout']
        write_predictions(reports_dir / 'holdout_predictions.csv', holdout, *predict_part(holdout))
        metrics['holdout'] = report_for(holdout, {alpha: conservative_model})

    if args.test:
        test = pd.read_csv(args.test, low_memory=False)
        started = time.perf_counter()
        predictions = predict_batch(test, load_bundle(models_dir))
        metrics['processing_time'] = {'rows': len(test), 'events': len(predictions),
                                      'seconds': round(time.perf_counter() - started, 3)}
        predictions.to_csv(reports_dir / 'test_predictions.csv', index=False)

    (reports_dir / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding='utf-8'
    )
    print(f'完成，结果在 {reports_dir / "metrics.json"}')


if __name__ == '__main__':
    main()
