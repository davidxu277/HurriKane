"""划分事件、跑基线、比较方案、训练模型并输出评估结果。

目标是官方的 final_tier。留出集只在加上 --evaluate-holdout 时才会使用，模型固定后只跑一次。
"""
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from ml import (
    CATEGORICAL_FEATURES, TIERS, alert_scores, apply_stale_rule, build_samples, combine_predictions,
    feature_columns, guard_conservative, load_bundle, predict_batch, predicted_tiers, tier_probabilities,
)
from validation import validate_input

ALPHAS = (0.8, 0.9, 0.95)
DEFAULT_ALPHA = 0.9
MIN_ALPHA_GAIN = 2           # 至少多抓这么多事件才换 alpha，否则视为样本波动
HIGH_RISK_LEVELS = (-4.0, -6.0)
SELECTION_LEVEL = -6.0
TARGET_FLOOR = -10.0         # 低于这个值都属于 Safe，差别没有操作意义
MIN_STRATIFY_EVENTS = 10
MIN_LEVEL_POSITIVES = 20
REVIEW_RECALL = 0.9
STALE_DAYS = 3.0            # 最新消息早于这个时间，就直接沿用当前等级
SEED = 42
TASK = 'final'

# 名称：(历史特征, 当前等级兜底, 训练时包含没有后续消息的事件, 时间规则)，按从简单到复杂排列
VARIANTS = {
    'B': (False, False, False, False),
    'C': (False, True, False, False),
    'D': (True, True, False, False),
    'D_all_events': (True, True, True, False),
    'D_stale': (True, True, False, True),
    'D_all_events_stale': (True, True, True, True),
}
SELECTION_MEASURES = ('critical_true_positives', 'accuracy_future', 'accuracy_all')

PREDICTION_COLUMNS = [
    'event_id', 'y_true', 'final_tier', 'has_future', 'current_tier', 'predicted_final_tier',
    'p_critical', 'alert_score', 'baseline_score', 'tier_log10_pc', 'predicted_log10_pc',
    'conservative_log10_pc', 'miss_distance_m', 'sigma_ratio',
]


def split_events(samples, seed=SEED):
    """按事件划分 70/15/15，按最终等级和是否有后续消息分层。"""
    strata = (samples['final_tier'].astype(str) + '/' + samples['has_future'].astype(str)).to_numpy()
    stratify = strata if pd.Series(strata).value_counts().min() >= MIN_STRATIFY_EVENTS else None

    train, rest, _, rest_strata = train_test_split(
        samples['event_id'].to_numpy(), strata, test_size=0.3, random_state=seed, stratify=stratify
    )
    validation, holdout = train_test_split(
        rest, test_size=0.5, random_state=seed,
        stratify=rest_strata if stratify is not None else None,
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


def choose_variant(results):
    """按从简单到复杂比较，三项成绩都不差于当前方案才替换。"""
    chosen = next(iter(results))
    for name, result in results.items():
        if all(result[measure] >= results[chosen][measure] for measure in SELECTION_MEASURES):
            chosen = name
    return chosen


def critical_rates(samples):
    """每个当前等级的事件，最终变成 Critical 的比例。"""
    rates = samples['final_tier'].eq('Critical').groupby(samples['current_tier']).mean()
    return {tier: float(rates.get(tier, 0.0)) for tier in TIERS}


def review_count(positive, scores, recall=REVIEW_RECALL):
    """从分数最高的事件往下看，抓到指定比例的目标事件时看了多少个。"""
    positive = np.asarray(positive, dtype=bool)
    if not positive.any():
        return None
    needed = math.ceil(recall * positive.sum())
    order = np.argsort(-np.asarray(scores, dtype=float), kind='stable')
    return int(np.searchsorted(np.cumsum(positive[order]), needed) + 1)


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


def tier_report(final, current, predicted, baseline_score, alert_score):
    """官方目标的成绩：和“最终等级等于当前等级”这个基线比较。"""
    final, current, predicted = np.asarray(final), np.asarray(current), np.asarray(predicted)
    positive = final == 'Critical'
    baseline_alert = current == 'Critical'
    negatives = int((~positive).sum())
    baseline_fpr = float((baseline_alert & ~positive).sum() / negatives) if negatives else 0.0
    return {
        'events': len(final),
        'accuracy': {'baseline': float((current == final).mean()), 'model': float((predicted == final).mean())},
        'confusion': pd.crosstab(pd.Series(final, name='final'), pd.Series(predicted, name='predicted'))
                       .to_dict(orient='index'),
        'critical': {
            'positive_events': int(positive.sum()),
            'baseline_fpr': baseline_fpr,
            'baseline': alert_report(positive, baseline_alert),
            'model': alert_report(positive, predicted == 'Critical'),
            'matched': {'baseline': recall_at_fpr(positive, baseline_score, baseline_fpr),
                        'model': recall_at_fpr(positive, alert_score, baseline_fpr)},
            'review_for_90_recall': {'baseline': review_count(positive, baseline_score),
                                     'model': review_count(positive, alert_score)},
        },
    }


def error_report(y, prediction):
    error = np.asarray(prediction, dtype=float) - y
    return {'rmse': float(np.sqrt(np.mean(error ** 2))), 'mae': float(np.mean(np.abs(error)))}


def clip_target(values):
    return np.maximum(np.asarray(values, dtype=float), TARGET_FLOOR)


def evaluate(y_raw, baseline, value_scores, rank_scores, constant):
    """risk 目标的成绩。value_scores 是 log10(Pc) 估计；rank_scores 只用于排序比较。"""
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
    return str(SELECTION_LEVEL)


def categorical_in(features):
    return [name for name in CATEGORICAL_FEATURES if name in features]


def fit_regressor(loss, pool, eval_pool, iterations):
    model = CatBoostRegressor(
        iterations=iterations, depth=6, learning_rate=0.05, loss_function=loss,
        random_seed=SEED, verbose=False, allow_writing_files=False,
    )
    return model.fit(pool, eval_set=eval_pool, early_stopping_rounds=50)


def fit_classifier(train, validation, features, iterations):
    model = CatBoostClassifier(
        iterations=iterations, depth=6, learning_rate=0.05, loss_function='MultiClass',
        random_seed=SEED, verbose=False, allow_writing_files=False,
    )
    cats = categorical_in(features)
    pool = Pool(train[features], train['final_tier'], cat_features=cats)
    eval_pool = Pool(validation[features], validation['final_tier'], cat_features=cats)
    return model.fit(pool, eval_set=eval_pool, early_stopping_rounds=50)


def compare_variants(train_all, train, validation, rates, iterations):
    """在验证集上比较各方案，误报率和当前等级基线对齐，主要看有后续消息的事件。"""
    future = validation['has_future'].to_numpy()
    final = validation['final_tier'].to_numpy()
    current = validation['current_tier'].to_numpy()
    positive = final[future] == 'Critical'
    baseline_alert = current[future] == 'Critical'
    baseline_fpr = float((baseline_alert & ~positive).sum() / max((~positive).sum(), 1))

    def summary(scores, predicted):
        return {
            **recall_at_fpr(positive, scores, baseline_fpr),
            'review_for_90_recall': review_count(positive, scores),
            'critical_true_positives': int((predicted[future][positive] == 'Critical').sum()),
            'critical_false_positives': int((predicted[future][~positive] == 'Critical').sum()),
            'accuracy_future': float((predicted[future] == final[future]).mean()),
            'accuracy_all': float((predicted == final).mean()),
        }

    fitted, results = {}, {}
    for name, (history, fallback, all_events, stale) in VARIANTS.items():
        features = feature_columns(TASK, history=history)
        if (history, all_events) not in fitted:
            fitted[(history, all_events)] = fit_classifier(
                train_all if all_events else train, validation[future], features, iterations
            )
        probabilities = tier_probabilities(fitted[(history, all_events)], validation[features])
        scores = alert_scores(probabilities['Critical'], validation['current_tier'], rates if fallback else None)
        threshold = recall_at_fpr(positive, scores[future], baseline_fpr)['threshold']
        predicted = apply_stale_rule(predicted_tiers(probabilities, scores, threshold), current,
                                     validation['time_to_tca'], STALE_DAYS if stale else None)
        results[name] = summary(scores[future], predicted)

    chosen = choose_variant(results)
    history, _, all_events, _ = VARIANTS[chosen]
    baseline = summary(validation['baseline_score'].to_numpy()[future], current)
    return chosen, fitted[(history, all_events)], {'baseline': baseline, **results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
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
    checks = {'task': TASK, 'features': feature_columns(TASK, history=True),
              'categorical_features': CATEGORICAL_FEATURES}
    valid, issues = validate_input(raw, checks)
    issues.to_csv(reports_dir / 'train_input_issues.csv', index=False, encoding='utf-8-sig')
    if issues['severity'].eq('ERROR').any():
        raise ValueError('训练输入存在 ERROR，请先检查 train_input_issues.csv')
    samples = build_samples(valid, TASK, args.horizon_days)
    print(f'原始 {len(raw)} 行、{raw["event_id"].nunique()} 个事件 -> 样本 {len(samples)} 个，'
          f'其中有后续消息 {int(samples["has_future"].sum())} 个')

    split = load_or_create_split(reports_dir / 'split_events.json', samples)
    parts = {name: samples[samples['event_id'].isin(ids)] for name, ids in split.items()}
    train_all = parts['train']
    train = train_all[train_all['has_future']]
    validation = parts['validation'][parts['validation']['has_future']]

    rates = critical_rates(train)
    variant, classifier, variant_report = compare_variants(
        train_all, train, parts['validation'], rates, args.iterations
    )
    history, fallback, _, stale = VARIANTS[variant]
    stale_days = STALE_DAYS if stale else None
    features = feature_columns(TASK, history=history)
    used_rates = rates if fallback else None
    print(f'选定方案 {variant}')

    train_y, validation_y = clip_target(train['y']), clip_target(validation['y'])
    cats = categorical_in(features)
    pool = Pool(train[features], train_y, cat_features=cats)
    eval_pool = Pool(validation[features], validation_y, cat_features=cats)

    # 最终 risk 大量堆在下限，中位数模型只会输出下限，所以最佳估计用 RMSE 模型
    estimate_model = fit_regressor('RMSE', pool, eval_pool, args.iterations)
    candidates = {alpha: fit_regressor(f'Quantile:alpha={alpha}', pool, eval_pool, args.iterations)
                  for alpha in ALPHAS}
    constant = float(train_y.mean())

    def risk_report(part, conservative_models):
        estimate = estimate_model.predict(part[features])
        rank = rank_scores(
            estimate,
            {alpha: model.predict(part[features]) for alpha, model in conservative_models.items()},
            tier_probabilities(classifier, part[features])['Critical'].to_numpy(),
        )
        return evaluate(part['y'], part['risk'], {'estimate_model': estimate}, rank, constant)

    validation_risk = risk_report(validation, candidates)
    level = selection_level(validation_risk)
    matched = validation_risk['levels'][level]['at_matched_fpr']
    alpha = choose_alpha({a: matched[f'conservative_{a}']['true_positives'] for a in ALPHAS})
    conservative_model = candidates[alpha]
    print(f'选定 alpha={alpha}（比较标签 y >= {level}）')

    safe_looking = train.loc[train['y'] < -6, 'sigma_ratio'].replace(np.inf, np.nan).dropna()
    metadata = {
        'task': TASK,
        'target': 'final_tier',
        'variant': variant,
        'features': features,
        'categorical_features': cats,
        'horizon_days': args.horizon_days,
        'label': '同事件最后一条消息（time_to_tca 最小）的 risk 和 miss_distance，按官方规则分级',
        'units': {'time_to_tca': 'days', 'miss_distance': 'm', 'relative_speed': 'm/s'},
        'target_floor': TARGET_FLOOR,
        'critical_rates': used_rates,
        'critical_threshold': variant_report[variant]['threshold'],
        'stale_days': stale_days,
        'split_seed': SEED,
        'alpha': alpha,
        'selection_level': float(level),
        'sigma_ratio_threshold': float(safe_looking.quantile(0.9)),
        'train_file': args.train,
        'events': {name: len(part) for name, part in parts.items()},
        'training_events': len(train_all) if VARIANTS[variant][2] else len(train),
        'model_version': datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'),
    }
    estimate_model.save_model(str(models_dir / 'estimate.cbm'))
    conservative_model.save_model(str(models_dir / 'conservative.cbm'))
    classifier.save_model(str(models_dir / 'classifier.cbm'))
    (models_dir / 'metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    def predict_part(part):
        result = combine_predictions(
            part,
            estimate_model.predict(part[features]),
            conservative_model.predict(part[features]),
            tier_probabilities(classifier, part[features]),
            used_rates,
            metadata['critical_threshold'],
            stale_days,
        )
        result['y_true'] = part['y'].to_numpy()
        result['final_tier'] = part['final_tier'].to_numpy()
        result['has_future'] = part['has_future'].to_numpy()
        return result

    def tiers_for(predictions):
        def report(frame):
            return tier_report(frame['final_tier'], frame['current_tier'], frame['predicted_final_tier'],
                               frame['baseline_score'], frame['alert_score'])
        return {'future': report(predictions[predictions['has_future']]), 'all': report(predictions)}

    validation_predictions = predict_part(parts['validation'])
    validation_predictions[PREDICTION_COLUMNS].to_csv(reports_dir / 'validation_predictions.csv', index=False)
    metrics = {'metadata': metadata, 'validation': {
        'variants': variant_report, 'tiers': tiers_for(validation_predictions), 'risk': validation_risk,
    }}

    if args.evaluate_holdout:
        holdout_predictions = predict_part(parts['holdout'])
        holdout_predictions[PREDICTION_COLUMNS].to_csv(reports_dir / 'holdout_predictions.csv', index=False)
        metrics['holdout'] = {'tiers': tiers_for(holdout_predictions),
                              'risk': risk_report(parts['holdout'], {alpha: conservative_model})}

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
