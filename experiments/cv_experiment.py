"""探索实验：只用训练集和验证集做 5 折交叉验证，不读取留出事件的标签。"""
import json
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, '.')
from ml import alert_scores, build_samples, feature_columns
from train import critical_rates, recall_at_fpr, review_count

ITERATIONS = 300
SEED = 42
raw = pd.read_csv('data/raw/train_data.csv', low_memory=False)
samples = build_samples(raw, 'final', 2.0)
split = json.load(open('reports/split_events.json', encoding='utf-8'))
dev_ids = set(split['train']) | set(split['validation'])
assert not dev_ids & set(split['holdout'])
dev = samples[samples['event_id'].isin(dev_ids) & samples['has_future']].reset_index(drop=True)
print('dev events with later messages', len(dev), 'critical', int((dev['final_tier'] == 'Critical').sum()), flush=True)

EXCLUDE = {
    'event_id', 'mission_id', 'y', 'final_miss_distance', 'final_tier', 'has_future', 'input_record_id',
    'baseline_score', 'known_log10_pc',
}
BASE = feature_columns('final', history=True)
numeric = [c for c in dev.columns if c not in EXCLUDE and c not in BASE and pd.api.types.is_numeric_dtype(dev[c])]
EXPANDED = BASE + numeric
print('features: D', len(BASE), '| expanded', len(EXPANDED), flush=True)
print('sample added:', numeric[:14], flush=True)
CATS = ['c_object_type', 'current_tier']
positive = (dev['final_tier'] == 'Critical').to_numpy()
baseline = dev['baseline_score'].to_numpy()
baseline_alert = (dev['current_tier'] == 'Critical').to_numpy()
fpr = (baseline_alert & ~positive).sum() / (~positive).sum()
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(dev, dev['final_tier']))


def classifier(train, valid, features, params):
    model = CatBoostClassifier(iterations=ITERATIONS, loss_function='MultiClass', random_seed=SEED, verbose=False,
                               allow_writing_files=False, **params)
    model.fit(Pool(train[features], train['final_tier'], cat_features=CATS))
    frame = pd.DataFrame(model.predict_proba(valid[features]), columns=[str(c) for c in model.classes_])
    return frame['Critical'].to_numpy()


def distance(train, valid, features, loss, params):
    model = CatBoostRegressor(iterations=ITERATIONS, loss_function=loss, random_seed=SEED, verbose=False,
                              allow_writing_files=False, **params)
    target = np.log10(train['final_miss_distance'].clip(lower=1))
    model.fit(Pool(train[features], target, cat_features=CATS))
    return -model.predict(valid[features])     # 预测距离越小，分数越高


def ranks(values):
    return pd.Series(values).rank(pct=True).to_numpy()


def score_all(names_fn):
    """返回 {方案: out-of-fold 分数}。"""
    out = {}
    for fold, (train_idx, valid_idx) in enumerate(folds):
        train, valid = dev.iloc[train_idx], dev.iloc[valid_idx]
        rates = critical_rates(train)
        for name, values in names_fn(train, valid, rates).items():
            out.setdefault(name, np.zeros(len(dev)))[valid_idx] = values
    return out


def report(scores):
    base = recall_at_fpr(positive, baseline, fpr)
    print(f"\n{'scheme':26s} {'tp':>4s} {'fp':>4s} {'gain':>5s} {'gain%':>6s} {'review90':>8s} fold gains", flush=True)
    print(f"{'baseline':26s} {base['true_positives']:4d} {base['false_positives']:4d} {'':5s} {'':6s} "
          f"{review_count(positive, baseline):8d}", flush=True)
    for name, values in scores.items():
        result = recall_at_fpr(positive, values, fpr)
        gains = []
        for _, valid_idx in folds:
            p, s, b = positive[valid_idx], values[valid_idx], baseline[valid_idx]
            gains.append(recall_at_fpr(p, s, fpr)['true_positives'] - recall_at_fpr(p, b, fpr)['true_positives'])
        gain = result['true_positives'] - base['true_positives']
        print(f"{name:26s} {result['true_positives']:4d} {result['false_positives']:4d} {gain:5d} "
              f"{gain / base['true_positives']:6.1%} {review_count(positive, values):8d} {gains}", flush=True)
    print('needed for +5%:', int(np.ceil(base['true_positives'] * 1.05)), flush=True)


DEFAULT = {'depth': 6, 'learning_rate': 0.05}


def first_pass(train, valid, rates):
    tier = valid['current_tier']
    p_d = classifier(train, valid, BASE, DEFAULT)
    p_x = classifier(train, valid, EXPANDED, DEFAULT)
    d_rmse = distance(train, valid, EXPANDED, 'RMSE', DEFAULT)
    d_low = distance(train, valid, EXPANDED, 'Quantile:alpha=0.2', DEFAULT)
    return {
        'D classifier': p_d,
        'D + fallback': alert_scores(p_d, tier, rates),
        'expanded classifier': p_x,
        'expanded + fallback': alert_scores(p_x, tier, rates),
        'distance RMSE': d_rmse,
        'distance low quantile': d_low,
        'classifier+distance rank': ranks(p_x) + ranks(d_rmse),
    }


if __name__ == '__main__':
    started = time.time()
    report(score_all(first_pass))
    print(f'\nseconds {time.time() - started:.0f}', flush=True)
