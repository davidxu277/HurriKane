"""用固定模型评估留出集：不训练、不改模型，结果已存在时拒绝重跑。"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ml import build_samples, combine_predictions, load_bundle, tier_probabilities
from train import PREDICTION_COLUMNS, tier_report
from validation import validate_input


def holdout_predictions(raw, bundle, holdout_events):
    metadata = bundle['metadata']
    valid, issues = validate_input(raw, metadata)
    if issues['severity'].eq('ERROR').any():
        raise ValueError('输入存在 ERROR，留出评估已停止')

    samples = build_samples(valid, metadata['task'], metadata['horizon_days'])
    part = samples[samples['event_id'].isin(holdout_events)]
    if len(part) != len(set(holdout_events)):
        raise ValueError('切分清单里的留出事件和数据对不上')

    features = part[metadata['features']]
    result = combine_predictions(
        part,
        bundle['estimate'].predict(features),
        bundle['conservative'].predict(features),
        tier_probabilities(bundle['classifier'], features),
        metadata['critical_rates'],
        metadata['critical_threshold'],
        metadata.get('stale_days'),
    )
    result['y_true'] = part['y'].to_numpy()
    result['final_tier'] = part['final_tier'].to_numpy()
    result['has_future'] = part['has_future'].to_numpy()
    return result


def report(frame):
    return tier_report(frame['final_tier'], frame['current_tier'], frame['predicted_final_tier'],
                       frame['baseline_score'], frame['alert_score'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--models', default='models')
    parser.add_argument('--split', default='reports/split_events.json')
    parser.add_argument('--reports', default='reports')
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    predictions_path = reports_dir / 'holdout_predictions.csv'
    if predictions_path.exists():
        raise SystemExit(f'{predictions_path} 已存在：留出集只评估一次，不重跑')

    bundle = load_bundle(args.models)
    split = json.loads(Path(args.split).read_text(encoding='utf-8'))
    raw = pd.read_csv(args.train, low_memory=False)
    predictions = holdout_predictions(raw, bundle, split['holdout'])

    predictions[PREDICTION_COLUMNS].to_csv(predictions_path, index=False)
    metrics = {
        'variant': bundle['metadata']['variant'],
        'model_version': bundle['metadata']['model_version'],
        'tiers': {'future': report(predictions[predictions['has_future']]), 'all': report(predictions)},
    }
    (reports_dir / 'holdout_metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding='utf-8'
    )
    print(f'完成，结果在 {reports_dir / "holdout_metrics.json"}')


if __name__ == '__main__':
    main()
