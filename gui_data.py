"""Frontend contract validation and display statistics; no risk inference."""
from collections.abc import Mapping
import math
import pandas as pd

EVENT_FIELDS = ['event_id', 'risk_level', 'time_to_tca_hours', 'miss_distance_m',
                'tier_pc', 'predicted_pc', 'conservative_pc', 'alert_score',
                'trigger_reason', 'data_quality_status', 'data_quality_reason',
                'sigma_ratio', 'relative_position_r', 'relative_position_t', 'relative_position_n']
HISTORY_FIELDS = ['event_id', 'time_to_tca_hours', 'risk', 'miss_distance_m']
ISSUE_FIELDS = ['row_number', 'field', 'problem']
LEVELS = ['CRITICAL', 'ELEVATED', 'SAFE', 'UNKNOWN']
COLORS = {'CRITICAL': '#ff5569', 'ELEVATED': '#ffc444', 'SAFE': '#00d9a1', 'UNKNOWN': '#8da5c3'}
SUMMARY_FIELDS = ['event_count', 'critical_count', 'elevated_count', 'safe_count',
                  'unknown_count', 'needs_review_count', 'window_24h_count', 'window_24h_note']


def field(obj, name, default=None):
    """Accept named mappings or attribute-based service results without changing names."""
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def normalize_result(result):
    warnings = []
    frames = {}
    for name, columns in [('events', EVENT_FIELDS), ('history', HISTORY_FIELDS), ('issues', ISSUE_FIELDS)]:
        value = field(result, name)
        if value is None:
            warnings.append('Backend result is missing the {} collection.'.format(name))
            frame = pd.DataFrame(columns=columns)
        else:
            frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            missing = [c for c in columns if c not in frame]
            if missing:
                warnings.append('{} fields unavailable: {}.'.format(name.title(), ', '.join(missing)))
            frame = frame.reindex(columns=columns)
        frames[name] = frame
    events = frames['events']
    if events['event_id'].isna().any() or events['event_id'].duplicated().any():
        raise ValueError('Events must have a unique, non-missing event_id; analysis was not displayed.')
    if not events.empty:
        ids = pd.to_numeric(events['event_id'], errors='coerce')
        if ids.isna().any() or (ids % 1 != 0).any():
            raise ValueError('The backend event_id contract requires integer identifiers.')
        events['event_id'] = ids.astype('int64')
    for col in EVENT_FIELDS:
        if col not in ['event_id', 'risk_level', 'trigger_reason', 'data_quality_status', 'data_quality_reason']:
            events[col] = pd.to_numeric(events[col], errors='coerce')
    for col in ['tier_pc', 'predicted_pc', 'conservative_pc', 'alert_score']:
        valid = events[col].between(0, 1) | events[col].isna()
        if not valid.all():
            warnings.append('Invalid {} values are displayed as N/A.'.format(col))
            events.loc[~valid, col] = float('nan')
    for col in ['time_to_tca_hours', 'miss_distance_m', 'relative_position_r',
                'relative_position_t', 'relative_position_n']:
        events[col] = events[col].where(events[col].map(lambda x: pd.isna(x) or math.isfinite(x)))
    events['miss_distance_m'] = events['miss_distance_m'].where(events['miss_distance_m'] >= 0)
    if not events['risk_level'].isin(LEVELS).all():
        warnings.append('Some current risk classifications are unavailable. They have not been inferred.')
    history = frames['history']
    for col in HISTORY_FIELDS:
        history[col] = pd.to_numeric(history[col], errors='coerce')
        history[col] = history[col].where(history[col].map(lambda x: pd.isna(x) or math.isfinite(x)))
    history['miss_distance_m'] = history['miss_distance_m'].where(history['miss_distance_m'] >= 0)
    summary = field(result, 'summary', {})
    metadata = field(result, 'metadata', {})
    return {**frames, 'summary': summary, 'metadata': metadata, 'warnings': warnings}


def priority_events(events, search='', risk='All risk levels', quality='All data quality', sort='Risk priority'):
    out = events.copy()
    if search:
        out = out[out['event_id'].astype(str).str.contains(search.strip(), regex=False)]
    if risk != 'All risk levels':
        out = out[out['risk_level'] == risk]
    if quality != 'All data quality':
        out = out[out['data_quality_status'] == quality]
    out['_priority'] = out['risk_level'].map({v: i for i, v in enumerate(LEVELS)}).fillna(4)
    if sort == 'Time to TCA':
        keys, ascending = ['time_to_tca_hours', 'event_id'], [True, True]
    elif sort == 'Ranking score':
        keys, ascending = ['alert_score', 'event_id'], [False, True]
    else:
        keys, ascending = ['_priority', 'alert_score', 'event_id'], [True, False, True]
    return out.sort_values(keys, ascending=ascending, na_position='last').drop(columns='_priority')


def seven_day_count(events):
    mask = events['time_to_tca_hours'].between(0, 168, inclusive='both') & events['risk_level'].isin(LEVELS[:2])
    return events.loc[mask, 'event_id'].nunique()


def time_counts(events):
    """[0,24], (24,48], (48,72], (72,120], (120,168]; no overlap."""
    data = events[events['time_to_tca_hours'].between(0, 168)].copy()
    labels = ['0–24h', '1–2d', '2–3d', '3–5d', '5–7d']
    data['window'] = pd.cut(data['time_to_tca_hours'], [0, 24, 48, 72, 120, 168],
                            labels=labels, include_lowest=True, right=True)
    return pd.crosstab(data['window'], data['risk_level']).reindex(index=labels, columns=LEVELS, fill_value=0)


def number(value, suffix='', scientific=False):
    try:
        value = float(value)
        if not math.isfinite(value):
            return 'N/A'
        return ('{:.1e}' if scientific else '{:,.2f}').format(value) + suffix
    except (TypeError, ValueError):
        return 'N/A'


def explanation(value):
    """Preserve safety text verbatim. English translation requires a verified backend mapping."""
    if value is None or pd.isna(value) or not str(value).strip():
        return 'N/A'
    return str(value)
