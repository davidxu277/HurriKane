"""Deterministic simulation fixtures in the exact public backend schema."""
import pandas as pd
from gui_data import EVENT_FIELDS, HISTORY_FIELDS, ISSUE_FIELDS


def mock_result():
    # These classifications and explanations are fixture values, not frontend predictions.
    fixtures = [
        (32, 'CRITICAL', 4.2, 420, 3.4e-4, 2.8e-4, 5.1e-4, .97, 'OK'),
        (178, 'CRITICAL', 12.6, 730, 1.6e-4, 1.2e-4, 2.3e-4, .94, 'NEEDS_REVIEW'),
        (55, 'CRITICAL', 18.3, 980, 1.2e-4, 8e-5, 1.8e-4, .89, 'OK'),
        (1024, 'ELEVATED', 52.5, 3200, 2.1e-5, 3e-9, 4.2e-4, .73, 'NEEDS_REVIEW'),
        (196, 'ELEVATED', 26.1, 4800, 7.3e-6, 4e-6, 1.1e-5, .68, 'OK'),
        (288, 'ELEVATED', 61.3, 2900, 5.6e-6, 3e-6, 9.1e-6, .61, 'OK'),
        (441, 'SAFE', 72.1, 7400, 8.2e-7, 4e-7, 1.3e-6, .32, 'NEEDS_REVIEW'),
        (520, 'UNKNOWN', 36.8, None, None, None, None, .21, 'NEEDS_REVIEW'),
        (731, 'SAFE', 90.2, 8100, 2.4e-7, 1e-7, 4.6e-7, .19, 'OK'),
        (815, 'UNKNOWN', 48.5, 6000, None, None, None, .18, 'NEEDS_REVIEW'),
        (933, 'SAFE', 102.4, 9600, 1.1e-7, 8e-8, 2e-7, .16, 'OK'),
        (1078, 'ELEVATED', 110.7, 4500, 6.3e-6, 3e-6, 7.5e-6, .15, 'NEEDS_REVIEW'),
        (1102, 'UNKNOWN', 118.3, None, None, None, None, None, 'NEEDS_REVIEW'),
        (1216, 'SAFE', 132.6, 12400, 6.4e-8, 3e-8, 1.2e-7, .11, 'OK'),
        (1350, 'SAFE', 161.1, 15200, 8.9e-8, 4e-8, 1.6e-7, .08, 'OK'),
        (1400, 'ELEVATED', 168, 4000, 2e-6, 1e-6, 4e-6, .42, 'OK'),
        (1401, 'CRITICAL', 0, 900, 2e-5, 1e-5, 5e-5, .82, 'OK'),
        (1402, 'CRITICAL', 168.1, 800, 2e-4, 1e-4, 3e-4, .8, 'OK'),
        (1403, 'ELEVATED', -1, 4800, 2e-6, 1e-6, 3e-6, .4, 'OK'),
        (1404, 'SAFE', None, 9200, 1e-8, 1e-8, 2e-8, None, 'OK'),
    ]
    # Distinct synthetic relative positions; no real satellite or orbit is implied.
    coordinates = {
        32: (120, -300, 268), 178: (-200, 650, -266), 55: (500, -720, 437),
        1024: (120, -2900, 850), 196: (-1200, 4500, 1166), 288: (600, -2700, -870),
        441: (1800, 7000, 1600), 731: (-2200, 7600, -1700), 815: (-2000, -5600, 800),
        933: (4500, -8000, 2800), 1078: (-500, -4300, 1230), 1216: (3200, -11400, 3700),
        1350: (-4500, 14000, 3900), 1400: (1500, -3500, -1200), 1401: (-600, 600, 300),
        1402: (300, -700, 245), 1403: (-2200, -4200, 750), 1404: (4000, 8000, -2000),
    }
    rows = []
    for eid, level, hours, distance, latest, predicted, conservative, score, quality in fixtures:
        reason = {
            'CRITICAL': 'Latest Pc or miss distance meets the Critical criterion.',
            'ELEVATED': 'Latest Pc or miss distance meets the Elevated criterion; not Critical.',
            'SAFE': 'Latest Pc is below 1e-6 and miss distance exceeds 5,000 m.',
            'UNKNOWN': 'Required Pc or distance information is unavailable; current risk cannot be classified.',
        }[level]
        review = 'Additional evidence requires operator review.' if quality == 'NEEDS_REVIEW' else ''
        if eid == 1024:
            reason = 'Latest Pc 2.1e-05 ≥ 1e-06; miss distance 3,200 m ≤ 5,000 m. Neither Critical criterion is met.'
            review = 'Conservative estimate 4.2e-04 exceeds the Critical Pc threshold of 1e-04.'
        if eid in (520, 815, 1102):
            review = 'Missing inputs prevent a reliable current risk classification.'
        if eid == 441:
            review = 'Conservative estimate 1.3e-06 crosses the Elevated Pc threshold of 1e-06.'
        rows.append(dict(zip(EVENT_FIELDS, [eid, level, hours, distance, latest, predicted,
                    conservative, score, reason, quality, review, .08 if eid == 1024 else .03,
                    *coordinates.get(eid, (None, None, None))])))
    events = pd.DataFrame(rows, columns=EVENT_FIELDS)
    # Fixture history is explicitly historical input, never generated from future model output.
    history = pd.DataFrame([
        (1024, 168, -7.1, 18000), (1024, 144, -6.8, 15000),
        (1024, 120, -6.3, 12000), (1024, 96, -5.7, 9000),
        (1024, 72, -5.0, 6000), (1024, 52.5, -4.67778, 3200),
        (32, 168, -4.2, 1600), (32, 120, -3.8, 900), (32, 48, -3.4685, 420),
    ], columns=HISTORY_FIELDS)
    # Additional synthetic INPUT observations, all at/before the 48-hour cutoff.
    # Keep the supplied #1024 history and the existing #32 example. Missing-history
    # fixtures remain missing; these are not predicted or future trajectories.
    import math
    histories = []
    for index, event in enumerate(events.itertuples(index=False)):
        if event.event_id in (32, 1024, 520, 815, 1102, 1404):
            continue
        cutoff = max(48, event.time_to_tca_hours)
        offsets = (-.7, -.25, -.4, 0) if index % 2 else (.5, .2, -.1, 0)
        distance_factors = (1.8, 1.35, 1.1, 1) if index % 2 else (1.3, 1.6, 1.2, 1)
        for hours, risk_offset, factor in zip((120, 72, 24, 0), offsets, distance_factors):
            histories.append((event.event_id, cutoff + hours,
                              math.log10(event.tier_pc) + risk_offset, event.miss_distance_m * factor))
    history = pd.concat([history, pd.DataFrame(histories, columns=HISTORY_FIELDS)], ignore_index=True)
    counts = events['risk_level'].value_counts()
    summary = {'event_count': len(events), 'needs_review_count': int(events['data_quality_status'].eq('NEEDS_REVIEW').sum()),
               'window_24h_count': None,
               'window_24h_note': 'No common observation timestamp is defined for this simulation; a synchronized 24-hour count is unavailable.'}
    summary.update({level.lower() + '_count': int(counts.get(level, 0)) for level in ['CRITICAL', 'ELEVATED', 'SAFE', 'UNKNOWN']})
    return {'events': events, 'history': history,
            'issues': pd.DataFrame([(8, 'miss_distance_m', 'Simulation example: distance unavailable.'),
                                    (10, 'tier_pc', 'Simulation example: latest Pc unavailable.')], columns=ISSUE_FIELDS),
            'summary': summary,
            'metadata': {'task': 'final', 'horizon_days': 2, 'alpha': .9, 'model_version': None}}
