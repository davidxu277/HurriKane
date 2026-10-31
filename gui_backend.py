"""Lazy public-service adapter. Never imported in simulation mode."""
from io import BytesIO
from pathlib import Path
import pandas as pd
from gui_data import normalize_service_result


def analyze_upload(payload):
    if not payload.strip():
        raise ValueError('The uploaded CSV is empty.')
    try:
        df = pd.read_csv(BytesIO(payload), low_memory=False, dtype={'event_id': str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError('CSV could not be read: {}'.format(exc)) from exc
    from service import analyze_csv
    result = analyze_csv(df, model_dir=Path(__file__).resolve().parent / 'models')
    return normalize_service_result(result, df)
