"""Lazy public-service adapter. Never imported in simulation mode."""
from io import BytesIO
import pandas as pd
from gui_data import normalize_result


def analyze_upload(payload):
    if not payload.strip():
        raise ValueError('The uploaded CSV is empty.')
    df = pd.read_csv(BytesIO(payload))
    if df.empty or len(df.columns) < 2:
        raise ValueError('Input must be a non-empty CSV with a valid column structure.')
    from service import analyze_csv
    return normalize_result(analyze_csv(df, 'models'))
