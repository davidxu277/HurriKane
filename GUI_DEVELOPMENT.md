# ORBITGUARD GUI MVP

## Run locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-gui.txt
.venv/bin/python -m streamlit run app.py
```

On Windows use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

The app starts in **SIMULATION DATA** mode without importing `service` or loading models. The 20 event fixtures include Event #1024, all risk/quality states, missing fields and histories, and seven-day boundary cases. Fixture history uses log10(Pc); fixture metadata is demonstration context, not a real model version.

## Backend integration

Select **Backend CSV**, upload raw messages, then select **Analyze CSV**. The only analysis entry point is `service.analyze_csv(raw_df, model_dir=Path(__file__).resolve().parent / 'models')`, called lazily by `gui_backend.py`. CSV parsing uses `low_memory=False` and string event IDs, preserving leading zeroes and nonnumeric IDs. Install `requirements-gui.txt` for Streamlit 1.50, pandas, Plotly, NumPy, and CatBoost. The frozen `models/metadata.json`, `estimate.cbm`, `conservative.cbm`, and `classifier.cbm` must be present. No training occurs.

Service 1.1 returns `records`, `events`, `history`, `issues`, `summary`, and `metadata`. Raw messages are retained separately for traceability. The GUI consumes service-provided `event_count`, `critical_count`, `elevated_count`, `safe_count`, `unknown_count`, and `needs_review_count` aliases directly; their original `*_events` totals remain available. The service-provided `window_24h_count`/`window_24h_note` and nested `window_24h` coverage status are preserved. Insufficient coverage is never interpreted as zero risk.

`events` supplies the list and details. `risk_level` remains current risk; `predicted_final_tier` and `prediction_status` are shown separately. `tier_pc` is Latest Known Pc in final mode (Current Estimated Pc in current mode). Missing backend numeric values display —, never zero; Simulation keeps its existing N/A presentation. Ranking Score is not Pc; finite nonnegative service scores may exceed one. SAFE and NEEDS_REVIEW remain separate badges. Original safety explanations, including Chinese text, remain unchanged and are escaped for HTML display.

Risk History uses the real service's `history` DataFrame directly. Service 1.1 returns only validated observations that actually participate in inference, excludes blocked events, and enforces the final-mode model horizon. Metadata `history_risk_scale` must equal `log10(Pc)` before the risk chart is shown. Smaller time to TCA means newer observations. No frontend reconstruction from raw input or prediction records occurs; missing or unconfirmed history displays unavailable data. The service history retains `input_record_id` and `row_position` for traceability. R/T/N uses the selected event's returned coordinates and remains a schematic, not an orbit. Current frozen metadata identifies final task, variant E, horizon 2 days, model version `20261031-064613`; the GUI displays metadata dynamically.

Input issues retain the complete real service schema (`input_record_id`, `row_position`, `event_id`, `column`, `code`, `severity`, `value`, `message`) plus the new `row_number`, `field`, and `problem` display columns. `row_position` is zero-based; `row_number` is one-based. ERROR and WARNING totals use separate UI states. Complete processing-result downloads use `records`; input-issues downloads use `issues`. The existing displayed-event export remains available. Downloads and metadata remain accessible for empty/invalid results.

Analysis is saved in per-session state and keyed to uploaded content. Navigating, selecting events, and filtering do not rerun inference. Changing the file invalidates its displayed result; a failed analysis never displays old or mock results. Service statuses `partial`, `invalid_input`, `no_eligible_input`, and `empty` show explicit messages. Model loading and parser failures appear in the Sidebar. **REAL BACKEND ANALYSIS** distinguishes processed uploads from **SIMULATION DATA**; using a synthetic upload still runs the real service but does not imply live telemetry.

## Display statistics

The seven-day card counts unique CRITICAL/ELEVATED event IDs with `0 <= time_to_tca_hours <= 168`, independently of the backend 24-hour statistic. It is a frontend display statistic, not an ML prediction horizon. A null backend 24-hour count displays its coverage note, never zero.

Time buckets are [0,24], (24,48], (48,72], (72,120], (120,168] hours. Negative, missing and later times are excluded. Donut counts come directly from backend summary; percentages are computed by Plotly from these counts. UNKNOWN is unclassified, not an official fourth Airbus tier.

Default ordering is CRITICAL, ELEVATED, SAFE, UNKNOWN, then descending ranking score; missing scores sort last within their tier. Event ID breaks ties. Filters and pagination affect the event list and its export; an already selected event can remain open while filtering.

## Verification

```bash
python -m unittest discover -s tests_gui -v
```

GUI tests cover contract adapters, counts, time boundaries, sorting, invalid probabilities, safe explanation rendering, navigation, filters, pagination, event details, missing data, backend error paths and non-invocation in simulation.

Run the unchanged backend service tests with `python -m pytest tests/test_service.py -q`. GUI integration tests also run frozen models on synthetic raw messages; no private data or model fitting is needed. Formal holdout reports are not loaded by the GUI.

Phase 1 modifies only GUI adapters, app presentation, dependency documentation, and GUI tests. Backend code, model artifacts, and reports remain unchanged. Visual fidelity is adapted to Streamlit's native interactive table; risk and quality badges appear in details and alert cards.
