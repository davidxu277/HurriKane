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

Select **Backend CSV**, upload a CSV, then select **Analyze CSV**. The sole analysis call is `service.analyze_csv(df, 'models')`. Result components retain the names `events`, `history`, `issues`, `summary`, and `metadata`. Frontend adaptation accepts mappings or objects with those named attributes; component collections accept DataFrames or row records. No backend implementation is present yet, so these container conventions require confirmation. Tuple results are not guessed.

Analysis is saved in per-session state and keyed to uploaded content. Navigating, selecting events, and filtering do not rerun inference. Changing the file invalidates its displayed result; a failed analysis does not display a previous successful result.

The backend remains authoritative for current risk levels, reasons and summary fields. Invalid display probabilities or ranking scores outside [0,1] become N/A with a warning. The frontend does not reclassify events. Original safety explanations, including Chinese text, remain unchanged; verified English backend explanations or an approved deterministic mapping are still required. HTML does not render backend explanations.

The risk-history axis uses an unverified backend-value label until the actual service confirms its scale. Only returned history rows are plotted; the backend must enforce its cutoff. No future observations are generated. R/T/N is a schematic relative position, not an orbit.

## Display statistics

The seven-day card counts unique CRITICAL/ELEVATED event IDs with `0 <= time_to_tca_hours <= 168`, independently of the backend 24-hour statistic. It is a frontend display statistic, not an ML prediction horizon. A null backend 24-hour count displays its coverage note, never zero.

Time buckets are [0,24], (24,48], (48,72], (72,120], (120,168] hours. Negative, missing and later times are excluded. Donut counts come directly from backend summary; percentages are computed by Plotly from these counts. UNKNOWN is unclassified, not an official fourth Airbus tier.

Default ordering is CRITICAL, ELEVATED, SAFE, UNKNOWN, then descending ranking score; missing scores sort last within their tier. Event ID breaks ties. Filters and pagination affect the event list and its export; an already selected event can remain open while filtering.

## Verification

```bash
python -m unittest discover -s tests_gui -v
```

GUI-only tests cover contract adapters, counts, time boundaries, sorting, invalid probabilities, safe explanation rendering, navigation, filters, pagination, event details, missing data, backend error paths and non-invocation in simulation.

No ML implementation, model training, CI, commits, pushes, or PRs are included. Remaining integration confirmations: actual service container types, risk-history scale/cutoff, English safety explanations, and metadata availability. Visual fidelity is adapted to Streamlit's native interactive table; risk and quality badges appear in details and alert cards.
