# Streamlit deployment

Deployment branch: `deploy/streamlit`. Stable source: `f30a9a3142eefe449e11f46c50bb45ebc6e45536`.

Deploy `davidxu277/HurriKane`, branch `deploy/streamlit`, entrypoint `app.py` on Streamlit Community Cloud. Select Python 3.12 in Advanced settings; the stable local demo currently uses Python 3.9, which is no longer supported by Community Cloud. No secrets are required by the application.

`requirements.txt` pins the five runtime dependencies to the installed stable demo versions. Application code, backend semantics, theme, and frozen models are identical to the stable source. This branch is developed in a separate worktree; do not merge it into the stable demo branch or main.

All four tracked model files must be present in `models/`: `metadata.json`, `estimate.cbm`, `conservative.cbm`, `classifier.cbm`. Runtime analysis resolves this directory relative to `gui_backend.py`, never from a developer-specific path. Reports and raw training data are not runtime inputs.

The initial page has no loaded data. Simulation loads only after selection. Backend CSV always calls the real service with frozen models. Keep the app's 24-hour coverage warnings and independent review badges.

Before sharing a public URL, verify empty startup, explicit Simulation, Backend CSV with a synthetic non-private raw-message CSV, event selection, separate current/final risk and review states, history/R/T/N updates, and processing/issue downloads. Do not upload private datasets or holdout labels for smoke tests.

Streamlit's default per-file uploader limit is 200 MB; this is an upload limit, not a guarantee that the cloud resource budget can analyze a file of that size. No limit or processing logic is changed by this branch. Start cloud checks with a small synthetic file. Community Cloud resource limits vary and idle apps may hibernate; measure actual remote behavior before relying on it for a live demo. The untouched local checkout remains the primary stable demo.
