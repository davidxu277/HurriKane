# HurriKane — Satellite Collision Risk Assessment

**Live demo:** [https://hurrikane-ox7w3xb4azkfwnzpdqmezd.streamlit.app/](https://hurrikane-ox7w3xb4azkfwnzpdqmezd.streamlit.app/)

A Python implementation for Autonomous SG 2026 · Airbus Challenge #1. The project uses historical messages for each conjunction event to predict its final risk tier and `log10(Pc)`, with input validation, current risk grading, insufficient-evidence flags, and review workload analysis.

This README describes the current code and bundled models, replacing the early six-hour development plan. The current model is **`E`**, version **`20261031-064613`**, with the **ORBITGUARD** Streamlit dashboard.

> **Models, thresholds, and event splits are frozen. Do not retrain or evaluate the holdout again.** The holdout has been inspected 3 times, and E is the final selection; see the [final model record](reports/final_model_summary.md). Competition materials are handed over to the teammate responsible for submission; see the [submission handoff](reports/submission_plan.md). A GitHub update does not constitute a valid competition submission.

## 1. Current implementation

- `validation.py`: validates raw messages, distinguishes ERROR from WARNING, and preserves issue locations and reasons.
- `ml.py`, `train.py`: construct event samples, train CatBoost models, and save models, validation predictions, and metrics.
- `service.py`: exposes `analyze_csv()`, combining validation, cached model inference, risk rules, history, and event summaries.
- `rules.py`: assigns current risk levels, trigger reasons, and `NEEDS_REVIEW` flags.
- `app.py`, `gui_*.py`: provide the ORBITGUARD dashboard with simulation, real CSV analysis, filters, event details, risk history, R/T/N schematics, and downloads.
- Evaluation scripts: analyze review workload, paired bootstrap intervals, evidence rules, and candidate comparisons from saved predictions. Historical training and holdout scripts remain available for traceability.

Dashboard dependencies are listed in [`requirements-gui.txt`](requirements-gui.txt). See the [GUI development guide](GUI_DEVELOPMENT.md) for integration details. The Python service interface also supports direct batch analysis.

## 2. Environment and data

Use Python 3.11 or 3.12 and run commands from the repository root.

Windows PowerShell:

```powershell
git clone https://github.com/davidxu277/HurriKane.git
Set-Location HurriKane
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-gui.txt
python -m pip install scikit-learn matplotlib pytest "pandas>=2.2,<2.4"
```

macOS / Linux:

```bash
git clone https://github.com/davidxu277/HurriKane.git
cd HurriKane
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gui.txt
python -m pip install scikit-learn matplotlib pytest "pandas>=2.2,<2.4"
```

The additional dependency command installs training, evaluation, and test dependencies, and selects pandas 2.2 or 2.3 to support `groupby.apply(include_groups=False)` in the evidence audit. The dependency file specifies version ranges; dependencies are not fully locked.

Start the dashboard after installation:

```bash
python -m streamlit run app.py
```

Select a data source in the sidebar:

1. **Simulation**: displays clearly labeled mock data for demonstration without loading the real service or models.
2. **Backend CSV**: upload raw message CSV data and click **Analyze CSV** to call the real service. Models are loaded from `models/` beside the app; no training occurs.

The Mission Overview and Collision Events pages show current levels, predicted final tiers, quality flags, actual input history, and R/T/N schematics. Processing results and input issues are downloadable. The app shows an empty state before a data source is selected. Changing the file invalidates the previous analysis; filtering and pagination do not rerun inference.

Obtain the organizer's raw data separately and place it here:

```text
Data_raw/
  train_data.csv
  test_data.csv
```

Commands in this README use `Data_raw/`. The `data/raw/` paths in older reports and model metadata record the historical training location; metadata does not need to be edited for inference.

Input CSV files must contain at least these columns, including historical `risk` for the bundled `final` model:

```text
event_id, time_to_tca, miss_distance, relative_speed,
relative_position_r, relative_position_t, relative_position_n,
c_object_type, c_sigma_r, t_sigma_r, max_risk_estimate, F10, AP, risk
```

| Field | Meaning / unit |
| --- | --- |
| `event_id` | Event identifier, used for grouping only, not as a model feature |
| `time_to_tca` | Days until closest approach; smaller values indicate newer messages |
| `risk` | Known historical `log10(Pc)`, not raw probability; must be ≤ 0 |
| `miss_distance`, `relative_position_r/t/n`, `c_sigma_r/t_sigma_r` | Meters |
| `relative_speed` | Meters/second |

E's physical features also read `c_sigma_t`, `t_sigma_t`, `c_sigma_n`, and `t_sigma_n`, so retain the original CSV's full set of columns where possible. Missing values or absent columns for these four inputs are treated as NaN. Derived feature columns do not need to be uploaded.

A missing required column causes a file-level ERROR. Except for event ID, time, and historical risk in final mode, permitted missing values remain NaN and produce WARNING entries rather than being filled with zero. Unparseable numeric values, infinities, negative distances, and negative sigma values produce ERROR entries.

## 3. Analyze a CSV with the saved models

`models/` contains `estimate.cbm`, `conservative.cbm`, `classifier.cbm`, and `metadata.json`. Inference with these models does not require retraining.

Run the following Python code from the repository root:

```python
from pathlib import Path

import pandas as pd

from service import analyze_csv

raw = pd.read_csv(
    "Data_raw/test_data.csv",
    low_memory=False,
    dtype={"event_id": str},
)
result = analyze_csv(raw, model_dir="models")

print(result["summary"])
print(result["events"][
    ["event_id", "risk_level", "predicted_final_tier",
     "data_quality_status", "prediction_status"]
].head())

output_dir = Path("reports/inference")
output_dir.mkdir(parents=True, exist_ok=True)
for name in ("records", "events", "history", "issues"):
    result[name].to_csv(
        output_dir / f"{name}.csv", index=False, encoding="utf-8-sig"
    )
```

Save this Python example as a local script and run `python <script_name>.py`. The export directory is separate from existing training and holdout reports.

The service returns six entries:

| Key | Type | Purpose |
| --- | --- | --- |
| `records` | DataFrame | Inference and rule results; one row per event in final mode, including placeholders for unpredicted events |
| `events` | DataFrame | Event table, ordered by current severity and then descending `alert_score` |
| `history` | DataFrame | Valid historical messages actually used for inference; `risk` retains its log10 scale |
| `issues` | DataFrame | Validation and service issues, including row position, event, code, and reason |
| `summary` | dict | Row and event counts, processing status, level counts, and 24-hour coverage |
| `metadata` | dict | A copy of model configuration with rule and service versions |

Models are cached within the process by directory, modification time, and file size. The service does not train models or automatically save reports. Missing or corrupt models raise exceptions.

## 4. Task definition and models

### Samples and labels

`train.py` currently fixes `TASK = 'final'` and has no `--task` argument.

1. Each event uses only messages with `time_to_tca >= horizon_days`; the default cutoff is **2 days** before TCA.
2. The latest eligible message supplies the main sample record. Deltas, message counts, and historical statistics use only messages in that input window.
3. Labels use the `risk` and `miss_distance` from the last message across the entire event, applying the grading thresholds to obtain `final_tier`. The code does not require the label message to be within 1 day of TCA.
4. `has_future=True` means the event has a later message with `time_to_tca < horizon_days`. For events without later messages, the final label is already present in the input, so the primary evaluation includes only events with later messages.

`ml.py` and the service retain support for `current` mode, but the bundled models and training CLI use final mode. Renaming an argument or editing metadata does not convert the saved model to current mode.

### Features and three models

E uses **37 features**: 12 base features, 5 final-mode features, 6 historical statistics, and 14 derived physical features. Physical features include combined sigma per axis, normalized relative positions, distance-threshold margins, a trend-based projected distance, and distance-range proportions computed using fixed random samples. See the feature constants in `ml.py` for the full list. Categorical features are `c_object_type` and `current_tier`.

| File | Model | Output meaning |
| --- | --- | --- |
| `estimate.cbm` | CatBoost RMSE regression | Point estimate of final `log10(Pc)`; **not a median model** |
| `conservative.cbm` | CatBoost quantile regression | Conservative estimate; alpha=0.8/0.9/0.95 are compared, and the bundled model uses 0.9 |
| `classifier.cbm` | CatBoost three-class classifier | Scores for final Safe / Elevated / Critical tiers |

Regression targets are floored at -10, and regression error evaluation uses the same clipping. `conservative_log10_pc` is clamped to at least the point estimate. The quantile output is not a calibrated confidence bound.

The default variant **E** uses physical features, selected classifier parameters, and a current-tier safeguard, with **no stale-message time rule**. Both `critical_threshold` and `stale_days` are null: final classification first selects the highest-scoring tier, then ensures an event currently classified as Critical cannot be downgraded.

`alert_score` still uses the training Critical rate for the current tier as a floor, with a small classifier-score contribution for ranking. E's discrete final tier is not determined by a threshold on this ranking score. Classifier parameters are depth=7, learning_rate=0.0652, l2_leaf_reg=6.7, and rsm=0.76; the tuned variant uses at most 300 iterations. The RMSE and quantile regressors use their own regression settings.

### Tier and probability fields

| Field | Meaning |
| --- | --- |
| `current_tier` | Tier computed from known risk and distance in the latest eligible input message |
| `predicted_final_tier` | Final tier predicted by the classifier and the current-Critical safeguard |
| `risk_level` | Current level assigned by service rules, including UNKNOWN, using uppercase names |
| `tier_pc` / `tier_log10_pc` | Evidence used for current grading; latest known risk in final mode |
| `predicted_pc` / `predicted_log10_pc` | Final risk predicted by the regressor |
| `conservative_pc` | Conservative regression estimate |
| `p_safe/p_elevated/p_critical` | Three-class model outputs, not physical collision probabilities |
| `alert_score` | Event ranking score, not collision probability |
| `baseline_score` | Baseline ranking score: current tier first, then closer distance within a tier |

Distinguish `risk_level` from `predicted_final_tier`: current rule-based grading does not replace historical observations with the final-risk regression estimate.

## 5. Current risk rules and data quality

`rules.py` applies these rules in priority order using the grading Pc and distance:

| Current level | Condition |
| --- | --- |
| CRITICAL | Pc ≥ 1e-4 or distance < 1000 m |
| ELEVATED | Not CRITICAL, and either Pc ≥ 1e-6 or distance ≤ 5000 m |
| SAFE | Pc < 1e-6 and distance > 5000 m |
| UNKNOWN | Available evidence cannot establish one of the levels above |

Risk level and quality status are separate. Input warnings, missing evidence, or a conservative estimate that would raise the level can trigger `NEEDS_REVIEW`. A SAFE event may still require review.

The evidence ratio is:

```text
sigma_ratio = sqrt(c_sigma_r² + t_sigma_r²) / miss_distance
```

Review is required when the latest known Pc < 1e-6 and this ratio exceeds the training threshold. The bundled threshold is **0.10816974945035213**; missing sigma must not be treated as zero. The threshold is the 90th percentile of finite ratios among training events with later messages and final label `y < -6`. This differs from the runtime condition based on latest known Pc; see the [holdout review report](reports/holdout_review_submission.md) for the audit.

Any validation ERROR within an event prevents inference for that entire event; a file-level ERROR blocks the entire batch. Duplicate message times within an event's eligible input window also block prediction for that event. Identifiable events without predictions retain `UNKNOWN / NEEDS_REVIEW` placeholders with `prediction_status=NOT_PREDICTED`.

The 24-hour statistic uses `0 < time_to_tca_hours <= 24` and represents only the window reported by each event's latest input message. It does not imply a shared observation time or complete coverage. Default final-mode inputs are at least 48 hours before TCA, so normal predicted records do not cover that window; display `INSUFFICIENT` as insufficient coverage. The dashboard separately counts unique CRITICAL/ELEVATED events with `0 <= time_to_tca_hours <= 168` for its seven-day display. This is separate from the service's 24-hour statistic and is not the model's prediction horizon.

## 6. Training design and frozen state

Events are split 70% / 15% / 15% into training, validation, and holdout sets with seed=42. When class counts are sufficient, splitting is stratified by final tier and whether later messages exist. An existing `split_events.json` is reused and checked for event-set consistency.

| Split | All events | Events with later messages |
| --- | ---: | ---: |
| Training | 8,359 | 6,224 |
| Validation | 1,791 | 1,333 |
| Holdout | 1,792 | 1,335 |

The three production models in the default E variant use training events with later messages. `train.py`, `experiments/`, and `evaluate_holdout.py` remain available to inspect training methods and historical experiments. Normal use only loads the saved models.

**Do not rerun training, create new splits, or generate new holdout predictions.** Historical holdout inspections were D_stale, E, and E_stale, for 3 inspections in total. E was selected; E_stale from the third inspection was not adopted. Repeated inspection may introduce selection bias, so saved results must not be presented as a single independent evaluation on unseen data.

## 7. Saved results and evaluation tools

The figures below come from E's [saved second-inspection metrics](reports/holdout_second_look/holdout_metrics.json) and the [current workload report](reports/workload_holdout.md). They were not produced by new training or evaluation for this README. `reports/holdout_metrics.json` and `reports/holdout_predictions.csv` contain historical first-inspection D_stale results and must not be mixed into final E materials.

The primary scope is `has_future=True`: 1,335 events, including 89 final Critical events.

| Metric | Current-tier / ranking baseline | E |
| --- | ---: | ---: |
| Final three-class accuracy | 91.61% | 92.51% |
| Three-class macro F1 (rounded saved-report values) | 0.853 | 0.861 |
| Critical recall from discrete tier predictions | 72/89 = 80.90% | 73/89 = 82.02% |
| Reviews needed to reach at least 90% recall by ranking | 125 | 125 |
| Critical events found at that review budget | 81/89 = 91.01% | 81/89 = 91.01% |

The primary scope shows **0%** workload reduction, with a paired-event bootstrap 95% interval of **[-13.69%, 26.95%]** from 1,000 resamples. The supplementary scope of all 1,792 events requires 153 → 150 reviews, a reduction of about 1.96%, with an interval of [-11.73%, 10.07%]. Both intervals cross zero, so these results do not establish a stable workload reduction. Macro F1 across all events is approximately 0.866, below the baseline's 0.890.

The saved E predictions have version **`20261031-063722`**, while current model metadata has version **`20261031-064613`**. Existing materials do not establish that the model binaries are identical. Do not describe saved-prediction metrics as measurements from new inference with the current bundle. See the [final model record](reports/final_model_summary.md).

A ranking budget is a retrospective measurement using known labels. It differs from discrete-tier recall and from the service's NEEDS_REVIEW count. The test CSV has no final ground truth and cannot establish final prediction accuracy.

To reproduce only the validation workload report, write outputs to a separate directory:

```bash
python evaluate_workload.py --predictions reports/validation_predictions.csv --split validation --target-recall 0.9 --bootstrap 1000 --seed 42 --output reports/reproduced/workload_validation.json
```

Other tools:

| Script | Behavior |
| --- | --- |
| `evaluate_workload.py` | Computes workload from saved predictions and exports JSON, Markdown, curve CSV, and PNG |
| `evaluate_evidence.py` | Checks saved predictions, input evidence, threshold provenance, and rules without training or model inference |
| `submission_review.py` | Compares saved first-version and candidate predictions and reports replacement eligibility without replacing models or submitting |

`evaluate_evidence.py` defaults to E's saved second-inspection predictions. Matching materials can be specified with `--predictions`, `--metrics`, `--models`, `--split`, and `--workload`; paths are relative to the working directory. Commands for reanalyzing saved predictions are in the [review report](reports/holdout_review_submission.md), including version provenance and the record of 3 inspections. Do not regenerate holdout predictions.

`submission_review.py` retains the historical replacement checks: with identical events, truth, and baseline, a candidate is eligible only if its macro F1 is no lower than the first model's and its Critical recall is no lower than the baseline's. The final selection is now frozen.

## 8. Repository layout and tests

```text
HurriKane/
  Data_raw/                 # Raw CSV data obtained separately
  models/                   # Three CatBoost models and metadata
  reports/                  # Splits, predictions, metrics, audits, and handoff
  tests/                    # ML, training, validation, rules, service, and analysis tests
  app.py                    # Streamlit entry point
  gui_backend.py            # Upload adapter that loads the service lazily
  gui_data.py               # Frontend contracts, sorting, and display statistics
  gui_charts.py             # Plotly charts
  gui_mock.py               # Simulation fixtures
  gui_style.py              # Dashboard styling
  requirements-gui.txt
  GUI_DEVELOPMENT.md
  analysis_provenance.py    # Saved-prediction provenance
  experiments/             # Historical cross-validation and tuning scripts
  tests_gui/               # GUI and real-service integration tests
  ml.py
  train.py
  validation.py
  rules.py
  service.py
  evaluate_holdout.py
  evaluate_workload.py
  evaluate_evidence.py
  submission_review.py
  README.md
```

After installing dependencies, run the existing tests:

```bash
python -m pytest tests -q
python -m unittest discover -s tests_gui -v
```

Training pipeline tests use synthetic data and temporary model directories. GUI integration tests use synthetic messages without reading actual holdout labels. See the [GUI development guide](GUI_DEVELOPMENT.md), [input validation interface](validation.py), and [service interface](service.py) for details.

## 9. Scope and limitations

The project predicts dataset risk indicators and final tiers to support event ranking and human review. It does not perform orbital simulation, maneuver planning, or automatic collision avoidance. R/T/N values describe relative positions, not reconstructed orbits. Classifier scores and regression quantiles must not be presented as physically validated collision probabilities or confidence guarantees.

E does not use a stale-message time rule. All-event metrics include events without later messages, and the final model was selected after repeated holdout inspection; disclose the evaluation scope and versions. Passing input validation only means the data can be processed. `summary.status == "ok"` does not mean all events are safe. Simulation is a mock demonstration, and real CSV analysis does not imply live telemetry. Competition materials and submission receipts are maintained by the teammate responsible for submission; see the [submission handoff](reports/submission_plan.md).
