# HurriKane — Satellite Collision Risk Assessment

Three-person, six-hour implementation plan for **Autonomous SG 2026 · Airbus Challenge #1**.

Updated: 31 October 2026. Adapted from the team's latest Chinese planning document.

> **Status:** This is an implementation plan, not a record of completed work. Dataset assumptions must be checked against this year's data and organizer instructions. All results remain `[TO BE MEASURED]` until evaluated. A has claimed the ML role; B and C still need named owners.

## 1. Objective and scope

Our assumed user is a satellite operator's on-duty collision avoidance analyst. The system prioritizes conjunction events for human review and identifies cases that cannot confidently be treated as safe.

**Goal:** Reduce the number of events analysts must review while detecting the same proportion of high-risk events, and make the system's limitations visible. Maneuver decisions remain with humans.

The model estimates the dataset's calculated risk indicator; it does not provide a collision probability validated against actual collisions. R/T/N plots show relative positions, not reconstructed orbits.

**Stack:** Python 3.11 or 3.12, pandas, NumPy, CatBoost, scikit-learn, Streamlit, Plotly, and pytest.

### Required deliverables

- CSV upload with minimum field and value validation.
- Constant and latest-risk baselines; use a `max_risk_estimate` baseline in `current` mode.
- Two CatBoost quantile models predicting `log10(Pc)`: a median estimate and a conservative estimate.
- A CatBoost classifier providing an alert-ranking score.
- Event-isolated training, validation, and holdout evaluation.
- Risk levels with explicit trigger reasons, insufficient-evidence flags, and threshold-crossing review flags.
- Review workload comparison at matched recall, including uncertainty estimates.
- One Streamlit page with summary metrics, event table, event details, R/T/N plot, risk history, and result download.
- A 24-hour event summary with honest coverage labels.
- One real failure case, reproducible local commands, and a three-minute demonstration.

**Optional after the full pipeline works at hour 3:** Top three CatBoost SHAP contributions. These explain the model, not physical causes.

**Out of scope:** Full orbital simulation, maneuver planning or execution, 3D Earth, separate frontend/backend, database, cloud deployment, authentication, live feeds, LLM chat, hyperparameter search, a physical-feature Ridge baseline, and batch acceleration benchmarks.

## 2. Ownership

| Role | Before hour 3 | After hour 3 | Owned files |
| --- | --- | --- | --- |
| **A — ML lead (claimed)** | Sample construction, event splits, baselines, quantile models, classifier, inference | One targeted improvement, model freeze, holdout evaluation, failure analysis | `ml.py`, `train.py`, `models/`, `reports/metrics.json`, `reports/validation_predictions.csv`, `reports/holdout_predictions.csv` |
| **B — UI and narrative (owner TBD)** | Streamlit page and integration with real outputs | README, demo script, failure-case presentation, rehearsals | `app.py`, `README.md`, `reports/demo_script.md` |
| **C — Data and integration (owner TBD)** | Git coordination, data audit, input validation, risk rules, service layer, integration | Workload analysis, evidence-rule validation, 24-hour statistics, submission checks | `validation.py`, `rules.py`, `service.py`, `evaluate_workload.py`, `tests/test_rules.py`, `reports/data_audit.md`, `reports/workload.json` |

A focuses exclusively on ML. B leads the presentation, C operates the application, and A answers model and validation questions. Confirm the presentation format with the organizers.

### Collaboration rules

1. Agree on task interpretation and interfaces together during the first 20 minutes.
2. Edit only your owned files. Announce interface changes before implementing them.
3. Hold a five-minute sync at each hour: working components, blockers, and next steps.
4. B may develop with clearly labeled mock outputs before hour 3. Demonstrations from hour 3 onward use real model outputs.
5. If the pipeline is broken at hour 3, everyone stops new features and fixes integration.
6. A and C must understand model inputs and outputs; B and C must understand `service.py`.
7. Pull before committing, and commit only your own changes. Agree on ownership of shared configuration files during kickoff.

## 3. Six-hour schedule

Times are relative to the actual development start.

| Time | A — ML | B — UI and narrative | C — Data and integration | Checkpoint |
| --- | --- | --- | --- | --- |
| 00:00–00:20 | Environment and task/interface agreement | Environment and page agreement | Repository coordination, data download, task interpretation | Task and interfaces agreed |
| 00:20–01:00 | Samples, event splits, two baselines | Page scaffold with labeled mock outputs | Data audit, rules, rule tests | Usable data and working baselines |
| 01:00–02:00 | Quantile models, classifier, metadata | Table, details, charts, download | Service layer, event aggregation, validation | Saved models ready for integration |
| 02:00–03:00 | Inference, validation metrics and predictions | Connect real service; fix fields and units | Upload-to-export integration | **One complete run on real data** |
| 03:00–04:00 | Analyze misses; make one targeted improvement | README and script draft; UI fixes | Validation workload and evidence-rule analysis | Credible preliminary findings |
| 04:00–05:00 | Freeze models; evaluate holdout | Add measured results and failure case; rehearse twice | Final holdout analysis, 24-hour counts, submission format | **Feature freeze** |
| 05:00–05:30 | Reproduce inference in a fresh process | Restart app and perform full rehearsal | Verify files and prepare offline backup | Deliverable ready |
| 05:30–06:00 | Troubleshooting support | Prepare presentation | Submit and confirm receipt | Submission complete |

One targeted improvement means inspecting validation failures and changing one relevant feature or weighting choice. It does not mean a hyperparameter search. If repeat submissions are allowed, submit a valid version at hour 5.

## 4. First 20 minutes

### Repository and environment

Use the existing private repository: [davidxu277/Hackers](https://github.com/davidxu277/Hackers). C coordinates access and data sharing. Keep `data/` and `.venv/` out of Git. Store model files in Git if practical; otherwise share them separately and announce their version.

Mac / Linux:

```bash
git clone https://github.com/davidxu277/Hackers.git
cd Hackers
python3 -m venv .venv
.venv/bin/python -m pip install pandas numpy scikit-learn catboost streamlit plotly pytest
```

Windows PowerShell:

```powershell
git clone https://github.com/davidxu277/Hackers.git
Set-Location Hackers
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pandas numpy scikit-learn catboost streamlit plotly pytest
```

### Data audit — C owns this

Obtain `Challenge #1/archive(1).zip` from the dataset link in the challenge PDF. Record the following in `reports/data_audit.md`:

- Filenames, row counts, unique event counts, and messages per event.
- Column lists and train/test differences, especially whether test data contains `risk`.
- Minimum test `time_to_tca`, units, and required submission granularity.
- Any official submission example.
- Risk distribution, including any pile-up at a lower bound such as -30 (**unverified**).
- Event counts with `risk >= -4` and `risk >= -6`.
- Fraction of events with `miss_distance < 1000`; alert the team if the distance rule dominates.

Expected ESA units, **pending confirmation**: `time_to_tca` in days, distance in meters, speed in meters/second, and `risk = log10(Pc)`. Internally use meters; display remaining time in hours and probabilities in scientific notation.

### Choose one task interpretation

1. Follow explicit organizer instructions, dataset documentation, or submission examples first.
2. Otherwise, test data containing `risk` suggests `final` mode: predicting the same row's risk would merely copy an input.
3. No test `risk` column plus required row-level predictions suggests `current` mode.
4. If still unresolved at minute 20, provisionally develop `final`, document the assumption, and continue seeking confirmation.

| Item | `current` — row-level estimation | `final` — final event risk prediction |
| --- | --- | --- |
| Input | Permitted fields of the current row | Messages available before the input cutoff |
| Target | Same row's `risk` | Last message's `risk` for the event |
| `risk` as a feature | **Forbidden: it is the target** | Allowed from historical input messages only |
| Ranking baseline | `max_risk_estimate` | Latest input `risk` |
| Regression baselines | Constant; single-variable regression on `max_risk_estimate` | Constant; latest input `risk` |

If the interpretation changes, rebuild samples, retrain, and reevaluate. Renaming output columns is insufficient.

## 5. Repository structure and contracts

The following files are planned deliverables, not a claim that they already exist.

```text
Hackers/
  data/raw/                  # Organizer data; excluded from Git
  models/                    # median.cbm, conservative.cbm, classifier.cbm, metadata.json
  reports/                   # Audit, metrics, predictions, workload, demo script
  tests/test_rules.py         # C
  ml.py                      # A
  train.py                   # A
  validation.py              # C
  rules.py                   # C
  service.py                 # C
  evaluate_workload.py        # C
  app.py                     # B
  README.md                  # B
  requirements.txt
```

### Shared interfaces

```python
# ml.py — A
build_samples(df, task: str, horizon_days: float) -> DataFrame
# Features, y, event_id; one sample per row in current, per event in final.
load_bundle(model_dir: str) -> dict
predict_batch(df, bundle: dict) -> DataFrame

# validation.py — C
validate_input(df, metadata: dict) -> tuple[DataFrame, DataFrame]
# Processable records and issue list; retain original row IDs, never silently drop rows.

# rules.py — C
apply_risk_rules(predictions, sigma_ratio_threshold: float) -> DataFrame
summarize_events(scored_df) -> DataFrame

# service.py — C; B calls this layer only
analyze_csv(df, model_dir: str) -> dict
# Returns records, events, issues, summary, metadata.
```

### Shared output fields

```text
input_record_id
event_id
time_to_tca_hours
miss_distance_m
predicted_log10_pc
predicted_pc
conservative_log10_pc
conservative_pc
alert_score
baseline_score
sigma_ratio
risk_level
trigger_reason
data_quality_status
data_quality_reason
```

- `predicted_pc`: median-model estimate, used for grading and display.
- `conservative_pc`: high-quantile estimate, displayed as **Conservative estimate**.
- `alert_score`: classifier score used for ranking; it is **not collision probability**.
- `summary`: counts deduplicated events.
- The page does not reimplement conversions or risk rules. Load models once; never retrain in the page.

### Handoff from A to C

A delivers `reports/validation_predictions.csv` before hour 3 and `reports/holdout_predictions.csv` after model freeze. Each contains one row per event with:

```text
event_id, y_true, baseline_score, alert_score,
predicted_log10_pc, conservative_log10_pc, miss_distance_m, sigma_ratio
```

For `current` mode, agree on and document event-level aggregation during kickoff so all event-level comparisons use the same definition.

`models/metadata.json` records task mode, feature order, units, cutoff, label definition, split seed, selected alpha, `sigma_ratio_threshold`, and model version.

## 6. A — ML execution checklist

| Time | Task | Acceptance criterion |
| --- | --- | --- |
| 00:00–00:20 | Confirm task with C | `current` or `final` recorded |
| 00:20–00:40 | Implement sample construction | Correct sample counts; no post-cutoff input |
| 00:40–01:00 | Split events and run baselines | No split overlap; baseline metrics recorded |
| 01:00–01:30 | Train RMSE comparison and median models | Comparison against the task-appropriate baseline |
| 01:30–02:00 | Train conservative model and classifier | All three production models reload successfully |
| 02:00–02:30 | Implement inference and hand off to C | Fresh-process inference works |
| 02:30–03:00 | Compare alpha values; export validation predictions | Alpha chosen and saved |
| 03:00–04:00 | Analyze misses and make one targeted improvement | Before/after validation comparison |
| 04:00–04:30 | Freeze models; evaluate holdout once | Metrics and holdout predictions saved |
| 04:30–05:00 | Analyze one real failure with B | Explain error, user-visible signal, and possible improvement |
| 05:00–05:30 | Reproduce saved predictions | Results match saved outputs |
| 05:30–06:00 | Support submission and prepare Q&A | Explain design and limitations |

### Features and samples

```python
BASE_FEATURES = [
    'time_to_tca', 'miss_distance', 'relative_speed',
    'relative_position_r', 'relative_position_t', 'relative_position_n',
    'c_object_type', 'c_sigma_r', 't_sigma_r',
    'max_risk_estimate', 'F10', 'AP',
]

# final only; compute exclusively from messages available at the cutoff.
FINAL_EXTRA_FEATURES = [
    'risk', 'risk_delta', 'miss_distance_delta', 'message_count',
]
```

`event_id` is for grouping only. With one historical message, leave deltas missing for CatBoost to handle.

- `current`: features and label come from the same valid row, excluding target `risk` from features.
- `final`: take the latest eligible input message with `time_to_tca >= 2`; take the last event message's risk as the label. Require label `time_to_tca <= 1` by default. **Both horizons require dataset confirmation.** Record exclusions.
- Smaller `time_to_tca` means a newer message. Do not reverse the order.
- Never remove a sample merely because its risk is extreme.

### Splits and models

Split **by event** into 70% training, 15% validation, and 15% holdout, seed 42. Stratify using `y >= -6`, verify positive examples in each split, and save event lists. Do not repeatedly resplit based on model performance. Use holdout only after the hour-4 model freeze.

If the audited risk distribution piles up at extremely low values, clip regression targets below -10. Apply the same clipping to baseline/model RMSE evaluation and disclose it.

```python
def make_regressor(loss):
    return CatBoostRegressor(
        iterations=400, depth=6, learning_rate=0.05,
        loss_function=loss, random_seed=42, verbose=100,
    )

median_model = make_regressor('Quantile:alpha=0.5')
conservative_model = make_regressor('Quantile:alpha=0.9')
rmse_model = make_regressor('RMSE')  # Evaluation only, not a production model.

classifier = CatBoostClassifier(
    iterations=400, depth=6, learning_rate=0.05,
    auto_class_weights='Balanced', random_seed=42, verbose=100,
)
```

Use validation-based early stopping and treat `c_object_type` as categorical. Quantile loss already penalizes underestimation asymmetrically; do not implement a custom loss. Compare alpha values 0.8, 0.9, and 0.95 on validation only. Prevent quantile crossing in outputs by taking the maximum of the median and conservative predictions.

The classifier's default target is `y >= -6`. Use `y >= -4` only if there are enough examples and the choice is made before holdout inspection. Select thresholds on validation only.

### Evaluation

Record in `reports/metrics.json`:

- Log10 RMSE and MAE for both baselines, the RMSE model, and the median model.
- Precision, Recall, and F2 for `Pc >= 1e-6` and `Pc >= 1e-4`.
- Positive event counts and missed-event counts alongside percentages.
- One measured total inference time: N records in X seconds.
- One genuine failure case.

For a fair alpha comparison, obtain the baseline false-positive rate on validation, then choose model thresholds matching that rate and compare recall. False-positive rate means the proportion of true negatives flagged. Evaluate Pc labels only; distance-rule detections are not evidence of model improvement. If alpha results are within sampling noise, prefer 0.9.

Report honestly if the model fails to beat the baseline. Do not tune after model freeze or invent accuracy for an unlabeled organizer test set.

A should be ready to explain model choice, why historical `risk` is valid in `final`, matched-FPR selection, the three output quantities, and known failure modes.

## 7. C — Data, rules, integration, and analysis

### Risk grading

Apply rules in priority order using `predicted_pc`:

| Level | Condition |
| --- | --- |
| CRITICAL | Pc ≥ 1e-4 **or** miss distance < 1000 m |
| ELEVATED | Not CRITICAL, and Pc ≥ 1e-6 **or** miss distance ≤ 5000 m |
| SAFE | Pc < 1e-6 **and** miss distance > 5000 m |
| UNKNOWN | Available evidence cannot establish a level above |

These are the challenge thresholds in the supplied plan, not universal operational standards. Reasons must include values, units, and thresholds. Invalid negative distances and infinite input values go to the issue list, not grading. Missing key fields must never silently imply safety.

### Required rule tests

| Pc | Distance (m) | Expected level | Quality status |
| --- | --- | --- | --- |
| 1e-4 | 6000 | CRITICAL | OK |
| 1e-6 | 6000 | ELEVATED | OK |
| 9.9e-7 | 5000 | ELEVATED | OK |
| 9.9e-7 | 5001 | SAFE | OK |
| 1e-7 | 999 | CRITICAL | OK |
| 1e-7 | 1000 | ELEVATED | OK |
| Missing | 800 | CRITICAL | NEEDS_REVIEW |
| 1e-5 | Missing | ELEVATED | NEEDS_REVIEW |
| 1e-7 | Missing | UNKNOWN | NEEDS_REVIEW |
| Missing | 6000 | UNKNOWN | NEEDS_REVIEW |
| Missing | Missing | UNKNOWN | NEEDS_REVIEW |

Threshold-crossing tests, all with distance 6000 m:

| Median estimate | Conservative estimate | Level | Quality status |
| --- | --- | --- | --- |
| 1e-5 | 3e-4 | ELEVATED | NEEDS_REVIEW |
| 1e-5 | 5e-5 | ELEVATED | OK |
| 1e-7 | 5e-6 | SAFE | NEEDS_REVIEW |
| 1e-7 | 5e-7 | SAFE | OK |

These isolated tests assume no additional quality flag is triggered.

### Insufficient-evidence rule

```text
sigma_ratio = sqrt(c_sigma_r² + t_sigma_r²) / miss_distance
```

Set `sigma_ratio_threshold` to the training-set 90th percentile among records with Pc < 1e-6, and save it in metadata. Flag cases with Pc < 1e-6 and a ratio above the threshold as `NEEDS_REVIEW`; preserve the risk level but do not display an unqualified green “safe” indicator. Explain the ratio and threshold. For zero distance, treat the derived ratio as infinity and above threshold.

This is a rough proxy using only two radial sigma values. On validation, compare the proportions becoming high-risk among flagged and unflagged events; confirm once on holdout using the same threshold. If holdout does not support the hypothesis, remove it from the main workflow and report it as an unsupported hypothesis. Do not retune to manufacture a result.

### Threshold-crossing review rule

Grade once using the median estimate and once using the conservative estimate with the same distance. If the conservative grade is higher, preserve the median-based level and set `NEEDS_REVIEW`. State the actual estimates and crossed threshold. Both this reason and an insufficient-evidence reason may apply simultaneously.

Measure the fraction flagged on validation. If excessive, consider a smaller alpha on validation and document the choice; never change it after seeing holdout results.

### Review workload

Develop `evaluate_workload.py` using validation predictions. Use only holdout results in the final demonstration:

1. Rank events separately by `baseline_score` and `alert_score`, descending.
2. Count how many events must be reviewed to capture 90% of true high-risk events.
3. Report both counts and the percentage reduction.
4. Bootstrap events 1,000 times to estimate a 95% interval for the reduction.
5. Plot reviewed-event count versus recall.
6. If there are fewer than 20 high-risk events, label the result as directional only.

Break score ties using `event_id` and save results to `reports/workload.json`. Use the same documented high-risk definition for both rankings.

### Aggregation, time coverage, and validation

- Count each event once, even with multiple messages.
- The 24-hour window is `0 < time_to_tca_hours <= 24`.
- Without a common message timestamp, label it “24-hour window reported by each event's latest message.”
- If `final` inputs are all at least 48 hours before TCA, the window may be empty; show insufficient coverage.
- Historical replay must be labeled and must use only information available at the replay time.
- Minimum validation checks required columns, numeric parsing, negative distance, and infinite values. Preserve row IDs and expose issues instead of silently dropping records.

## 8. B — Page and presentation

### Page layout

1. Title, upload, task mode, and data coverage.
2. Event count, critical count, elevated count, review-required count, and 24-hour summary.
3. Risk-level and time-range filters.
4. Event table: ID, level, time remaining, distance, median estimate, conservative estimate, ranking score, and trigger reasons.
5. Event details: evidence, risk history, quality warnings, model and rule versions.
6. R/T/N relative-position plot with units and a schematic label.
7. Evaluation: baseline comparison, workload curve, and a failure case.
8. Results download.

Sort by severity, then descending `alert_score`. Explain missing fields and unavailable models. If blocked at hour 4, remove filters and charts first; retain the table, details, metrics, and export.

After hour 3, update this README with actual data placement, training, startup, export, task interpretation, and limitations. Write `reports/demo_script.md`, select a real holdout failure with A, and add only measured results.

### Three-minute demonstration

| Time | Content |
| --- | --- |
| 0:00–0:25 | User, current workflow, and pain point |
| 0:25–1:05 | Import data and show ranked events |
| 1:05–1:40 | Inspect a critical event: estimates, distance, reasons |
| 1:40–2:15 | Workload comparison, recall, misses, and sample counts |
| 2:15–2:40 | Evidence-rule validation and one failure case |
| 2:40–3:00 | Scope, limitations, and human decision authority |

Core statement: “While detecting `[TO BE MEASURED]` of high-risk events, the number of events requiring review changed from `[TO BE MEASURED]` to `[TO BE MEASURED]`.” Use “reduced” only if the measurements support it.

Do not present mock data, historical replay, or training performance as unseen-data performance. All members should be able to answer: “Where does this fail, and how does the user know?”

## 9. Contingency plan

| Checkpoint | Problem | Action |
| --- | --- | --- |
| 00:20 | Task unresolved | Provisionally use `final`, document it, continue confirmation |
| 01:00 | Data unreadable | A and C debug; B continues UI; do not fabricate data |
| 02:00 | No model | Use base features, fewer iterations, median model first |
| 03:00 | Broken end-to-end flow | Everyone stops features and fixes upload-to-export |
| 04:00 | Weak model performance | Freeze and report honestly; prioritize credible validation |
| 04:30 | Evidence rule unvalidated | Remove it from the demo; make no unsupported claims |
| 05:00 | Features unfinished | Freeze; fix only prediction, display, export, or startup blockers |
| 05:30 | Submission incomplete | Submit a valid version first and keep a local backup |

Distance rules may provide explicitly labeled fallback warnings if model inference fails. They do not constitute completed AI risk prediction.

## 10. Acceptance checklist

### Correctness

- [ ] Confirm task mode, input cutoff, units, and submission granularity.
- [ ] Exclude `risk` as a feature in `current`; use only pre-cutoff history in `final`.
- [ ] Exclude `event_id` from features.
- [ ] Verify no event overlap among train, validation, and holdout.
- [ ] Pass all rule tests, including boundaries and missing data.
- [ ] Never label missing key evidence as safe.
- [ ] Deduplicate event counts.
- [ ] Distinguish median estimate, conservative estimate, and ranking score in the UI.
- [ ] Ensure `conservative_log10_pc >= predicted_log10_pc`.
- [ ] Compare model recall at matched false-positive rates using Pc labels only.
- [ ] Evaluate holdout once after model freeze.
- [ ] Keep displayed and exported results consistent.

### Evaluation and presentation

- [ ] Explain the user, current workflow, and cost of the problem.
- [ ] Handle valid files beyond the rehearsed example.
- [ ] Include baselines, measured metrics, sample counts, and a genuine failure.
- [ ] Show evidence and uncertainty for alerts; retain human decision authority.
- [ ] Report workload changes and distinguish measurement from assumptions.
- [ ] Complete the presentation in three minutes and submit on time.

### Delivery

- [ ] Reload saved models in a fresh process.
- [ ] Reproduce the workflow using this README.
- [ ] Verify submission column names, count, and granularity against this year's requirements.
- [ ] Keep an offline CSV and screenshots or recording, accurately labeled.
- [ ] Complete at least two full rehearsals.

## 11. Planned command interface

These commands become executable once the relevant scripts are implemented. B must verify and update them before delivery.

```bash
.venv/bin/python -m pytest tests/test_rules.py -v
.venv/bin/python train.py --train data/raw/train_data.csv --task final --horizon-days 2
.venv/bin/python evaluate_workload.py --predictions reports/holdout_predictions.csv
.venv/bin/python -m streamlit run app.py
```

For `current`, replace `--task final --horizon-days 2` with `--task current` and update metadata. On Windows, replace `.venv/bin/python` with `.\.venv\Scripts\python.exe`.

After the environment is stable, record dependencies:

```bash
.venv/bin/python -m pip freeze > requirements.txt
```

CSV export is provided through the application's download button. Formal submission format remains subject to organizer confirmation.

## 12. References

This year's **Autonomous SG 2026_Airbus Track.pdf**, dataset documentation, and organizer clarifications take precedence. The following links are retained from the supplied plan for implementation reference:

- [ESA dataset description](https://kelvins.esa.int/collision-avoidance-challenge/data/)
- [ESA original challenge](https://kelvins.esa.int/collision-avoidance-challenge/challenge/)
- [ESA original submission rules](https://kelvins.esa.int/collision-avoidance-challenge/submission-rules/) — reference only
- [CatBoost quick start](https://catboost.ai/docs/en/concepts/python-quickstart)
- [Streamlit installation](https://docs.streamlit.io/get-started/installation/command-line)
- [GroupShuffleSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)

**Kickoff:** Assign B and C, verify repository access, download the data, and settle task interpretation within 20 minutes. Run the complete real-data pipeline by hour 3 and freeze features at hour 5.
