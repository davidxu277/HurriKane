"""Synthetic saved artifacts only: never train or run holdout inference."""
import hashlib
import json
import sys

import pandas as pd
import pytest

import evaluate_evidence
import evaluate_workload
from analysis_provenance import saved_prediction_context


@pytest.fixture
def saved(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"variant": "E", "model_version": "bundle-version"}))
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"look": "second", "variant": "E", "model_version": "saved-version"}))
    return metadata, metrics


def test_saved_version_is_not_silently_relabelled_as_current_bundle(saved):
    metadata, metrics = saved
    context = saved_prediction_context(metrics, metadata, 3, "1d52485")
    assert context["prediction_model_version"] == "saved-version"
    assert context["current_bundle_model_version"] == "bundle-version"
    assert not context["model_version_matches_current_bundle"]
    assert context["holdout_look_count"] == 3
    assert context["prediction_source_look"] == "second"
    assert not context["new_training_or_inference"]
    assert len(context["warnings"]) == 2
    assert context["metrics_sha256"] == hashlib.sha256(metrics.read_bytes()).hexdigest()


def test_old_variant_cannot_be_audited_as_e(saved):
    metadata, metrics = saved
    metrics.write_text(json.dumps({"variant": "D_stale", "model_version": "old"}))
    with pytest.raises(ValueError, match="variant"):
        saved_prediction_context(metrics, metadata, 3, "1d52485")


def test_matching_versions_still_disclose_three_looks(saved):
    metadata, metrics = saved
    metadata.write_text(json.dumps({"variant": "E", "model_version": "saved-version"}))
    context = saved_prediction_context(metrics, metadata, 3, "1d52485")
    assert context["model_version_matches_current_bundle"]
    assert len(context["warnings"]) == 1
    with pytest.raises(ValueError):
        saved_prediction_context(metrics, metadata, 0, "1d52485")


def test_workload_cli_keeps_saved_provenance_and_three_look_disclosure(saved, tmp_path, monkeypatch):
    metadata, metrics = saved
    source = tmp_path / "holdout_predictions.csv"
    frame = pd.DataFrame({"event_id": [1, 2], "final_tier": ["Critical", "Safe"],
                          "has_future": [True, False], "baseline_score": [1, 0], "alert_score": [1, 0]})
    frame.to_csv(source, index=False)
    before = source.read_bytes()
    expected, _ = evaluate_workload.evaluate_workload(frame, bootstrap_iterations=5)
    data = json.loads(metrics.read_text())
    data["tiers"] = {scope: {"events": report["events"], "critical": {
        "positive_events": report["positive_events"],
        "review_for_90_recall": {key: report[key]["reviews"] for key in ("baseline", "model")}}}
        for scope, report in (("future", expected), ("all", expected["supplementary_all"]))}
    metrics.write_text(json.dumps(data))
    output = tmp_path / "workload.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_workload.py", "--predictions", str(source),
        "--output", str(output), "--bootstrap", "5", "--metrics", str(metrics),
        "--model-metadata", str(metadata), "--holdout-look-count", "3", "--source-commit", "1d52485"])
    monkeypatch.setattr(evaluate_workload, "save_curve", lambda *args: None)
    evaluate_workload.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["purpose"] == "saved_prediction_reanalysis"
    assert report["provenance"]["holdout_look_count"] == 3
    text = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "留出集共看了 3 次" in text
    assert "--holdout-look-count 3" in text
    assert source.read_bytes() == before


@pytest.mark.parametrize("bad_source,bad_context", [(True, False), (False, True)])
def test_evidence_rejects_stale_workload_before_reading_data(saved, tmp_path, monkeypatch, bad_source, bad_context):
    metadata, metrics = saved
    workload = tmp_path / "workload.json"
    context = saved_prediction_context(metrics, metadata, 3, "1d52485")
    workload.write_text(json.dumps({"source_sha256": "wrong" if bad_source else "saved-hash",
                                     "provenance": {} if bad_context else context}))
    monkeypatch.setattr(evaluate_evidence, "sha256", lambda path: "saved-hash")
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Should reject before reading raw data or predictions"))
    monkeypatch.setattr(sys, "argv", ["evaluate_evidence.py", "--models", str(tmp_path),
        "--metrics", str(metrics), "--workload", str(workload)])
    with pytest.raises(AssertionError, match="Workload"):
        evaluate_evidence.main()


def test_evidence_cannot_overwrite_saved_prediction_file(tmp_path, monkeypatch):
    source = tmp_path / "predictions.csv"
    source.write_text("must remain unchanged")
    monkeypatch.setattr(sys, "argv", ["evaluate_evidence.py", "--predictions", str(source), "--output", str(source)])
    with pytest.raises(SystemExit):
        evaluate_evidence.main()
    assert source.read_text() == "must remain unchanged"
