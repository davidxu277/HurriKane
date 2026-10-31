import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import service
from ml import feature_columns


class Regressor:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value)


class Classifier:
    classes_ = np.array(["Critical", "Elevated", "Safe"])

    def predict_proba(self, features):
        return np.tile([0.2, 0.3, 0.5], (len(features), 1))


@pytest.fixture
def models(tmp_path, monkeypatch):
    service.clear_model_cache()
    for name in service.MODEL_FILES:
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    bundle = {
        "metadata": {
            "task": "final", "horizon_days": 2.0, "features": feature_columns("final"),
            "sigma_ratio_threshold": 0.1, "model_version": "test",
            "critical_rates": None, "critical_threshold": 0.5,
        },
        "estimate": Regressor(-9), "conservative": Regressor(-8),
        "classifier": Classifier(),
    }
    calls = []

    def load(directory):
        calls.append(directory)
        return bundle

    monkeypatch.setattr(service, "load_bundle", load)
    yield tmp_path, bundle, calls
    service.clear_model_cache()


def messages():
    return pd.DataFrame({
        "event_id": ["a", "a", "b", "b"],
        "time_to_tca": [3.0, 2.5, 3.0, 2.5], "risk": [-8.0, -7.0, -8.0, -5.0],
        "miss_distance": [6000.0] * 4, "relative_speed": [11000.0] * 4,
        "relative_position_r": [1.0, 2.0, 3.0, 4.0], "relative_position_t": [2.0] * 4,
        "relative_position_n": [3.0] * 4, "c_object_type": ["DEBRIS"] * 4,
        "c_sigma_r": [3.0] * 4, "t_sigma_r": [4.0] * 4,
        "max_risk_estimate": [-4.0] * 4, "F10": [80.0] * 4, "AP": [12.0] * 4,
    }, index=[10, 11, 12, 13])


def analyze(source, models):
    return service.analyze_csv(source, models[0])


def test_full_chain_uses_known_risk_deduplicates_and_preserves_source_details(models):
    source = messages()
    before = source.copy(deep=True)
    result = analyze(source, models)
    assert set(result) == {"records", "events", "issues", "summary", "metadata"}
    assert_frame_equal(source, before)
    assert result["issues"].empty
    events = result["events"].set_index("event_id")
    assert result["events"].event_id.tolist() == ["b", "a"]
    assert events.loc["b", "risk_level"] == "ELEVATED"
    assert events.loc["a", "risk_level"] == "SAFE"
    assert events.loc["b", "tier_pc"] == pytest.approx(1e-5)
    assert events.loc["b", "predicted_pc"] == pytest.approx(1e-9)
    assert events.message_count.tolist() == [2, 2]
    assert events.input_message_count.tolist() == [2, 2]
    assert events.loc["b", "relative_position_r"] == 4
    assert events.loc["b", "input_record_id"] == 13
    assert events.loc["b", "row_position"] == 3
    summary = result["summary"]
    assert summary["input_rows"] == summary["valid_rows"] == 4
    assert summary["total_events"] == summary["predicted_events"] == 2
    assert summary["elevated_events"] == summary["safe_events"] == 1
    assert summary["window_24h"]["total_events"] == 0
    assert summary["window_24h"]["coverage_status"] == "INSUFFICIENT"


def test_cache_reused_invalidated_on_update_and_results_are_not_shared(models):
    first = analyze(messages(), models)
    first["metadata"]["features"].append("bad")
    first["records"].loc[:, "risk_level"] = "BAD"
    second = analyze(messages(), models)
    assert len(models[2]) == 1
    assert "bad" not in second["metadata"]["features"]
    assert "BAD" not in second["records"].risk_level.tolist()
    (models[0] / "metadata.json").write_text("updated fixture", encoding="utf-8")
    analyze(messages(), models)
    assert len(models[2]) == 2


def test_bad_latest_message_blocks_whole_event_instead_of_using_older_safe_message(models):
    source = messages().astype(object)
    source.loc[11, "risk"] = "bad"
    result = analyze(source, models)
    events = result["events"].set_index("event_id")
    assert events.loc["a", "risk_level"] == "UNKNOWN"
    assert events.loc["a", "prediction_status"] == "NOT_PREDICTED"
    assert pd.isna(events.loc["a", "predicted_pc"])
    assert events.loc["a", "data_quality_status"] == "NEEDS_REVIEW"
    assert events.loc["b", "prediction_status"] == "PREDICTED"
    assert result["summary"]["rejected_rows"] == 1
    assert result["summary"]["unpredicted_events"] == 1


@pytest.mark.parametrize("missing", ["risk", "F10", "event_id"])
def test_missing_columns_return_issues_and_stable_result_schemas(models, missing):
    result = analyze(messages().drop(columns=missing), models)
    assert "MISSING_COLUMN" in result["issues"].code.tolist()
    assert result["summary"]["predicted_events"] == 0
    assert result["summary"]["status"] == "invalid_input"
    assert "time_coverage_status" in result["events"]
    if missing != "event_id":
        assert len(result["events"]) == 2
        assert result["events"].risk_level.eq("UNKNOWN").all()


@pytest.mark.parametrize("column", ["event_id", "risk"])
def test_duplicate_columns_do_not_crash_or_infer(models, column):
    source = pd.concat([messages(), messages()[[column]]], axis=1)
    result = analyze(source, models)
    assert result["summary"]["predicted_events"] == 0
    assert result["issues"].code.tolist() == ["DUPLICATE_COLUMN"]


def test_warning_on_history_is_propagated_to_event_and_not_lost_in_prediction(models):
    source = messages()
    source.loc[10, "F10"] = np.nan
    result = analyze(source, models)
    a = result["events"].set_index("event_id").loc["a"]
    assert a.risk_level == "SAFE"
    assert a.data_quality_status == "NEEDS_REVIEW"
    assert "F10" in a.data_quality_reason
    assert result["summary"]["needs_review_events"] == 1


def test_missing_sigma_at_zero_distance_stays_unknown_not_zero(models):
    source = messages()
    source.loc[11, "miss_distance"] = 0
    source.loc[11, "c_sigma_r"] = np.nan
    a = analyze(source, models)["events"].set_index("event_id").loc["a"]
    assert a.risk_level == "CRITICAL"
    assert pd.isna(a.sigma_ratio)
    assert a.data_quality_status == "NEEDS_REVIEW"


@pytest.mark.parametrize("all_events", [False, True])
def test_no_eligible_messages_keep_unknown_events_without_empty_batch_prediction(models, all_events):
    source = messages()
    selected = source.index if all_events else source.index[source.event_id.eq("b")]
    source.loc[selected, "time_to_tca"] = [1.5, 1.0] * (2 if all_events else 1)
    result = analyze(source, models)
    b = result["events"].set_index("event_id").loc["b"]
    assert b.risk_level == "UNKNOWN"
    assert b.message_count == 0 and b.input_message_count == 2
    assert "NO_ELIGIBLE_MESSAGES" in result["issues"].code.tolist()
    assert len(result["events"]) == 2
    assert result["summary"]["predicted_events"] == (0 if all_events else 1)


def test_final_ignores_post_cutoff_values_and_their_warnings(models):
    source = messages()
    future = source.iloc[[1]].copy()
    future.index = [14]
    future["time_to_tca"] = 0.5
    future["risk"] = -2
    future["F10"] = np.nan
    result = analyze(pd.concat([source, future]), models)
    a = result["events"].set_index("event_id").loc["a"]
    assert a.tier_pc == pytest.approx(1e-7)
    assert a.data_quality_status == "OK"
    assert a.message_count == 2 and a.input_message_count == 3
    assert len(result["issues"]) == 1


def test_tied_message_times_block_ambiguous_event(models):
    source = messages()
    source.loc[10, "time_to_tca"] = 2.5
    result = analyze(source, models)
    assert result["events"].set_index("event_id").loc["a", "risk_level"] == "UNKNOWN"
    assert "AMBIGUOUS_MESSAGE_TIME" in result["issues"].code.tolist()


def test_duplicate_source_indices_are_mapped_by_position(models):
    source = messages()
    source.index = [7, 7, 7, 7]
    records = analyze(source, models)["records"].set_index("event_id")
    assert records.input_record_id.tolist() == [7, 7]
    assert records.row_position.tolist() == [1, 3]
    assert records.relative_position_r.tolist() == [2, 4]


@pytest.mark.parametrize("hours,expected", [(0, False), (-1, False), (24, True), (0.1, True), (24.01, False)])
def test_current_uses_each_message_but_window_counts_unique_events(models, hours, expected):
    models[1]["metadata"].update(task="current", features=feature_columns("current"))
    source = messages().drop(columns="risk")
    source.loc[[11, 13], "time_to_tca"] = hours / 24
    result = analyze(source, models)
    assert len(result["records"]) == 4
    assert len(result["events"]) == 2
    assert result["summary"]["window_24h"]["total_events"] == (2 if expected else 0)
    assert result["summary"]["safe_events"] == 2


def test_empty_input_returns_empty_results(models):
    result = analyze(messages().iloc[:0], models)
    assert result["records"].empty and result["events"].empty
    assert result["summary"]["status"] == "empty"
    assert result["summary"]["total_events"] == 0


def test_unidentified_row_is_reported_without_inventing_an_event(models):
    source = messages().astype(object)
    source.loc[11, "event_id"] = None
    result = analyze(source, models)
    assert result["summary"]["unidentified_event_rows"] == 1
    assert len(result["events"]) == 2
    assert result["issues"].iloc[0].input_record_id == 11


def test_missing_models_raise_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="模型未就绪"):
        service.analyze_csv(messages(), tmp_path)


def test_events_carry_predicted_final_tier_for_the_page(models):
    events = analyze(messages(), models)["events"].set_index("event_id")
    assert events.loc["b", "current_tier"] == "Elevated"
    assert events.loc["b", "predicted_final_tier"] == "Safe"
    assert events.loc["b", "p_critical"] == 0.2
    assert events.loc["b", "p_safe"] == 0.5
