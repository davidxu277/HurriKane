"""Provenance for reanalysis of saved predictions; never load or run models."""
import hashlib
import json
from pathlib import Path


def saved_prediction_context(metrics_path, metadata_path, holdout_look_count, source_commit):
    if holdout_look_count < 1:
        raise ValueError("holdout_look_count must be positive")
    metrics_path, metadata_path = Path(metrics_path), Path(metadata_path)
    metrics_bytes, metadata_bytes = metrics_path.read_bytes(), metadata_path.read_bytes()
    metrics = json.loads(metrics_bytes)
    metadata = json.loads(metadata_bytes)
    if metrics["variant"] != metadata["variant"]:
        raise ValueError("Saved prediction variant differs from selected model metadata")
    warnings = ["留出集共看了 %d 次；多次查看后的结果可能偏乐观，不是未见数据上的独立验证。" % holdout_look_count]
    same_version = metrics["model_version"] == metadata["model_version"]
    if not same_version:
        warnings.append("保存预测的 model_version 与当前模型 metadata 不同；只分析指定预测，不重新推理，不宣称已证明当前模型二进制与该预测一致。")
    return {
        "source_commit": source_commit,
        "variant": metrics["variant"],
        "prediction_model_version": metrics["model_version"],
        "current_bundle_model_version": metadata["model_version"],
        "model_version_matches_current_bundle": same_version,
        "prediction_source_look": metrics.get("look"),
        "holdout_look_count": holdout_look_count,
        "operation": "reanalysis_of_saved_predictions_only",
        "new_training_or_inference": False,
        "metrics_file": str(metrics_path),
        "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "metadata_file": str(metadata_path),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "warnings": warnings,
    }
