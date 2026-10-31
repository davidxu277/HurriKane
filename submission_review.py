"""Compare saved predictions after A's handoff. No training or inference.

Only run on the real holdout after the user announces A's push. This is the
second inspection; passing these necessary checks never replaces any model.
"""
import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path


TIERS = ("Safe", "Elevated", "Critical")


def _index(rows):
    result = {}
    for original in rows:
        row = dict(original)
        event = str(row.get("event_id", "")).strip()
        if not event or event in result:
            raise ValueError("Missing or duplicate event_id")
        for name in ("final_tier", "current_tier", "predicted_final_tier"):
            if row.get(name) not in TIERS:
                raise ValueError(f"Invalid {name} for event {event}")
        future = str(row.get("has_future", "")).lower()
        if future not in ("true", "false", "1", "0"):
            raise ValueError(f"Invalid has_future for event {event}")
        row["has_future"] = future in ("true", "1")
        result[event] = row
    if not result:
        raise ValueError("No events")
    return result


def _metrics(rows, prediction):
    confusion = {actual: {predicted: 0 for predicted in TIERS} for actual in TIERS}
    for row in rows:
        confusion[row["final_tier"]][row[prediction]] += 1
    f1 = {}
    for tier in TIERS:
        tp = confusion[tier][tier]
        fn = sum(confusion[tier].values()) - tp
        fp = sum(confusion[actual][tier] for actual in TIERS) - tp
        denominator = 2 * tp + fp + fn
        f1[tier] = Fraction(2 * tp, denominator) if denominator else Fraction(0)
    critical_total = sum(confusion["Critical"].values())
    if not critical_total:
        raise ValueError("No Critical truth events in evaluation scope; recall is undefined")
    macro = sum(f1.values()) / len(TIERS)
    recall = Fraction(confusion["Critical"]["Critical"], critical_total)
    return {"macro_f1": float(macro), "f1_by_tier": {k: float(v) for k, v in f1.items()},
            "critical_recall": float(recall), "critical_events": critical_total,
            "confusion_actual_rows_predicted_columns": confusion}, macro, recall


def _scope(existing, candidate):
    old, old_f1, _ = _metrics(existing, "predicted_final_tier")
    baseline, _, baseline_recall = _metrics(existing, "current_tier")
    new, new_f1, new_recall = _metrics(candidate, "predicted_final_tier")
    return {"events": len(existing), "existing_model": old, "baseline": baseline,
            "candidate_model": new}, {
                "macro_f1_not_below_existing": new_f1 >= old_f1,
                "critical_recall_not_below_baseline": new_recall >= baseline_recall,
            }


def compare_predictions(existing_rows, candidate_rows):
    """Use the same saved cohort and truth; main result is has_future=True."""
    existing, candidate = _index(existing_rows), _index(candidate_rows)
    if existing.keys() != candidate.keys():
        raise ValueError("Existing and candidate event sets differ")
    for event in existing:
        for key in ("final_tier", "current_tier", "has_future"):
            if existing[event][key] != candidate[event][key]:
                raise ValueError(f"Changed {key} for event {event}")
    future = [event for event in existing if existing[event]["has_future"]]
    if not future:
        raise ValueError("No future events")
    primary, checks = _scope([existing[e] for e in future], [candidate[e] for e in future])
    supplementary, _ = _scope(list(existing.values()), list(candidate.values()))
    return {
        "inspection_label": "第二次查看留出集",
        "primary_scope": "has_future=True",
        "macro_f1_definition": "Unweighted mean over Safe, Elevated, Critical; zero_division=0",
        "baseline_definition": "Saved current_tier on the identical events",
        "checks": checks,
        "replacement_eligible": all(checks.values()),
        "decision_note": "Necessary metric checks only; no automatic model replacement or submission.",
        "primary": primary,
        "supplementary_all": supplementary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve the inspection record")
    report_inputs = {}
    frames = []
    for role, path in (("existing", args.existing), ("candidate", args.candidate)):
        # Hash and parse the same bytes to bind the evidence to the measured input.
        payload = path.read_bytes()
        report_inputs[role] = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
        frames.append(list(csv.DictReader(payload.decode("utf-8-sig").splitlines())))
    report = compare_predictions(*frames)
    report["inputs"] = report_inputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"第二次查看留出集: replacement_eligible={report['replacement_eligible']}; {args.output}")


if __name__ == "__main__":
    main()
