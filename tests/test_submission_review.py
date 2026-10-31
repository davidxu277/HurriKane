"""Replacement checks use synthetic rows only; never read the real holdout."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from submission_review import compare_predictions, main


def rows(predictions=("Safe", "Elevated", "Critical"), current=None):
    truth = ("Safe", "Elevated", "Critical")
    current = current or truth
    return [dict(event_id=str(i), final_tier=y, current_tier=b,
                 predicted_final_tier=p, has_future=True)
            for i, (y, b, p) in enumerate(zip(truth, current, predictions))]


class ReplacementGateTests(unittest.TestCase):
    def test_equal_metrics_are_allowed(self):
        report = compare_predictions(rows(), rows())
        self.assertTrue(report["replacement_eligible"])
        self.assertEqual(report["inspection_label"], "第二次查看留出集")
        self.assertEqual(report["primary"]["existing_model"]["macro_f1"], 1)

    def test_lower_f1_blocks_even_when_critical_recall_matches(self):
        candidate = rows(("Elevated", "Elevated", "Critical"))
        report = compare_predictions(rows(), candidate)
        self.assertFalse(report["replacement_eligible"])
        self.assertTrue(report["checks"]["critical_recall_not_below_baseline"])
        self.assertFalse(report["checks"]["macro_f1_not_below_existing"])

    def test_improved_f1_cannot_offset_recall_below_baseline(self):
        existing = rows(("Elevated", "Safe", "Critical"))
        candidate = rows(("Safe", "Elevated", "Safe"))
        report = compare_predictions(existing, candidate)
        self.assertTrue(report["checks"]["macro_f1_not_below_existing"])
        self.assertFalse(report["checks"]["critical_recall_not_below_baseline"])
        self.assertFalse(report["replacement_eligible"])

    def test_compare_recall_to_baseline_rather_than_existing_model(self):
        baseline = ("Safe", "Elevated", "Elevated")
        existing = rows(current=baseline)
        candidate = rows(("Safe", "Elevated", "Elevated"), current=baseline)
        report = compare_predictions(existing, candidate)
        self.assertTrue(report["checks"]["critical_recall_not_below_baseline"])
        self.assertFalse(report["replacement_eligible"])

    def test_reordering_events_does_not_change_result(self):
        self.assertTrue(compare_predictions(rows(), rows()[::-1])["replacement_eligible"])

    def test_known_label_events_cannot_hide_future_event_regression(self):
        existing = rows()
        candidate = rows(("Elevated", "Elevated", "Critical"))
        for i in range(3, 30):
            known = dict(event_id=str(i), final_tier="Safe", current_tier="Safe",
                         predicted_final_tier="Safe", has_future=False)
            existing.append(dict(known, predicted_final_tier="Elevated"))
            candidate.append(known)
        report = compare_predictions(existing, candidate)
        self.assertGreater(report["supplementary_all"]["candidate_model"]["macro_f1"],
                           report["supplementary_all"]["existing_model"]["macro_f1"])
        self.assertFalse(report["replacement_eligible"])
        self.assertEqual(report["primary"]["events"], 3)

    def test_changed_cohort_or_truth_is_rejected(self):
        for field, value in (("event_id", "99"), ("final_tier", "Critical"),
                             ("current_tier", "Critical"), ("has_future", False)):
            with self.subTest(field=field):
                candidate = rows()
                candidate[0][field] = value
                with self.assertRaises(ValueError):
                    compare_predictions(rows(), candidate)

    def test_duplicate_ids_and_invalid_values_are_rejected(self):
        for field, value in (("event_id", "1"), ("predicted_final_tier", "UNKNOWN"),
                             ("has_future", "maybe"), ("event_id", "")):
            with self.subTest(field=field):
                candidate = rows()
                candidate[0][field] = value
                with self.assertRaises(ValueError):
                    compare_predictions(rows(), candidate)

    def test_no_future_critical_events_cannot_pass_gate(self):
        existing = rows()
        existing[2]["has_future"] = False
        with self.assertRaises(ValueError):
            compare_predictions(existing, existing)

    def test_csv_boolean_strings_are_supported(self):
        existing = rows()
        for row in existing:
            row["has_future"] = "True"
        self.assertTrue(compare_predictions(existing, rows())["replacement_eligible"])

    def test_cli_preserves_inputs_and_records_their_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ("existing.csv", "candidate.csv"):
                with (base / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows()[0]))
                    writer.writeheader()
                    writer.writerows(rows())
            before = {name: (base / name).read_bytes() for name in ("existing.csv", "candidate.csv")}
            output = base / "second_review.json"
            argv = ["submission_review.py", "--existing", str(base / "existing.csv"),
                    "--candidate", str(base / "candidate.csv"), "--output", str(output)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["replacement_eligible"])
            for role in ("existing", "candidate"):
                payload = before[f"{role}.csv"]
                self.assertEqual(report["inputs"][role]["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual((base / f"{role}.csv").read_bytes(), payload)
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
