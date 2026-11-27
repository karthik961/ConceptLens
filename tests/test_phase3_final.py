"""Self-contained offline tests of the frozen Phase 3 evaluation protocol.

Constructs synthetic fixtures; does not require private review CSV or Ollama.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_phase3_final import read_review
from scripts.run_phase3_final import load_contexts
from scripts.score_phase3_final import wilson95

LABELS = ("correct", "incomplete", "incorrect", "off_topic")


def fixture_review_csv(path):
    columns = (
        "case_id", "subject", "topic", "question", "student_answer",
        "expected_assessment", "teacher_rationale", "reference_source",
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        i = 0
        for subject in ("computer_science", "electronics"):
            for label in LABELS:
                for _ in range(12):
                    i += 1
                    writer.writerow({
                        "case_id": f"FIX{i:03}", "subject": subject,
                        "topic": "synthetic-test-topic",
                        "question": f"Synthetic question {i}?",
                        "student_answer": f"Synthetic answer {i}",
                        "expected_assessment": label,
                        "teacher_rationale": "Fixture only; not student data",
                        "reference_source": f"{subject}/01_fixture.md",
                    })


class FinalProtocolTests(unittest.TestCase):
    def test_review_balance_without_private_data(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fixture.csv"
            fixture_review_csv(p)
            self.assertEqual(len(read_review(p)), 96)

    def test_reject_duplicate_review_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fixture.csv"
            fixture_review_csv(p)
            with p.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows[1]["case_id"] = rows[0]["case_id"]
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                read_review(p)

    def test_contexts_never_contain_gold(self):
        rows = [
            {"case_id": f"Z{i:03}", "subject": "computer_science", "k": 4,
             "question": f"Q{i}", "answer": "A", "sources": [{}, {}, {}, {}]}
            for i in range(96)
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "contexts.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(len(load_contexts(p)), 96)
            rows[0]["expected_assessment"] = "correct"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaises(ValueError):
                load_contexts(p)

    def test_wilson_interval(self):
        self.assertEqual(wilson95(0, 0), (None, None))
        lo, hi = wilson95(84, 96)
        self.assertLess(lo, 84 / 96)
        self.assertGreater(hi, 84 / 96)


if __name__ == "__main__":
    unittest.main()
