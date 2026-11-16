"""Prepare/score a *separate* relevance audit for hybrid retrieval.

Usage:
  python -m scripts.audit_hybrid prepare
  # Review blank relevance/rationale cells in eval/retrieval_audit_hybrid.csv
  python -m scripts.audit_hybrid score

Requires the original scripts/audit_retrieval.py installed in this project.
Carries over *unchanged* query/source judgments ONLY when exact query, source,
and passage text all match the previous audited worksheet.
"""
import argparse
import csv
from pathlib import Path

from scripts import audit_retrieval as base

ROOT = Path(__file__).resolve().parents[1]
BASE_CSV = ROOT / "eval/retrieval_audit.csv"
base.RESULTS = ROOT / "eval/results_hybrid.json"
base.WORKSHEET = ROOT / "eval/retrieval_audit_hybrid.csv"
base.REPORT = ROOT / "eval/retrieval_audit_hybrid_report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "score"])
    args = parser.parse_args()
    if args.mode == "score":
        base.score()
        return

    # Won't overwrite an existing hybrid worksheet; protects manual judgments.
    base.prepare(force=False)
    if not BASE_CSV.exists():
        print(f"No original audit CSV found at {BASE_CSV}; label all rows manually.")
        return
    with BASE_CSV.open(encoding="utf-8-sig", newline="") as f:
        originals = {
            (row["case_id"], row["query"], row["source"], row["passage"]): row
            for row in csv.DictReader(f)
        }
    with base.WORKSHEET.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    reused = 0
    new = 0
    for row in rows:
        original = originals.get((row["case_id"], row["query"], row["source"], row["passage"]))
        if original and original.get("relevance") in {"0", "1", "2"} and original.get("rationale", "").strip():
            row["relevance"] = original["relevance"]
            row["rationale"] = original["rationale"]
            reused += 1
        else:
            new += 1
    with base.WORKSHEET.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reused {reused} prior AI-assisted judgments; {new} new judgments require review.")
    print("Review all reused labels too before calling these independently human-validated.")


if __name__ == "__main__":
    main()
