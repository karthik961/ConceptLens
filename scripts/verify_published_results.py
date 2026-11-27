"""Validate published Phase 3 summaries against case-level archived output.

Does NOT rerun inference, verify labels, or check citation semantics.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"correct", "incomplete", "incorrect", "off_topic"}


def main() -> None:
    ev = ROOT / "eval"
    summary = json.loads((ev / "phase3_final_report_summary.json").read_text(encoding="utf-8"))
    with (ev / "phase3_final_report_case_results.csv").open(encoding="utf-8", newline="") as f:
        cases = list(csv.DictReader(f))
    with (ev / "ConceptLens_Phase3_Final_Citation_Audit_AI_Reviewed.csv").open(encoding="utf-8", newline="") as f:
        audit = list(csv.DictReader(f))
    with (ev / "phase3_final_report_citation_audit.csv").open(encoding="utf-8", newline="") as f:
        raw = list(csv.DictReader(f))
    assert len(cases) == len(audit) == len(raw) == 96, "Expected 96 cases in each file"
    ids = [c["case_id"] for c in cases]
    assert len(set(ids)) == 96, "Duplicate case IDs"
    assert set(ids) == {a["case_id"] for a in audit} == {a["case_id"] for a in raw}, "Mismatched case IDs"
    assert all(c['expected'] in LABELS for c in cases), "Unexpected gold label"
    correct = sum(c["expected"] == c["predicted"] for c in cases)
    confusion = Counter(f'{c["expected"]}->{c["predicted"]}' for c in cases)
    abstentions = sum(c["predicted"] == "cannot_assess" for c in cases)
    assert correct == summary["correct"] == 77, f"Unexpected correct count: {correct}"
    assert summary["total"] == 96 and abs(summary["accuracy"] - correct/96) < 1e-10
    assert abstentions == summary["abstentions"] == 4
    assert dict(confusion) == summary["confusion"], "Confusion matrix mismatch"
    for subject in ("computer_science", "electronics"):
        subset = [c for c in cases if c["subject"] == subject]
        count = sum(c["expected"] == c["predicted"] for c in subset)
        assert len(subset) == summary["by_group"][subject]["total"] == 48
        assert count == summary["by_group"][subject]["correct"]
    support = Counter(row["supports_central_claim_0_1_2"] for row in audit)
    cited = [row for row in audit if row["citation_count"].strip() and int(row["citation_count"]) > 0]
    assert len(cited) == 75, f"Unexpected cited count: {len(cited)}"
    cited_support = Counter(r["supports_central_claim_0_1_2"] for r in cited)
    assert cited_support == {'0':4,'1':19,'2':52}, f"Citation score count mismatch: {cited_support}"
    print(f"Published Phase 3 case metrics verified: {correct}/96 = {100*correct/96:.1f}%")
    print(f"Abstentions verified: {abstentions}; cited feedback: {len(cited)}")
    print(f"AI-assisted citation scores (0/1/2): {dict(sorted(cited_support.items()))}")
    print("NOTE: Arithmetic only; no independent labels, grading quality or citation entailment verification.")


if __name__ == "__main__":
    main()
