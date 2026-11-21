"""Blind (method-hidden) feedback-quality review; score 0/1/2 on each criterion.

Usage:
    python -m scripts.review_feedback_dev prepare
    # Human completes eval/feedback_review.csv
    python -m scripts.review_feedback_dev score

No correctness or groundedness label is inferred by this script.
"""
import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from src.settings import ROOT

FIELDS = ["review_id", "subject", "question", "answer", "assessment", "possible_misconception", "explanation", "hint", "cited_evidence", "factual_correctness", "usefulness", "diagnosis_specificity", "evidence_support", "reviewer_notes"]
CSV_PATH = ROOT / "eval/feedback_review.csv"
MAP_PATH = ROOT / "eval/feedback_review_key.json"
REPORT = ROOT / "eval/feedback_review_report.json"
RESULTS = ROOT / "eval/feedback_dev_results.jsonl"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare():
    if CSV_PATH.exists() or MAP_PATH.exists():
        raise SystemExit("Review files already exist; back them up before preparing again.")
    if not RESULTS.exists():
        raise SystemExit(f"First run: python -m scripts.evaluate_feedback_dev (missing {RESULTS})")
    original = read_jsonl(RESULTS)
    rng = random.Random(678)
    rng.shuffle(original)
    rows, mapping = [], {}
    for row in original:
        if "error" in row:
            print("Skipping failed generation:", row["case_id"], row["mode"], row["error"])
            continue
        feedback = row["result"]
        rid = hashlib.sha256(f"{row['case_id']}:{row['mode']}".encode()).hexdigest()[:12]
        mapping[rid] = {"mode": row["mode"], "case_id": row["case_id"], "expected_assessment": row["expected_assessment"]}
        cited = "\n".join(f"[{s['ref']}] {s['source']}: {s['text']}" for s in feedback.get("sources", []) if s["ref"] in feedback.get("evidence_ids", []))
        rows.append({"review_id": rid, "subject": row["subject"], "question": row["question"], "answer": row["answer"],
                     "assessment": feedback["assessment"], "possible_misconception": feedback["misconception"],
                     "explanation": feedback["explanation"], "hint": feedback["hint"], "cited_evidence": cited,
                     "factual_correctness": "", "usefulness": "", "diagnosis_specificity": "", "evidence_support": "", "reviewer_notes": ""})
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    MAP_PATH.write_text(json.dumps(mapping, indent=2)+"\n", encoding="utf-8")
    print(f"Prepared {len(rows)} blinded rows: {CSV_PATH}")
    print("Score correctness, usefulness, specificity 0–2 for ALL methods; score evidence_support only when cited_evidence is populated.")
    print("IMPORTANT: evidence_support can be scored only from the cited passage text, not the model's unverified assertion.")


def score():
    if not CSV_PATH.exists() or not MAP_PATH.exists():
        raise SystemExit("Run prepare before score.")
    mapping = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {r["review_id"] for r in rows} != set(mapping) or len(rows) != len(mapping):
        raise SystemExit("Review IDs do not match the generated mapping; do not add or remove rows.")
    grouped = defaultdict(list)
    for row in rows:
        k = row["review_id"]
        mode = mapping[k]["mode"]
        for key in ("factual_correctness", "usefulness", "diagnosis_specificity"):
            if row[key].strip() not in {"0","1","2"}:
                raise SystemExit(f"Missing/invalid {key} for {k}")
        if mode == "rag" and row["cited_evidence"].strip() and row["evidence_support"].strip() not in {"0","1","2"}:
            raise SystemExit(f"Missing/invalid evidence_support for cited RAG {k}")
        grouped[mode].append({**{q:int(row[q]) for q in ("factual_correctness", "usefulness", "diagnosis_specificity")},
                              "evidence_support": (int(row["evidence_support"]) if row["evidence_support"].strip() in {"0","1","2"} else 0) if mode == "rag" else None,
                              "assessment_match": row["assessment"] == mapping[k]["expected_assessment"], "review_id":k})
    summary = {}
    for mode, items in sorted(grouped.items()):
        report = {"count":len(items),"mean_scores_out_of_2":{key:statistics.mean(i[key] for i in items) for key in ("factual_correctness", "usefulness", "diagnosis_specificity")},
                  "development_assessment_match_fraction":statistics.mean(i["assessment_match"] for i in items)}
        if mode == "rag":
            report["mean_scores_out_of_2"]["evidence_support"] = statistics.mean(i["evidence_support"] for i in items)
        summary[mode] = report
    REPORT.write_text(json.dumps({"description":"Human-reviewed Phase 3 development feedback quality; not independent test performance", "modes":summary},indent=2)+"\n",encoding="utf-8")
    for mode, stats in summary.items():
        print(mode, stats)
    print("Saved:", REPORT)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare","score"))
    args=p.parse_args()
    {"prepare":prepare,"score":score}[args.action]()

if __name__=="__main__":
    main()
