"""Create a model-blinded relevance worksheet and score human judgments.

Usage:
  python -m scripts.review_heldout prepare
  # Manually judge ALL rows in eval/heldout_audit.csv (0, 1, or 2) and explain why.
  python -m scripts.review_heldout score

Rows are de-duplicated across methods; the worksheet does not show which
retriever returned a passage or at what rank. Both methods are scored using
identical human judgments for the exact same question + source + passage.
"""
import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

from src.settings import ROOT

INPUT = ROOT / "eval/heldout_run.json"
QUESTIONS = ROOT / "eval/heldout_questions.jsonl"
WORKSHEET = ROOT / "eval/heldout_audit.csv"
OUTPUT = ROOT / "eval/heldout_comparison.json"
FIELDS = ["case_id", "subject", "query", "source", "passage", "relevance", "rationale"]


def read_input():
    run = json.loads(INPUT.read_text(encoding="utf-8"))
    digest = hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()
    if digest != run["questions_sha256"]:
        raise ValueError("Held-out question set changed after the run. Restore the frozen question file.")
    return run


def key(case_id, source, passage):
    return (case_id, source, passage)


def expected_rows(run):
    cases = {c["id"]: c for c in run["cases"]}
    by_key = {}
    for method in ("semantic", "hybrid"):
        for result in run["results"][method]:
            c = cases[result["id"]]
            for hit in result["hits"]:
                k = key(c["id"], hit["source"], hit["passage"])
                by_key[k] = {"case_id": c["id"], "subject": c["subject"], "query": c["query"],
                             "source": hit["source"], "passage": hit["passage"],
                             "relevance": "", "rationale": ""}
    return by_key


def prepare():
    if WORKSHEET.exists():
        raise SystemExit(f"Audit already exists: {WORKSHEET}. Won't overwrite human review.")
    rows = list(expected_rows(read_input()).values())
    random.Random(678).shuffle(rows)  # remove model and rank ordering as judgment cues
    with WORKSHEET.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {WORKSHEET} with {len(rows)} unique query/passage judgments.")
    print("Judge each row: 2 fully answers; 1 partially supports; 0 irrelevant. Fill rationale too.")


def metric(rows, scores):
    per_case = []
    for result in rows:
        grades = [scores[key(result["id"], h["source"], h["passage"])] for h in result["hits"]]
        rank = next((i for i, g in enumerate(grades, 1) if g == 2), None)
        per_case.append({"id": result["id"], "grades": grades,
                         "sources": [h["source"] for h in result["hits"]],
                         "hit_at_3": int(rank is not None),
                         "mrr_at_3": 1 / rank if rank is not None else 0.0,
                         "precision_at_3": sum(g == 2 for g in grades) / 3,
                         "latency_ms": result["latency_ms"]})
    return {"n": len(per_case), "hit_at_3": statistics.mean(x["hit_at_3"] for x in per_case),
            "mrr_at_3": statistics.mean(x["mrr_at_3"] for x in per_case),
            "precision_at_3": statistics.mean(x["precision_at_3"] for x in per_case),
            "mean_retrieval_latency_ms": statistics.mean(x["latency_ms"] for x in per_case),
            "misses": [x["id"] for x in per_case if not x["hit_at_3"]],
            "per_case": per_case}


def score():
    run = read_input()
    expected = expected_rows(run)
    with WORKSHEET.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(expected):
        raise ValueError(f"Expected exactly {len(expected)} rows, got {len(rows)}")
    scores = {}
    for r in rows:
        k = key(r["case_id"], r["source"], r["passage"])
        if k not in expected or k in scores:
            raise ValueError("Unknown/duplicate query-source-passage row; do not edit identity fields")
        original = expected[k]
        if any(r[x] != original[x] for x in ("subject", "query", "source", "passage")):
            raise ValueError("Worksheet row was changed; edit relevance and rationale only")
        value = r["relevance"].strip()
        if value not in {"0", "1", "2"} or not r["rationale"].strip():
            raise ValueError(f"Incomplete judgment for {r['case_id']} / {r['source']}")
        scores[k] = int(value)
    report = {"description": "Human-reviewed held-out retrieval evaluation; candidate pool includes only top-3 outputs of the compared models",
              "questions_sha256": run["questions_sha256"],
              "limitations": ["20 project-authored questions; limited power to establish generalization",
                              "All judgments cover pooled top-3 passages, not the full corpus",
                              "Hit@3 rather than exhaustive Recall@3",
                              "No model changes after inspecting held-out outputs"],
              "models": {m: metric(run["results"][m], scores) for m in ("semantic", "hybrid")}}
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {OUTPUT}")
    for name, r in report["models"].items():
        print(f"{name}: Hit@3={r['hit_at_3']:.3f} MRR@3={r['mrr_at_3']:.3f} Precision@3={r['precision_at_3']:.3f} mean latency={r['mean_retrieval_latency_ms']:.1f} ms; misses={r['misses']}")
    print("Do NOT use these final results to retune the retrievers.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "score"])
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else score)()


if __name__ == "__main__":
    main()
