"""Human relevance audit for ConceptLens Phase 2.

Place in the project's scripts/ directory, then use:
  python scripts/audit_retrieval.py prepare
  # Fill the 'relevance' and 'rationale' columns in eval/retrieval_audit.csv.
  python scripts/audit_retrieval.py score

No third-party dependencies. Does not edit the indexed notes, retrieved results,
or existing gold_sources. The audit scores the FROZEN results.json retrieval run.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval/results.json"
WORKSHEET = ROOT / "eval/retrieval_audit.csv"
REPORT = ROOT / "eval/retrieval_audit_report.json"
FIELDS = [
    "case_id", "subject", "category", "query", "source", "rank",
    "originally_gold", "relevance", "rationale", "passage",
]


def read_result():
    if not RESULTS.exists():
        raise SystemExit(f"Missing {RESULTS}. Run python -m scripts.evaluate_retrieval first.")
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    if not result.get("per_case"):
        raise SystemExit(f"No per_case entries in {RESULTS}")
    return result


def prepare(force=False):
    if WORKSHEET.exists() and not force:
        raise SystemExit(f"{WORKSHEET} already exists. Refusing to overwrite human labels."
                         " To regenerate deliberately, rename/delete it first, or pass --force.")
    source_rows = read_result()["per_case"]
    all_rows = []
    for case in source_rows:
        returned = case["retrieved"]
        sources = list(dict.fromkeys([*returned, *case["gold_sources"]]))
        for source in sources:
            source_path = (ROOT / "knowledge" / source).resolve()
            knowledge_dir = (ROOT / "knowledge").resolve()
            if not source_path.is_relative_to(knowledge_dir) or not source_path.is_file():
                raise SystemExit(f"Invalid or missing source: {source}")
            all_rows.append({
                "case_id": case["id"],
                "subject": case["subject"],
                "category": case.get("category", ""),
                "query": case["query"],
                "source": source,
                "rank": str(returned.index(source) + 1) if source in returned else "",
                "originally_gold": "yes" if source in case["gold_sources"] else "no",
                "relevance": "",  # Human must enter 0, 1, or 2.
                "rationale": "",  # Explain the judgment briefly.
                "passage": source_path.read_text(encoding="utf-8-sig"),
            })
    WORKSHEET.parent.mkdir(parents=True, exist_ok=True)
    with WORKSHEET.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Created {WORKSHEET} with {len(all_rows)} query/source judgments ")
    print("Review every row. Fill relevance: 2 = fully answers, 1 = partially supports, 0 = not relevant.")
    print("Fill rationale. Do not change IDs, sources, ranks, or passage text.")


def summarize(per_case, key):
    result = {}
    for val in sorted({case[key] for case in per_case}):
        group = [c for c in per_case if c[key] == val]
        result[val] = {
            "n": len(group),
            "full_hit_at_3": statistics.mean(c["full_hit_at_3"] for c in group),
            "full_mrr_at_3": statistics.mean(c["full_reciprocal_rank"] for c in group),
        }
    return result


def score():
    existing = read_result()
    if not WORKSHEET.exists():
        raise SystemExit(f"Missing {WORKSHEET}. Run prepare first.")
    judgments = {}
    missing = []
    invalid = []
    with WORKSHEET.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["case_id"], row["source"])
            if key in judgments:
                raise SystemExit(f"Duplicate worksheet entry: {key}")
            grade = row.get("relevance", "").strip()
            if grade not in ("0", "1", "2"):
                (missing if not grade else invalid).append(key)
                continue
            if not row.get("rationale", "").strip():
                missing.append(key)
                continue
            judgments[key] = {"grade": int(grade), "reason": row["rationale"].strip()}
    if missing or invalid:
        raise SystemExit(f"Audit incomplete: {len(missing)} blank grade/reason pairs, "
                         f"{len(invalid)} invalid grades. First pending: "
                         f"{(missing + invalid)[:6]}")

    scored = []
    originally_gold_rejected = []
    for case in existing["per_case"]:
        id_ = case["id"]
        ranked = case["retrieved"][:3]
        for source in [*ranked, *case["gold_sources"]]:
            if (id_, source) not in judgments:
                raise SystemExit(f"Missing judgment for {id_} {source}. Regenerate worksheet if results changed.")
        grades = [judgments[(id_, src)]["grade"] for src in ranked]
        # Fully relevant means sufficient alone; partial is useful context only.
        first_full = next((i + 1 for i, grade in enumerate(grades) if grade == 2), None)
        first_support = next((i + 1 for i, grade in enumerate(grades) if grade >= 1), None)
        for source in case["gold_sources"]:
            if judgments[(id_, source)]["grade"] == 0:
                originally_gold_rejected.append({"case_id": id_, "source": source})
        scored.append({
            "id": id_, "subject": case["subject"], "category": case.get("category", ""),
            "query": case["query"], "retrieved": ranked, "grades": grades,
            "full_hit_at_3": int(first_full is not None),
            "support_hit_at_3": int(first_support is not None),
            "full_reciprocal_rank": 1 / first_full if first_full else 0,
            "support_reciprocal_rank": 1 / first_support if first_support else 0,
            "fully_relevant_precision_at_3": sum(g == 2 for g in grades) / 3,
        })
    report = {
        "description": "Human-judged audit of frozen retrieval results; non-exhaustive candidate pool.",
        "count": len(scored),
        "full_hit_at_3": statistics.mean(c["full_hit_at_3"] for c in scored),
        "full_mrr_at_3": statistics.mean(c["full_reciprocal_rank"] for c in scored),
        "fully_relevant_precision_at_3": statistics.mean(c["fully_relevant_precision_at_3"] for c in scored),
        "support_hit_at_3": statistics.mean(c["support_hit_at_3"] for c in scored),
        "support_mrr_at_3": statistics.mean(c["support_reciprocal_rank"] for c in scored),
        "by_subject": summarize(scored, "subject"),
        "by_category": summarize(scored, "category"),
        "original_gold_labeled_irrelevant": originally_gold_rejected,
        "misses_without_fully_relevant_evidence": [c["id"] for c in scored if not c["full_hit_at_3"]],
        "cases": scored,
        "limitations": [
            "Audited current top-3 plus previous gold source, not all possible relevant notes.",
            "Hit@3 and MRR@3 are measured on this judged set; this is not exhaustive Recall@3.",
            "Questions and notes are project-authored and constitute development data, not an independent blind test.",
            "Do not interpret millisecond search latency as full answer generation latency.",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Human-audited 40-query benchmark: {report['count']} cases")
    print(f"Full hit@3: {report['full_hit_at_3']:.3f} | Full MRR@3: {report['full_mrr_at_3']:.3f}")
    print(f"Fully relevant Precision@3: {report['fully_relevant_precision_at_3']:.3f}")
    print(f"Full+partial hit@3: {report['support_hit_at_3']:.3f}")
    print("Cases without fully sufficient retrieved source:", report["misses_without_fully_relevant_evidence"])
    print("Saved:", REPORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "score"])
    parser.add_argument("--force", action="store_true", help="Overwrite existing worksheet in prepare mode (DESTROYS HUMAN LABELS)")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.force)
    else:
        score()


if __name__ == "__main__":
    main()
