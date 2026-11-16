"""Run *frozen* held-out questions using both existing ConceptLens retrievers.

Usage: python -m scripts.run_heldout
Output: eval/heldout_run.json. This script makes no model or index changes.
"""
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from src.hybrid_retrieval import hybrid_sources_search
from src.knowledge import create_store
from src.retrieval import unique_sources_search
from src.settings import ROOT

QUESTIONS = ROOT / "eval" / "heldout_questions.jsonl"
OUTPUT = ROOT / "eval" / "heldout_run.json"


def load_questions(path=QUESTIONS):
    raw = Path(path).read_bytes()
    cases = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = [c["id"] for c in cases]
    if len(cases) != 20 or len(set(ids)) != len(ids):
        raise ValueError("Expected 20 unique held-out cases")
    if sum(c["subject"] == "electronics" for c in cases) != 10 or sum(c["subject"] == "computer_science" for c in cases) != 10:
        raise ValueError("Expected 10 questions per subject")
    if any("gold_sources" in c or not c["query"].strip() for c in cases):
        raise ValueError("Questions must be nonempty and have no gold labels")
    return cases, hashlib.sha256(raw).hexdigest()


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Held-out result already exists: {OUTPUT}. Do not rerun or overwrite a final test without documenting why.")
    cases, digest = load_questions()
    store = create_store()
    results = {}
    for method, search in (("semantic", unique_sources_search), ("hybrid", hybrid_sources_search)):
        entries = []
        for case in cases:
            response = search(store, case["query"], case["subject"], k=3)
            hits = [{"rank": i, "source": h["source"], "passage": h["document"].page_content}
                    for i, h in enumerate(response["hits"], 1)]
            entries.append({"id": case["id"], "latency_ms": response["latency_ms"], "hits": hits})
        results[method] = entries
    report = {
        "description": "Frozen 20-question project-authored held-out retrieval comparison; no relevance labels in the question file",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "questions_sha256": digest,
        "cases": cases,
        "results": results,
        "method_notes": "Two fixed retrieval configurations; neither is tuned on these test queries.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {OUTPUT}")
    for method, rows in results.items():
        times = [r["latency_ms"] for r in rows]
        print(f"{method}: {len(rows)} questions, mean retrieval latency {statistics.mean(times):.1f} ms")
    print("Next: python -m scripts.review_heldout prepare")
    print("Do not change the retrievers after seeing these results.")


if __name__ == "__main__":
    main()
