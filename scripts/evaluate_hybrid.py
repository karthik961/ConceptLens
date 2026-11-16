"""Rerun the original 40 development queries through an *experimental* hybrid retriever.

Usage: python -m scripts.evaluate_hybrid
Creates eval/results_hybrid.json without altering eval/results.json or existing audit.
The reported gold-source overlap is a diagnostic, not human-audited retrieval quality.
"""
import json
import statistics
from src.knowledge import create_store
from src.hybrid_retrieval import hybrid_sources_search
from src.settings import ROOT
from scripts.evaluate_retrieval import compute_metrics


def main():
    path = ROOT / "eval/retrieval_cases.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    store = create_store()
    retrieved = {}
    timings = []
    for case in cases:
        result = hybrid_sources_search(store, case["query"], case["subject"], k=3)
        retrieved[case["id"]] = [hit["source"] for hit in result["hits"]]
        timings.append(result["latency_ms"])
    report = compute_metrics(cases, retrieved, k=3)
    report["mean_latency_ms"] = statistics.mean(timings)
    report["median_latency_ms"] = statistics.median(timings)
    report["benchmark_note"] = (
        "Development-set original gold-source overlap only. Re-audit any newly retrieved "
        "question/source pairs before claiming improved Hit@3, MRR@3, or Precision@3. "
        "Hybrid latency includes local lexical scoring and semantic search."
    )
    output = ROOT / "eval/results_hybrid.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Cases: {len(cases)} | Original-gold overlap@3: {report['recall_at_3']:.3f} | Original-gold MRR@3: {report['mrr_at_3']:.3f}")
    print(f"Mean hybrid retrieval latency: {report['mean_latency_ms']:.1f} ms")
    print(f"Saved: {output}")
    item = next((c for c in report["per_case"] if c["id"] == "R032"), None)
    if item:
        print("R032 retrieved:", item["retrieved"])
        print("R032 original gold in top 3:", bool(item["recall_at_3"]))


if __name__ == "__main__":
    main()
