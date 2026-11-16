"""Evaluate indexed retrieval against preauthored source relevance labels.
Usage: python -m scripts.evaluate_retrieval --output eval/results.json
"""
import argparse
import json
import statistics
from pathlib import Path
from src.knowledge import create_store
from src.retrieval import unique_sources_search
from src.settings import ROOT


def compute_metrics(cases, hits_by_id, k=3):
    if not cases:
        raise ValueError("No evaluation cases")
    recall_values, reciprocal_ranks = [], []
    per_case = []
    for item in cases:
        found = hits_by_id[item["id"]][:k]
        gold = set(item["gold_sources"])
        recall = len(gold.intersection(found)) / len(gold)
        reciprocal = next((1 / rank for rank, src in enumerate(found, 1) if src in gold), 0.0)
        recall_values.append(recall)
        reciprocal_ranks.append(reciprocal)
        per_case.append({**item, "retrieved": found, "recall_at_3": recall, "reciprocal_rank": reciprocal})
    return {"count": len(cases), "recall_at_3": statistics.mean(recall_values),
            "mrr_at_3": statistics.mean(reciprocal_ranks), "per_case": per_case}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='eval/results.json')
    parser.add_argument('--k', type=int, default=3)
    args = parser.parse_args()
    if args.k != 3:
        raise SystemExit('This benchmark currently reports Recall@3 and MRR@3: use --k 3')
    cases_path = ROOT / 'eval/retrieval_cases.jsonl'
    cases = [json.loads(line) for line in cases_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    store = create_store()
    hits_by_id = {}
    times = []
    for c in cases:
        result = unique_sources_search(store, c['query'], c['subject'], k=3)
        hits_by_id[c['id']] = [hit['source'] for hit in result['hits']]
        times.append(result['latency_ms'])
    result = compute_metrics(cases, hits_by_id)
    result['mean_latency_ms'] = statistics.mean(times)
    result['median_latency_ms'] = statistics.median(times)
    result['benchmark_note'] = 'Small project-authored development set. Gold labels are source-file relevance judgments, not BEETLE answers. Do not present as an independent blind test.'
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')
    print(f"Cases: {result['count']} | Recall@3: {result['recall_at_3']:.3f} | MRR@3: {result['mrr_at_3']:.3f}")
    print(f"Mean query latency: {result['mean_latency_ms']:.1f} ms | Median: {result['median_latency_ms']:.1f} ms")
    print('Saved per-question results:', out)
    misses = [row for row in result['per_case'] if row['recall_at_3'] == 0]
    print('Misses:', len(misses))
    for row in misses[:10]:
        print(' ', row['id'], 'expected', row['gold_sources'], 'got', row['retrieved'])

if __name__ == '__main__':
    main()
