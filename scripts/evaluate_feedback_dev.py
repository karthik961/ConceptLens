"""Run local model baselines on a project-authored Phase 3 DEVELOPMENT set.

No official BEETLE test_ua/test_uq examples or labels are used. No scores are
claimed without a human rubric review. This script records model outputs only.
"""
import argparse
import json
import time
from pathlib import Path
from src.feedback_engine import MODES, give_feedback
from src.settings import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="First N cases for smoke testing")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--output", default="eval/feedback_dev_results.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data = ROOT / "eval/feedback_dev_cases.jsonl"
    cases = [json.loads(x) for x in data.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        cases = cases[:args.limit]
    output = ROOT / args.output
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists. Use a different --output or --overwrite.")
    # Initialize once; even llm-only baseline is evaluated on the same environment.
    store = None
    if "rag" in args.modes:
        from src.knowledge import create_store
        store = create_store()
    results = []
    for case in cases:
        for mode in args.modes:
            start = time.perf_counter()
            try:
                feedback = give_feedback(
                    store if mode == "rag" else None,
                    case["subject"], case["question"], case["answer"], mode=mode,
                )
                row = {"case_id": case["id"], "subject": case["subject"], "question": case["question"],
                       "answer": case["answer"], "expected_assessment": case["expected_assessment"],
                       "category": case["category"], "mode": mode, "result": feedback,
                       "end_to_end_ms": (time.perf_counter()-start)*1000}
                print(case["id"], mode, "->", feedback["assessment"])
            except Exception as exc:
                row = {"case_id": case["id"], "subject": case["subject"], "question": case["question"],
                       "answer": case["answer"], "expected_assessment": case["expected_assessment"],
                       "category": case["category"], "mode": mode, "error": f"{type(exc).__name__}: {exc}",
                       "end_to_end_ms": (time.perf_counter()-start)*1000}
                print(case["id"], mode, "ERROR", row["error"])
            results.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in results), encoding="utf-8")
    print(f"Saved {len(results)} records to {output}. No model-quality claims without human review.")

if __name__ == "__main__":
    main()
