"""Small post-fix GPU diagnostic on existing DEVELOPMENT cases only.

In particular, compares the raw structured assessment returned by Ollama with
what the application exposes after citation validation. It does not read the
frozen held-out model-comparison dataset or change the model/benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from src.feedback_engine import give_feedback
from src.settings import ROOT

DEFAULT_IDS = ("F006", "F014", "F001", "F002")


class StructuredCallRecorder:
    def __init__(self, wrapped, trace):
        self.wrapped, self.trace = wrapped, trace

    def invoke(self, messages):
        response = self.wrapped.invoke(messages)
        draft = response if isinstance(response, dict) else response.model_dump()
        self.trace.append({
            "assessment": draft.get("assessment"),
            "evidence_ids": draft.get("evidence_ids"),
        })
        return response


class ObservedModel:
    def __init__(self, model):
        self.model = model
        self.trace = []

    def with_structured_output(self, *args, **kwargs):
        return StructuredCallRecorder(self.model.with_structured_output(*args, **kwargs), self.trace)


def read_saved_dev_cases():
    path = ROOT / "eval/feedback_dev_results.jsonl"
    if not path.exists():
        raise RuntimeError(f"Missing development input: {path}")
    cases = {}
    for text in path.read_text(encoding="utf-8").splitlines():
        if not text.strip():
            continue
        row = json.loads(text)
        if row.get("mode") != "rag" or "error" in row:
            continue
        saved = row.get("result", {})
        sources = saved.get("sources", [])
        if not sources:
            continue
        cases[row["case_id"]] = {"row": row, "sources": sources}
    return cases


def cached_retriever(item):
    row = item["row"]
    sources = item["sources"]

    def retrieve(_store, query, subject, k=3):
        if query != row["question"] or subject != row["subject"]:
            raise ValueError("Development evidence query mismatch")
        hits = []
        for s in sources[:k]:
            metadata = {"source": s["source"], "topic": s.get("topic", ""),
                        "review_status": s.get("review_status", ""),
                        "source_url": s.get("source_url", "")}
            hits.append({"document": SimpleNamespace(page_content=s["text"], metadata=metadata),
                         "distance": s.get("distance", 0.0)})
        return {"hits": hits, "latency_ms": row["result"].get("retrieval_latency_ms")}

    return retrieve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("qwen2.5:3b", "qwen2.5:7b"))
    p.add_argument("--output", required=True)
    p.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS))
    args = p.parse_args()
    saved = read_saved_dev_cases()
    ids = list(dict.fromkeys(args.ids))
    if any(cid not in saved for cid in ids):
        p.error(f"Requested IDs absent from saved development results: {[cid for cid in ids if cid not in saved]}")
    out = ROOT / args.output
    if out.exists():
        p.error(f"Refusing to overwrite {out}")
    from langchain_ollama import ChatOllama
    model = ChatOllama(model=args.model, temperature=0, num_ctx=4096,
                      base_url="http://" + os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        for cid in ids:
            item = saved[cid]
            row = item["row"]
            observed = ObservedModel(model)
            start = time.perf_counter()
            result = {"case_id": cid, "model": args.model, "expected_assessment": row["expected_assessment"],
                      "original_post_validation": row["result"]["assessment"]}
            try:
                feedback = give_feedback(object(), row["subject"], row["question"], row["answer"],
                                         k=len(item["sources"]), mode="rag", model=observed,
                                         retriever=cached_retriever(item))
                result.update({"raw_structured_calls": observed.trace,
                               "final_assessment": feedback["assessment"],
                               "final_evidence_ids": feedback["evidence_ids"],
                               "final_explanation": feedback["explanation"]})
            except Exception as ex:
                result["error"] = f"{type(ex).__name__}: {ex}"
            result["generation_ms"] = round((time.perf_counter() - start)*1000, 1)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(cid, args.model,
                  "raw:", result.get("raw_structured_calls"),
                  "final:", result.get("final_assessment", result.get("error")),
                  flush=True)
    print("Saved", out, flush=True)


if __name__ == "__main__":
    main()
