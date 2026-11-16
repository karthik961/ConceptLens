# Phase 2 — Frozen retrieval evaluation

## Scope and methodology

ConceptLens uses a local knowledge corpus of 32 original, project-authored notes (18 electronics, 14 computer science). The application uses subject filtering, MiniLM embeddings, and ChromaDB. The semantic-only retriever is `src/retrieval.py`; the experimental hybrid version (`src/hybrid_retrieval.py`) combines semantic ranking with in-memory BM25 lexical ranking.

All recorded results below were produced on the **pre-provenance-edit** corpus. The repository's tested source revision was `cb9a846c4d41b5f80c303391db63b85c8870dbc4`. After that revision, `knowledge/catalog.json` was updated to add supporting reference links and review status. The underlying educational passage files, retrieval functions, evaluation questions, and previously saved scores were **not** altered as part of that metadata update. Because catalog metadata is also attached to Chroma documents during indexing, a newly rebuilt index will have revised metadata; the preserved scores refer to the frozen preceding revision.

## Results

Human-reviewed **development** set: 40 project-authored questions; pooled judgments of retrieved document sufficiency (0 irrelevant, 1 partial, 2 fully sufficient).

| Metric (full relevance = 2) | Semantic | Hybrid |
|---|---:|---:|
| Hit@3 | 0.975 | 1.000 |
| MRR@3 | 0.867 | 0.842 |
| Precision@3 | 0.425 | 0.433 |
| Mean retrieval latency | 5.8 ms | 7.9 ms |

Human-reviewed **held-out** set: 20 new project-authored questions, 10 electronics and 10 computer science, not used to tune either method. Results: `eval/heldout_comparison.json`; question-set hash is stored in that JSON.

| Metric (full relevance = 2) | Semantic | Hybrid |
|---|---:|---:|
| Hit@3 | 0.850 (17/20) | 0.850 (17/20) |
| MRR@3 | 0.817 | 0.800 |
| Precision@3 | 0.350 | 0.333 |
| Mean retrieval latency | 8.9 ms | 6.1 ms |
| Missed test IDs | H005, H010, H012 | H005, H010, H012 |

The small difference in single-run latency is not robust evidence of a speed advantage. Both methods missed the same three questions. We select **semantic-only** for the default application on the grounds of slightly better held-out ranking and precision, while keeping the hybrid code and its comparative evaluation as an experiment. The development-set R032 miss was rescued by hybrid retrieval, but this did not yield improved held-out coverage.

**Terminology:** Hit@3 means at least one retrieved passage was judged individually fully sufficient. It is **not exhaustive Recall@3**: the audit covers the pooled top-three outputs rather than every relevant passage in the corpus. MRR@3 uses the rank of the first fully sufficient passage; Precision@3 is the fraction of the three returned passages judged fully sufficient.

## Research limitations

- Only 20 held-out questions. Both query sets and educational passages are project-authored, not externally sampled.
- Relevance labels were initially AI-assisted and then manually verified by the project author. No independently blind second assessor or inter-annotator agreement is claimed.
- No further model tuning should be made *against these held-out questions*. Future revisions require a new evaluation set for final performance claims.
- Latency excludes local language-model generation and end-to-end UI response time.
- Evaluation covers evidence retrieval; it does not demonstrate reliable misconception diagnosis or high-quality educational feedback.
- Corpus passages were written originally for ConceptLens, not quoted from textbooks. Supporting readings have been linked in `knowledge/catalog.json`. Those references do not imply that a subject expert has independently approved every claim. Review status is `reference_linked_pending_human_content_review` until full manual checking.

## Reproduction

For clean replication, checkout the frozen revision separately so later metadata edits cannot interfere with comparison:

```bash
git clone https://github.com/karthik961/ConceptLens.git conceptlens-phase2-frozen
cd conceptlens-phase2-frozen
git checkout cb9a846c4d41b5f80c303391db63b85c8870dbc4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.index_knowledge --reset
python -m scripts.evaluate_retrieval
python -m scripts.evaluate_hybrid
python -m scripts.run_heldout
```

**Caution:** `--reset` deletes the local Chroma database of the current checkout. Run in a clean clone, not in an environment with uploads to preserve.

For the recorded human-judged metrics, inspect the committed `eval/retrieval_audit_report.json`, `eval/retrieval_audit_hybrid_report.json` and `eval/heldout_comparison.json`. Rerunning retrieval from scratch may produce minor differences across dependency/model versions or hardware; `requirements.txt` is not an exact lockfile.

## Source provenance and corpus review

`knowledge/catalog.json` has one entry per concept note, with the original authoring provenance, a supplementary `source_url`, and a review status. Links are contextual readings, **not verbatim note origins**. The original text remains in `knowledge/{electronics,computer_science}/`. A final content-verification pass should check each note against its linked reading, particularly nuanced or safety-sensitive claims, and record any corrections separately from the frozen experiments.

## Transition to Phase 3

The Phase 2 retrieval study is complete as a reproducible development + held-out comparison. Phase 3 will separately evaluate correctness of short-answer judgments and feedback, including the known open-circuit/battery-depletion misclassification. Do not interpret retrieval Hit@3 as classification accuracy or feedback quality.
