# ConceptLens Phase 2 — Frozen held-out retrieval comparison

This pack adds three files to the existing `ConceptLens_Starter2` project:
- `eval/heldout_questions.jsonl`: 20 new scenario/misconception queries (10 electronics, 10 computer science). NO gold labels or BEETLE student/test answers.
- `scripts/run_heldout.py`: runs **existing** semantic and hybrid retrievers once on the same frozen questions, saves individual top-3 passages and timings.
- `scripts/review_heldout.py`: creates one **retriever-blinded** CSV with unique question-source-passage pairs for human relevance judgments; scores both methods with identical judgments.

## Copy into project

Move files to matching locations under your main `ConceptLens_Starter2/` folder, **not** inside a separate extracted subfolder. Do not overwrite existing scripts.

Ensure your prior index is available (`.chromadb/`) and `.venv` is active. Do not reindex or edit the retrievers based on these questions.

```bash
python -m scripts.run_heldout
python -m scripts.review_heldout prepare
```

Open `eval/heldout_audit.csv`. Judge every passage, including any passages that appear less relevant than the others:
- 2 — passage contains enough evidence by itself to answer the query correctly
- 1 — passage is relevant background but not sufficient by itself
- 0 — unrelated or misleading for this query

Write a brief `rationale` for every row. `case_id`, `subject`, `query`, `source`, and `passage` **must stay unchanged**. Save CSV in UTF-8.

```bash
python -m scripts.review_heldout score
```

The script creates `eval/heldout_comparison.json` with paired Hit@3, MRR@3, Precision@3, and mean retrieval latency. Do not call it Recall@3: not all relevant passages in the corpus were exhaustively judged. Latency measures only retrieval, not LLM generation.

## Scientific validity

1. Freeze the 20-question set **before the first run**. Never revise it in response to retrieval results. Code checks the raw query-file hash on scoring.
2. Both retrievers and their weights must remain exactly as evaluated on the 40-query development set. Do not optimize for held-out cases after inspection.
3. For rigor, conduct your human judgments before looking at method rankings and aggregate scores. The shared CSV hides the retrieval method and rank for each candidate.
4. This is a *project-authored*, pre-frozen test, not a randomly sampled independent real-student dataset. Avoid overclaiming generalization.
5. Human checking of concept notes against authoritative references is a **separate remaining task**. The original notes are labeled as unvetted demonstration material.
6. Preserve the baseline results alongside the final result in GitHub, after human review; no special model files or downloaded datasets should be committed.
7. A score difference on only 20 questions is noisy. Report both methods rather than calling small differences a proof.
