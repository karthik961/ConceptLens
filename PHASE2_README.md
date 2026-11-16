# ConceptLens Phase 2: Knowledge and retrieval

This is an **upgraded complete starter project**, not just a patch. It keeps `app.py` and the local Ollama generator.

## 1. Install and migrate

**Back up your working directory before copying this version over it.** From the new `ConceptLens_Starter` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.index_knowledge --reset
python -m scripts.evaluate_retrieval
streamlit run app.py
```

The `--reset` argument **DELETES ALL local ChromaDB collections AND uploaded notes in that database**, including uploads you indexed from the old app. Keep copies of any custom notes before resetting. The BEETLE dataset folder is not touched. You can also simply create a fresh folder and environment with the new ZIP instead of overwriting the old one.

## What changed

- 32 self-authored concept notes (8 from starter, 24 new) with `knowledge/catalog.json` metadata.
- 40 manually specified retrieval prompts, with source-file ground-truth labels in `eval/retrieval_cases.jsonl`.
- `src/retrieval.py` restricts by subject and deduplicates source files in the top 3 results. Chroma scores are distances, **not likelihoods or accuracy**.
- `scripts/evaluate_retrieval.py` reports **Recall@3** (fraction of relevant sources retrieved) and **MRR@3** (mean reciprocal rank of first relevant source), plus latency, and saves per-case failures to `eval/results.json`.
- `src/rag.py` uses the same deduplicated retrieval as evaluation.

## Research limitations / integrity

- Content is written for this project and **not independently expert-reviewed**; `review_status` flags this.
- Metadata `source` is a local file path, not a textbook citation. Do not imply third-party educational authority.
- Development benchmark questions and relevance judgments were created with knowledge of the notes, so this is a **smoke/evaluation development benchmark**, not an unbiased external test.
- BEETLE records are kept **out** of retrieval indexing. Evaluation of student feedback is a separate phase.
- Retrieval success does **not** establish correct misconception assessment. The starter example where a depleted battery answer was marked 'correct but incomplete' is a known failure to investigate in Phase 3/4.

## Improvements after initial measurement

Inspect misses in `eval/results.json`, revise your queries or notes **only against a separate development split**, then collect new held-out queries evaluated once. Compare question-only, question-plus-answer, keyword matching, and alternative chunk sizes using fixed labels before selecting the production approach.

## Recommended background reading (not copied into notes)

- LangChain Chroma docs: https://python.langchain.com/docs/integrations/vectorstores/chroma/
- Chroma reference: https://docs.trychroma.com/
- OpenStax College Physics 2e: https://openstax.org/details/books/college-physics-2e

Review scientific explanations against these or similarly reliable references, and document exactly which passages were actually validated before claiming vetted evidence.
