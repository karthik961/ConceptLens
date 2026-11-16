# ConceptLens Phase 2: Human Retrieval Audit

This is a **patch**, not a full project. Copy `scripts/audit_retrieval.py` into your existing `ConceptLens_starter2/scripts/` folder. It needs no new dependencies and **does not modify the Chroma index**.

## 1. Freeze your baseline
Run the existing evaluation once, before changing any knowledge, model, index, or labels:

```bash
python -m scripts.evaluate_retrieval
cp eval/results.json eval/results_baseline.json
```

Keep a backup of `eval/retrieval_cases.jsonl` as well.

## 2. Prepare the audit sheet

```bash
python scripts/audit_retrieval.py prepare
```

Open `eval/retrieval_audit.csv` in Excel/Numbers or a CSV editor. Every row is a question plus a candidate passage (the retrieved top three and the originally labeled expected passage). Read the **actual passage**. Fill:

- `relevance`: `2` = the passage *alone* can fully answer the question; `1` = the passage contains useful related facts but is insufficient alone; `0` = irrelevant or misleading.
- `rationale`: a short human explanation, e.g. `Explains that a broken series path stops current in all bulbs`.

Do not automatically mark originally gold passages as relevant or retrieved passages as irrelevant. Check against authoritative subject references where necessary. Note that the CSV is UTF-8 with BOM for easy spreadsheet import; edit it as a CSV preserving column names.

**This file covers a small candidate pool, not every possible relevant source in all 32 notes.** For fuller gold judgments, inspect more candidates or all notes in the same subject, then annotate accordingly. Never use an LLM alone to create evaluation ground truth.

## 3. Score reviewed results

```bash
python scripts/audit_retrieval.py score
```

Produces `eval/retrieval_audit_report.json` with fully relevant Hit@3, MRR@3, Precision@3, partial-support hit rate, and breakdowns by subject/category. It checks all candidate labels are filled. It evaluates the **same frozen results.json**, not a new retrieval run.

This deliberately does not report true `Recall@3`, because one or more relevant sources outside the judged candidate pool may not be identified.

## 4. Improve the real retrieval failure

Inspect `R032` and `knowledge/computer_science/03_data_structures.md`. Test a simple query rewrite or a hybrid BM25+dense retriever; keep your original query and baseline for comparison. Do not alter R032's labels simply because it was missed. Re-run the evaluator only after making a deliberate retriever change. Any new retrieved sources need human judgments too.

## 5. Independent hold-out (manual work)

Before tuning retrieval further, prepare **20 new, previously unseen questions**, ideally 10 electronics and 10 computer science. Use natural student language, include indirect questions and tricky negatives, and write reference sources independently of the retriever's output. Freeze these questions and candidate-gold judgments **before** testing the modified system. Save a copy of the index configuration and any model versions. Do not use this held-out set to tune retrieval repeatedly; otherwise it becomes another development set.

## 6. Documentation
Record for your report: corpus size, chunking parameters, MiniLM model identifier, top-k settings, label rubric, evaluator version, number of reviewed candidate pairs, scores per domain, disagreements, and examples of successful and failed retrieval.
