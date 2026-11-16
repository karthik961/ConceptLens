# ConceptLens — CS 678 NLP Project

**A local, subject-switchable learning assistant that uses retrieval-augmented generation (RAG) to give grounded feedback on short student answers.**

This is a **starter implementation**, not a validated grading system. Main approach: classification-free RAG. The separate five-way classifier is a **future ablation**, not a mandatory gate in the inference pipeline.

## Tech stack
- Python 3.11 or 3.12, Streamlit
- LangChain core, `langchain-chroma`, `langchain-huggingface`, `langchain-ollama`, `langchain-text-splitters`
- Chroma persistent vector database; `sentence-transformers/all-MiniLM-L6-v2` (local embeddings)
- Ollama `qwen2.5:3b` (local inference; check Qwen Research License for your usage)
- BEETLE dataset (`nkazi/Beetle`) for later benchmarking; it is **not** automatically indexed

## 1. Install prerequisites on your Mac

1. Install **Python 3.11 or 3.12**, e.g., `brew install python@3.12`. Check `python3 --version`. If `python3` resolves to a different version, use `python3.12` below.
2. Install **Ollama** from https://ollama.com/download and launch it. In a fresh Terminal:

```bash
ollama pull qwen2.5:3b
ollama run qwen2.5:3b 'Reply with one sentence explaining a closed circuit.'
```

Ollama runs at `http://localhost:11434` by default. If requests to localhost fail, open the Ollama application or run `ollama serve` in a separate terminal if it is not already running.

## 2. Set up your project

```bash
cd ConceptLens_Starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To inspect the *benchmark* dataset separately (requires internet once):

```bash
python -m scripts.inspect_beetle
```

This fetches the licensed dataset from Hugging Face, prints sizes/label distribution, and caches it under `data/beetle/`. The directory is deliberately NOT fed into Chroma. Keep official `test_ua` and `test_uq` untouched until final experiments.

## 3. Index the sample knowledge notes

```bash
python -m scripts.index_knowledge --reset
```

This downloads the embedding model on first run and indexes the **illustrative, project-authored** lessons in `knowledge/`. `--reset` removes the existing local index (including anything previously uploaded); use it ONLY when you want to rebuild from scratch. For later additions use the script without `--reset`.

## 4. Launch and try the app

```bash
streamlit run app.py
```

In your browser, choose **Electronics** and try:

- Question: `Why does a bulb stop glowing when a circuit is open?`
- Student answer: `Because the battery has run out of electricity.`

The system should retrieve locally indexed explanations, call Qwen2.5 3B, and display a short feedback response with the retrieved source passages. This is a smoke test, NOT measured educational correctness.

Try **Computer Science** as a second subject. You may upload `.md`, `.txt`, or text-based `.pdf` learning notes; they are added to the **local**, persisted index for the selected subject. Scanned/image-only PDFs need OCR, which is deliberately not included.

## Repo contents

```text
ConceptLens_Starter/
├── app.py                       # Streamlit UI, subject selection, uploads
├── requirements.txt
├── src/
│   ├── settings.py              # Model/config/paths
│   ├── knowledge.py             # Load/split/index notes; deterministic IDs
│   ├── rag.py                   # Retrieval → cited prompt → Ollama feedback
│   └── formatting.py            # Prompt context, domain validation, hashing
├── scripts/
│   ├── index_knowledge.py       # Build local Chroma DB
│   └── inspect_beetle.py        # Obtain and inspect held-out dataset
├── knowledge/
│   ├── electronics/             # Original demonstrative notes
│   └── computer_science/        # Original demonstrative notes
└── tests/                       # Stdlib tests for prompt formatting
```

## Important scientific restrictions

1. **Prevent benchmark leakage.** Do NOT build the RAG knowledge store from official BEETLE student answers, reference answers tied to test items, gold labels, or test files. Curate independent, attributable concept-level educational passages. `knowledge/` is only an illustration, not your final 20–40-passage corpus.
2. **Do not mistake RAG output for a correct judgment.** The model can hallucinate even when sources are retrieved. Human-review at least 40–60 outputs, including correct, partially correct, and incorrect answers, and report failures.
3. **Do not present classifier accuracy for the classification-free RAG system.** Add a separate classifier later and compare it as an ablation; the main system is evaluated by retrieval relevance, groundedness, and feedback usefulness.
4. **Make controlled comparisons:** template feedback, the same Qwen model without retrieval, Qwen + RAG, and (if time) Qwen + RAG + predicted classification.
5. **Dataset split integrity:** Train and validate on `train`; reserve `test_ua` and `test_uq` for the final reported held-out tests. For the RAG evaluation, use independently curated topical evidence not answer-label lookups.
6. **License and sources:** BEETLE's hosted release is CC BY 4.0 (credit source); Qwen2.5 3B has a Qwen Research License, not Apache 2.0.
7. **Domain independence ≠ equal benchmark coverage.** The example CS topic switch works, but only Electronics can be quantitatively benchmarked against BEETLE until a comparable CS evaluation set is created.

## Suggested next experiments

- Expand to 20–40 independently curated and cited concept passages covering 8–12 BEETLE-compatible electricity questions.
- Label 30–50 question/passage relevance pairs and calculate Recall@3 / MRR.
- Add a deterministic no-retrieval Qwen baseline and a template baseline, using identical test questions and judge rubric.
- Sample 40–60 prompts for human review: factual correctness, helpfulness, coverage, and unsupported claims.
- Train sentence-embedding + logistic-regression **five-way** classifier using only train split as an optional ablation.
- Report failures explicitly. Do not claim the system reliably identifies misconceptions before testing.

## Troubleshooting

- `ModuleNotFoundError`: activate `.venv` and reinstall dependencies.
- `Connection refused` on port `11434`: start Ollama.
- `model not found`: run `ollama pull qwen2.5:3b`.
- First index slow: Sentence Transformers downloads MiniLM once and computes embeddings locally.
- No relevant passages: confirm `python -m scripts.index_knowledge --reset` ran and you're using the matching subject.
- Apple Silicon/Python compatibility issues: try supported Python 3.11/3.12 in a clean venv. Avoid installing an unspecified bleeding-edge Python version.
- Tests (no model download): `python -m unittest discover -s tests -v`.

## Sources

- Dataset: https://huggingface.co/datasets/nkazi/Beetle
- SemEval-2013 Task 7 paper: https://aclanthology.org/S13-2045/
- LangChain Chroma: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- LangChain ChatOllama: https://docs.langchain.com/oss/python/integrations/chat/ollama
- LangChain local Hugging Face embeddings: https://docs.langchain.com/oss/python/integrations/embeddings/huggingfacehub
- Ollama Qwen2.5: https://ollama.com/library/qwen2.5
