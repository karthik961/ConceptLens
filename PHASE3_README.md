# ConceptLens Phase 3 — Structured misconception-aware feedback

**What this patch changes:** replaces `app.py` and the tiny `src/rag.py` facade; adds `src/feedback_engine.py`, `src/feedback_schema.py`, two feedback evaluation scripts, 16 **development** cases and offline unit tests. It **does not** change Phase 2 retrieval code, the Chroma index, the knowledge notes or earlier evaluation results.

## Installation into your existing working project (not as its own app)

> First commit or back up local edits. The GitHub README/catalog were updated from ChatGPT in Phase 2; run `git pull --ff-only origin main` on a clean local branch before applying this patch.

After unzipping this patch, from your **existing** `ConceptLens_Starter2/` directory (with `.venv` activated):

```bash
# Path below assumes ZIP was extracted to ~/Downloads/ConceptLens_Phase3_Patch
cp -R ~/Downloads/ConceptLens_Phase3_Patch/src/. src/
cp -R ~/Downloads/ConceptLens_Phase3_Patch/scripts/. scripts/
cp -R ~/Downloads/ConceptLens_Phase3_Patch/tests/. tests/
cp -R ~/Downloads/ConceptLens_Phase3_Patch/eval/. eval/
cp ~/Downloads/ConceptLens_Phase3_Patch/app.py ./app.py
cp ~/Downloads/ConceptLens_Phase3_Patch/PHASE3_README.md ./PHASE3_README.md

python -m unittest discover -s tests -v
streamlit run app.py
```

**No `pip install` or reset required** if your Phase 2 Python environment and Ollama model are already working. The new code requires `pydantic`, which is installed through `langchain-core`. Use the same local `qwen2.5:3b` model. If using a different set of package versions, `ChatOllama.with_structured_output(..., method="json_schema")` needs a compatible Ollama/LangChain installation.

### First manual regression check

In Streamlit select Electronics → **Evidence-grounded RAG** and use:

- Question: `Why does a bulb stop glowing when a circuit is open?`
- Student answer: `Because the battery has run out of electricity.`

A sound result is **incorrect**, pointing to the broken conducting loop, not mere battery discharge. It should cite a retrieved note. **This is an expected behavior, not a verified result:** the Mac runtime still needs to be tested. If it reports `incomplete`, `correct`, or no evidence, record it as a failure rather than assuming the prompt fixed it.

Then test the exact same question with Qwen without retrieval and generic template. This is a within-question comparison, not a proof of effectiveness.

## Three comparison conditions

1. `template`: generic next-step prompt, deliberately **not diagnostic**. Outputs `cannot_assess` because it does not inspect subject evidence.
2. `llm_only`: Qwen2.5:3b gets the same question and student answer but no retrieved passages. It can use pretrained knowledge but must not fabricate citations.
3. `rag`: Phase 2 subject-filtered semantic retriever selects three unique sources from Chroma. Qwen2.5:3b is instructed to use retrieved passages for claims; it returns validated evidence indices. Unsupported citations or missing evidence force a `cannot_assess` abstention.

`assessment` can be one of `correct`, `incomplete`, `incorrect`, `off_topic`, `cannot_assess`. **These are model-generated judgments, not a trained BEETLE classifier.** The validation code verifies schema and citation bounds, **not** whether a citation truly supports a statement.

## Evaluate on DEVELOPMENT cases first

Start with a small inference smoke test:

```bash
python -m scripts.evaluate_feedback_dev --limit 3 --modes rag --output eval/phase3_smoke.jsonl
```

Then run the full 16 × 3 comparison (32 LLM calls, plus 16 cheap template calls):

```bash
python -m scripts.evaluate_feedback_dev
python -m scripts.review_feedback_dev prepare
```

Open `eval/feedback_review.csv`. It is **blinded** to the method used. Manually score:

- `factual_correctness`: 0 wrong/misleading; 1 partly right; 2 factually correct and appropriate judgment.
- `usefulness`: 0 not useful; 1 some useful context; 2 actionable correction or learning guidance.
- `diagnosis_specificity`: 0 generic/incorrect; 1 somewhat targeted; 2 specifically addresses the response or accurately identifies no misconception.
- `evidence_support`: 0 unsupported, 1 partial, 2 fully supported. Score this *only if the `cited_evidence` column contains passage text*. If there is no cited passage, leave it blank; the scorer treats an evidence-free RAG case as zero support.

Finish every `factual_correctness`, `usefulness`, `diagnosis_specificity` field, then:

```bash
python -m scripts.review_feedback_dev score
```

You should receive `eval/feedback_review_report.json`, grouped by condition. The author-provided `expected_assessment` labels are **development-only** sanity checks, **not validated ground truth**, and are deliberately not passed to the generator. No BEETLE `test_ua` or `test_uq` records enter this pipeline.

## Interpretation and research integrity

- Source notes currently have project-written text and reference links but have not passed expert review. Do not call feedback "verified" based on displaying `[1]`.
- Since the 16 development questions and an explicit open-circuit prompt example are part of this design, **do not present their performance as held-out generalization**.
- The generative model may still label answers incorrectly; schema validation cannot guarantee semantic truth. Preserve observed failures for analysis.
- We will use the BEETLE `train` split for later development / validation, and leave the official `test_ua` / `test_uq` splits untouched for the final feedback evaluation.
- Future Phase 3 steps: evaluate 40–60 human-reviewed answer/feedback examples, compare conditions, examine unsupported claims and confusion, and only then report reliability.
- To keep final GitHub reproducible, commit source files, evaluation questions, scripts and sanitized results; exclude local model weights, `.chromadb`, `.venv`, and downloaded datasets.
