# ConceptLens — Evidence-Grounded Short-Answer Feedback

**ConceptLens** is a local educational NLP prototype that analyzes a student's short answer and generates a **provisional** assessment, a possible misconception, a concise explanation, and a revision hint. When evidence is available, it also displays the retrieved teaching passages selected by the model.

Built for **CS 678** using a retrieval-augmented generation (RAG) pipeline. **It is not an automated grading system** and its assessments and citations can be wrong.

## Research result at a glance

| Measure | Observed result | Scope |
| --- | ---: | --- |
| **Assessment agreement** | **80.2% (77/96)** | One-time, frozen, project-authored/author-reviewed short-answer test |
| Computer Science | 79.2% (38/48) | Same 96-case test |
| Electronics | 81.2% (39/48) | Same 96-case test |
| Incorrect-answer agreement | 95.8% (23/24) | Category-level agreement |
| Off-topic agreement | 70.8% (17/24) | Category-level agreement |
| Abstentions | 4/96 | `cannot_assess` predictions |
| **Fully supported citations** | **69.3% (52/75)** | **Preliminary AI-assisted review** of responses with citations; NOT an independently verified human metric |

The project's aspirational assessment target was 85–90%; **it was not achieved**. The descriptive 95% Wilson interval for assessment agreement is 71.1%–86.9%; the examples are topic-clustered and not independently sampled real student submissions, so this interval does **not** establish real-world accuracy.

Read [the final evaluation and limitations](docs/FINAL_EVALUATION.md) and [the reproducibility guide](docs/REPRODUCIBILITY.md).

## How it works

```text
Student's subject + question + answer
                |
                v
MiniLM embeddings -> Chroma vector search (subject-filtered)
                |
        top 4 distinct passages
                |
                v
Qwen2.5 7B via Ollama + checklist assessment prompt
                |
                v
Structured response (Pydantic schema)
                |
                v
Citation-ID and output validation (structural, NOT factual entailment)
                |
                v
Streamlit: assessment / possible misconception / explanation / revision hint / source passages
```

**Final evaluated configuration**: `qwen2.5:7b`, temperature `0`, `prompt_variant="checklist"`, semantic retrieval `k=4`, 1,100-character chunks with 100-character overlap, `sentence-transformers/all-MiniLM-L6-v2`, and Chroma persisted at `.chromadb_1100/`. The app explicitly passes the evaluated model and prompt in `app.py`. The older shared default in `src/settings.py` refers to the historical 3B experimental baseline and should not be interpreted as the Streamlit configuration.

Subjects: **Computer Science** (14 project-authored concept notes) and **Electronics** (18 notes). Supplementary reading links in `knowledge/catalog.json` do **not** mean every note has been independently checked by a subject expert.

## Run locally (Mac or compatible workstation)

Prerequisites: Python **3.11 or 3.12** recommended by this repo's requirements, Ollama installed, and sufficient RAM for the approximately 4.7-GB quantized Qwen2.5 7B model. A fast GPU/Apple Silicon helps; CPU-only may be slow.

```bash
# Clone if necessary, then cd into the repository
python3.12 -m venv .venv                # use an available supported Python interpreter
source .venv/bin/activate
python -m pip install -r requirements.txt

ollama pull qwen2.5:7b
ollama list

# From a clean clone, index the 32 curated notes into .chromadb_1100/
python -m scripts.index_knowledge

python -m streamlit run app.py
```

Open `http://localhost:8501`. If you already have a populated `.chromadb_1100/` database, you ordinarily **do not** need to reindex it. Avoid `python -m scripts.index_knowledge --reset` unless you intend to destroy that local index, including previously uploaded notes.

### Example input

- **Subject:** Computer Science
- **Question:** Why might two threads need a mutex when updating shared data?
- **Student answer:** Because threads always run one after another.
- **Expected qualitative behavior:** `incorrect`; explain that threads can overlap and need synchronization to avoid race conditions, with relevant passages displayed.

For demonstration, use the **Evidence-grounded RAG** mode. `Qwen without retrieval` and `Generic feedback template` are comparison modes and should not be presented as equivalent to the evaluated RAG configuration. Scanned PDFs are not supported; uploaded learning notes are stored in the local Chroma database and **change the retrieval corpus**, so they are not part of the frozen 96-case result.

## Repository map

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit user interface, explicitly uses 7B + checklist + `k=4` |
| `src/feedback_engine.py` | RAG retrieval, structured model generation, retry and validation |
| `src/feedback_schema.py` | Output schema and citation-ID validation |
| `src/assessment_checklist.py` | Question-coverage and assessment-label checklist |
| `src/retrieval.py` | Subject-filtered semantic search with unique sources |
| `src/knowledge.py` | Curated-note ingestion and 1,100-character chunking |
| `knowledge/` | 32 project-authored teaching notes plus source catalog |
| `scripts/index_knowledge.py` | Build local Chroma index |
| `scripts/prepare_phase3_final.py` | Freeze final test contexts and manifest from reviewed labels |
| `scripts/run_phase3_final.py` | Hopper inference on frozen contexts |
| `scripts/score_phase3_final.py` | Score downloaded predictions against local gold labels |
| `eval/phase3_final_report_summary.json` | Observed final label metrics |
| `docs/FINAL_EVALUATION.md` | Final experimental results, audit, and limitations |
| `docs/REPRODUCIBILITY.md` | How to reproduce a **new** run and interpret archived outputs |

## Run tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m scripts.verify_published_results
python -m py_compile app.py
```

`verify_published_results` checks consistency between the archived case-level results and reported metrics; **it does not rerun Qwen or independently validate gold labels**.

## Research limitations

1. **Assessment agreement is not human-verified grading accuracy.** The 96-case test uses project-authored, topic-clustered answers with author-reviewed gold labels. It is not a representative sample of real classroom responses.
2. **Citation existence is not citation support.** The validator checks citation indices; it cannot determine whether the cited text entails the model's claims. The 69.3% figure is an AI-assisted, preliminary qualitative assessment.
3. **False positives and contradictions occur.** In one final case, an Ohm's-law response was labeled `correct` while the explanation contradicted that label. The system can overstate omissions and mishandle some off-topic answers.
4. **Experimental provenance matters.** The 16-case development result was 16/16 and the 40-case post-fix diagnostic result was 31/40; neither is the final unseen estimate. The final one-time test was 77/96.
5. **Do not tune against the final test.** Improvements to the model, prompt, corpus or retrieval require a separate future development set and fresh evaluation. UI-only changes do not alter the *historical* result but still require functional tests.
6. **Local-only runtime.** A public GitHub repo or a public Streamlit URL does not imply the Ollama model can be served without an accessible local inference backend.

## Project history and references

- [Phase 2 retrieval study](PHASE2_RESULTS.md) — historical semantic vs hybrid retrieval experiments.
- [Phase 3 development notes](PHASE3_README.md) — **historical** instructions for an earlier 3B variant, not final setup.
- [BEETLE dataset](https://huggingface.co/datasets/nkazi/Beetle) — background resource; the final 96-case benchmark is **not** a BEETLE test result.
- [LangChain / Ollama integration](https://python.langchain.com/docs/integrations/chat/ollama/) and [Ollama Qwen2.5](https://ollama.com/library/qwen2.5).

This repository currently has no explicit license file. Public source availability does not, by itself, confer an open-source reuse license.
