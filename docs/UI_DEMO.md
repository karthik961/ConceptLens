# ConceptLens — Streamlit demo (Milestone 2)

The UI is presentation-only: it does **not** modify the frozen Phase 3 inference, retrieval, knowledge corpus, or evaluation harness.

## Start

In the project root, with Ollama running and `qwen2.5:7b` already downloaded:

```bash
source .venv/bin/activate
python3 -m streamlit run app.py
```

Open `http://localhost:8501`.

## Three-minute demonstration

1. Leave **Subject = Computer Science** and **Feedback method = Evidence-grounded RAG**.
2. Use the default **Mutex · common misconception** example. Click **Generate feedback**.
3. Explain the provisional assessment, possible misconception, explanation, and next-step hint.
4. Open **Evidence explorer**. Distinguish **Cited passages** (chosen by the model) from **Other retrieved** (retrieved but not cited); emphasize that citations are *not* independently checked.
5. Choose another example such as **Binary search · correct idea**; generate another result.
6. Switch the subject to **Electronics**, load **Open circuit · common misconception**, and generate feedback.
7. Click **Download feedback (.md)** to export the currently displayed assessment.

A generated label is not guaranteed for each example; model judgments can be wrong. This is an interface demo, not a fresh accuracy test.

## Controls

- **Try an example**: populates the question and answer for the selected subject.
- **Reset inputs**: clears both text fields and the previous result.
- **Feedback method**: `rag` is the evaluated system; `llm_only` and `template` are experimental baselines.
- **Knowledge library**: optional PDF/TXT/MD uploads. Uploading changes the local corpus and invalidates any direct attribution of the historical evaluation to the modified corpus.
- **Download feedback**: creates a local Markdown file with the response and model-selected citation references.

## Evaluation provenance

The historical 96-case experiment measured 77/96 exact assessment agreement (80.2%) with Qwen2.5 7B, checklist prompt and semantic retrieval `k=4`. This is **not** a confidence score for the live UI response. Preliminary citation-support ratings were AI-assisted, not independently expert-verified.

## Run UI tests

```bash
python3 -m unittest discover -s tests -p 'test_ui_helpers.py' -v
python3 -m unittest discover -s tests -v
python3 -m scripts.verify_published_results
```

The UI helper tests are offline. A live Ollama + Streamlit smoke test must still be performed on a machine with the required dependencies and model.
