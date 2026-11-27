"""ConceptLens: polished Streamlit UI for the frozen Qwen2.5 7B RAG demo.

This file changes only presentation and interaction. The evaluated Phase 3
feedback engine, prompts, retrieval and index settings are not altered.
"""
from __future__ import annotations

import time

import streamlit as st
from langchain_ollama import ChatOllama

from src.feedback_engine import give_feedback
from src.knowledge import create_store, index_documents, load_uploaded_file
from src.settings import SUBJECTS
from src.ui_helpers import ASSESSMENTS, EXAMPLES, MODES, make_feedback_markdown, partition_sources

# Explicit frozen final-demo settings; do not use src.settings.LLM_MODEL (historical 3B).
DEMO_MODEL = "qwen2.5:7b"
DEMO_RETRIEVAL_K = 4
DEMO_PROMPT_VARIANT = "checklist"

st.set_page_config(page_title="ConceptLens | Answer feedback", page_icon="📚", layout="wide")

st.markdown("""
<style>
  .block-container {max-width: 1300px; padding-top: 2rem; padding-bottom: 3rem;}
  [data-testid="stAppViewContainer"] {background: #f7f9fc; color: #16283c;}
  [data-testid="stSidebar"] {background: #edf3f7; border-right: 1px solid #dce5ed;}
  [data-testid="stHeader"] {background: transparent;}
  h1, h2, h3 {letter-spacing: -0.025em;}
  h1 {color: #142a42;}
  [data-testid="stButton"] button[kind="primary"],
  [data-testid="stFormSubmitButton"] button[kind="primary"] {
      border: 0; background: #137d79; color: #ffffff; font-weight: 650;}
  [data-testid="stButton"] button[kind="primary"]:hover,
  [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
      background: #0d6663; color: #fff;}
  div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
  .cl-eyebrow {font-size: .72rem; font-weight: 750; letter-spacing: .13em;
     text-transform: uppercase; color: #147a76; margin-bottom: .4rem;}
  .cl-subtitle {font-size: 1.03rem; color: #52677a; margin-top: -.45rem; max-width: 760px;}
  .cl-tag {display: inline-block; border: 1px solid #cbdde0;
     background: #eaf6f5; border-radius: 999px; padding: .22rem .65rem;
     font-size: .77rem; color: #225b5a; font-weight: 600; margin: .2rem .3rem .2rem 0;}
  .cl-rule {height: 1px; background: #e0e8ee; margin: 1.1rem 0 1.5rem 0;}
  .cl-footnote {font-size: .79rem; color: #607387; line-height: 1.5;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading local embeddings and the Chroma knowledge index...")
def local_store():
    return create_store()


def _clear_previous_result():
    st.session_state.pop("cl_result", None)
    st.session_state.pop("cl_snapshot", None)
    st.session_state.pop("cl_error", None)
    st.session_state.pop("cl_elapsed_ms", None)


def _load_example():
    chosen = st.session_state["cl_example"]
    for title, question, answer in EXAMPLES[st.session_state["cl_subject"]]:
        if title == chosen:
            st.session_state["cl_question"] = question
            st.session_state["cl_answer"] = answer
            _clear_previous_result()
            return


def _subject_changed():
    st.session_state["cl_example"] = EXAMPLES[st.session_state["cl_subject"]][0][0]
    _load_example()


def _reset_input():
    st.session_state["cl_question"] = ""
    st.session_state["cl_answer"] = ""
    _clear_previous_result()


def _source_panel(source: dict, *, expanded: bool = False):
    title = f"[{source['ref']}] {source['source']}"
    with st.expander(title, expanded=expanded):
        if source.get("topic"):
            st.caption(f"Topic: {source['topic']}")
        st.write(source["text"])
        if source.get("source_url"):
            st.markdown(f"Supporting reading: {source['source_url']}")
        if source.get("review_status"):
            st.caption(f"Source review status: {source['review_status']}")
        if source.get("distance") is not None:
            st.caption(f"Vector distance: {float(source['distance']):.3f} (ranking value, not confidence)")


# Explicit session defaults; subsequent sample selections update the text inputs.
if "cl_subject" not in st.session_state:
    st.session_state["cl_subject"] = "computer_science"
if "cl_example" not in st.session_state:
    st.session_state["cl_example"] = EXAMPLES["computer_science"][0][0]
if "cl_question" not in st.session_state:
    st.session_state["cl_question"] = EXAMPLES["computer_science"][0][1]
if "cl_answer" not in st.session_state:
    st.session_state["cl_answer"] = EXAMPLES["computer_science"][0][2]
if "cl_mode" not in st.session_state:
    st.session_state["cl_mode"] = "rag"

with st.sidebar:
    st.markdown('<div class="cl-eyebrow">Workspace setup</div>', unsafe_allow_html=True)
    subject = st.selectbox("Subject", list(SUBJECTS), format_func=lambda x: SUBJECTS[x],
                           key="cl_subject", on_change=_subject_changed)
    mode = st.selectbox("Feedback method", list(MODES), format_func=lambda m: MODES[m],
                        key="cl_mode", on_change=_clear_previous_result)
    st.divider()
    st.markdown("**Research configuration**")
    st.caption("Qwen2.5 7B · Ollama")
    st.caption("Checklist assessment prompt · Temperature 0")
    st.caption("MiniLM + semantic RAG · Top 4 distinct sources")
    st.caption("Local inference · No cloud API required")
    if mode != "rag":
        st.info("Comparison mode selected. The 80.2% result applies to the final RAG configuration, not this baseline.")
    st.divider()
    st.caption("CS 678 · Educational NLP research prototype")

st.markdown('<div class="cl-eyebrow">Educational AI · evidence-first feedback</div>',
            unsafe_allow_html=True)
st.title("ConceptLens")
st.markdown('<p class="cl-subtitle">Understand what a short answer gets right, what may be missing, and how to improve it—with inspectable retrieved teaching notes.</p>', unsafe_allow_html=True)
st.markdown('<span class="cl-tag">Computer Science + Electronics</span>'
            '<span class="cl-tag">Local Qwen2.5 7B</span>'
            '<span class="cl-tag">Provisional assessment</span>', unsafe_allow_html=True)
st.markdown('<div class="cl-rule"></div>', unsafe_allow_html=True)

left, right = st.columns([1, 1.12], gap="large")

with left:
    st.subheader("01 / Student response")
    st.caption("Choose a demonstration case or enter your own teacher question and student answer.")
    example_names = [item[0] for item in EXAMPLES[subject]]
    st.selectbox("Try an example", example_names, key="cl_example", on_change=_load_example)
    st.button("Reset inputs", on_click=_reset_input, use_container_width=False)

    with st.form("cl_feedback_form", clear_on_submit=False):
        st.text_area("Teacher's question", key="cl_question", height=110,
                     placeholder="What concept are you assessing?")
        st.text_area("Student's answer", key="cl_answer", height=150,
                     placeholder="Paste the student's short answer here...")
        submitted = st.form_submit_button("Generate feedback", type="primary",
                                          use_container_width=True)

    st.caption("No numerical grades or calibrated confidence scores are generated.")

    with st.expander("Knowledge library · optional notes", expanded=False):
        st.write("The default local index contains 32 project-authored notes: 14 Computer Science and 18 Electronics.")
        st.warning("Adding notes modifies your local retrieval corpus. Historical evaluation results do not apply to this modified corpus.")
        uploaded = st.file_uploader("Trusted TXT, MD or text-based PDF", type=["pdf", "txt", "md"],
                                    accept_multiple_files=True, key="cl_uploads")
        if st.button("Index uploaded notes", disabled=not uploaded):
            total = 0
            try:
                store = local_store()
                for file in uploaded or []:
                    if file.size > 5_000_000:
                        st.warning(f"{file.name}: file exceeds 5 MB and was skipped.")
                        continue
                    try:
                        documents = load_uploaded_file(file.name, file.getvalue(), subject)
                        if not documents:
                            st.warning(f"{file.name}: no extractable text; scanned PDFs need OCR.")
                            continue
                        total += index_documents(store, documents)
                    except Exception as exc:
                        st.error(f"Could not index {file.name}: {exc}")
                if total:
                    st.success(f"Indexed {total} chunks for {SUBJECTS[subject]}.")
                    _clear_previous_result()
            except Exception as exc:
                st.error(f"Knowledge indexing failed: {exc}")

# Form submits rerun the script. Keep successful results in session state so source
# expanders and downloads remain accessible after normal UI interactions.
if submitted:
    _clear_previous_result()
    question = st.session_state["cl_question"].strip()
    answer = st.session_state["cl_answer"].strip()
    if not question or not answer:
        st.session_state["cl_error"] = "Please enter both a question and a student answer."
    else:
        try:
            with st.spinner("Retrieving evidence and preparing feedback..."):
                started = time.perf_counter()
                result = give_feedback(
                    local_store() if mode == "rag" else None,
                    subject, question, answer,
                    k=DEMO_RETRIEVAL_K,
                    mode=mode,
                    model=ChatOllama(model=DEMO_MODEL, temperature=0) if mode != "template" else None,
                    prompt_variant=DEMO_PROMPT_VARIANT,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
            st.session_state["cl_result"] = result
            st.session_state["cl_elapsed_ms"] = elapsed_ms
            st.session_state["cl_snapshot"] = (subject, mode, question, answer)
        except Exception as exc:
            st.session_state["cl_error"] = f"Feedback failed: {exc}"

with right:
    st.subheader("02 / Feedback")
    st.caption("Structured explanation and evidence selected by the model, not an instructor-issued grade.")

    if st.session_state.get("cl_error"):
        st.error(st.session_state["cl_error"])
        st.caption("Check Ollama is running, the qwen2.5:7b model is installed, and the local index has been built.")

    result = st.session_state.get("cl_result")
    snapshot = (subject, mode, st.session_state["cl_question"].strip(),
                st.session_state["cl_answer"].strip())
    if result and st.session_state.get("cl_snapshot") != snapshot:
        st.info("Your inputs have changed. Select Generate feedback to assess the updated answer.")
    elif not result:
        with st.container(border=True):
            st.markdown("### Ready when you are")
            st.write("Select an example or write a short answer, then generate feedback to view the provisional assessment, explanation, next-step hint and retrieved sources.")
            st.caption("Tip: Start with the mutex misconception example in Computer Science.")
    else:
        assessment, definition = ASSESSMENTS.get(result["assessment"],
                                                  (result["assessment"].replace("_", " ").title(), ""))
        with st.container(border=True):
            st.caption("PROVISIONAL ASSESSMENT")
            st.markdown(f"### {assessment}")
            if definition:
                st.caption(definition)
            if result.get("abstained"):
                st.info("The system withheld an assessment. This is not a judgment that the answer is wrong.")

        if result.get("misconception"):
            with st.container(border=True):
                st.markdown("**Possible misconception**")
                st.write(result["misconception"])
        with st.container(border=True):
            st.markdown("**Explanation**")
            st.write(result["explanation"])
        with st.container(border=True):
            st.markdown("**Next step for the student**")
            st.write(result["hint"])

        elapsed = st.session_state.get("cl_elapsed_ms")
        if elapsed is not None:
            st.caption(f"Response time: {elapsed/1000:.1f} s · Local run, not a standardized latency benchmark")
        if result.get("retrieval_latency_ms") is not None:
            st.caption(f"Retrieval time: {result['retrieval_latency_ms']:.1f} ms (excludes model generation)")

        sources = result.get("sources", [])
        cited, additional = partition_sources(sources, result.get("evidence_ids", []))
        if sources:
            st.markdown("#### Evidence explorer")
            st.caption("Citations below are model-selected, not independently fact-checked. The other passages were retrieved but not cited.")
            cited_tab, additional_tab = st.tabs([
                f"Cited passages ({len(cited)})", f"Other retrieved ({len(additional)})"
            ])
            with cited_tab:
                if not cited:
                    st.info("No passages were cited for this response.")
                for item in cited:
                    _source_panel(item, expanded=len(cited) == 1)
            with additional_tab:
                if not additional:
                    st.caption("No additional retrieved passages.")
                for item in additional:
                    _source_panel(item)

        st.download_button(
            "Download feedback (.md)",
            data=make_feedback_markdown(SUBJECTS[subject], snapshot[2], snapshot[3], result),
            file_name="conceptlens_feedback.md", mime="text/markdown",
            use_container_width=True,
        )
        st.info("Research prototype: assessments and citation support may be wrong. Do not use as verified or autonomous grading.")

st.markdown('<div class="cl-rule"></div>', unsafe_allow_html=True)
st.markdown('<p class="cl-footnote">Evaluated Qwen2.5 7B + checklist + semantic RAG configuration: 77/96 exact assessment agreement (80.2%) on project-authored cases. This is a historical research result, not a confidence score for any particular answer. Source support was only provisionally AI-reviewed.</p>', unsafe_allow_html=True)
