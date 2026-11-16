import streamlit as st

from src.knowledge import create_store, index_documents, load_uploaded_file
from src.rag import give_feedback
from src.settings import SUBJECTS

st.set_page_config(page_title="ConceptLens", page_icon="📚", layout="centered")
st.title("ConceptLens")
st.caption("CS 678 • Local, evidence-grounded short-answer feedback • Research prototype")

@st.cache_resource(show_spinner="Loading local sentence embeddings and Chroma index...")
def local_store():
    return create_store()

subject = st.selectbox("Subject", options=list(SUBJECTS), format_func=lambda x: SUBJECTS[x])
example = {
    "electronics": ("Why does a bulb stop glowing when a circuit is open?", "Because the battery has run out of electricity."),
    "computer_science": ("Why might two threads need a mutex when updating shared data?", "Because threads always run one after another."),
}
question = st.text_area("Teacher's question", value=example[subject][0], height=80, key=f"question_{subject}")
answer = st.text_area("Student's answer", value=example[subject][1], height=100, key=f"answer_{subject}")

with st.expander("Add your own learning notes (optional)"):
    st.write("Upload trusted, text-based PDF, TXT, or Markdown notes. They are indexed locally under the selected subject and persist until the database is reset.")
    uploaded = st.file_uploader("Notes", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Index uploaded notes", disabled=not uploaded):
        store = local_store()
        total = 0
        for file in uploaded or []:
            if file.size > 5_000_000:
                st.warning(f"{file.name}: maximum size is 5 MB; skipped.")
                continue
            try:
                docs = load_uploaded_file(file.name, file.getvalue(), subject)
                if not docs:
                    st.warning(f"{file.name}: no extractable text found (scanned PDFs need OCR).")
                    continue
                total += index_documents(store, docs)
            except Exception as exc:
                st.error(f"Could not index {file.name}: {exc}")
        st.success(f"Processed {total} text chunks for {SUBJECTS[subject]}.")

if st.button("Generate feedback", type="primary"):
    if not question.strip() or not answer.strip():
        st.error("Enter a question and student answer first.")
    else:
        try:
            with st.spinner("Retrieving evidence and generating feedback locally..."):
                result = give_feedback(local_store(), subject, question, answer)
            st.subheader("Feedback")
            st.markdown(result["feedback"])
            if result["sources"]:
                st.subheader("Retrieved evidence")
                for item in result["sources"]:
                    with st.expander(f'[{item["ref"]}] {item["source"]}'):
                        st.caption(f"Topic: {item.get('topic') or 'Unspecified'} | Review: {item.get('review_status') or 'Unreviewed'} | Vector distance: {item.get('distance', 0):.3f} (lower is closer)")
                        st.write(item["text"] )
            st.info("This is unvalidated model-generated feedback; verify factual claims against the cited passages.")
        except Exception as exc:
            st.error(f"Feedback failed: {exc}")
            st.write("Check whether Ollama is running and `qwen2.5:3b` has been downloaded.")
