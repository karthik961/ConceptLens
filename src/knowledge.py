"""Prepare subject-filtered, persistently indexed learning material."""
import io
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.formatting import document_id, validate_subject
from src.settings import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, KNOWLEDGE_DIR, SUBJECTS


def create_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def load_text(path: Path, subject: str) -> list[Document]:
    validate_subject(subject)
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    source = str(path.relative_to(KNOWLEDGE_DIR))
    catalog = json.loads((KNOWLEDGE_DIR / "catalog.json").read_text(encoding="utf-8"))
    if source not in catalog:
        raise ValueError(f"No provenance metadata for knowledge passage: {source}")
    metadata = {"source": source, "subject": subject, **catalog[source]}
    return [Document(page_content=text, metadata=metadata)]


def load_sample_notes() -> list[Document]:
    docs = []
    for subject in SUBJECTS:
        for path in sorted((KNOWLEDGE_DIR / subject).glob("*")):
            if path.suffix.lower() in {".txt", ".md"}:
                docs.extend(load_text(path, subject))
    return docs


def load_uploaded_file(filename: str, raw: bytes, subject: str) -> list[Document]:
    validate_subject(subject)
    # Filename is metadata only, never used to construct a destination path.
    filename = Path(filename).name
    ext = Path(filename).suffix.lower()
    source = f"upload:{filename}"
    if ext in {".txt", ".md"}:
        text = raw.decode("utf-8-sig", errors="replace").replace("\x00", " ")
        return [Document(page_content=text, metadata={"source": source, "subject": subject, "origin": "user_upload"})] if text.strip() else []
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(Document(page_content=text, metadata={"source": source, "subject": subject, "origin": "user_upload", "page": i}))
        return pages
    raise ValueError("Only .md, .txt and text-based .pdf files are supported")


def index_documents(store: Chroma, docs: list[Document]) -> int:
    if not docs:
        return 0
    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=100)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    ids = [document_id(c.metadata["subject"], c.metadata["source"], str(c.metadata.get("page", "")), c.page_content) for c in chunks]
    if chunks:
        # Stable IDs make repeated indexing idempotent in Chroma's add_documents path.
        store.add_documents(documents=chunks, ids=ids)
    return len(chunks)
