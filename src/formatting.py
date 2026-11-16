"""Pure functions: do not require an Ollama server or embeddings."""
import hashlib

VALID_SUBJECTS = frozenset({"electronics", "computer_science"})


def validate_subject(subject: str) -> str:
    if subject not in VALID_SUBJECTS:
        raise ValueError(f"Unknown subject {subject!r}. Expected one of {sorted(VALID_SUBJECTS)}")
    return subject


def document_id(subject: str, source: str, page: str, content: str) -> str:
    payload = f"{subject}\x00{source}\x00{page}\x00{content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def source_name(metadata: dict) -> str:
    name = str(metadata.get("source", "Unknown source"))
    if metadata.get("page"):
        name += f" (p. {metadata['page']})"
    return name


def make_context(documents: list) -> str:
    return "\n\n".join(
        f"[{i}] Source: {source_name(doc.metadata)}\n{doc.page_content.strip()}"
        for i, doc in enumerate(documents, start=1)
    )
