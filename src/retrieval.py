"""Retrieval utilities shared by the UI and the evaluation harness."""
import time
from src.formatting import validate_subject


def unique_sources_search(store, query: str, subject: str, k: int = 3, fetch_k: int = 12):
    """Rank chunks by Chroma distance and retain at most one chunk per source.

    The metadata subject filter prevents cross-domain results. A larger candidate
    pool mitigates multiple chunks from one source crowding out other sources.
    Chroma's returned number is distance, NOT a calibrated confidence score.
    """
    validate_subject(subject)
    if not query.strip() or k < 1:
        raise ValueError("Query must be nonempty and k must be positive")
    start = time.perf_counter()
    rows = store.similarity_search_with_score(query.strip(), k=max(k, fetch_k), filter={"subject": subject})
    hits = []
    seen = set()
    for doc, distance in rows:
        source = doc.metadata.get("source", "")
        if source in seen:
            continue
        seen.add(source)
        hits.append({"document": doc, "distance": float(distance), "source": source})
        if len(hits) == k:
            break
    return {"hits": hits, "latency_ms": (time.perf_counter() - start) * 1000}
