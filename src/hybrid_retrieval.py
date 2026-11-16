"""Experimental small-corpus hybrid retriever (MiniLM semantic + BM25 lexical).

Uses existing Chroma documents. BM25 is computed in memory for each query,
which is appropriate for the project's ~32 knowledge notes, not large corpora.
This is a *candidate retrieval* experiment; do not tune on held-out tests.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict

from src.formatting import validate_subject

# General-purpose stopwords, used for lexical matching only.
_STOP = frozenset("a an the and or of to in on for from by with as at is are was were be been being this that these those it its why how what which when where who does do did have has had can could should would rather than into through about one two each both other".split())
# Expand standard subject abbreviations into their written-out forms.
_ABBREVIATIONS = {
    "fifo": ("first", "in", "first", "out"),
    "lifo": ("last", "in", "first", "out"),
}


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    expanded = []
    for word in words:
        expanded.append(word)
        expanded.extend(_ABBREVIATIONS.get(word, ()))
    return [w for w in expanded if w not in _STOP and len(w) > 1]


def bm25_rank(query: str, source_texts: dict[str, str]) -> tuple[list[str], dict[str, float]]:
    """Small-corpus, document-level BM25 rank. Zero-overlap sources excluded."""
    if not source_texts:
        return [], {}
    counts = {source: Counter(tokens(text)) for source, text in source_texts.items()}
    lengths = {source: sum(counter.values()) for source, counter in counts.items()}
    average_length = sum(lengths.values()) / len(lengths) or 1
    df = Counter()
    for counter in counts.values():
        df.update(counter.keys())
    queries = Counter(tokens(query))
    scores = {}
    n = len(source_texts)
    k1, b = 1.5, 0.75
    for source, c in counts.items():
        score = 0.0
        length = lengths[source]
        for word, repeats in queries.items():
            tf = c[word]
            if not tf:
                continue
            idf = math.log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average_length)) * min(repeats, 2)
        if score > 0:
            scores[source] = score
    ranked = sorted(scores, key=lambda source: (-scores[source], source))
    return ranked, scores


def hybrid_sources_search(store, query: str, subject: str, k: int = 3):
    """Blend normalized semantic source rank and normalized lexical BM25 score.

    Returns the same high-level result shape as unique_sources_search plus
    diagnostic ranks/scores. All candidates are subject-filtered. Preserves the
    original semantic retriever; experimental method is opt-in.
    """
    validate_subject(subject)
    query = query.strip()
    if not query or k < 1:
        raise ValueError("Query must be nonempty and k must be positive")
    start = time.perf_counter()
    corpus = store.get(where={"subject": subject}, include=["documents", "metadatas"])
    metadata_rows = corpus.get("metadatas") or []
    document_rows = corpus.get("documents") or []
    if not document_rows:
        return {"hits": [], "latency_ms": (time.perf_counter() - start) * 1000}

    # Source-level lexical ranking; chunks from the same source become one document.
    group = defaultdict(list)
    for text, metadata in zip(document_rows, metadata_rows):
        source = (metadata or {}).get("source", "")
        if source and text:
            group[source].append(text)
    source_texts = {source: "\n".join(parts) for source, parts in group.items()}
    lexical, bm25_scores = bm25_rank(query, source_texts)
    lexical_ranks = {source: i + 1 for i, source in enumerate(lexical)}

    # Request all matching indexed chunks for this small benchmark. In a larger
    # application, use an approximate index and independently maintained BM25 index.
    semantic_rows = store.similarity_search_with_score(
        query, k=len(document_rows), filter={"subject": subject}
    )
    semantic_ranks = {}
    best_chunk = {}
    for doc, distance in semantic_rows:
        source = doc.metadata.get("source", "")
        if source and source not in semantic_ranks:
            semantic_ranks[source] = len(semantic_ranks) + 1
            best_chunk[source] = (doc, float(distance))

    # A lexical winner may need a different chunk than its best semantic chunk.
    lexical_best_chunk = {}
    query_terms = set(tokens(query))
    for doc, distance in semantic_rows:
        source = doc.metadata.get("source", "")
        overlap = sum(tokens(doc.page_content).count(t) for t in query_terms)
        if source and (source not in lexical_best_chunk or overlap > lexical_best_chunk[source][0]):
            lexical_best_chunk[source] = (overlap, doc, float(distance))

    # Equal-weight score blend: 0.5 normalized semantic rank + 0.5 normalized
    # BM25 score. Strong exact keyword evidence can rescue a low semantic rank.
    # This is a development-set experiment, NOT a calibrated relevance probability.
    source_count = max(len(semantic_ranks), 1)
    max_bm25 = max(bm25_scores.values(), default=0.0)
    fused = {}
    for source in set(semantic_ranks) | set(lexical_ranks):
        semantic_strength = (
            (source_count - semantic_ranks[source]) / max(source_count - 1, 1)
            if source in semantic_ranks else 0.0
        )
        lexical_strength = (bm25_scores.get(source, 0.0) / max_bm25) if max_bm25 else 0.0
        fused[source] = 0.5 * semantic_strength + 0.5 * lexical_strength
    ordered = sorted(fused, key=lambda source: (-fused[source], semantic_ranks.get(source, 10**9), source))
    hits = []
    for source in ordered:
        if source not in best_chunk:  # Chroma's semantic search should cover indexed sources.
            continue
        doc, distance = best_chunk[source]
        if lexical_ranks.get(source, 10**9) < semantic_ranks.get(source, 10**9):
            _, doc, distance = lexical_best_chunk.get(source, (None, doc, distance))
        hits.append({
            "document": doc, "distance": distance, "source": source,
            "semantic_rank": semantic_ranks.get(source),
            "lexical_rank": lexical_ranks.get(source),
            "bm25": bm25_scores.get(source, 0.0),
            "hybrid_score": fused[source],
        })
        if len(hits) >= k:
            break
    return {"hits": hits, "latency_ms": (time.perf_counter() - start) * 1000}
