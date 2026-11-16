"""Dependency-free tests for the experimental lexical+semantic blend."""
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.hybrid_retrieval import bm25_rank, hybrid_sources_search
from src.settings import KNOWLEDGE_DIR


class FakeStore:
    def __init__(self, sources, semantic_order):
        self.sources = sources
        self.semantic_order = semantic_order

    def get(self, *, where, include):
        filtered = {s: t for s, t in self.sources.items() if s.startswith(where["subject"] + "/")}
        return {"documents": list(filtered.values()), "metadatas": [{"source": source} for source in filtered]}

    def similarity_search_with_score(self, query, *, k, filter):
        return [
            (SimpleNamespace(metadata={"source": source}, page_content=self.sources[source]), rank * 0.1)
            for rank, source in enumerate(self.semantic_order[:k], 1)
        ]


class HybridTests(unittest.TestCase):
    def test_bm25_recognizes_fifo_data_structures(self):
        docs = {
            str(f.relative_to(KNOWLEDGE_DIR)): f.read_text(encoding="utf-8")
            for f in (KNOWLEDGE_DIR / "computer_science").glob("*.md")
        }
        ranked, _ = bm25_rank("Why choose a queue rather than a stack for FIFO processing?", docs)
        self.assertEqual(ranked[0], "computer_science/03_data_structures.md")

    def test_hybrid_recovers_source_even_when_semantic_ranks_it_twelfth(self):
        docs = {
            str(f.relative_to(KNOWLEDGE_DIR)): f.read_text(encoding="utf-8")
            for f in (KNOWLEDGE_DIR / "computer_science").glob("*.md")
        }
        target = "computer_science/03_data_structures.md"
        first = [
            "computer_science/06_process_vs_thread.md",
            "computer_science/07_stack_heap.md",
            "computer_science/01_threads_mutex.md",
        ]
        rest = [s for s in sorted(docs) if s not in set(first + [target])]
        order = first + rest
        order.insert(11, target)
        result = hybrid_sources_search(
            FakeStore(docs, order), "Why choose a queue rather than a stack for FIFO processing?",
            "computer_science", k=3
        )
        self.assertIn(target, [hit["source"] for hit in result["hits"]])

    def test_empty_subject(self):
        result = hybrid_sources_search(FakeStore({}, []), "a valid query", "electronics")
        self.assertEqual(result["hits"], [])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            hybrid_sources_search(FakeStore({}, []), " ", "electronics")
        with self.assertRaises(ValueError):
            hybrid_sources_search(FakeStore({}, []), "hi", "invalid")


if __name__ == "__main__":
    unittest.main()
