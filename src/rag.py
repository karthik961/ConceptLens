"""Backward-compatible Phase 3 entry point.

Original Phase 2 inference (semantic retrieval) remains the RAG default.
"""
from src.feedback_engine import give_feedback

__all__ = ["give_feedback"]
