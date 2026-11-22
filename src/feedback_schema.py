"""Validation of generated educational feedback; no model calls.

Assessment categories are *generative judgments*, not predictions from a separately
trained five-way BEETLE classifier.
"""
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class FeedbackDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessment: Literal["correct", "incomplete", "incorrect", "off_topic", "cannot_assess"]
    misconception: str = Field(description="A possible misconception, or empty if none is supported", max_length=400)
    explanation: str = Field(description="One to four concise, student-facing sentences", min_length=1, max_length=1300)
    next_step_hint: str = Field(description="One constructive revision prompt or next step", min_length=1, max_length=500)
    evidence_ids: list[int] = Field(..., description="Required field. For RAG correct/incomplete/incorrect, cite supporting 1-indexed passage IDs; for off_topic or cannot_assess, [] is allowed; without retrieval, [].")


def validate_draft(draft: FeedbackDraft, mode: str, source_count: int) -> FeedbackDraft:
    """Reject source hallucination and unsupported RAG assessments rather than hiding it.

    This is structural validation, not semantic fact checking. A citation number being
    in range does not imply that the cited note actually supports the claim.
    """
    if mode not in {"rag", "llm_only", "template"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if source_count < 0:
        raise ValueError("source_count must be nonnegative")
    # Do not let arbitrary citation-looking tokens generated inside free text
    # bypass the numeric evidence_ids check. Only the renderer adds references.
    def clean(text: str) -> str:
        return re.sub(r"\[\s*\d+\s*\]", "", text).strip()
    draft = draft.model_copy(update={
        "misconception": clean(draft.misconception),
        "explanation": clean(draft.explanation),
        "next_step_hint": clean(draft.next_step_hint),
    })
    if not draft.explanation or not draft.next_step_hint:
        return abstention("The generated explanation was not sufficiently well formed to evaluate.")
    if mode != "rag":
        if draft.evidence_ids:
            return draft.model_copy(update={"evidence_ids": []})
        return draft
    bad_ids = any(not (1 <= value <= source_count) for value in draft.evidence_ids)
    if bad_ids:
        return abstention("The model referenced a passage that was not retrieved. Its assessment cannot be verified.")
    # Topic mismatch is determined by comparing the question and student answer.
    # It does not require evidence about the subject matter. A generated factual
    # correction, if provided, should still cite a source (prompt-level guidance).
    if draft.assessment in {"correct", "incomplete", "incorrect"} and not draft.evidence_ids:
        return abstention("The model gave no supporting passage for its assessment.")
    # Deduplicate while preserving first-seen citation order.
    deduped = list(dict.fromkeys(draft.evidence_ids))
    return draft.model_copy(update={"evidence_ids": deduped})


def abstention(reason: str) -> FeedbackDraft:
    return FeedbackDraft(
        assessment="cannot_assess",
        misconception="",
        explanation=reason,
        next_step_hint="Consult a reliable lesson or ask your teacher to clarify the expected concept.",
        evidence_ids=[],
    )
