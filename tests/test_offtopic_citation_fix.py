"""Regression checks for the distinct off-topic/evidence behavior.

No GPU, Ollama, Chroma, or internet required.
"""
import unittest
from dataclasses import dataclass

from src.feedback_engine import give_feedback
from src.feedback_schema import FeedbackDraft, validate_draft


@dataclass
class Doc:
    page_content: str
    metadata: dict


def retriever(_store, _query, subject, k=3):
    return {
        "hits": [{
            "source": f"{subject}/test.md",
            "distance": 0.25,
            "document": Doc("A stack is a LIFO data structure.", {"source": f"{subject}/test.md"}),
        }],
        "latency_ms": 1.0,
    }


def draft(assessment, evidence_ids, explanation="The response does not answer the question."):
    return FeedbackDraft(
        assessment=assessment,
        misconception="",
        explanation=explanation,
        next_step_hint="Try answering the question instead.",
        evidence_ids=evidence_ids,
    )


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def with_structured_output(self, schema, method="json_schema"):
        assert schema is FeedbackDraft
        assert method == "json_schema"
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.responses[len(self.calls) - 1]


class OffTopicCitationTests(unittest.TestCase):
    def test_off_topic_no_citation_is_valid(self):
        out = validate_draft(draft("off_topic", []), mode="rag", source_count=1)
        self.assertEqual(out.assessment, "off_topic")
        self.assertEqual(out.evidence_ids, [])

    def test_uncited_off_topic_does_not_trigger_retry(self):
        m = FakeModel([draft("off_topic", [])])
        result = give_feedback(
            object(), "computer_science",
            "What data structure is last in first out?",
            "The ocean is salty.", model=m, retriever=retriever,
        )
        self.assertEqual(result["assessment"], "off_topic")
        self.assertEqual(result["evidence_ids"], [])
        self.assertFalse(result["abstained"])
        self.assertEqual(len(m.calls), 1)
        self.assertIn("off_topic", m.calls[0][0][1])
        self.assertIn("evidence_ids=[] is valid", m.calls[0][0][1])

    def test_subject_matter_categories_still_need_citations(self):
        for label in ("correct", "incomplete", "incorrect"):
            with self.subTest(label=label):
                out = validate_draft(draft(label, []), mode="rag", source_count=1)
                self.assertEqual(out.assessment, "cannot_assess")

    def test_missing_subject_matter_citation_still_triggers_retry(self):
        m = FakeModel([draft("incorrect", []), draft("incorrect", [1])])
        out = give_feedback(
            object(), "computer_science", "What is a stack?", "It's a queue.",
            model=m, retriever=retriever,
        )
        self.assertEqual(out["assessment"], "incorrect")
        self.assertEqual(out["evidence_ids"], [1])
        self.assertEqual(len(m.calls), 2)

    def test_invalid_off_topic_citation_is_rejected(self):
        out = validate_draft(draft("off_topic", [99]), mode="rag", source_count=1)
        self.assertEqual(out.assessment, "cannot_assess")

    def test_valid_off_topic_citation_is_preserved(self):
        out = validate_draft(draft("off_topic", [1, 1]), mode="rag", source_count=1)
        self.assertEqual(out.assessment, "off_topic")
        self.assertEqual(out.evidence_ids, [1])

    def test_cannot_assess_without_evidence_is_still_valid(self):
        out = validate_draft(draft("cannot_assess", []), mode="rag", source_count=1)
        self.assertEqual(out.assessment, "cannot_assess")

    def test_no_retrieved_passages_still_abstains(self):
        def no_hits(_store, _query, _subject, k=3):
            return {"hits": [], "latency_ms": 1.0}
        m = FakeModel([draft("off_topic", [])])
        out = give_feedback(object(), "computer_science", "What is a stack?",
                            "The ocean is salty.", model=m, retriever=no_hits)
        self.assertEqual(out["assessment"], "cannot_assess")
        self.assertEqual(len(m.calls), 0)


if __name__ == "__main__":
    unittest.main()
