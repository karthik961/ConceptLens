import unittest
from pydantic import ValidationError
from src.feedback_schema import FeedbackDraft, validate_draft


def draft(**kwargs):
    values=dict(assessment="incorrect", misconception="May confuse energy with conduction",
                explanation="An open switch interrupts the path.", next_step_hint="Trace the circuit path.", evidence_ids=[1])
    values.update(kwargs)
    return FeedbackDraft(**values)


class SchemaTests(unittest.TestCase):
    def test_bad_assessment_rejected(self):
        with self.assertRaises(ValidationError):
            draft(assessment="probably_wrong")

    def test_invalid_citation_abstains(self):
        checked = validate_draft(draft(evidence_ids=[4]), "rag", 3)
        self.assertEqual(checked.assessment, "cannot_assess")
        self.assertEqual(checked.evidence_ids, [])

    def test_missing_citation_abstains(self):
        self.assertEqual(validate_draft(draft(evidence_ids=[]), "rag", 2).assessment,"cannot_assess")

    def test_good_citations_deduplicate(self):
        self.assertEqual(validate_draft(draft(evidence_ids=[2,1,2]), "rag", 2).evidence_ids,[2,1])

    def test_free_text_unverified_citations_are_removed(self):
        checked = validate_draft(draft(explanation="An open switch stops current [99].", evidence_ids=[1]), "rag", 1)
        self.assertNotIn("[99]", checked.explanation)
        self.assertEqual(checked.evidence_ids, [1])

    def test_llm_only_must_have_no_sources(self):
        self.assertEqual(validate_draft(draft(), "llm_only",0).evidence_ids,[])

if __name__ == '__main__':
    unittest.main()
