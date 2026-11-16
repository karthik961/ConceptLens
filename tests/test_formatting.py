import unittest
from dataclasses import dataclass

from src.formatting import document_id, make_context, validate_subject


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict


class FormattingTests(unittest.TestCase):
    def test_subject_validation(self):
        self.assertEqual(validate_subject("electronics"), "electronics")
        with self.assertRaises(ValueError):
            validate_subject("history")

    def test_stable_document_ids(self):
        a = document_id("electronics", "note.md", "", "a passage")
        self.assertEqual(a, document_id("electronics", "note.md", "", "a passage"))
        self.assertNotEqual(a, document_id("computer_science", "note.md", "", "a passage"))

    def test_numbered_context(self):
        docs = [FakeDocument("A circuit requires a closed path.", {"source": "open.md", "page": 2})]
        self.assertEqual(make_context(docs), "[1] Source: open.md (p. 2)\nA circuit requires a closed path.")


if __name__ == "__main__":
    unittest.main()
