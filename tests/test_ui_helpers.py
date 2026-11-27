"""Offline checks for presentation semantics, without Ollama or Streamlit."""
import unittest
from src.ui_helpers import ASSESSMENTS, EXAMPLES, MODES, make_feedback_markdown, partition_sources


class UIRenderingTests(unittest.TestCase):
    def test_expected_subjects_examples_and_categories(self):
        self.assertEqual(set(EXAMPLES), {"computer_science", "electronics"})
        self.assertEqual(set(ASSESSMENTS), {"correct", "incomplete", "incorrect", "off_topic", "cannot_assess"})
        self.assertEqual(set(MODES), {"rag", "llm_only", "template"})
        for entries in EXAMPLES.values():
            self.assertGreaterEqual(len(entries), 3)
            self.assertTrue(all(all(str(v).strip() for v in entry) for entry in entries))

    def test_cited_vs_uncited_preserves_source_order(self):
        sources = [
            {"ref": 1, "source": "one"},
            {"ref": 2, "source": "two"},
            {"ref": 3, "source": "three"},
            {"ref": 4, "source": "four"},
        ]
        cited, other = partition_sources(sources, [4, 2, 2])
        self.assertEqual([s["ref"] for s in cited], [2, 4])
        self.assertEqual([s["ref"] for s in other], [1, 3])
        self.assertEqual(len(cited) + len(other), len(sources))

    def test_no_citations_does_not_fabricate_source_support(self):
        cited, extra = partition_sources([{"ref": 1, "source": "note"}], [])
        self.assertEqual(cited, [])
        self.assertEqual(len(extra), 1)

    def test_export_retains_warning_and_only_cited_references(self):
        result = {
            "assessment": "incorrect", "mode": "rag", "misconception": "Confuses timing.",
            "explanation": "Threads can overlap.", "hint": "Explain critical sections.",
            "evidence_ids": [2], "sources": [
                {"ref": 1, "source": "not cited"},
                {"ref": 2, "source": "mutex lesson"},
            ],
        }
        output = make_feedback_markdown("Computer Science", "Why mutex?", "Because sequential.", result)
        self.assertIn("Provisional", output)
        self.assertIn("[2] mutex lesson", output)
        self.assertNotIn("[1] not cited", output)
        self.assertIn("not independently validated", output)
        self.assertIn("Not a verified grade", output)

    def test_export_without_sources_or_misconception(self):
        result = {"assessment": "cannot_assess", "mode": "template", "misconception": "",
                  "explanation": "No evidence.", "hint": "Review notes.", "evidence_ids": [], "sources": []}
        out = make_feedback_markdown("Electronics", "Q", "A", result)
        self.assertIn("Cannot assess", out)
        self.assertNotIn("Possible misconception", out)
        self.assertNotIn("Model-selected citations", out)

    def test_app_matches_frozen_final_configuration(self):
        # Static check: the UI should not silently fall back to the old 3B defaults.
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertIn('DEMO_MODEL = "qwen2.5:7b"', src)
        self.assertIn('DEMO_RETRIEVAL_K = 4', src)
        self.assertIn('DEMO_PROMPT_VARIANT = "checklist"', src)
        self.assertIn('model=ChatOllama(model=DEMO_MODEL, temperature=0)', src)
        self.assertIn('k=DEMO_RETRIEVAL_K', src)
        self.assertIn('prompt_variant=DEMO_PROMPT_VARIANT', src)
        compile(src, "app.py", "exec")


if __name__ == "__main__":
    unittest.main()
