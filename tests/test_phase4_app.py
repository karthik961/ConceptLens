"""Offline checks for the narrow, reversible Streamlit integration."""
import unittest
from pathlib import Path
from scripts.enable_phase4_app import OLD_IMPORT, OLD_CALL, OLD_CAPTION, NEW_IMPORT, NEW_CALL, transform


class Phase4AppPatchTests(unittest.TestCase):
    @staticmethod
    def sample():
        return f'import streamlit as st\n{OLD_IMPORT}{OLD_CAPTION}\n\nif True:\n{OLD_CALL}\n'

    def test_inserts_final_configuration(self):
        changed = transform(self.sample())
        self.assertIn('model="qwen2.5:7b"', changed)
        self.assertIn('prompt_variant="checklist"', changed)
        self.assertIn('k=4', changed)
        self.assertIn(NEW_IMPORT, changed)
        self.assertIn(NEW_CALL, changed)
        compile(changed, "app.py", "exec")

    def test_idempotent(self):
        once = transform(self.sample())
        self.assertEqual(transform(once), once)

    def test_fail_closed_when_old_call_differs(self):
        with self.assertRaisesRegex(ValueError, "give_feedback call"):
            transform(self.sample().replace('mode=mode,', 'mode="rag",'))

    def test_preserves_other_code(self):
        source = self.sample() + '\nprint("preserved")\n'
        self.assertIn('print("preserved")', transform(source))


if __name__ == "__main__":
    unittest.main()
