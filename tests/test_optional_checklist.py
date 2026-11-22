import unittest
from src import feedback_engine as engine
from src.assessment_checklist import RUBRIC_ADDENDUM

class PromptVariantTests(unittest.TestCase):
    def test_baseline_unchanged(self):
        messages=engine._messages('electronics','Q?','answer','rag','[1] evidence')
        self.assertEqual(messages[0][1], engine.SYSTEM_PROMPT + '\n' + engine.RAG_POLICY)
        self.assertIn('[1] evidence', messages[1][1])

    def test_checklist_changes_only_system_prompt(self):
        b=engine._messages('electronics','Q?','answer','rag','[1] evidence')
        c=engine._messages('electronics','Q?','answer','rag','[1] evidence', prompt_variant='checklist')
        self.assertEqual(b[1],c[1])
        self.assertIn(RUBRIC_ADDENDUM,c[0][1])
        self.assertIn(engine.RAG_POLICY,c[0][1])
        self.assertNotIn(RUBRIC_ADDENDUM,b[0][1])

    def test_unknown_variant_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown prompt variant'):
            engine._messages('electronics','Q','A','rag','context',prompt_variant='unknown')
        with self.assertRaisesRegex(ValueError,'Unknown prompt variant'):
            engine.give_feedback(None,'electronics','Q','A',mode='template',prompt_variant='unknown')

    def test_template_baseline_unchanged(self):
        r=engine.give_feedback(None,'electronics','Question?','Answer',mode='template')
        self.assertEqual(r['assessment'],'cannot_assess')

if __name__=='__main__':
    unittest.main()
