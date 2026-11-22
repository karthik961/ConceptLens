"""Fail-closed, idempotent version-aware edit to ConceptLens src/feedback_engine.py.

Only adds an opt-in prompt_variant keyword; it does not change existing defaults.
Review the diff after applying. This script should run from the repo root.
"""
from pathlib import Path
import argparse
import difflib

IMPORT_OLD = 'from src.feedback_schema import FeedbackDraft, abstention, validate_draft\n'
IMPORT_NEW = IMPORT_OLD + 'from src.assessment_checklist import RUBRIC_ADDENDUM\n'
CONST_OLD = 'MODES = ("rag", "llm_only", "template")\n'
CONST_NEW = CONST_OLD + 'PROMPT_VARIANTS = ("baseline", "checklist")\n'
MESSAGES_OLD = '''def _messages(subject: str, question: str, answer: str, mode: str, context: str) -> list[tuple[str, str]]:
    policy = RAG_POLICY if mode == "rag" else NO_RAG_POLICY
    human = (f"Subject: {subject}\\nQuestion: {question}\\n"
             f"Student answer: {answer}\\n\\n"
             f"Retrieved evidence:\\n{context if mode == 'rag' else '(none)'}")
    return [("system", SYSTEM_PROMPT + "\\n" + policy), ("human", human)]
'''
MESSAGES_NEW = '''def _messages(subject: str, question: str, answer: str, mode: str, context: str,
              prompt_variant: str = "baseline") -> list[tuple[str, str]]:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {prompt_variant}")
    policy = RAG_POLICY if mode == "rag" else NO_RAG_POLICY
    system_prompt = SYSTEM_PROMPT
    if prompt_variant == "checklist":
        system_prompt += "\\n\\n" + RUBRIC_ADDENDUM
    human = (f"Subject: {subject}\\nQuestion: {question}\\n"
             f"Student answer: {answer}\\n\\n"
             f"Retrieved evidence:\\n{context if mode == 'rag' else '(none)'}")
    return [("system", system_prompt + "\\n" + policy), ("human", human)]
'''
GIVE_SIGNATURE_OLD = '''def give_feedback(store, subject: str, question: str, answer: str,
                  k: int = 3, mode: str = "rag", model=None,
                  retriever=unique_sources_search) -> dict:
'''
GIVE_SIGNATURE_NEW = '''def give_feedback(store, subject: str, question: str, answer: str,
                  k: int = 3, mode: str = "rag", model=None,
                  retriever=unique_sources_search, *, prompt_variant: str = "baseline") -> dict:
'''
VALIDATE_OLD='''    validate_subject(subject)
    if mode not in MODES:
        raise ValueError(f"Unknown feedback mode: {mode}")
'''
VALIDATE_NEW='''    validate_subject(subject)
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {prompt_variant}")
    if mode not in MODES:
        raise ValueError(f"Unknown feedback mode: {mode}")
'''
CALL_OLD='''    messages = _messages(subject, question, answer, mode, make_context(docs) if docs else "")
'''
CALL_NEW='''    messages = _messages(subject, question, answer, mode, make_context(docs) if docs else "",
                         prompt_variant=prompt_variant)
'''
REPLACEMENTS = [(IMPORT_OLD, IMPORT_NEW), (CONST_OLD, CONST_NEW),
                (MESSAGES_OLD, MESSAGES_NEW), (GIVE_SIGNATURE_OLD, GIVE_SIGNATURE_NEW),
                (VALIDATE_OLD, VALIDATE_NEW), (CALL_OLD, CALL_NEW)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo', default='.', help='ConceptLens repository root')
    ap.add_argument('--check', action='store_true', help='Check applicability, print diff, do not edit')
    args=ap.parse_args()
    path=Path(args.repo).resolve()/'src'/'feedback_engine.py'
    new_mod=Path(args.repo).resolve()/'src'/'assessment_checklist.py'
    if not path.is_file() or not new_mod.is_file():
        ap.error('Expected src/feedback_engine.py and src/assessment_checklist.py')
    original=path.read_text(encoding='utf-8')
    if 'from src.assessment_checklist import RUBRIC_ADDENDUM' in original:
        print('Already installed; no changes made')
        return
    revised=original
    for old,new in REPLACEMENTS:
        count=revised.count(old)
        if count!=1:
            ap.error(f'Unexpected feedback engine revision: expected one unique occurrence of {old[:75]!r}, found {count}. Nothing modified.')
        revised=revised.replace(old,new,1)
    diff=''.join(difflib.unified_diff(original.splitlines(True), revised.splitlines(True),
            fromfile='src/feedback_engine.py (before)',tofile='src/feedback_engine.py (after)'))
    print(diff)
    if args.check:
        print('Check passed; no files were modified.')
        return
    backup=path.with_name(path.name+'.pre_checklist.bak')
    if backup.exists():
        ap.error(f'Backup already exists: {backup}. Refusing overwrite.')
    backup.write_text(original, encoding='utf-8')
    path.write_text(revised, encoding='utf-8')
    print(f'Installed optional variant. Original backup: {backup}')

if __name__=='__main__':
    main()
