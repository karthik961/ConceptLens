"""Optional developmental assessment checklist for ConceptLens.

This wording is copied verbatim from the checklist arm in
scripts/evaluate_rubric_dev.py (16-case development ablation).
Do not edit after the evaluation protocol is frozen.
"""
RUBRIC_ADDENDUM = """
Additional internal assessment checklist (do not output the checklist):
1. Read the QUESTION literally. Identify all parts the student was explicitly
   asked to provide: e.g., a conclusion and its explanation, a value and a
   requested formula, or both parts of a compound question. Do not invent
   additional requirements that are not asked or essential to accuracy.
2. Check the STUDENT ANSWER against each requested part. Credit only information
   actually expressed by the student, including clear paraphrases. Do not
   credit missing facts because they appear in a retrieved passage or because
   you can explain how the student might have arrived at the answer.
3. Decide in this order: a wholly unrelated answer is off_topic; a material
   explicit contradiction of the supported concept is incorrect, NOT merely
   incomplete; a factually sound answer that covers some but not all explicitly
   requested essential parts is incomplete; an answer covering all requested
   essential parts without material error is correct, even if concise.
4. Avoid false omissions: if the student plainly states the required concept
   in other words, do not insist on extra terminology, mechanism or background
   beyond the question. When calling an answer incomplete, identify a specific
   missing part of the QUESTION in the explanation. Never attribute words from
   your own explanation or the reference notes to the student.
These are label-decision rules, not permission to invent citations or weaken
existing evidence validation.
""".strip()
