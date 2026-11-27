"""Pure presentation helpers for the ConceptLens Streamlit demo.

Never infer correctness, citation support, or model confidence here. The frozen
Phase 3 feedback engine and its evaluation code remain unchanged.
"""
from __future__ import annotations

ASSESSMENTS = {
    "correct": ("Correct", "An answer that addresses the key parts of the question."),
    "incomplete": ("Incomplete", "Some relevant understanding, but an essential part is missing."),
    "incorrect": ("Incorrect", "The answer conflicts with an important part of the concept."),
    "off_topic": ("Off topic", "The answer does not address the question."),
    "cannot_assess": ("Cannot assess", "The available evidence is not sufficient for a judgment."),
}

EXAMPLES = {
    "computer_science": (
        ("Mutex · common misconception",
         "Why might two threads need a mutex when updating shared data?",
         "Because threads always run one after another."),
        ("Binary search · correct idea",
         "Why does binary search require sorted data?",
         "The sorted order lets us discard half of the remaining search range after each comparison."),
        ("BFS · incomplete explanation",
         "How does breadth-first search explore a graph?",
         "It uses a queue."),
        ("Mutex · off-topic answer",
         "Why might two threads need a mutex when updating shared data?",
         "My favorite food is pizza."),
    ),
    "electronics": (
        ("Open circuit · common misconception",
         "Why does a bulb stop glowing when a switch opens?",
         "Because the battery has run out of electricity."),
        ("Ohm's law · correct idea",
         "What happens to current if resistance doubles while voltage stays constant?",
         "The current is halved because I = V/R."),
        ("Parallel circuit · incomplete explanation",
         "How does voltage behave across branches in a parallel circuit?",
         "The branches have the same voltage."),
        ("Circuit · off-topic answer",
         "Why does a bulb stop glowing when a switch opens?",
         "I enjoy playing video games."),
    ),
}

MODES = {
    "rag": "Evidence-grounded RAG",
    "llm_only": "LLM only (comparison)",
    "template": "Generic template (comparison)",
}


def partition_sources(sources: list[dict], evidence_ids: list[int]) -> tuple[list[dict], list[dict]]:
    """Separate model-cited passages from merely retrieved passages.

    Membership in *cited* is not verification that the source entails the claim.
    Do not drop, invent, or reorder sources.
    """
    referenced = set(evidence_ids)
    cited = [item for item in sources if item["ref"] in referenced]
    additional = [item for item in sources if item["ref"] not in referenced]
    return cited, additional


def make_feedback_markdown(subject: str, question: str, answer: str, result: dict) -> str:
    """Export the displayed response without asserting independently verified grading."""
    assessment = ASSESSMENTS.get(result["assessment"], (result["assessment"], ""))[0]
    lines = [
        "# ConceptLens — Provisional feedback",
        "",
        f"**Subject:** {subject}",
        f"**Method:** {MODES.get(result['mode'], result['mode'])}",
        "",
        "## Teacher question",
        question.strip(),
        "",
        "## Student answer",
        answer.strip(),
        "",
        f"## Provisional assessment: {assessment}",
        "",
    ]
    if result.get("misconception"):
        lines.extend(["### Possible misconception", result["misconception"], ""])
    lines.extend([
        "### Explanation", result["explanation"], "",
        "### Next-step hint", result["hint"], "",
    ])
    cited, _ = partition_sources(result.get("sources", []), result.get("evidence_ids", []))
    if cited:
        lines.append("### Model-selected citations (not independently validated)")
        for item in cited:
            lines.extend([f"- [{item['ref']}] {item['source']}", ""])
    lines.extend([
        "---", "",
        "Research prototype: assessments and model-selected citations may be incorrect.",
        "Not a verified grade or a calibrated confidence score.", "",
    ])
    return "\n".join(lines)
