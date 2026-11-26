"""Safely select the evaluated ConceptLens feedback configuration in app.py.

Run from the project root. Preserves a local backup. Does not alter the engine,
retrieval implementation, Chroma indexes, or evaluation artifacts.
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path

OLD_IMPORT = "from src.rag import give_feedback\n"
NEW_IMPORT = "from src.feedback_engine import give_feedback\nfrom langchain_ollama import ChatOllama\n"
OLD_CALL = '''                result = give_feedback(
                    local_store() if mode == "rag" else None,
                    subject, question, answer, mode=mode,
                )'''
NEW_CALL = '''                result = give_feedback(
                    local_store() if mode == "rag" else None,
                    subject, question, answer,
                    k=4,
                    mode=mode,
                    model=ChatOllama(model="qwen2.5:7b", temperature=0) if mode != "template" else None,
                    prompt_variant="checklist",
                )'''
OLD_CAPTION = 'st.caption("CS 678 · Local, evidence-grounded student feedback · Research prototype")'
NEW_CAPTION = 'st.caption("CS 678 · Qwen2.5 7B · Checklist prompt · Semantic RAG (k=4) · Research prototype")'


def transform(source: str) -> str:
    if NEW_IMPORT in source and NEW_CALL in source:
        return source  # idempotent
    for name, value in (("import", OLD_IMPORT), ("give_feedback call", OLD_CALL), ("caption", OLD_CAPTION)):
        if source.count(value) != 1:
            raise ValueError(f"Expected exactly one original {name}; found {source.count(value)}. No files changed.")
    return source.replace(OLD_IMPORT, NEW_IMPORT, 1).replace(OLD_CALL, NEW_CALL, 1).replace(OLD_CAPTION, NEW_CAPTION, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true", help="Preview and validate without writing")
    args = parser.parse_args()
    root = args.repo.resolve()
    path = root / "app.py"
    if not (root / "src/feedback_engine.py").exists() or not (root / "src/assessment_checklist.py").exists():
        raise SystemExit("Missing expected Phase 3 engine/checklist files in project root.")
    original = path.read_text(encoding="utf-8")
    modified = transform(original)
    if original == modified:
        print("Phase 4 integration already applied; no change.")
        return
    print("".join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True), fromfile="app.py (old)", tofile="app.py (candidate)")))
    compile(modified, str(path), "exec")
    if args.check:
        print("Check passed; app.py unchanged.")
        return
    backup = root / "app.py.pre_phase4.bak"
    if backup.exists():
        raise SystemExit(f"Backup already exists: {backup}. Review it before applying; app.py unchanged.")
    backup.write_text(original, encoding="utf-8")
    path.write_text(modified, encoding="utf-8")
    print(f"Installed evaluated configuration. Backup: {backup}")


if __name__ == "__main__":
    main()
