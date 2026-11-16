import argparse
import shutil

from src.knowledge import create_store, index_documents, load_sample_notes
from src.settings import CHROMA_DIR


def main():
    parser = argparse.ArgumentParser(description="Index project-authored demonstration concept notes")
    parser.add_argument("--reset", action="store_true", help="Delete local database before indexing (including user uploads)")
    args = parser.parse_args()
    if args.reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("Deleted old local Chroma index.")
    docs = load_sample_notes()
    if not docs:
        raise SystemExit("No curated notes in knowledge/ folder")
    store = create_store()
    n = index_documents(store, docs)
    print(f"Indexed {n} chunks from {len(docs)} files into {CHROMA_DIR}")
    print("Start the UI with: streamlit run app.py")


if __name__ == "__main__":
    main()
