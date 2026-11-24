from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge"
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / ".chromadb_1100"
COLLECTION_NAME = "conceptlens_local_v1"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:3b"
SUBJECTS = {
    "electronics": "Electronics",
    "computer_science": "Computer Science",
}
