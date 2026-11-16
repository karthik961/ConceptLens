"""Download & inspect BEETLE. NEVER add these answers to the RAG store."""
from collections import Counter

from datasets import load_dataset

from src.settings import DATA_DIR


def main():
    ds = load_dataset("nkazi/Beetle")
    target = DATA_DIR / "beetle"
    target.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(target))
    names = ds["train"].features["label"].names
    print("BEETLE dataset saved at:", target)
    for split in ("train", "test_ua", "test_uq"):
        counts = Counter(ds[split]["label"])
        print(f"\n{split}: {len(ds[split])} answers")
        for i, name in enumerate(names):
            print(f"  {name}: {counts[i]}")
    print("\nTraining example (do not index as RAG concept evidence):")
    first = ds["train"][0]
    print("Q:", first["question"])
    print("A:", first["student_answer"])
    print("Label:", names[first["label"]])
    print("\nThe test sets are reserved for later evaluation.")


if __name__ == "__main__":
    main()
