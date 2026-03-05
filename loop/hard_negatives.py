# loop/hard_negatives.py
import os
import pandas as pd

HARD_NEG_PATH = "data/evolution/hard_negatives.csv"
MERGED_TRAIN_PATH = "data/evolution/train_plus_hard.csv"
BASE_TRAIN_PATH = "data/processed/train.csv"

def _ensure_dirs():
    os.makedirs(os.path.dirname(HARD_NEG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MERGED_TRAIN_PATH), exist_ok=True)

def save_hard_negative(text: str, label: int = 1):
    """
    Save a hard negative sample. For generated phishing, label=1 (phish).
    """
    _ensure_dirs()
    row = pd.DataFrame([{"text": text, "label": label}])
    if os.path.exists(HARD_NEG_PATH):
        row.to_csv(HARD_NEG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(HARD_NEG_PATH, index=False)

def build_merged_train() -> str:
    """
    Merge base train.csv with hard_negatives.csv (if present).
    Returns path to merged CSV.
    """
    _ensure_dirs()
    if not os.path.exists(BASE_TRAIN_PATH):
        raise FileNotFoundError(f"Base train not found: {BASE_TRAIN_PATH}")

    base = pd.read_csv(BASE_TRAIN_PATH)
    if os.path.exists(HARD_NEG_PATH):
        hn = pd.read_csv(HARD_NEG_PATH)
        merged = pd.concat([base, hn], ignore_index=True)
    else:
        merged = base

    # Simple dedupe if any
    merged = merged.drop_duplicates(subset=["text", "label"])
    merged.to_csv(MERGED_TRAIN_PATH, index=False)
    return MERGED_TRAIN_PATH

def hard_neg_count() -> int:
    if not os.path.exists(HARD_NEG_PATH):
        return 0
    return sum(1 for _ in open(HARD_NEG_PATH, "r")) - 1 if os.path.getsize(HARD_NEG_PATH) > 0 else 0
