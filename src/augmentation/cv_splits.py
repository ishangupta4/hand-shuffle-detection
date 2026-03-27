"""Cross-validation split strategies for the hand-shuffle dataset.

KEY RULE: Splits are defined at the original video level. All augmented
copies of a video must stay in the same fold as the original. This
prevents data leakage between train and validation.

Provides:
    - Leave-One-Out CV (LOOCV): 19 folds, each holds out 1 video
    - K-Fold Stratified CV: groups by end_hand label for balanced folds
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_labels(labels_path: str = "data/labels.csv") -> pd.DataFrame:
    """Load video labels and derive the 'switched' column."""
    df = pd.read_csv(labels_path, dtype={"video_id": str})
    df["switched"] = (df["start_hand"] != df["end_hand"]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Leave-One-Out CV
# ---------------------------------------------------------------------------

def leave_one_out_splits(
    video_ids: list[str],
) -> list[dict]:
    """Generate LOOCV splits. Each fold holds out one video.

    Returns list of dicts with keys:
        fold: int
        train_ids: list of video IDs for training
        val_ids: list of video IDs for validation (single video)
    """
    splits = []
    for i, val_id in enumerate(video_ids):
        train_ids = [vid for vid in video_ids if vid != val_id]
        splits.append({
            "fold": i,
            "train_ids": train_ids,
            "val_ids": [val_id],
        })
    return splits


# ---------------------------------------------------------------------------
# K-Fold Stratified CV
# ---------------------------------------------------------------------------

def stratified_kfold_splits(
    video_ids: list[str],
    labels_df: pd.DataFrame,
    n_folds: int = 5,
    stratify_col: str = "end_hand",
    seed: int = 42,
) -> list[dict]:
    """Generate stratified K-fold splits.

    Stratifies by the target label (end_hand) to ensure each fold
    has roughly the same class distribution.
    """
    rng = np.random.default_rng(seed)

    # Group IDs by class
    id_to_label = dict(zip(labels_df["video_id"], labels_df[stratify_col]))
    classes = {}
    for vid in video_ids:
        cls = id_to_label.get(vid, "unknown")
        classes.setdefault(cls, []).append(vid)

    # Shuffle within each class
    for cls in classes:
        rng.shuffle(classes[cls])

    # Distribute into folds round-robin, one class at a time
    fold_assignments = {}
    for cls, ids in classes.items():
        for i, vid in enumerate(ids):
            fold_assignments[vid] = i % n_folds

    # Build split dicts
    splits = []
    for fold in range(n_folds):
        val_ids = [vid for vid, f in fold_assignments.items() if f == fold]
        train_ids = [vid for vid, f in fold_assignments.items() if f != fold]
        splits.append({
            "fold": fold,
            "train_ids": sorted(train_ids),
            "val_ids": sorted(val_ids),
        })

    return splits


# ---------------------------------------------------------------------------
# Save / load splits for reproducibility
# ---------------------------------------------------------------------------

def save_splits(splits: list[dict], output_path: str):
    """Save split definitions to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved {len(splits)} fold definitions to {output_path}")


def load_splits(splits_path: str) -> list[dict]:
    """Load split definitions from JSON."""
    with open(splits_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_split_summary(splits: list[dict], labels_df: pd.DataFrame | None = None):
    """Print a summary of fold composition."""
    id_to_end = {}
    if labels_df is not None:
        id_to_end = dict(zip(labels_df["video_id"], labels_df["end_hand"]))

    for split in splits:
        fold = split["fold"]
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]

        if id_to_end:
            train_dist = {}
            for vid in train_ids:
                cls = id_to_end.get(vid, "?")
                train_dist[cls] = train_dist.get(cls, 0) + 1
            val_dist = {}
            for vid in val_ids:
                cls = id_to_end.get(vid, "?")
                val_dist[cls] = val_dist.get(cls, 0) + 1
            print(f"  Fold {fold}: train={len(train_ids)} {train_dist}, "
                  f"val={len(val_ids)} {val_dist}")
        else:
            print(f"  Fold {fold}: train={len(train_ids)}, val={len(val_ids)}")


if __name__ == "__main__":
    # Demo with the actual labels
    labels_path = "data/labels.csv"
    if os.path.exists(labels_path):
        df = load_labels(labels_path)
    else:
        # Fallback for testing without data dir
        df = pd.DataFrame({
            "video_id": [f"{i:05d}" for i in range(1, 20)],
            "start_hand": ["right"] * 10 + ["left"] * 9,
            "end_hand": ["left"] * 8 + ["right"] * 11,
        })
        df["switched"] = (df["start_hand"] != df["end_hand"]).astype(int)

    video_ids = df["video_id"].tolist()
    print(f"Dataset: {len(video_ids)} videos")
    print(f"End hand distribution: {df['end_hand'].value_counts().to_dict()}")
    print(f"Switched distribution: {df['switched'].value_counts().to_dict()}")

    # LOOCV
    loo_splits = leave_one_out_splits(video_ids)
    print(f"\nLOOCV: {len(loo_splits)} folds")
    print_split_summary(loo_splits[:3], df)
    print("  ...")

    # 5-Fold stratified
    kfold_splits = stratified_kfold_splits(video_ids, df, n_folds=5)
    print(f"\n5-Fold Stratified CV:")
    print_split_summary(kfold_splits, df)

    # Save
    os.makedirs("data", exist_ok=True)
    save_splits(loo_splits, "data/splits_loocv.json")
    save_splits(kfold_splits, "data/splits_5fold.json")