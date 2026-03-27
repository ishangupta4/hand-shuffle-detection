"""PyTorch Dataset for hand-shuffle feature sequences.

Loads feature .npy files and labels, applies augmentation on-the-fly
for training, and handles padding/masking for batching.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.augmentation.augmentations import (
    horizontal_flip,
    time_warp,
    gaussian_jitter,
    slight_rotation,
    feature_dropout,
)
from src.augmentation.pipeline import AugmentationConfig, augment_single
from src.augmentation.sequence_utils import pad_sequence, resample_sequence


LABEL_MAP = {"left": 0, "right": 1}


def _normalize_vid(vid) -> str:
    """Normalize a video ID to a zero-padded 5-char string.

    Handles int (1), unpadded str ("1"), or already-padded ("00001").
    """
    return f"{int(vid):05d}"


class HandShuffleDataset(Dataset):
    """Dataset for hand-shuffle prediction.

    In training mode, augmentations are applied on-the-fly to generate
    diverse samples each epoch. In eval mode, returns clean originals.

    Args:
        video_ids: List of video IDs to include (e.g. from a CV split).
        features_dir: Path to directory with feature .npy files.
        labels_df: DataFrame with video_id, start_hand, end_hand columns.
        augment: Whether to apply augmentations (True for train, False for val).
        aug_config: Augmentation configuration.
        seq_mode: How to handle variable lengths: 'pad', 'resample', or 'raw'.
        target_length: Fixed length for pad/resample modes. None = max in batch.
        target_col: Which label to predict ('end_hand' or 'switched').
        epoch_seed_offset: Added to base seed each epoch for variety.
    """

    def __init__(
        self,
        video_ids: list[str],
        features_dir: str = "data/features",
        labels_df: pd.DataFrame = None,
        augment: bool = False,
        aug_config: AugmentationConfig | None = None,
        seq_mode: str = "pad",
        target_length: int | None = None,
        target_col: str = "end_hand",
        seed: int = 42,
    ):
        # Normalize video IDs to zero-padded strings (e.g. 1 -> "00001")
        self.video_ids = sorted([_normalize_vid(v) for v in video_ids])
        self.features_dir = features_dir
        self.augment = augment
        self.aug_config = aug_config or AugmentationConfig()
        self.seq_mode = seq_mode
        self.target_length = target_length
        self.target_col = target_col
        self.seed = seed
        self._epoch = 0

        # Load all sequences and labels
        self.sequences = []
        self.labels = []
        self.raw_labels = []

        if labels_df is None:
            labels_df = pd.read_csv("data/labels.csv", dtype={"video_id": str})

        # Ensure label lookup keys are also normalized
        labels_df = labels_df.copy()
        labels_df["video_id"] = labels_df["video_id"].apply(_normalize_vid)
        label_lookup = labels_df.set_index("video_id").to_dict("index")

        for vid in self.video_ids:
            feat_path = os.path.join(features_dir, f"{vid}.npy")
            seq = np.load(feat_path).astype(np.float32)
            # Replace any remaining NaNs with 0 for model input
            seq = np.nan_to_num(seq, nan=0.0)
            self.sequences.append(seq)

            row = label_lookup[vid]
            label_dict = {
                "start_hand": row["start_hand"],
                "end_hand": row["end_hand"],
            }
            self.raw_labels.append(label_dict)

            if target_col == "switched":
                self.labels.append(int(row["start_hand"] != row["end_hand"]))
            else:
                self.labels.append(LABEL_MAP[row[target_col]])

        # Pre-generate augmented samples if augmenting
        self._build_samples()

        # Compute target_length if not specified (max across originals)
        if self.target_length is None:
            all_lengths = [s.shape[0] for s in self.sequences]
            if self.augment:
                all_lengths += [s.shape[0] for s, _, _ in self._aug_cache]
            self.target_length = max(all_lengths)

    def _build_samples(self):
        """Pre-generate augmented samples for this epoch."""
        if not self.augment:
            self._aug_cache = []
            return

        rng = np.random.default_rng(self.seed + self._epoch)
        self._aug_cache = []

        for seq, label_dict in zip(self.sequences, self.raw_labels):
            variants = augment_single(seq, label_dict, self.aug_config, rng=rng)
            # Skip the first one (original) since we already have it
            for v_seq, v_label, v_desc in variants[1:]:
                v_seq = np.nan_to_num(v_seq, nan=0.0).astype(np.float32)
                if self.target_col == "switched":
                    numeric_label = int(v_label["start_hand"] != v_label["end_hand"])
                else:
                    numeric_label = LABEL_MAP[v_label[self.target_col]]
                self._aug_cache.append((v_seq, numeric_label, v_desc))

    def set_epoch(self, epoch: int):
        """Update epoch for augmentation variety. Call before each epoch."""
        self._epoch = epoch
        self._build_samples()

    def __len__(self):
        n = len(self.sequences)
        if self.augment:
            n += len(self._aug_cache)
        return n

    def __getitem__(self, idx):
        if idx < len(self.sequences):
            seq = self.sequences[idx]
            label = self.labels[idx]
        else:
            aug_idx = idx - len(self.sequences)
            seq, label, _ = self._aug_cache[aug_idx]

        actual_length = seq.shape[0]

        # Handle sequence length
        if self.seq_mode == "pad":
            seq_out, mask = pad_sequence(seq, self.target_length)
        elif self.seq_mode == "resample":
            seq_out = resample_sequence(seq, self.target_length)
            mask = np.ones(self.target_length, dtype=np.float32)
        else:  # raw — return as-is (caller handles batching)
            return {
                "features": torch.from_numpy(seq),
                "label": torch.tensor(label, dtype=torch.long),
                "length": actual_length,
            }

        return {
            "features": torch.from_numpy(seq_out),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label, dtype=torch.long),
            "length": actual_length,
        }


def collate_variable_length(batch: list[dict]) -> dict:
    """Custom collate for raw (variable-length) mode.

    Pads to the max length in the batch, adds masks.
    Use with DataLoader(collate_fn=collate_variable_length).
    """
    max_len = max(item["length"] for item in batch)
    F = batch[0]["features"].shape[1]

    features = torch.zeros(len(batch), max_len, F)
    masks = torch.zeros(len(batch), max_len)
    labels = torch.zeros(len(batch), dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["length"]
        features[i, :L] = item["features"]
        masks[i, :L] = 1.0
        labels[i] = item["label"]
        lengths[i] = L

    return {
        "features": features,
        "mask": masks,
        "label": labels,
        "length": lengths,
    }


if __name__ == "__main__":
    # Quick demo — requires actual data files
    labels_path = "data/labels.csv"
    features_dir = "data/features"

    if not os.path.exists(labels_path):
        print("No data files found. Creating synthetic demo...")
        os.makedirs(features_dir, exist_ok=True)

        # Create fake data
        rng = np.random.default_rng(42)
        records = []
        for i in range(1, 6):
            vid = f"{i:05d}"
            T = rng.integers(30, 60)
            np.save(os.path.join(features_dir, f"{vid}.npy"),
                    rng.standard_normal((T, 39)).astype(np.float32))
            records.append({
                "video_id": vid,
                "start_hand": rng.choice(["left", "right"]),
                "end_hand": rng.choice(["left", "right"]),
            })
        pd.DataFrame(records).to_csv(labels_path, index=False)
        print(f"Created {len(records)} synthetic samples.")

    df = pd.read_csv(labels_path, dtype={"video_id": str})
    video_ids = df["video_id"].tolist()[:5]  # Use subset for demo

    # Test pad mode (default)
    ds = HandShuffleDataset(
        video_ids=video_ids,
        features_dir=features_dir,
        labels_df=df,
        augment=True,
        seq_mode="pad",
    )
    print(f"\nDataset size: {len(ds)} (from {len(video_ids)} originals)")
    sample = ds[0]
    print(f"Sample: features={sample['features'].shape}, "
          f"mask_sum={sample['mask'].sum():.0f}, label={sample['label']}")

    # Test DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"Batch: features={batch['features'].shape}, labels={batch['label']}")

    # Test variable-length mode
    ds_raw = HandShuffleDataset(
        video_ids=video_ids,
        features_dir=features_dir,
        labels_df=df,
        augment=False,
        seq_mode="raw",
    )
    loader_raw = DataLoader(ds_raw, batch_size=3, collate_fn=collate_variable_length)
    batch_raw = next(iter(loader_raw))
    print(f"Variable batch: features={batch_raw['features'].shape}, "
          f"lengths={batch_raw['length']}")

    print("\nDataset tests passed.")