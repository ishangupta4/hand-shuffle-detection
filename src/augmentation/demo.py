#!/usr/bin/env python3
"""Demo script: exercises the full Phase 4 augmentation pipeline.

Loads real feature data, generates augmented samples, sets up CV splits,
and shows a complete training-ready DataLoader. Run from repo root:

    python -m src.augmentation.demo
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.augmentation.augmentations import (
    horizontal_flip, time_warp, gaussian_jitter,
    slight_rotation, feature_dropout, load_feature_metadata,
)
from src.augmentation.pipeline import (
    AugmentationConfig, augment_single, augment_dataset,
    estimate_augmentation_factor,
)
from src.augmentation.sequence_utils import (
    pad_batch, resample_batch, compute_length_stats,
)
from src.augmentation.cv_splits import (
    leave_one_out_splits, stratified_kfold_splits,
    load_labels, save_splits, print_split_summary,
)
from src.augmentation.dataset import HandShuffleDataset, collate_variable_length


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    features_dir = "data/features"
    labels_path = "data/labels.csv"
    metadata_path = "data/features/feature_metadata.json"

    # ------------------------------------------------------------------
    section("1. Loading data")
    # ------------------------------------------------------------------

    labels_df = load_labels(labels_path)
    print(f"Labels loaded: {len(labels_df)} videos")
    print(f"  End hand distribution: {labels_df['end_hand'].value_counts().to_dict()}")
    print(f"  Switched distribution: {labels_df['switched'].value_counts().to_dict()}")

    video_ids = labels_df["video_id"].tolist()

    sequences = []
    for vid in video_ids:
        seq = np.load(os.path.join(features_dir, f"{vid}.npy")).astype(np.float32)
        sequences.append(seq)

    stats = compute_length_stats(sequences)
    print(f"  Sequence lengths: min={stats['min']}, max={stats['max']}, "
          f"mean={stats['mean']:.1f}, median={stats['median']:.0f}")

    metadata = load_feature_metadata(metadata_path)
    print(f"  Features: {metadata['num_features']} "
          f"({metadata['num_static']} static + {metadata['num_dynamic']} dynamic)")

    # ------------------------------------------------------------------
    section("2. Individual augmentation demos")
    # ------------------------------------------------------------------

    sample_seq = sequences[0]
    sample_label = {
        "start_hand": labels_df.iloc[0]["start_hand"],
        "end_hand": labels_df.iloc[0]["end_hand"],
    }
    print(f"Sample: video {video_ids[0]}, shape={sample_seq.shape}, label={sample_label}")

    # Horizontal flip
    flipped, fl = horizontal_flip(sample_seq, sample_label, metadata_path=metadata_path)
    print(f"\n  Horizontal flip:")
    print(f"    Shape: {flipped.shape}, label: {fl}")
    print(f"    Max diff from original: {np.nanmax(np.abs(flipped - sample_seq)):.4f}")

    # Time warp
    for speed in [0.8, 1.0, 1.2]:
        warped, _ = time_warp(sample_seq, sample_label, speed_factor=speed)
        print(f"  Time warp {speed}x: {sample_seq.shape[0]} -> {warped.shape[0]} frames")

    # Gaussian jitter
    rng = np.random.default_rng(42)
    jittered, _ = gaussian_jitter(sample_seq, sample_label, sigma_fraction=0.02, rng=rng)
    diffs = np.abs(jittered - sample_seq)
    print(f"  Gaussian jitter: mean_diff={np.nanmean(diffs):.5f}, "
          f"max_diff={np.nanmax(diffs):.5f}")

    # Slight rotation
    rotated, _ = slight_rotation(sample_seq, sample_label, metadata_path=metadata_path, rng=rng)
    diffs_r = np.abs(rotated - sample_seq)
    print(f"  Slight rotation: mean_diff={np.nanmean(diffs_r):.5f}")

    # Feature dropout
    dropped, _ = feature_dropout(sample_seq, sample_label, drop_rate=0.07, rng=rng)
    n_valid = np.sum(~np.isnan(sample_seq))
    n_zeroed = np.sum((dropped == 0) & (sample_seq != 0) & ~np.isnan(sample_seq))
    print(f"  Feature dropout: {n_zeroed}/{n_valid} values zeroed "
          f"({n_zeroed/n_valid*100:.1f}%)")

    # ------------------------------------------------------------------
    section("3. Augmentation pipeline")
    # ------------------------------------------------------------------

    config = AugmentationConfig(metadata_path=metadata_path)
    factor = estimate_augmentation_factor(config)
    print(f"Default config augmentation factor: {factor}x")
    print(f"Expected total: {len(video_ids)} * {factor} = {len(video_ids) * factor} samples")

    # Augment a single sample to show all variants
    variants = augment_single(sample_seq, sample_label, config, rng=rng)
    print(f"\nSingle-sample variants: {len(variants)}")
    descs = [v[2] for v in variants]
    # Show a subset
    for desc in descs[:10]:
        print(f"    {desc}")
    if len(descs) > 10:
        print(f"    ... and {len(descs)-10} more")

    # Full dataset augmentation
    all_labels = [
        {"start_hand": row["start_hand"], "end_hand": row["end_hand"]}
        for _, row in labels_df.iterrows()
    ]

    t0 = time.time()
    aug_seqs, aug_labels, aug_ids = augment_dataset(
        sequences, all_labels, video_ids, config=config
    )
    elapsed = time.time() - t0
    print(f"\nFull augmentation: {len(video_ids)} -> {len(aug_seqs)} samples "
          f"({len(aug_seqs)/len(video_ids):.1f}x) in {elapsed:.2f}s")

    aug_lengths = [s.shape[0] for s in aug_seqs]
    print(f"  Augmented length range: {min(aug_lengths)}-{max(aug_lengths)} frames")

    # Label distribution in augmented set
    end_hands = [l["end_hand"] for l in aug_labels]
    from collections import Counter
    print(f"  Augmented end_hand distribution: {dict(Counter(end_hands))}")

    # ------------------------------------------------------------------
    section("4. Cross-validation splits")
    # ------------------------------------------------------------------

    # LOOCV
    loo = leave_one_out_splits(video_ids)
    print(f"LOOCV: {len(loo)} folds")
    print_split_summary(loo[:3], labels_df)
    print("  ...")

    # 5-Fold
    kfold = stratified_kfold_splits(video_ids, labels_df, n_folds=5)
    print(f"\n5-Fold Stratified CV:")
    print_split_summary(kfold, labels_df)

    # Save splits
    os.makedirs("data", exist_ok=True)
    save_splits(loo, "data/splits_loocv.json")
    save_splits(kfold, "data/splits_5fold.json")

    # ------------------------------------------------------------------
    section("5. PyTorch Dataset & DataLoader")
    # ------------------------------------------------------------------

    # Use fold 0 of 5-fold as an example
    fold = kfold[0]
    print(f"Fold 0: train={len(fold['train_ids'])} videos, "
          f"val={len(fold['val_ids'])} videos")

    # Training dataset (with augmentation)
    train_ds = HandShuffleDataset(
        video_ids=fold["train_ids"],
        features_dir=features_dir,
        labels_df=labels_df,
        augment=True,
        aug_config=config,
        seq_mode="pad",
        target_col="end_hand",
    )
    print(f"\nTraining dataset: {len(train_ds)} samples "
          f"(from {len(fold['train_ids'])} originals)")

    # Validation dataset (no augmentation)
    val_ds = HandShuffleDataset(
        video_ids=fold["val_ids"],
        features_dir=features_dir,
        labels_df=labels_df,
        augment=False,
        seq_mode="pad",
        target_length=train_ds.target_length,
        target_col="end_hand",
    )
    print(f"Validation dataset: {len(val_ds)} samples")

    # DataLoader
    try:
        import torch
        from torch.utils.data import DataLoader

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

        batch = next(iter(train_loader))
        print(f"\nTrain batch:")
        print(f"  features: {batch['features'].shape}")
        print(f"  mask: {batch['mask'].shape} (sum per sample: "
              f"{batch['mask'].sum(dim=1)[:4].int().tolist()}...)")
        print(f"  labels: {batch['label'].tolist()}")

        val_batch = next(iter(val_loader))
        print(f"\nVal batch:")
        print(f"  features: {val_batch['features'].shape}")
        print(f"  labels: {val_batch['label'].tolist()}")

    except ImportError:
        print("PyTorch not installed, skipping DataLoader demo.")

    # ------------------------------------------------------------------
    section("6. Epoch-to-epoch augmentation variety")
    # ------------------------------------------------------------------

    # Show that augmentations change between epochs
    sample_0 = train_ds[len(fold["train_ids"])]  # First augmented sample
    feats_epoch0 = sample_0["features"].numpy().copy()

    train_ds.set_epoch(1)
    sample_1 = train_ds[len(fold["train_ids"])]
    feats_epoch1 = sample_1["features"].numpy()

    diff = np.abs(feats_epoch0 - feats_epoch1)
    print(f"Same augmented index, different epochs:")
    print(f"  Mean absolute difference: {np.mean(diff):.6f}")
    print(f"  (Should be > 0, confirming epoch variety)")

    # ------------------------------------------------------------------
    section("7. Leakage check")
    # ------------------------------------------------------------------

    # Verify augmented data doesn't leak across folds
    print("Checking for data leakage across folds...")
    for i, fold in enumerate(kfold):
        train_set = set(fold["train_ids"])
        val_set = set(fold["val_ids"])
        overlap = train_set & val_set
        assert len(overlap) == 0, f"Fold {i}: leakage detected! {overlap}"
    print("  No leakage detected across all 5 folds.")

    # Verify all videos appear in exactly one val fold
    all_val = []
    for fold in kfold:
        all_val.extend(fold["val_ids"])
    assert sorted(all_val) == sorted(video_ids), "Not all videos covered by folds!"
    print("  All videos appear exactly once in validation across folds.")

    print(f"\n{'='*60}")
    print("  Phase 4 pipeline demo complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()