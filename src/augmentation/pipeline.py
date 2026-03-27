"""Composable augmentation pipeline.

Applies augmentation techniques in combination to generate diverse
training samples from a small set of original videos. Augmentation
is only applied to training folds — never to validation data.

Target: 19 originals → ~400-600 augmented samples.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.augmentation.augmentations import (
    horizontal_flip,
    time_warp,
    gaussian_jitter,
    slight_rotation,
    feature_dropout,
)


@dataclass
class AugmentationConfig:
    """Controls which augmentations to apply and how many variants."""

    # Whether each augmentation is enabled
    enable_flip: bool = True
    enable_time_warp: bool = True
    enable_jitter: bool = True
    enable_rotation: bool = True
    enable_dropout: bool = True

    # Time warp speeds (1.0 = original, always included implicitly)
    warp_speeds: list[float] = field(default_factory=lambda: [0.8, 1.2])

    # Number of jitter variants per sample
    num_jitter_variants: int = 2
    jitter_sigma: float = 0.02

    # Number of rotation variants per sample
    num_rotation_variants: int = 2
    rotation_max_angle: float = 7.0

    # Number of dropout variants per sample
    num_dropout_variants: int = 2
    dropout_rate: float = 0.07

    # Path to feature metadata (needed for flip and rotation)
    metadata_path: str = "data/features/feature_metadata.json"


def augment_single(
    sequence: np.ndarray,
    label: dict,
    config: AugmentationConfig,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, dict, str]]:
    """Generate all augmented variants of a single sample.

    Returns list of (augmented_sequence, label, description) tuples.
    The original sample is always included.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = [(sequence.copy(), label.copy(), "original")]

    # Start with base variants: original + flip
    bases = [(sequence.copy(), label.copy(), "orig")]
    if config.enable_flip:
        flipped_seq, flipped_label = horizontal_flip(
            sequence, label, metadata_path=config.metadata_path
        )
        bases.append((flipped_seq, flipped_label, "flip"))

    # Apply time warping to each base
    warped_variants = []
    for base_seq, base_label, base_desc in bases:
        # Always include the unwarped version
        warped_variants.append((base_seq, base_label, base_desc))

        if config.enable_time_warp:
            for speed in config.warp_speeds:
                w_seq, w_label = time_warp(base_seq, base_label, speed_factor=speed)
                warped_variants.append((w_seq, w_label, f"{base_desc}_warp{speed}"))

    # Apply jitter to each warped variant
    jittered_variants = []
    for w_seq, w_label, w_desc in warped_variants:
        jittered_variants.append((w_seq, w_label, w_desc))

        if config.enable_jitter:
            for j in range(config.num_jitter_variants):
                j_seq, j_label = gaussian_jitter(
                    w_seq, w_label, sigma_fraction=config.jitter_sigma, rng=rng
                )
                jittered_variants.append((j_seq, j_label, f"{w_desc}_jit{j}"))

    # From the jittered variants, randomly apply rotation or dropout
    # (not composing these exhaustively to avoid combinatorial explosion)
    final = []
    for j_seq, j_label, j_desc in jittered_variants:
        final.append((j_seq, j_label, j_desc))

    # Add rotation variants from a subset of bases
    if config.enable_rotation:
        for base_seq, base_label, base_desc in bases:
            for r in range(config.num_rotation_variants):
                r_seq, r_label = slight_rotation(
                    base_seq, base_label,
                    max_angle_deg=config.rotation_max_angle,
                    metadata_path=config.metadata_path, rng=rng,
                )
                final.append((r_seq, r_label, f"{base_desc}_rot{r}"))

    # Add dropout variants from a subset of bases
    if config.enable_dropout:
        for base_seq, base_label, base_desc in bases:
            for d in range(config.num_dropout_variants):
                d_seq, d_label = feature_dropout(
                    base_seq, base_label, drop_rate=config.dropout_rate, rng=rng
                )
                final.append((d_seq, d_label, f"{base_desc}_drop{d}"))

    return final


def augment_dataset(
    sequences: list[np.ndarray],
    labels: list[dict],
    video_ids: list[str],
    config: AugmentationConfig | None = None,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[dict], list[str]]:
    """Augment an entire dataset (typically just the training fold).

    Returns:
        aug_sequences: All augmented sequences.
        aug_labels: Corresponding labels.
        aug_source_ids: Source video ID for each augmented sample
                        (for tracking lineage / preventing leakage).
    """
    if config is None:
        config = AugmentationConfig()

    rng = np.random.default_rng(seed)

    aug_sequences = []
    aug_labels = []
    aug_source_ids = []

    for seq, label, vid_id in zip(sequences, labels, video_ids):
        variants = augment_single(seq, label, config, rng=rng)
        for v_seq, v_label, v_desc in variants:
            aug_sequences.append(v_seq)
            aug_labels.append(v_label)
            aug_source_ids.append(vid_id)

    return aug_sequences, aug_labels, aug_source_ids


def estimate_augmentation_factor(config: AugmentationConfig | None = None) -> int:
    """Estimate the multiplication factor for a given config."""
    if config is None:
        config = AugmentationConfig()

    # bases: 1 (orig) + 1 (flip) = 2
    n_bases = 2 if config.enable_flip else 1

    # warps per base: 1 (unwarped) + len(warp_speeds)
    n_warps = 1 + (len(config.warp_speeds) if config.enable_time_warp else 0)

    # jitter per warped: 1 (unjittered) + num_jitter_variants
    n_jitter = 1 + (config.num_jitter_variants if config.enable_jitter else 0)

    # Main compositional branch
    n_main = n_bases * n_warps * n_jitter

    # Additional rotation and dropout variants (from bases only)
    n_rotation = n_bases * config.num_rotation_variants if config.enable_rotation else 0
    n_dropout = n_bases * config.num_dropout_variants if config.enable_dropout else 0

    return n_main + n_rotation + n_dropout


if __name__ == "__main__":
    config = AugmentationConfig()
    factor = estimate_augmentation_factor(config)
    print(f"Default config augmentation factor: {factor}x")
    print(f"19 videos -> ~{19 * factor} samples")

    # Test with reduced config
    small_config = AugmentationConfig(
        enable_rotation=False,
        enable_dropout=False,
        num_jitter_variants=1,
    )
    small_factor = estimate_augmentation_factor(small_config)
    print(f"\nReduced config factor: {small_factor}x")
    print(f"19 videos -> ~{19 * small_factor} samples")

    # Quick test with synthetic data
    rng = np.random.default_rng(0)
    fake_seqs = [rng.standard_normal((40 + i * 2, 39)).astype(np.float32) for i in range(3)]
    fake_labels = [
        {"start_hand": "right", "end_hand": "left"},
        {"start_hand": "left", "end_hand": "right"},
        {"start_hand": "right", "end_hand": "right"},
    ]
    fake_ids = ["00001", "00002", "00003"]

    # Need a metadata file for flip/rotation — create a temp one
    import tempfile, os
    meta = {"num_features": 39, "num_static": 23, "num_dynamic": 16,
            "feature_names": [f"f{i}" for i in range(39)], "features": {}}
    for i in range(5):
        meta["features"][f"left_curl_{i}"] = {"index": i, "category": "left_hand", "type": "static"}
        meta["features"][f"right_curl_{i}"] = {"index": i + 5, "category": "right_hand", "type": "static"}
    meta["features"]["asymmetry_test"] = {"index": 17, "category": "inter_hand", "type": "static"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(meta, f)
        tmp_path = f.name

    test_config = AugmentationConfig(metadata_path=tmp_path)
    aug_seqs, aug_labs, aug_ids = augment_dataset(
        fake_seqs, fake_labels, fake_ids, config=test_config
    )
    os.unlink(tmp_path)

    print(f"\n3 samples -> {len(aug_seqs)} augmented samples ({len(aug_seqs)/3:.1f}x)")
    print(f"Source IDs distribution: {dict(zip(*np.unique(aug_ids, return_counts=True)))}")