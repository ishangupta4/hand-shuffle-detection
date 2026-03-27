"""Assemble all features into a single matrix per video.

Combines static and dynamic features, saves feature matrices and
a metadata JSON describing each column.
"""

import argparse
import json
import os
import sys

import numpy as np

from src.features.static_features import get_static_feature_names
from src.features.dynamic_features import get_dynamic_feature_names


def classify_feature(name: str) -> str:
    """Classify a feature as left_hand, right_hand, or inter_hand."""
    if name.startswith("left_"):
        return "left_hand"
    elif name.startswith("right_"):
        return "right_hand"
    else:
        return "inter_hand"


def build_feature_metadata(
    static_names: list[str],
    dynamic_names: list[str],
) -> dict:
    """Build metadata dict describing all feature columns."""
    all_names = static_names + dynamic_names
    features = {}
    for i, name in enumerate(all_names):
        features[name] = {
            "index": i,
            "category": classify_feature(name),
            "type": "static" if i < len(static_names) else "dynamic",
        }

    return {
        "num_features": len(all_names),
        "num_static": len(static_names),
        "num_dynamic": len(dynamic_names),
        "feature_names": all_names,
        "features": features,
    }


def assemble_video_features(
    static_feats: np.ndarray,
    dynamic_feats: np.ndarray,
) -> np.ndarray:
    """Concatenate static and dynamic features.

    Both should have shape (T, *), and T must match.

    Returns:
        Shape (T, num_static + num_dynamic).
    """
    assert static_feats.shape[0] == dynamic_feats.shape[0], (
        f"Frame count mismatch: static={static_feats.shape[0]}, "
        f"dynamic={dynamic_feats.shape[0]}"
    )
    return np.column_stack([static_feats, dynamic_feats])


def main():
    parser = argparse.ArgumentParser(
        description="Assemble all features into per-video matrices."
    )
    parser.add_argument(
        "--static-dir", default="data/features_static",
        help="Directory with static feature .npy files.",
    )
    parser.add_argument(
        "--dynamic-dir", default="data/features_dynamic",
        help="Directory with dynamic feature .npy files.",
    )
    parser.add_argument(
        "--output-dir", default="data/features",
        help="Directory for assembled feature .npy files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    static_names = get_static_feature_names()
    dynamic_names = get_dynamic_feature_names()
    metadata = build_feature_metadata(static_names, dynamic_names)

    # Save metadata
    meta_path = os.path.join(args.output_dir, "feature_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Feature metadata: {metadata['num_features']} features "
          f"({metadata['num_static']} static + {metadata['num_dynamic']} dynamic)")
    print(f"Saved metadata to {meta_path}\n")

    static_files = sorted([
        f for f in os.listdir(args.static_dir) if f.endswith(".npy")
    ])

    if not static_files:
        print(f"No static feature files found in {args.static_dir}")
        sys.exit(1)

    for sf in static_files:
        video_name = sf.replace(".npy", "")
        static_path = os.path.join(args.static_dir, sf)
        dynamic_path = os.path.join(args.dynamic_dir, sf)

        if not os.path.exists(dynamic_path):
            print(f"  Skipping {video_name}: no dynamic features found")
            continue

        static_feats = np.load(static_path)
        dynamic_feats = np.load(dynamic_path)
        combined = assemble_video_features(static_feats, dynamic_feats)

        out_path = os.path.join(args.output_dir, sf)
        np.save(out_path, combined)
        print(f"  {video_name}: {combined.shape}")

    print(f"\nAssembled features saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
