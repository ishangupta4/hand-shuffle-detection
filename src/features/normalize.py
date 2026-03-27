"""Normalize hand keypoints: center on wrist, scale by hand size.

For each frame and each hand:
    1. Subtract wrist (keypoint 0) so the wrist sits at origin.
    2. Divide by distance from wrist to middle finger MCP (keypoint 9)
       to make features scale-invariant.

NaN frames (hand not detected) pass through unchanged.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def normalize_hand(hand_kp: np.ndarray) -> np.ndarray:
    """Normalize a single hand's keypoints for one frame.

    Args:
        hand_kp: Shape (21, 3). May contain NaN if hand wasn't detected.

    Returns:
        Normalized keypoints, same shape. NaN if input is NaN.
    """
    if np.any(np.isnan(hand_kp)):
        return hand_kp.copy()

    wrist = hand_kp[0]
    centered = hand_kp - wrist

    # Scale by wrist-to-middle-finger-MCP distance
    scale = np.linalg.norm(centered[9])
    if scale < 1e-8:
        return centered

    return centered / scale


def normalize_video(keypoints: np.ndarray) -> np.ndarray:
    """Normalize all frames of a video.

    Args:
        keypoints: Shape (T, 2, 21, 3).

    Returns:
        Normalized keypoints, same shape.
    """
    T = keypoints.shape[0]
    result = np.empty_like(keypoints)

    for t in range(T):
        for h in range(2):
            result[t, h] = normalize_hand(keypoints[t, h])

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Normalize cleaned keypoints (center + scale)."
    )
    parser.add_argument(
        "--input-dir", default="data/keypoints_cleaned",
        help="Directory with cleaned .npy files.",
    )
    parser.add_argument(
        "--output-dir", default="data/keypoints_normalized",
        help="Directory for normalized .npy files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    kp_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith(".npy")
        and not f.endswith("_mask.npy")
        and not f.endswith("_meta.npy")
    ])

    if not kp_files:
        print(f"No .npy files found in {args.input_dir}")
        sys.exit(1)

    print(f"Normalizing {len(kp_files)} videos...\n")

    for kp_file in kp_files:
        video_name = kp_file.replace(".npy", "")
        keypoints = np.load(os.path.join(args.input_dir, kp_file))
        normalized = normalize_video(keypoints)

        out_path = os.path.join(args.output_dir, kp_file)
        np.save(out_path, normalized)
        print(f"  {video_name}: {keypoints.shape[0]} frames normalized")

    print(f"\nNormalized keypoints saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
