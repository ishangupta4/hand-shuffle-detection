"""Static per-frame features extracted from hand keypoints.

Features computed per frame:
    - Finger curl angles (5 per hand, 10 total)
    - Fist compactness (1 per hand, 2 total)
    - Hand bounding box volume (1 per hand, 2 total)
    - Fingertip spread (1 per hand, 2 total)
    - Inter-hand distance (1 total)
    - Left-right asymmetry (compactness diff + curl diffs = 6 total)

All features return NaN for frames where the relevant hand wasn't detected.
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np


# MediaPipe keypoint indices
WRIST = 0
FINGERTIPS = [4, 8, 12, 16, 20]
PALM_LANDMARKS = [0, 5, 9, 13, 17]

# MCP, PIP, TIP triplets for curl angle computation
# Thumb uses CMC→MCP→TIP (indices 1→2→4) since it has different anatomy
FINGER_JOINTS = [
    (1, 2, 4),    # Thumb: CMC → MCP → TIP
    (5, 6, 8),    # Index: MCP → PIP → TIP
    (9, 10, 12),  # Middle: MCP → PIP → TIP
    (13, 14, 16), # Ring: MCP → PIP → TIP
    (17, 18, 20), # Pinky: MCP → PIP → TIP
]

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
HAND_NAMES = ["left", "right"]


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between two 3D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.arccos(cos_angle)


def finger_curl_angles(hand_kp: np.ndarray) -> np.ndarray:
    """Compute curl angle for each finger.

    The curl angle is the angle at the PIP joint between the
    MCP→PIP and PIP→TIP segments. A straight finger ≈ π, a
    curled finger ≈ small angle.

    Args:
        hand_kp: Shape (21, 3).

    Returns:
        Array of 5 angles in radians, or NaN if hand not detected.
    """
    if np.any(np.isnan(hand_kp)):
        return np.full(5, np.nan)

    angles = np.empty(5)
    for i, (mcp, pip, tip) in enumerate(FINGER_JOINTS):
        v1 = hand_kp[mcp] - hand_kp[pip]  # PIP → MCP direction
        v2 = hand_kp[tip] - hand_kp[pip]  # PIP → TIP direction
        angles[i] = _angle_between(v1, v2)

    return angles


def fist_compactness(hand_kp: np.ndarray) -> float:
    """Average distance from each fingertip to the palm center.

    Palm center = mean of wrist + 4 finger MCP joints.
    Lower values → tighter fist.
    """
    if np.any(np.isnan(hand_kp)):
        return np.nan

    palm_center = hand_kp[PALM_LANDMARKS].mean(axis=0)
    tips = hand_kp[FINGERTIPS]
    dists = np.linalg.norm(tips - palm_center, axis=1)
    return dists.mean()


def bounding_box_volume(hand_kp: np.ndarray) -> float:
    """Product of x, y, z coordinate ranges across all 21 keypoints."""
    if np.any(np.isnan(hand_kp)):
        return np.nan

    ranges = hand_kp.max(axis=0) - hand_kp.min(axis=0)
    return float(np.prod(ranges))


def fingertip_spread(hand_kp: np.ndarray) -> float:
    """Average pairwise distance between the 5 fingertips."""
    if np.any(np.isnan(hand_kp)):
        return np.nan

    tips = hand_kp[FINGERTIPS]
    dists = []
    for i, j in combinations(range(5), 2):
        dists.append(np.linalg.norm(tips[i] - tips[j]))
    return float(np.mean(dists))


def inter_hand_distance(left_kp: np.ndarray, right_kp: np.ndarray) -> float:
    """Euclidean distance between the two wrist keypoints."""
    if np.any(np.isnan(left_kp)) or np.any(np.isnan(right_kp)):
        return np.nan
    return float(np.linalg.norm(left_kp[WRIST] - right_kp[WRIST]))


def compute_static_features_frame(
    left_kp: np.ndarray,
    right_kp: np.ndarray,
) -> np.ndarray:
    """Compute all static features for a single frame.

    Returns:
        1D array of features (see get_static_feature_names for ordering).
    """
    features = []

    # Curl angles: 5 left + 5 right
    left_curls = finger_curl_angles(left_kp)
    right_curls = finger_curl_angles(right_kp)
    features.extend(left_curls)
    features.extend(right_curls)

    # Compactness: 1 left + 1 right
    left_compact = fist_compactness(left_kp)
    right_compact = fist_compactness(right_kp)
    features.append(left_compact)
    features.append(right_compact)

    # Bounding box volume: 1 left + 1 right
    features.append(bounding_box_volume(left_kp))
    features.append(bounding_box_volume(right_kp))

    # Fingertip spread: 1 left + 1 right
    features.append(fingertip_spread(left_kp))
    features.append(fingertip_spread(right_kp))

    # Inter-hand distance
    features.append(inter_hand_distance(left_kp, right_kp))

    # Asymmetry features
    features.append(abs(left_compact - right_compact))  # compactness diff
    for i in range(5):
        features.append(abs(left_curls[i] - right_curls[i]))  # curl diffs

    return np.array(features, dtype=np.float64)


def compute_static_features_video(keypoints: np.ndarray) -> np.ndarray:
    """Compute static features for all frames.

    Args:
        keypoints: Shape (T, 2, 21, 3).

    Returns:
        Shape (T, num_static_features).
    """
    T = keypoints.shape[0]
    sample = compute_static_features_frame(keypoints[0, 0], keypoints[0, 1])
    num_features = len(sample)

    result = np.empty((T, num_features), dtype=np.float64)
    result[0] = sample

    for t in range(1, T):
        result[t] = compute_static_features_frame(keypoints[t, 0], keypoints[t, 1])

    return result


def get_static_feature_names() -> list[str]:
    """Ordered list of static feature names matching column indices."""
    names = []

    # Curl angles
    for hand in HAND_NAMES:
        for finger in FINGER_NAMES:
            names.append(f"{hand}_curl_{finger}")

    # Compactness
    for hand in HAND_NAMES:
        names.append(f"{hand}_compactness")

    # Bounding box volume
    for hand in HAND_NAMES:
        names.append(f"{hand}_bbox_volume")

    # Fingertip spread
    for hand in HAND_NAMES:
        names.append(f"{hand}_fingertip_spread")

    # Inter-hand distance
    names.append("inter_hand_distance")

    # Asymmetry
    names.append("asymmetry_compactness")
    for finger in FINGER_NAMES:
        names.append(f"asymmetry_curl_{finger}")

    return names


def main():
    parser = argparse.ArgumentParser(
        description="Compute static per-frame features from keypoints."
    )
    parser.add_argument(
        "--input-dir", default="data/keypoints_cleaned",
        help="Directory with .npy keypoint files.",
    )
    parser.add_argument(
        "--output-dir", default="data/features_static",
        help="Directory for static feature .npy files.",
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

    names = get_static_feature_names()
    print(f"Computing {len(names)} static features per frame...\n")

    for kp_file in kp_files:
        video_name = kp_file.replace(".npy", "")
        keypoints = np.load(os.path.join(args.input_dir, kp_file))
        features = compute_static_features_video(keypoints)

        out_path = os.path.join(args.output_dir, kp_file)
        np.save(out_path, features)
        print(f"  {video_name}: {features.shape}")

    print(f"\nStatic features saved to {args.output_dir}/")
    print(f"Feature names: {names}")


if __name__ == "__main__":
    main()
