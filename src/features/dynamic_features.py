"""Dynamic (temporal) features derived from static features.

Computes frame-to-frame derivatives with optional smoothing:
    - Wrist velocity and acceleration (2 + 2 = 4 features)
    - Finger curl velocity (10 features)
    - Compactness change rate (2 features)

Smoothing is applied before differentiation to reduce noise amplification.
"""

import argparse
import os
import sys

import numpy as np
from scipy.signal import savgol_filter


HAND_NAMES = ["left", "right"]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def smooth_column(signal: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smooth a 1D signal, handling NaN segments.

    Only smooths contiguous non-NaN segments that are long enough
    for the filter window.
    """
    result = signal.copy()
    valid = ~np.isnan(result)

    # Find contiguous valid segments
    segments = []
    i = 0
    n = len(valid)
    while i < n:
        if valid[i]:
            start = i
            while i < n and valid[i]:
                i += 1
            segments.append((start, i))
        else:
            i += 1

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len < window:
            continue
        wl = min(window, seg_len)
        if wl % 2 == 0:
            wl -= 1
        if wl < 3:
            continue
        po = min(polyorder, wl - 1)
        result[seg_start:seg_end] = savgol_filter(
            result[seg_start:seg_end], wl, po
        )

    return result


def frame_derivative(signal: np.ndarray) -> np.ndarray:
    """First-order finite difference, preserving NaN structure.

    Returns array of same length (first value is NaN).
    """
    deriv = np.full_like(signal, np.nan)
    for i in range(1, len(signal)):
        if not np.isnan(signal[i]) and not np.isnan(signal[i - 1]):
            deriv[i] = signal[i] - signal[i - 1]
    return deriv


def compute_wrist_velocity(keypoints: np.ndarray, smooth_window: int = 7) -> np.ndarray:
    """Frame-to-frame wrist displacement for each hand.

    Uses the raw (non-normalized) keypoints so velocity is in
    coordinate space.

    Args:
        keypoints: Shape (T, 2, 21, 3).
        smooth_window: Savitzky-Golay window for pre-smoothing.

    Returns:
        Shape (T, 2) — velocity per hand.
    """
    T = keypoints.shape[0]
    velocities = np.full((T, 2), np.nan)

    for h in range(2):
        wrist = keypoints[:, h, 0, :]  # (T, 3)

        # Smooth each coordinate before computing displacement
        smoothed = np.empty_like(wrist)
        for c in range(3):
            smoothed[:, c] = smooth_column(wrist[:, c], window=smooth_window)

        for t in range(1, T):
            if not np.any(np.isnan(smoothed[t])) and not np.any(np.isnan(smoothed[t - 1])):
                velocities[t, h] = np.linalg.norm(smoothed[t] - smoothed[t - 1])

    return velocities


def compute_wrist_acceleration(velocities: np.ndarray) -> np.ndarray:
    """Derivative of wrist velocity.

    Args:
        velocities: Shape (T, 2).

    Returns:
        Shape (T, 2).
    """
    T = velocities.shape[0]
    accel = np.full((T, 2), np.nan)
    for h in range(2):
        accel[:, h] = frame_derivative(velocities[:, h])
    return accel


def compute_curl_velocity(
    static_features: np.ndarray,
    curl_indices: list[int],
    smooth_window: int = 7,
) -> np.ndarray:
    """Rate of change of finger curl angles.

    Args:
        static_features: Shape (T, num_static_features).
        curl_indices: Column indices for the 10 curl angle features.
        smooth_window: Pre-smoothing window.

    Returns:
        Shape (T, 10).
    """
    T = static_features.shape[0]
    n_curls = len(curl_indices)
    result = np.full((T, n_curls), np.nan)

    for i, col in enumerate(curl_indices):
        smoothed = smooth_column(static_features[:, col], window=smooth_window)
        result[:, i] = frame_derivative(smoothed)

    return result


def compute_compactness_velocity(
    static_features: np.ndarray,
    compact_indices: list[int],
    smooth_window: int = 7,
) -> np.ndarray:
    """Rate of change of fist compactness.

    Args:
        static_features: Shape (T, num_static_features).
        compact_indices: Column indices for left/right compactness.
        smooth_window: Pre-smoothing window.

    Returns:
        Shape (T, 2).
    """
    T = static_features.shape[0]
    result = np.full((T, 2), np.nan)

    for i, col in enumerate(compact_indices):
        smoothed = smooth_column(static_features[:, col], window=smooth_window)
        result[:, i] = frame_derivative(smoothed)

    return result


def compute_dynamic_features(
    keypoints: np.ndarray,
    static_features: np.ndarray,
    curl_indices: list[int],
    compact_indices: list[int],
    smooth_window: int = 7,
) -> np.ndarray:
    """Compute all dynamic features for one video.

    Args:
        keypoints: Shape (T, 2, 21, 3) — cleaned (non-normalized) keypoints.
        static_features: Shape (T, num_static).
        curl_indices: Column indices for 10 curl features in static_features.
        compact_indices: Column indices for 2 compactness features.
        smooth_window: Pre-smoothing window.

    Returns:
        Shape (T, num_dynamic_features).
    """
    wrist_vel = compute_wrist_velocity(keypoints, smooth_window)
    wrist_acc = compute_wrist_acceleration(wrist_vel)
    curl_vel = compute_curl_velocity(static_features, curl_indices, smooth_window)
    compact_vel = compute_compactness_velocity(static_features, compact_indices, smooth_window)

    return np.column_stack([wrist_vel, wrist_acc, curl_vel, compact_vel])


def get_dynamic_feature_names() -> list[str]:
    """Ordered list of dynamic feature names."""
    names = []

    for hand in HAND_NAMES:
        names.append(f"{hand}_wrist_velocity")
    for hand in HAND_NAMES:
        names.append(f"{hand}_wrist_acceleration")
    for hand in HAND_NAMES:
        for finger in FINGER_NAMES:
            names.append(f"{hand}_curl_velocity_{finger}")
    for hand in HAND_NAMES:
        names.append(f"{hand}_compactness_velocity")

    return names


def main():
    parser = argparse.ArgumentParser(
        description="Compute dynamic (temporal) features from keypoints and static features."
    )
    parser.add_argument(
        "--keypoints-dir", default="data/keypoints_cleaned",
        help="Directory with cleaned .npy keypoint files.",
    )
    parser.add_argument(
        "--static-dir", default="data/features_static",
        help="Directory with static feature .npy files.",
    )
    parser.add_argument(
        "--output-dir", default="data/features_dynamic",
        help="Directory for dynamic feature .npy files.",
    )
    parser.add_argument(
        "--smooth-window", type=int, default=7,
        help="Savitzky-Golay smoothing window before differentiation.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Import static feature names to find curl/compactness column indices
    from src.features.static_features import get_static_feature_names
    static_names = get_static_feature_names()
    curl_indices = [i for i, n in enumerate(static_names) if "curl_" in n and "asymmetry" not in n and "velocity" not in n]
    compact_indices = [i for i, n in enumerate(static_names) if n in ("left_compactness", "right_compactness")]

    kp_files = sorted([
        f for f in os.listdir(args.keypoints_dir)
        if f.endswith(".npy")
        and not f.endswith("_mask.npy")
        and not f.endswith("_meta.npy")
    ])

    if not kp_files:
        print(f"No .npy files found in {args.keypoints_dir}")
        sys.exit(1)

    names = get_dynamic_feature_names()
    print(f"Computing {len(names)} dynamic features per frame...\n")

    for kp_file in kp_files:
        video_name = kp_file.replace(".npy", "")
        keypoints = np.load(os.path.join(args.keypoints_dir, kp_file))
        static_path = os.path.join(args.static_dir, kp_file)

        if not os.path.exists(static_path):
            print(f"  Skipping {video_name}: no static features found")
            continue

        static_feats = np.load(static_path)
        dynamic_feats = compute_dynamic_features(
            keypoints, static_feats,
            curl_indices, compact_indices,
            args.smooth_window,
        )

        out_path = os.path.join(args.output_dir, kp_file)
        np.save(out_path, dynamic_feats)
        print(f"  {video_name}: {dynamic_feats.shape}")

    print(f"\nDynamic features saved to {args.output_dir}/")
    print(f"Feature names: {names}")


if __name__ == "__main__":
    main()