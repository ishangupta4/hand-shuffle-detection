"""Clean and interpolate raw MediaPipe keypoint data.

Cleaning philosophy:
    Hands leaving the frame is NORMAL — those NaN gaps are left untouched.
    We only clean two types of problems:

    1. Detection flicker: short gaps (configurable, default 1-5 frames)
       where a hand was detected before and after, meaning MediaPipe
       briefly lost tracking on a hand that was still visible. These
       get interpolated.

    2. Noisy/jumping keypoints: when a hand IS detected but the
       coordinates are jittery or jump suddenly (e.g. MediaPipe swaps
       left/right labels for a frame). These get smoothed or replaced.

    Long gaps where no hand is detected are assumed to be the hands
    genuinely off-frame and are preserved as NaN.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# --- Flicker detection ---

def classify_gaps(
    hand_mask: np.ndarray,
    flicker_max_gap: int = 5,
) -> np.ndarray:
    """Classify each NaN frame as 'flicker' (should interpolate) or 'off-frame' (leave alone).

    A gap is flicker if:
    - It is <= flicker_max_gap frames long
    - There is a detection on BOTH sides (hand was in frame before and after)

    Args:
        hand_mask: 1D boolean array — True where hand is detected.
        flicker_max_gap: Max gap length to count as flicker.

    Returns:
        Boolean mask (same length) — True for frames that are flicker gaps.
    """
    flicker = np.zeros(len(hand_mask), dtype=bool)
    i = 0
    n = len(hand_mask)
    while i < n:
        if not hand_mask[i]:
            gap_start = i
            while i < n and not hand_mask[i]:
                i += 1
            gap_len = i - gap_start
            if gap_len <= flicker_max_gap and gap_start > 0 and i < n:
                flicker[gap_start:i] = True
        else:
            i += 1
    return flicker


# --- Interpolation (only for flicker gaps) ---

def interpolate_flicker(
    signal: np.ndarray,
    flicker_mask: np.ndarray,
    use_spline: bool = True,
) -> tuple[np.ndarray, int]:
    """Interpolate only the frames marked as flicker.

    Frames that are NaN but NOT flicker (off-frame) are left as NaN.

    Args:
        signal: 1D array, may contain NaN.
        flicker_mask: Boolean mask — True for flicker frames to interpolate.
        use_spline: Use cubic spline (True) or linear (False).

    Returns:
        Tuple of (interpolated signal, count of frames interpolated).
    """
    result = signal.copy()
    flicker_indices = np.where(flicker_mask & np.isnan(result))[0]

    if len(flicker_indices) == 0:
        return result, 0

    valid_mask = ~np.isnan(result)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        return result, 0

    count = 0

    if use_spline and len(valid_indices) >= 4:
        try:
            cs = CubicSpline(valid_indices, result[valid_indices])
            for idx in flicker_indices:
                result[idx] = cs(idx)
                count += 1
            return result, count
        except ValueError:
            pass  # fall through to linear

    # Linear fallback: interpolate each flicker frame from nearest neighbors
    for idx in flicker_indices:
        left = valid_indices[valid_indices < idx]
        right = valid_indices[valid_indices > idx]

        if len(left) > 0 and len(right) > 0:
            li, ri = left[-1], right[0]
            t = (idx - li) / (ri - li)
            result[idx] = result[li] + t * (result[ri] - result[li])
            count += 1

    return result, count


# --- Outlier/jump detection ---

def detect_jumps(
    signal: np.ndarray,
    detection_mask: np.ndarray,
    threshold_factor: float = 3.0,
) -> np.ndarray:
    """Detect sudden jumps in keypoint position between consecutive detected frames.

    Only flags jumps in frames where the hand IS detected — these are the
    real quality issues (e.g. hand label swaps, noise spikes).

    Returns:
        Boolean mask — True for frames flagged as outlier jumps.
    """
    outliers = np.zeros(len(signal), dtype=bool)

    if np.sum(detection_mask) < 3:
        return outliers

    # Compute velocity only between consecutive detected frames
    velocity = np.full(len(signal), np.nan)
    for i in range(1, len(signal)):
        if detection_mask[i] and detection_mask[i - 1]:
            if not np.isnan(signal[i]) and not np.isnan(signal[i - 1]):
                velocity[i] = abs(signal[i] - signal[i - 1])

    valid_vel = velocity[~np.isnan(velocity)]
    if len(valid_vel) == 0:
        return outliers

    median_vel = np.median(valid_vel)
    threshold = max(threshold_factor * median_vel, 0.01)

    for i in range(1, len(signal)):
        if not np.isnan(velocity[i]) and velocity[i] > threshold:
            outliers[i] = True

    return outliers


def fix_outliers(
    signal: np.ndarray,
    outliers: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Replace outlier frames with linearly interpolated values."""
    result = signal.copy()
    count = 0

    outlier_indices = np.where(outliers)[0]
    if len(outlier_indices) == 0:
        return result, 0

    valid = ~np.isnan(result) & ~outliers
    valid_indices = np.where(valid)[0]

    if len(valid_indices) < 2:
        return result, 0

    for idx in outlier_indices:
        left = valid_indices[valid_indices < idx]
        right = valid_indices[valid_indices > idx]

        if len(left) > 0 and len(right) > 0:
            li, ri = left[-1], right[0]
            t = (idx - li) / (ri - li)
            result[idx] = result[li] + t * (result[ri] - result[li])
            count += 1
        elif len(left) > 0:
            result[idx] = result[left[-1]]
            count += 1
        elif len(right) > 0:
            result[idx] = result[right[0]]
            count += 1

    return result, count


# --- Temporal smoothing (only on detected segments) ---

def smooth_signal(
    signal: np.ndarray,
    detection_mask: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing only to contiguous detected segments.

    Off-frame NaN regions are untouched.
    """
    result = signal.copy()

    # Build a mask of "smoothable" frames: detected AND not NaN
    smoothable = detection_mask & ~np.isnan(result)

    # Find contiguous smoothable segments
    segments = []
    i = 0
    while i < len(smoothable):
        if smoothable[i]:
            start = i
            while i < len(smoothable) and smoothable[i]:
                i += 1
            segments.append((start, i))
        else:
            i += 1

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len < window_length:
            continue

        wl = min(window_length, seg_len)
        if wl % 2 == 0:
            wl -= 1
        if wl < 3:
            continue

        po = min(polyorder, wl - 1)
        result[seg_start:seg_end] = savgol_filter(
            result[seg_start:seg_end], wl, po
        )

    return result


# --- Full pipeline ---

def clean_video_keypoints(
    keypoints: np.ndarray,
    detection_mask: np.ndarray,
    flicker_max_gap: int = 5,
    jump_threshold: float = 3.0,
    smooth_window: int = 7,
    smooth_polyorder: int = 2,
) -> tuple[np.ndarray, dict]:
    """Run the full cleaning pipeline on one video's keypoints.

    Only interpolates flicker gaps (short dropouts between detections).
    Long gaps (hands off-frame) are left as NaN.

    Args:
        keypoints: Shape (T, 2, 21, 3).
        detection_mask: Shape (T, 2).
        flicker_max_gap: Max gap to classify as flicker vs off-frame.
        jump_threshold: Velocity multiplier for outlier detection.
        smooth_window: Savitzky-Golay window length.
        smooth_polyorder: Savitzky-Golay polynomial order.

    Returns:
        Tuple of (cleaned keypoints, stats dict).
    """
    T, num_hands, num_kp, num_coords = keypoints.shape
    cleaned = keypoints.copy()

    total_flicker_filled = 0
    total_outliers_fixed = 0

    for hand in range(num_hands):
        hand_mask = detection_mask[:, hand]
        flicker_mask = classify_gaps(hand_mask, flicker_max_gap)
        flicker_frames_this_hand = int(np.sum(flicker_mask))

        for kp in range(num_kp):
            for coord in range(num_coords):
                signal = cleaned[:, hand, kp, coord].copy()

                # Ensure undetected frames are NaN
                signal[~hand_mask] = np.nan

                # Step 1: Interpolate ONLY flicker gaps
                if flicker_frames_this_hand > 0:
                    signal, n_filled = interpolate_flicker(signal, flicker_mask)
                    if kp == 0 and coord == 0:
                        # Count once per hand, not per keypoint*coord
                        total_flicker_filled += n_filled

                # Step 2: Detect and fix outlier jumps in detected frames
                outliers = detect_jumps(signal, hand_mask, jump_threshold)
                signal, n_fixed = fix_outliers(signal, outliers)
                if kp == 0 and coord == 0:
                    total_outliers_fixed += n_fixed

                # Step 3: Smooth only detected segments
                signal = smooth_signal(signal, hand_mask, smooth_window, smooth_polyorder)

                cleaned[:, hand, kp, coord] = signal

    stats = {
        "total_frames": T,
        "flicker_frames_filled": total_flicker_filled,
        "outlier_frames_fixed": total_outliers_fixed,
    }

    return cleaned, stats


# --- Visualization ---

def plot_raw_vs_cleaned(
    raw_kp: np.ndarray,
    cleaned_kp: np.ndarray,
    raw_mask: np.ndarray,
    video_name: str,
    output_path: str,
) -> None:
    """Plot raw vs cleaned trajectories side by side for wrist and index tip."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Raw vs Cleaned: {video_name}", fontsize=13)

    kp_targets = {"Wrist (x)": (0, 0), "Index Tip (x)": (8, 0)}

    for col, (label, (kp_idx, coord_idx)) in enumerate(kp_targets.items()):
        for hand_idx, (hand_name, color) in enumerate(
            [("Left", "tab:blue"), ("Right", "tab:red")]
        ):
            raw = raw_kp[:, hand_idx, kp_idx, coord_idx].copy()
            raw[~raw_mask[:, hand_idx]] = np.nan

            axes[0, col].plot(raw, color=color, alpha=0.6, linewidth=0.8,
                              label=f"{hand_name} (raw)")
            axes[0, col].set_title(f"{label} -- Raw")
            axes[0, col].set_ylabel("Coordinate")

            cln = cleaned_kp[:, hand_idx, kp_idx, coord_idx]
            axes[1, col].plot(cln, color=color, alpha=0.8, linewidth=0.8,
                              label=f"{hand_name} (cleaned)")
            axes[1, col].set_title(f"{label} -- Cleaned")
            axes[1, col].set_ylabel("Coordinate")
            axes[1, col].set_xlabel("Frame")

    # Shade flicker regions on raw plots
    for hand_idx in range(2):
        flicker = classify_gaps(raw_mask[:, hand_idx], flicker_max_gap=5)
        if np.any(flicker):
            for ax in axes[0, :]:
                _shade_flicker(ax, flicker)

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _shade_flicker(ax, mask: np.ndarray, color: str = "red", alpha: float = 0.1):
    """Shade flicker regions on a matplotlib axis."""
    in_region = False
    start = 0
    for i in range(len(mask)):
        if mask[i] and not in_region:
            start = i
            in_region = True
        elif not mask[i] and in_region:
            ax.axvspan(start, i, color=color, alpha=alpha)
            in_region = False
    if in_region:
        ax.axvspan(start, len(mask), color=color, alpha=alpha)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Clean and interpolate raw MediaPipe keypoint data."
    )
    parser.add_argument(
        "--keypoints-dir",
        default="data/keypoints",
        help="Directory with raw .npy keypoint files (default: data/keypoints).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/keypoints_cleaned",
        help="Directory for cleaned .npy files (default: data/keypoints_cleaned).",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs",
        help="Directory for cleaning report (default: outputs).",
    )
    parser.add_argument(
        "--viz-dir",
        default="outputs/cleaning_viz",
        help="Directory for raw vs cleaned plots (default: outputs/cleaning_viz).",
    )
    parser.add_argument(
        "--flicker-max-gap",
        type=int,
        default=5,
        help="Max gap (frames) to classify as flicker vs off-frame (default: 5).",
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=3.0,
        help="Velocity multiplier for outlier detection (default: 3.0).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Savitzky-Golay filter window length (default: 7).",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order (default: 2).",
    )
    parser.add_argument(
        "--viz-samples",
        type=int,
        default=3,
        help="Number of videos to generate comparison plots for (default: 3).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)

    kp_files = sorted([
        f for f in os.listdir(args.keypoints_dir)
        if f.endswith(".npy")
        and not f.endswith("_mask.npy")
        and not f.endswith("_meta.npy")
    ])

    if not kp_files:
        print(f"No keypoint .npy files found in {args.keypoints_dir}")
        sys.exit(1)

    print(f"Found {len(kp_files)} videos to clean.\n")

    report_rows = []
    viz_candidates = []

    for kp_file in kp_files:
        video_name = kp_file.replace(".npy", "")
        mask_file = f"{video_name}_mask.npy"

        kp_path = os.path.join(args.keypoints_dir, kp_file)
        mask_path = os.path.join(args.keypoints_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"  Skipping {video_name}: no mask file found")
            continue

        keypoints = np.load(kp_path)
        detection_mask = np.load(mask_path)

        print(f"Cleaning: {video_name} ({keypoints.shape[0]} frames)...")

        cleaned, stats = clean_video_keypoints(
            keypoints,
            detection_mask,
            flicker_max_gap=args.flicker_max_gap,
            jump_threshold=args.jump_threshold,
            smooth_window=args.smooth_window,
            smooth_polyorder=args.smooth_polyorder,
        )

        # Save cleaned keypoints
        out_path = os.path.join(args.output_dir, kp_file)
        np.save(out_path, cleaned)

        # Copy meta file if it exists
        meta_src = os.path.join(args.keypoints_dir, f"{video_name}_meta.npy")
        if os.path.exists(meta_src):
            shutil.copy2(meta_src, os.path.join(args.output_dir, f"{video_name}_meta.npy"))

        # Also copy the mask (unchanged — off-frame gaps stay as-is)
        shutil.copy2(mask_path, os.path.join(args.output_dir, mask_file))

        stats["video"] = video_name
        report_rows.append(stats)
        viz_candidates.append((video_name, keypoints, cleaned, detection_mask, stats))

        print(
            f"  Flicker frames filled: {stats['flicker_frames_filled']}, "
            f"outlier frames fixed: {stats['outlier_frames_fixed']}"
        )

    if not report_rows:
        print("No videos processed.")
        sys.exit(1)

    # Save cleaning report
    report_df = pd.DataFrame(report_rows)
    cols = [
        "video", "total_frames",
        "flicker_frames_filled", "outlier_frames_fixed",
    ]
    report_df = report_df[cols]

    csv_path = os.path.join(args.report_dir, "cleaning_report.csv")
    report_df.to_csv(csv_path, index=False)

    # Markdown report
    md_lines = [
        "# Keypoint Cleaning Report\n",
        f"**Videos cleaned:** {len(report_df)}\n",
        "## Cleaning Philosophy\n",
        "Only two types of issues are corrected:\n",
        "1. **Flicker gaps** -- short dropouts (1-{} frames) where a hand was ".format(args.flicker_max_gap),
        "   detected before and after, indicating MediaPipe briefly lost a visible hand.",
        "   These are interpolated (cubic spline or linear).",
        "2. **Outlier jumps** -- sudden coordinate spikes in detected frames (e.g.",
        "   hand label swaps). These are replaced with interpolated values.",
        "",
        "Long gaps where no hand is detected are assumed to be hands leaving the",
        "frame and are **left as NaN** -- this is normal behavior in shuffle videos.\n",
        "All detected segments also receive light Savitzky-Golay smoothing to reduce jitter.\n",
        "## Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Avg flicker frames filled | {report_df['flicker_frames_filled'].mean():.1f} |",
        f"| Avg outlier frames fixed | {report_df['outlier_frames_fixed'].mean():.1f} |",
        "",
        "## Per-Video Details\n",
        "| Video | Frames | Flicker Filled | Outliers Fixed |",
        "|-------|--------|---------------|----------------|",
    ]
    for _, row in report_df.iterrows():
        md_lines.append(
            f"| {row['video']} | {row['total_frames']} | "
            f"{row['flicker_frames_filled']} | {row['outlier_frames_fixed']} |"
        )

    md_lines.append("\n## Parameters Used\n")
    md_lines.append(f"- Flicker max gap: {args.flicker_max_gap} frames")
    md_lines.append(f"- Jump threshold: {args.jump_threshold}x median velocity")
    md_lines.append(f"- Smoothing window: {args.smooth_window}")
    md_lines.append(f"- Smoothing polyorder: {args.smooth_polyorder}")

    md_path = os.path.join(args.report_dir, "cleaning_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # Generate comparison visualizations
    # Prioritize videos that had the most cleaning work done
    viz_candidates.sort(
        key=lambda x: x[4]["flicker_frames_filled"] + x[4]["outlier_frames_fixed"],
        reverse=True,
    )
    num_viz = min(args.viz_samples, len(viz_candidates))

    for video_name, raw_kp, cleaned_kp, mask, _ in viz_candidates[:num_viz]:
        viz_path = os.path.join(args.viz_dir, f"{video_name}_raw_vs_cleaned.png")
        plot_raw_vs_cleaned(raw_kp, cleaned_kp, mask, video_name, viz_path)
        print(f"\nSaved comparison plot: {viz_path}")

    print(f"\nCleaning report saved to {csv_path} and {md_path}")
    print(f"Cleaned keypoints saved to {args.output_dir}/")


if __name__ == "__main__":
    main()