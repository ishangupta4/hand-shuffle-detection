"""Visualize extracted keypoints for quality checking.

Overlays keypoints on original frames, plots keypoint trajectories,
and identifies videos with actual quality problems (detection flicker,
noisy keypoints) — NOT simply low detection coverage, which is normal
when hands leave the frame.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extraction.process_all_videos import (
    count_flicker_frames,
    compute_wrist_jitter,
)

# MediaPipe hand connections for drawing skeleton lines
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

# Colors: left hand = blue, right hand = red
HAND_COLORS = [(66, 133, 244), (234, 67, 53)]


def overlay_keypoints_on_frame(
    frame_bgr: np.ndarray,
    keypoints_frame: np.ndarray,
    detection_mask_frame: np.ndarray,
) -> np.ndarray:
    """Draw hand keypoints and skeleton on a single BGR frame.

    Args:
        frame_bgr: Original BGR frame.
        keypoints_frame: Shape (2, 21, 3) — normalized coords.
        detection_mask_frame: Shape (2,) — which hands are detected.

    Returns:
        Annotated BGR frame.
    """
    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    for hand_idx in range(2):
        if not detection_mask_frame[hand_idx]:
            continue

        color = HAND_COLORS[hand_idx]
        kp = keypoints_frame[hand_idx]  # (21, 3)

        px = (kp[:, 0] * w).astype(int)
        py = (kp[:, 1] * h).astype(int)

        for start, end in HAND_CONNECTIONS:
            pt1 = (int(px[start]), int(py[start]))
            pt2 = (int(px[end]), int(py[end]))
            cv2.line(annotated, pt1, pt2, color, 1, cv2.LINE_AA)

        for j in range(21):
            cv2.circle(annotated, (int(px[j]), int(py[j])), 3, color, -1)

    return annotated


def create_frame_grid(
    video_path: str,
    keypoints: np.ndarray,
    detection_mask: np.ndarray,
    output_path: str,
    num_samples: int = 8,
) -> None:
    """Create a grid of annotated frames sampled evenly from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return

    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if idx < len(keypoints):
            frame = overlay_keypoints_on_frame(
                frame, keypoints[idx], detection_mask[idx]
            )

        cv2.putText(
            frame, f"#{idx}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        frames.append(frame)

    cap.release()

    if not frames:
        return

    target_h = 240
    target_w = int(target_h * frames[0].shape[1] / frames[0].shape[0])
    resized = [cv2.resize(f, (target_w, target_h)) for f in frames]

    cols = (len(resized) + 1) // 2
    while len(resized) < cols * 2:
        resized.append(np.zeros_like(resized[0]))

    row1 = np.hstack(resized[:cols])
    row2 = np.hstack(resized[cols : cols * 2])
    grid = np.vstack([row1, row2])

    cv2.imwrite(output_path, grid)


def plot_trajectories(
    keypoints: np.ndarray,
    detection_mask: np.ndarray,
    video_name: str,
    output_path: str,
) -> None:
    """Plot x/y trajectories of wrist and index fingertip over time.

    Shades flicker regions (short gaps between detections) in red to
    distinguish them from normal off-frame gaps.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Keypoint Trajectories: {video_name}", fontsize=13)

    kp_indices = {"Wrist": 0, "Index Tip": 8}
    hand_labels = ["Left (subject)", "Right (subject)"]
    colors = ["tab:blue", "tab:red"]

    for col, (kp_name, kp_idx) in enumerate(kp_indices.items()):
        for hand_idx in range(2):
            x_vals = keypoints[:, hand_idx, kp_idx, 0].copy()
            x_vals[~detection_mask[:, hand_idx]] = np.nan
            axes[0, col].plot(x_vals, color=colors[hand_idx], alpha=0.7,
                              label=hand_labels[hand_idx], linewidth=0.8)
            axes[0, col].set_ylabel("x (normalized)")
            axes[0, col].set_title(f"{kp_name} — X")

            y_vals = keypoints[:, hand_idx, kp_idx, 1].copy()
            y_vals[~detection_mask[:, hand_idx]] = np.nan
            axes[1, col].plot(y_vals, color=colors[hand_idx], alpha=0.7,
                              label=hand_labels[hand_idx], linewidth=0.8)
            axes[1, col].set_ylabel("y (normalized)")
            axes[1, col].set_xlabel("Frame")
            axes[1, col].set_title(f"{kp_name} — Y")

    # Shade flicker gaps on all axes
    for hand_idx in range(2):
        mask_1d = detection_mask[:, hand_idx]
        flicker_mask = _build_flicker_mask(mask_1d, max_gap=5)
        if np.any(flicker_mask):
            for ax in axes.flat:
                _shade_mask(ax, flicker_mask, color="red", alpha=0.08)

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _build_flicker_mask(hand_mask: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Return a boolean mask that is True for frames in flicker gaps."""
    flicker = np.zeros(len(hand_mask), dtype=bool)
    i = 0
    n = len(hand_mask)
    while i < n:
        if not hand_mask[i]:
            gap_start = i
            while i < n and not hand_mask[i]:
                i += 1
            gap_len = i - gap_start
            if gap_len <= max_gap and gap_start > 0 and i < n:
                flicker[gap_start:i] = True
        else:
            i += 1
    return flicker


def _shade_mask(ax, mask: np.ndarray, color: str = "red", alpha: float = 0.1):
    """Add vertical shading over True regions in a boolean mask."""
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


def summarize_quality(keypoints_dir: str) -> list[dict]:
    """Summarize quality across all videos using flicker + jitter metrics."""
    summaries = []
    for fname in sorted(os.listdir(keypoints_dir)):
        if not fname.endswith("_mask.npy"):
            continue

        video_name = fname.replace("_mask.npy", "")
        mask = np.load(os.path.join(keypoints_dir, fname))
        kp_path = os.path.join(keypoints_dir, f"{video_name}.npy")
        if not os.path.exists(kp_path):
            continue

        keypoints = np.load(kp_path)
        total = len(mask)

        flicker_l = count_flicker_frames(mask[:, 0])
        flicker_r = count_flicker_frames(mask[:, 1])
        flicker_total = flicker_l + flicker_r
        jitter = compute_wrist_jitter(keypoints, mask)

        issues = []
        if flicker_total > 15:
            issues.append(f"flicker: {flicker_total} frames")
        if jitter > 0.05:
            issues.append(f"jitter: {jitter:.4f}")

        status = "OK" if not issues else "REVIEW"

        summaries.append({
            "video": video_name,
            "total_frames": total,
            "flicker_total": flicker_total,
            "wrist_jitter": jitter,
            "issues": issues,
            "status": status,
        })

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Visualize extracted keypoints for quality verification."
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing original video files.",
    )
    parser.add_argument(
        "--keypoints-dir",
        default="data/keypoints",
        help="Directory containing .npy keypoint files (default: data/keypoints).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visualizations",
        help="Directory for output visualizations (default: outputs/visualizations).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of sample videos to visualize in detail (default: 3).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Quality summary
    print("Quality summary (based on flicker + jitter, NOT detection coverage):")
    print("-" * 70)
    summaries = summarize_quality(args.keypoints_dir)

    if not summaries:
        print(f"No keypoint files found in {args.keypoints_dir}")
        sys.exit(1)

    flagged = []
    for s in summaries:
        issue_str = ", ".join(s["issues"]) if s["issues"] else "OK"
        marker = " <-- REVIEW" if s["status"] == "REVIEW" else ""
        print(
            f"  {s['video']}: flicker={s['flicker_total']}, "
            f"jitter={s['wrist_jitter']:.4f}  [{issue_str}]{marker}"
        )
        if s["status"] == "REVIEW":
            flagged.append(s["video"])

    if flagged:
        print(f"\nFlagged for review ({len(flagged)} videos): {', '.join(flagged)}")
        print("These have detection quality issues (not just hands off-frame).")
    else:
        print("\nAll videos have acceptable extraction quality.")

    # Detailed visualization for a subset
    all_videos = [s["video"] for s in summaries]

    # Prioritize flagged videos, then fill remaining slots
    sample_videos = flagged[:args.num_samples]
    remaining_slots = args.num_samples - len(sample_videos)
    good_videos = [v for v in all_videos if v not in flagged]
    if remaining_slots > 0 and good_videos:
        step = max(1, len(good_videos) // remaining_slots)
        sample_videos += good_videos[::step][:remaining_slots]

    print(f"\nGenerating detailed visualizations for: {sample_videos}\n")

    from src.extraction.process_all_videos import find_videos as find_video_files
    video_files = {Path(v).stem: v for v in find_video_files(args.video_dir)}

    for video_name in sample_videos:
        kp_file = os.path.join(args.keypoints_dir, f"{video_name}.npy")
        mask_file = os.path.join(args.keypoints_dir, f"{video_name}_mask.npy")

        if not os.path.exists(kp_file) or not os.path.exists(mask_file):
            print(f"  Skipping {video_name}: keypoint files not found")
            continue

        keypoints = np.load(kp_file)
        mask = np.load(mask_file)

        if video_name in video_files:
            grid_path = os.path.join(args.output_dir, f"{video_name}_grid.png")
            create_frame_grid(video_files[video_name], keypoints, mask, grid_path)
            print(f"  Saved frame grid: {grid_path}")

        traj_path = os.path.join(args.output_dir, f"{video_name}_trajectories.png")
        plot_trajectories(keypoints, mask, video_name, traj_path)
        print(f"  Saved trajectory plot: {traj_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()