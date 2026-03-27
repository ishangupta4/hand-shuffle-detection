"""Batch processing of all videos through the MediaPipe extraction pipeline.

Runs keypoint extraction on every video, saves .npy files, and generates
a detection quality report (CSV + markdown).

Quality model:
    Hands leaving the frame is NORMAL in shuffle videos — it does not
    indicate bad quality. The report distinguishes between:
    - Detection coverage: how many frames have 0/1/2 hands detected
      (informational, not a quality flag)
    - Detection flicker: short dropout gaps (1-5 frames) where a hand
      was detected before and after, suggesting MediaPipe briefly lost
      tracking on a visible hand
    - Keypoint jitter: high frame-to-frame variance in wrist position
      when a hand IS detected, suggesting noisy/unreliable coordinates
    These last two are the actual quality signals.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extraction.frame_extractor import extract_frames
from src.extraction.keypoint_extractor import extract_keypoints

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def find_videos(directory: str) -> list[str]:
    """Find all video files in a directory, sorted by name."""
    videos = []
    for entry in sorted(os.listdir(directory)):
        if Path(entry).suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(os.path.join(directory, entry))
    return videos


def count_flicker_frames(hand_mask: np.ndarray, max_gap: int = 5) -> int:
    """Count frames that look like detection flicker, not hands leaving frame.

    A flicker gap is a short run of False (hand not detected) that is
    bordered on BOTH sides by True (hand detected). This means the hand
    was likely still in frame but MediaPipe briefly lost it.

    Args:
        hand_mask: 1D boolean array for one hand across all frames.
        max_gap: Maximum gap length to count as flicker.

    Returns:
        Total number of frames classified as flicker dropouts.
    """
    flicker_count = 0
    i = 0
    n = len(hand_mask)
    while i < n:
        if not hand_mask[i]:
            gap_start = i
            while i < n and not hand_mask[i]:
                i += 1
            gap_len = i - gap_start
            # Flicker = short gap with valid detection on both sides
            if gap_len <= max_gap and gap_start > 0 and i < n:
                flicker_count += gap_len
        else:
            i += 1
    return flicker_count


def compute_wrist_jitter(keypoints: np.ndarray, detection_mask: np.ndarray) -> float:
    """Compute median frame-to-frame wrist displacement across both hands.

    Only considers consecutive frames where the same hand is detected in
    both. High jitter suggests noisy keypoint coordinates.

    Returns:
        Median wrist displacement (normalized coords), or 0 if not enough data.
    """
    displacements = []
    for hand_idx in range(2):
        wrist = keypoints[:, hand_idx, 0, :2]  # (T, 2) — x, y only
        mask = detection_mask[:, hand_idx]
        for i in range(1, len(wrist)):
            if mask[i] and mask[i - 1]:
                d = np.linalg.norm(wrist[i] - wrist[i - 1])
                displacements.append(d)

    if len(displacements) < 2:
        return 0.0
    return float(np.median(displacements))


def compute_detection_stats(
    keypoints: np.ndarray,
    detection_mask: np.ndarray,
    flicker_max_gap: int = 5,
) -> dict:
    """Compute per-video detection and quality statistics.

    Args:
        keypoints: Shape (T, 2, 21, 3).
        detection_mask: Shape (T, 2).
        flicker_max_gap: Max gap length to classify as flicker.

    Returns:
        Dict with coverage counts, flicker counts, and jitter metric.
    """
    total = len(detection_mask)
    both = int(np.sum(detection_mask[:, 0] & detection_mask[:, 1]))
    left_only = int(np.sum(detection_mask[:, 0] & ~detection_mask[:, 1]))
    right_only = int(np.sum(~detection_mask[:, 0] & detection_mask[:, 1]))
    neither = int(np.sum(~detection_mask[:, 0] & ~detection_mask[:, 1]))

    # Flicker = short dropout gaps that are NOT hands leaving frame
    flicker_left = count_flicker_frames(detection_mask[:, 0], flicker_max_gap)
    flicker_right = count_flicker_frames(detection_mask[:, 1], flicker_max_gap)

    # Jitter = noisy coordinates when hand IS detected
    jitter = compute_wrist_jitter(keypoints, detection_mask)

    return {
        "total_frames": total,
        "both_hands": both,
        "both_hands_pct": both / total * 100 if total > 0 else 0,
        "left_only": left_only,
        "left_only_pct": left_only / total * 100 if total > 0 else 0,
        "right_only": right_only,
        "right_only_pct": right_only / total * 100 if total > 0 else 0,
        "no_hands": neither,
        "no_hands_pct": neither / total * 100 if total > 0 else 0,
        "flicker_left": flicker_left,
        "flicker_right": flicker_right,
        "flicker_total": flicker_left + flicker_right,
        "wrist_jitter": jitter,
    }


def assess_quality(stats: dict, flicker_threshold: int = 15, jitter_threshold: float = 0.05) -> str:
    """Classify overall extraction quality for a video.

    Args:
        stats: Output of compute_detection_stats.
        flicker_threshold: Flag if total flicker frames exceed this.
        jitter_threshold: Flag if median wrist jitter exceeds this.

    Returns:
        "Good", "Acceptable", or "Poor".
    """
    issues = 0
    if stats["flicker_total"] > flicker_threshold:
        issues += 1
    if stats["wrist_jitter"] > jitter_threshold:
        issues += 1

    if issues == 0:
        return "Good"
    elif issues == 1:
        return "Acceptable"
    else:
        return "Poor"


def generate_markdown_report(report_df: pd.DataFrame, output_path: str) -> None:
    """Write the detection quality report as markdown."""
    lines = [
        "# Keypoint Detection Quality Report\n",
        f"**Videos processed:** {len(report_df)}\n",
    ]

    # Summary
    avg_jitter = report_df["wrist_jitter"].mean()
    avg_flicker = report_df["flicker_total"].mean()
    lines.append("## Summary\n")
    lines.append(f"- Average wrist jitter: {avg_jitter:.4f}")
    lines.append(f"- Average flicker frames per video: {avg_flicker:.1f}")
    lines.append("")

    # Quality breakdown
    for rating in ["Good", "Acceptable", "Poor"]:
        count = len(report_df[report_df["quality"] == rating])
        lines.append(f"- **{rating}:** {count} videos")
    lines.append("")

    # Flag videos that need attention
    needs_attention = report_df[report_df["quality"] == "Poor"]
    if not needs_attention.empty:
        lines.append("## Videos Needing Attention\n")
        lines.append(
            "These videos have high detection flicker and/or noisy keypoints, "
            "which indicates MediaPipe struggled with the video quality "
            "(not simply hands leaving the frame).\n"
        )
        for _, row in needs_attention.iterrows():
            reasons = []
            if row["flicker_total"] > 15:
                reasons.append(f"flicker: {row['flicker_total']} frames")
            if row["wrist_jitter"] > 0.05:
                reasons.append(f"jitter: {row['wrist_jitter']:.4f}")
            lines.append(f"- `{row['video']}`: {', '.join(reasons)}")
        lines.append("")

    # Coverage table (informational)
    lines.append("## Detection Coverage (informational)\n")
    lines.append(
        "Hands leaving the frame is normal during shuffles. "
        "These numbers show detection coverage, not quality.\n"
    )
    lines.append(
        "| Video | Frames | Both Hands | Left Only | Right Only | No Hands | Quality |"
    )
    lines.append(
        "|-------|--------|-----------|-----------|------------|----------|---------|"
    )
    for _, row in report_df.iterrows():
        lines.append(
            f"| {row['video']} | {row['total_frames']} | "
            f"{row['both_hands_pct']:.1f}% | "
            f"{row['left_only_pct']:.1f}% | "
            f"{row['right_only_pct']:.1f}% | "
            f"{row['no_hands_pct']:.1f}% | "
            f"{row['quality']} |"
        )

    # Quality metrics table
    lines.append("\n## Quality Metrics\n")
    lines.append(
        "| Video | Flicker (L) | Flicker (R) | Flicker Total | Wrist Jitter | Quality |"
    )
    lines.append(
        "|-------|------------|------------|---------------|-------------|---------|"
    )
    for _, row in report_df.iterrows():
        lines.append(
            f"| {row['video']} | {row['flicker_left']} | {row['flicker_right']} | "
            f"{row['flicker_total']} | {row['wrist_jitter']:.4f} | {row['quality']} |"
        )

    lines.append("\n## Terminology\n")
    lines.append(
        "- **Detection coverage:** Percentage of frames where 0, 1, or 2 hands "
        "were detected. Hands leaving the frame is expected and does not indicate "
        "a problem."
    )
    lines.append(
        "- **Flicker:** Short gaps (1-5 frames) where a hand was detected before "
        "and after, but not during. This suggests MediaPipe briefly lost a visible "
        "hand — a real quality issue that the cleaning step will interpolate."
    )
    lines.append(
        "- **Wrist jitter:** Median frame-to-frame wrist displacement when the "
        "hand IS detected. High jitter means noisy coordinates, often caused by "
        "motion blur or low resolution."
    )
    lines.append(
        "- **Quality:** 'Good' = no issues. 'Acceptable' = minor flicker or "
        "jitter. 'Poor' = both significant flicker and jitter — consider "
        "excluding or reviewing manually."
    )
    lines.append(
        "\n- 'Left'/'Right' refers to the **subject's** perspective "
        "(MediaPipe labels are flipped)."
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand keypoints from all videos using MediaPipe."
    )
    parser.add_argument(
        "video_dir",
        help="Directory containing the raw video files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/keypoints",
        help="Directory for output .npy files (default: data/keypoints).",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs",
        help="Directory for the detection quality report (default: outputs).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to hand_landmarker.task model file (auto-downloaded if omitted).",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe min detection confidence (default: 0.5).",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe min tracking confidence (default: 0.5).",
    )
    parser.add_argument(
        "--flicker-max-gap",
        type=int,
        default=5,
        help="Max gap length (frames) to classify as flicker vs off-frame (default: 5).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    videos = find_videos(args.video_dir)
    if not videos:
        print(f"No video files found in {args.video_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} videos. Starting extraction...\n")

    report_rows = []

    for video_path in videos:
        video_name = Path(video_path).stem
        print(f"Processing: {os.path.basename(video_path)}...")

        try:
            frames, fps = extract_frames(video_path)
        except ValueError as e:
            print(f"  SKIPPED: {e}")
            continue

        keypoints, detection_mask = extract_keypoints(
            frames,
            fps=fps,
            model_path=args.model_path,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )

        # Save keypoints and mask
        kp_path = os.path.join(args.output_dir, f"{video_name}.npy")
        mask_path = os.path.join(args.output_dir, f"{video_name}_mask.npy")
        np.save(kp_path, keypoints)
        np.save(mask_path, detection_mask)

        # Save FPS for later use
        meta_path = os.path.join(args.output_dir, f"{video_name}_meta.npy")
        np.save(meta_path, np.array([fps]))

        stats = compute_detection_stats(keypoints, detection_mask, args.flicker_max_gap)
        stats["video"] = video_name
        stats["quality"] = assess_quality(stats)
        report_rows.append(stats)

        print(
            f"  {keypoints.shape[0]} frames | "
            f"both: {stats['both_hands_pct']:.1f}%, "
            f"flicker: {stats['flicker_total']}, "
            f"jitter: {stats['wrist_jitter']:.4f} | "
            f"{stats['quality']}"
        )

    if not report_rows:
        print("No videos were successfully processed.")
        sys.exit(1)

    # Save reports
    report_df = pd.DataFrame(report_rows)
    cols = [
        "video", "total_frames",
        "both_hands", "both_hands_pct",
        "left_only", "left_only_pct",
        "right_only", "right_only_pct",
        "no_hands", "no_hands_pct",
        "flicker_left", "flicker_right", "flicker_total",
        "wrist_jitter", "quality",
    ]
    report_df = report_df[cols]

    csv_path = os.path.join(args.report_dir, "detection_quality_report.csv")
    md_path = os.path.join(args.report_dir, "detection_quality_report.md")

    report_df.to_csv(csv_path, index=False)
    generate_markdown_report(report_df, md_path)

    print(f"\nDone. Keypoints saved to {args.output_dir}/")
    print(f"Report saved to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()