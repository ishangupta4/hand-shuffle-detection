"""Assess video quality for the hand shuffle dataset.

Extracts metadata, saves sample frames, and generates a quality report
for each video in the dataset.
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Quality thresholds
MIN_RESOLUTION = (480, 360)  # width, height
GOOD_RESOLUTION = (1280, 720)
MIN_FPS = 15
GOOD_FPS = 24
BLUR_THRESHOLD_POOR = 50  # Laplacian variance below this = blurry
BLUR_THRESHOLD_GOOD = 100
BRIGHTNESS_LOW = 40
BRIGHTNESS_HIGH = 220


@dataclass
class VideoAssessment:
    filename: str
    path: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int = 0
    duration: float = 0.0
    codec: str = ""
    blur_scores: list = field(default_factory=list)
    brightness_scores: list = field(default_factory=list)
    quality_rating: str = "Unknown"
    issues: list = field(default_factory=list)
    readable: bool = True


def get_video_metadata(video_path: str) -> VideoAssessment:
    """Extract basic metadata from a video file."""
    assessment = VideoAssessment(
        filename=os.path.basename(video_path), path=video_path
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        assessment.readable = False
        assessment.quality_rating = "Unreadable"
        assessment.issues.append("Could not open video file")
        return assessment

    assessment.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    assessment.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assessment.fps = cap.get(cv2.CAP_PROP_FPS)
    assessment.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    assessment.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    if assessment.fps > 0:
        assessment.duration = assessment.total_frames / assessment.fps

    cap.release()
    return assessment


def compute_blur_score(frame: np.ndarray) -> float:
    """Laplacian variance as a measure of image sharpness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_brightness(frame: np.ndarray) -> float:
    """Mean brightness of the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def extract_sample_frames(
    video_path: str, output_dir: str, video_name: str
) -> list[tuple[str, np.ndarray]]:
    """Extract frames from beginning, middle, and end of the video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    # Sample at 10%, 50%, 90% to avoid blank first/last frames
    positions = {
        "begin": max(0, int(total * 0.10)),
        "middle": int(total * 0.50),
        "end": min(total - 1, int(total * 0.90)),
    }

    frames = []
    for label, frame_idx in positions.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            fname = f"{video_name}_{label}.jpg"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, frame)
            frames.append((fpath, frame))

    cap.release()
    return frames


def rate_video(assessment: VideoAssessment) -> None:
    """Assign a quality rating and note issues."""
    if not assessment.readable:
        return

    issues = []
    score = 0  # 0-6 scale

    # Resolution check
    if assessment.width >= GOOD_RESOLUTION[0] and assessment.height >= GOOD_RESOLUTION[1]:
        score += 2
    elif assessment.width >= MIN_RESOLUTION[0] and assessment.height >= MIN_RESOLUTION[1]:
        score += 1
    else:
        issues.append(f"Low resolution ({assessment.width}x{assessment.height})")

    # FPS check
    if assessment.fps >= GOOD_FPS:
        score += 2
    elif assessment.fps >= MIN_FPS:
        score += 1
    else:
        issues.append(f"Low frame rate ({assessment.fps:.1f} fps)")

    # Blur check (average across sampled frames)
    if assessment.blur_scores:
        avg_blur = np.mean(assessment.blur_scores)
        if avg_blur >= BLUR_THRESHOLD_GOOD:
            score += 1
        elif avg_blur < BLUR_THRESHOLD_POOR:
            issues.append(f"Blurry (Laplacian var: {avg_blur:.1f})")
    
    # Brightness check
    if assessment.brightness_scores:
        avg_bright = np.mean(assessment.brightness_scores)
        if BRIGHTNESS_LOW <= avg_bright <= BRIGHTNESS_HIGH:
            score += 1
        elif avg_bright < BRIGHTNESS_LOW:
            issues.append(f"Too dark (mean brightness: {avg_bright:.1f})")
        else:
            issues.append(f"Overexposed (mean brightness: {avg_bright:.1f})")

    # Short duration might indicate incomplete captures
    if assessment.duration < 1.0:
        issues.append(f"Very short duration ({assessment.duration:.2f}s)")

    # Assign rating
    if score >= 5:
        assessment.quality_rating = "Good"
    elif score >= 3:
        assessment.quality_rating = "Acceptable"
    else:
        assessment.quality_rating = "Poor"

    assessment.issues = issues


def generate_report(assessments: list[VideoAssessment], output_path: str) -> None:
    """Generate a markdown quality report."""
    good = sum(1 for a in assessments if a.quality_rating == "Good")
    acceptable = sum(1 for a in assessments if a.quality_rating == "Acceptable")
    poor = sum(1 for a in assessments if a.quality_rating == "Poor")
    unreadable = sum(1 for a in assessments if a.quality_rating == "Unreadable")

    readable = [a for a in assessments if a.readable]
    avg_duration = np.mean([a.duration for a in readable]) if readable else 0
    avg_fps = np.mean([a.fps for a in readable]) if readable else 0

    lines = []
    lines.append("# Video Quality Assessment Report\n")
    lines.append(f"**Total videos:** {len(assessments)}\n")

    lines.append("## Dataset Statistics\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total videos | {len(assessments)} |")
    lines.append(f"| Good | {good} |")
    lines.append(f"| Acceptable | {acceptable} |")
    lines.append(f"| Poor | {poor} |")
    lines.append(f"| Unreadable | {unreadable} |")
    lines.append(f"| Average duration | {avg_duration:.2f}s |")
    lines.append(f"| Average FPS | {avg_fps:.1f} |")
    lines.append("")

    # Per-video table
    lines.append("## Per-Video Assessment\n")
    lines.append(
        "| Video | Resolution | FPS | Frames | Duration | "
        "Blur (avg) | Brightness (avg) | Rating | Issues |"
    )
    lines.append("|-------|-----------|-----|--------|----------|" "-----------|-----------------|--------|--------|")

    for a in assessments:
        if not a.readable:
            lines.append(
                f"| {a.filename} | - | - | - | - | - | - | {a.quality_rating} | "
                f"{'; '.join(a.issues)} |"
            )
            continue

        avg_blur = f"{np.mean(a.blur_scores):.1f}" if a.blur_scores else "-"
        avg_bright = f"{np.mean(a.brightness_scores):.1f}" if a.brightness_scores else "-"
        issue_str = "; ".join(a.issues) if a.issues else "None"

        lines.append(
            f"| {a.filename} | {a.width}x{a.height} | {a.fps:.1f} | "
            f"{a.total_frames} | {a.duration:.2f}s | {avg_blur} | {avg_bright} | "
            f"{a.quality_rating} | {issue_str} |"
        )

    # Recommendations
    lines.append("\n## Recommendations\n")

    exclude = [a for a in assessments if a.quality_rating in ("Poor", "Unreadable")]
    if exclude:
        lines.append("**Consider excluding these videos:**\n")
        for a in exclude:
            reasons = "; ".join(a.issues) if a.issues else "Poor overall quality"
            lines.append(f"- `{a.filename}`: {reasons}")
        lines.append("")
    else:
        lines.append("All videos meet minimum quality standards. No exclusions recommended.\n")

    # General notes
    lines.append("## Notes\n")
    lines.append(
        "- Blur is measured via Laplacian variance (higher = sharper). "
        f"Threshold: <{BLUR_THRESHOLD_POOR} is poor, >{BLUR_THRESHOLD_GOOD} is good."
    )
    lines.append(
        "- Brightness is mean pixel intensity (0-255). "
        f"Range {BRIGHTNESS_LOW}-{BRIGHTNESS_HIGH} is acceptable."
    )
    lines.append(
        "- Sample frames are saved in `outputs/sample_frames/` for manual inspection."
    )
    lines.append(
        "- Hand visibility and occlusion should be verified manually using the sample frames."
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def find_videos(directory: str) -> list[str]:
    """Find all video files in a directory."""
    videos = []
    for entry in sorted(os.listdir(directory)):
        if Path(entry).suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(os.path.join(directory, entry))
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Assess video quality for the hand shuffle dataset."
    )
    parser.add_argument(
        "video_dir",
        help="Directory containing the raw video files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for output files (default: outputs).",
    )
    args = parser.parse_args()

    video_dir = args.video_dir
    output_dir = args.output_dir
    frames_dir = os.path.join(output_dir, "sample_frames")
    os.makedirs(frames_dir, exist_ok=True)

    videos = find_videos(video_dir)
    if not videos:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} videos in {video_dir}\n")

    assessments = []
    for vpath in videos:
        vname = Path(vpath).stem
        print(f"Processing: {os.path.basename(vpath)}...")

        assessment = get_video_metadata(vpath)
        if not assessment.readable:
            assessments.append(assessment)
            continue

        # Extract and analyze sample frames
        frames = extract_sample_frames(vpath, frames_dir, vname)
        for _, frame in frames:
            assessment.blur_scores.append(compute_blur_score(frame))
            assessment.brightness_scores.append(compute_brightness(frame))

        rate_video(assessment)
        assessments.append(assessment)

        print(
            f"  {assessment.width}x{assessment.height} @ {assessment.fps:.1f}fps, "
            f"{assessment.duration:.2f}s -> {assessment.quality_rating}"
        )

    # Generate report
    report_path = os.path.join(output_dir, "video_quality_report.md")
    generate_report(assessments, report_path)
    print(f"\nReport saved to {report_path}")
    print(f"Sample frames saved to {frames_dir}/")


if __name__ == "__main__":
    main()
