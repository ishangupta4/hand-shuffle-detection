"""Frame extraction from video files using OpenCV.

Reads all frames from a video, converts them to RGB (for MediaPipe),
and returns them alongside the video's FPS.
"""

import cv2
import numpy as np


def extract_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    """Extract all frames from a video file as RGB arrays.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (list of RGB frames as numpy arrays, FPS of the video).

    Raises:
        ValueError: If the video cannot be opened or contains no frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")

    return frames, fps