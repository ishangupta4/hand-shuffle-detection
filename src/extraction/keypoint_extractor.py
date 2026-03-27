"""MediaPipe hand keypoint extraction using the Tasks API.

Extracts 21 keypoints (x, y, z) for up to 2 hands per frame using
the MediaPipe HandLandmarker (Tasks API). Outputs a (T, 2, 21, 3)
array per video.

Hand assignment convention:
    Index 0 = "Left" hand (from the SUBJECT's perspective)
    Index 1 = "Right" hand (from the SUBJECT's perspective)

    MediaPipe labels hands from the CAMERA's perspective, which is mirrored.
    So MediaPipe's "Left" = subject's Right, and MediaPipe's "Right" = subject's Left.
    We flip this so that index 0/1 match the subject's actual left/right.

Model file:
    This module requires `hand_landmarker.task` — a pre-trained model bundle.
    It will be auto-downloaded on first use if not found at the expected path.
    Download URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Where to store the model file
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
_MODEL_FILENAME = "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe hand landmark indices for reference
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def ensure_model(model_path: str | None = None) -> str:
    """Ensure the hand_landmarker.task model file exists, downloading if needed.

    Args:
        model_path: Explicit path to the .task file. If None, uses the
            default location next to this source file.

    Returns:
        Resolved path to the model file.
    """
    if model_path and os.path.isfile(model_path):
        return model_path

    default_path = _DEFAULT_MODEL_DIR / _MODEL_FILENAME
    if default_path.is_file():
        return str(default_path)

    print(f"Downloading hand_landmarker.task model...")
    _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, str(default_path))
    print(f"  Saved to {default_path}")
    return str(default_path)


def extract_keypoints(
    frames: list[np.ndarray],
    fps: float,
    model_path: str | None = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract hand keypoints from a sequence of RGB frames.

    Uses the MediaPipe Tasks API (HandLandmarker) in VIDEO mode, which
    enables cross-frame tracking for better temporal consistency.

    Args:
        frames: List of RGB frames (H, W, 3) as uint8 numpy arrays.
        fps: Frames per second of the source video (used for timestamps).
        model_path: Path to hand_landmarker.task. Auto-downloaded if None.
        min_detection_confidence: Detection confidence threshold.
        min_tracking_confidence: Tracking confidence threshold.

    Returns:
        keypoints: Array of shape (T, 2, 21, 3). NaN where a hand is not detected.
        detection_mask: Boolean array of shape (T, 2). True if hand was detected.
    """
    resolved_model = ensure_model(model_path)

    num_frames = len(frames)
    keypoints = np.full((num_frames, 2, 21, 3), np.nan, dtype=np.float32)
    detection_mask = np.zeros((num_frames, 2), dtype=bool)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=resolved_model),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Compute ms-per-frame from FPS
    ms_per_frame = 1000.0 / fps if fps > 0 else 33.33

    with HandLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            timestamp_ms = int(i * ms_per_frame)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame,
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.hand_landmarks or not result.handedness:
                continue

            for hand_landmarks, handedness_info in zip(
                result.hand_landmarks, result.handedness
            ):
                # MediaPipe label (from camera's perspective)
                mp_label = handedness_info[0].category_name

                # Flip to subject's perspective:
                # MediaPipe "Left" (camera) = Subject's Right -> index 1
                # MediaPipe "Right" (camera) = Subject's Left -> index 0
                hand_idx = 0 if mp_label == "Right" else 1

                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                    dtype=np.float32,
                )
                keypoints[i, hand_idx] = coords
                detection_mask[i, hand_idx] = True

    return keypoints, detection_mask