"""FastAPI inference server for hand-shuffle prediction.

Receives a base64-encoded JPEG frame from the browser, runs the full
pipeline (MediaPipe → features → CNN1D), and returns a prediction with
confidence scores.

Usage:
    pip install fastapi uvicorn python-multipart
    python app.py
    python app.py --model outputs/models/cnn1d_best.pt --port 8000

The server maintains a rolling buffer of recent frames so the CNN1D
always gets a full temporal sequence to work with.
"""

import argparse
import base64
import collections
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# app.py lives at src/app/app.py — project root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn1d import CNN1DClassifier
from src.extraction.keypoint_extractor import extract_keypoints, ensure_model
from src.extraction.clean_keypoints import clean_video_keypoints
from src.features.static_features import compute_static_features_video, get_static_feature_names
from src.features.dynamic_features import compute_dynamic_features, get_dynamic_feature_names
from src.features.normalize import normalize_video

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_NAMES = {0: "left", 1: "right"}
BUFFER_SIZE = 110         # 15s game + buffer headroom at ~7fps (150ms send interval)
MIN_FRAMES_FOR_PRED = 10  # minimum frames needed before making a prediction
FPS_ASSUMPTION = 30.0     # assumed webcam FPS for MediaPipe timestamps

# Column indices used by dynamic feature computation — derived from static names
_STATIC_NAMES = get_static_feature_names()
CURL_INDICES = [
    i for i, n in enumerate(_STATIC_NAMES)
    if "curl_" in n and "asymmetry" not in n and "velocity" not in n
]
COMPACT_INDICES = [
    i for i, n in enumerate(_STATIC_NAMES)
    if n in ("left_compactness", "right_compactness")
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load CNN1D from a checkpoint saved by train_and_save.py."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_kwargs = ckpt["model_kwargs"]
    model = CNN1DClassifier(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    target_length = ckpt.get("target_length", 60)
    meta = {
        "loocv_accuracy": ckpt.get("loocv_mean_accuracy"),
        "loocv_std": ckpt.get("loocv_std_accuracy"),
        "target_length": target_length,
    }
    return model, target_length, meta


# ---------------------------------------------------------------------------
# Feature pipeline (mirrors training exactly)
# ---------------------------------------------------------------------------

def extract_features_from_frames(
    frames: list[np.ndarray],
    mp_model_path: str,
    fps: float = FPS_ASSUMPTION,
) -> np.ndarray | None:
    """Run the full feature pipeline on a list of RGB frames.

    Returns:
        Feature array of shape (T, 39), or None if extraction fails.
    """
    if len(frames) < 2:
        return None

    try:
        keypoints, detection_mask = extract_keypoints(
            frames, fps=fps, model_path=mp_model_path
        )
    except Exception as e:
        print(f"MediaPipe error: {e}")
        return None

    # Skip if virtually nothing was detected
    if detection_mask.sum() < len(frames) * 0.1:
        return None

    # Clean keypoints (flicker interpolation + outlier removal)
    cleaned_kp, _ = clean_video_keypoints(keypoints, detection_mask)

    # Normalize (center on wrist, scale by hand size) — for static features
    normalized_kp = normalize_video(cleaned_kp)

    # Static features
    static_feats = compute_static_features_video(normalized_kp)

    # Dynamic features use RAW (non-normalized) keypoints for velocity
    dynamic_feats = compute_dynamic_features(
        cleaned_kp, static_feats, CURL_INDICES, COMPACT_INDICES
    )

    features = np.column_stack([static_feats, dynamic_feats]).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0)
    return features


def pad_or_truncate(features: np.ndarray, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad or truncate to target_length. Returns (padded, mask)."""
    T, F = features.shape
    if T >= target_length:
        return features[:target_length], np.ones(target_length, dtype=np.float32)

    pad = np.zeros((target_length - T, F), dtype=np.float32)
    padded = np.vstack([features, pad])
    mask = np.array([1.0] * T + [0.0] * (target_length - T), dtype=np.float32)
    return padded, mask


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class InferenceState:
    def __init__(self):
        self.model = None
        self.target_length = 60
        self.mp_model_path = None
        self.frame_buffer: collections.deque = collections.deque(maxlen=BUFFER_SIZE)
        self.model_meta = {}
        self.device = "cpu"

state = InferenceState()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Hand Shuffle AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend static assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    if (FRONTEND_DIR / "css").exists():
        app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    if (FRONTEND_DIR / "js").exists():
        app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


class FrameRequest(BaseModel):
    frame: str  # base64-encoded JPEG


class PredictionResponse(BaseModel):
    prediction: str          # "left" or "right"
    confidence: float        # probability of the predicted class
    prob_left: float
    prob_right: float
    num_frames_buffered: int
    ready: bool              # False if not enough frames yet


@app.get("/")
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "running", "note": "No frontend found. Place index.html in ./frontend/"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "meta": state.model_meta,
    }


@app.post("/reset")
def reset_buffer():
    state.frame_buffer.clear()
    return {"status": "buffer cleared"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: FrameRequest):
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_predict_sync, req)


def _predict_sync(req: FrameRequest) -> PredictionResponse:
    # Decode base64 JPEG → numpy RGB frame
    try:
        img_bytes = base64.b64decode(req.frame)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return PredictionResponse(
            prediction="unknown", confidence=0.5,
            prob_left=0.5, prob_right=0.5,
            num_frames_buffered=0, ready=False,
        )

    state.frame_buffer.append(frame_rgb)
    n_buffered = len(state.frame_buffer)

    if state.model is None or n_buffered < MIN_FRAMES_FOR_PRED:
        return PredictionResponse(
            prediction="unknown", confidence=0.5,
            prob_left=0.5, prob_right=0.5,
            num_frames_buffered=n_buffered, ready=False,
        )

    frames = list(state.frame_buffer)
    features = extract_features_from_frames(frames, state.mp_model_path)

    if features is None or features.shape[0] < 2:
        return PredictionResponse(
            prediction="unknown", confidence=0.5,
            prob_left=0.5, prob_right=0.5,
            num_frames_buffered=n_buffered, ready=False,
        )

    padded, mask = pad_or_truncate(features, state.target_length)

    x = torch.from_numpy(padded).unsqueeze(0)
    m = torch.from_numpy(mask).unsqueeze(0)

    with torch.no_grad():
        logits = state.model(x, mask=m)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    pred_idx = int(np.argmax(probs))
    prediction = LABEL_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        prob_left=float(probs[0]),
        prob_right=float(probs[1]),
        num_frames_buffered=n_buffered,
        ready=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hand Shuffle inference server.")
    parser.add_argument("--model", default="outputs/models/cnn1d_best.pt",
                        help="Path to trained model checkpoint.")
    parser.add_argument("--mp-model", default=None,
                        help="Path to hand_landmarker.task (auto-downloaded if omitted).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}.")
        print("Run train_and_save.py first to generate the checkpoint.")
        sys.exit(1)

    print(f"Loading model from {args.model}...")
    state.model, state.target_length, state.model_meta = load_model(args.model, args.device)
    state.device = args.device
    print(f"  target_length={state.target_length}")
    if state.model_meta.get("loocv_accuracy"):
        print(f"  LOOCV accuracy: {state.model_meta['loocv_accuracy']:.3f} "
              f"+/- {state.model_meta['loocv_std']:.3f}")

    # Ensure MediaPipe model
    state.mp_model_path = ensure_model(args.mp_model)
    print(f"  MediaPipe model: {state.mp_model_path}")

    print(f"\nServer running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()