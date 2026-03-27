---
title: Hand Shuffle AI
emoji: 🤚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# hand-shuffle-detection

An AI pipeline that predicts which hand holds a hidden object after a hand-shuffle sequence. Given a video of two hands shuffling, the model extracts skeletal keypoints frame-by-frame, engineers motion and pose features, and classifies the final hand using a 1D CNN trained on temporal sequences.

Built as part of a research project studying minimum-data regimes for skeletal behavioral inference — using the shuffle game as a constrained proving ground.

**Live demo [WIP]:** [ishangupta4-hand-shuffle-ai.hf.space](https://ishangupta4-hand-shuffle-ai.hf.space)

---

## How It Works

```
Video / Webcam
     ↓
MediaPipe HandLandmarker  (21 keypoints × 2 hands per frame)
     ↓
Cleaning + Normalization  (interpolation, Savitzky-Golay smoothing)
     ↓
Feature Engineering       (39 features/frame: curl angles, compactness,
                           inter-hand distance, wrist velocity, asymmetry…)
     ↓
CNN1D Classifier          (temporal sequence → left / right prediction)
```

Trained on 19 labeled videos with leave-one-out cross-validation (LOOCV) and on-the-fly data augmentation (horizontal flip, time warp, Gaussian jitter, 3D rotation).

---

## Stack

| Component | Tool |
|---|---|
| Keypoint extraction | MediaPipe HandLandmarker (Tasks API) |
| Deep learning | PyTorch |
| Inference server | FastAPI + Uvicorn |
| Data processing | NumPy, Pandas, SciPy |
| Deployment | Hugging Face Spaces (Docker) |

---

## Project Structure

```
hand-shuffle-detection/
├── src/
│   ├── app/                  # FastAPI server + browser frontend
│   ├── extraction/           # MediaPipe keypoint extraction + cleaning
│   ├── features/             # Static and dynamic feature engineering
│   ├── models/               # CNN1D, Bi-LSTM, Transformer architectures
│   ├── augmentation/         # On-the-fly augmentation + CV splits
│   ├── training/             # LOOCV training loop + hyperparameter search
│   └── evaluation/           # Metrics, error analysis, report generation
├── data/
│   ├── labels.csv            # Ground truth (video_id, start_hand, end_hand)
│   ├── splits_loocv.json     # LOOCV fold definitions
│   └── splits_5fold.json
├── outputs/
│   └── models/
│       └── cnn1d_best.pt     # Trained model weights (Git LFS)
├── configs/
│   ├── config.yaml
│   └── search_space.yaml
├── run_training.py
├── run_evaluation.py
├── Dockerfile                # For Hugging Face Spaces deployment
└── requirements.txt
```

---

## Setup

**Prerequisites:** Python 3.10+, Git LFS (for model weights)

```bash
# 1. Clone the repo
git clone https://github.com/ishangupta4/hand-shuffle-detection.git
cd hand-shuffle-detection

# 2. Pull model weights via Git LFS
git lfs pull

# 3. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Running the Inference App

```bash
python -m src.app.app --model outputs/models/cnn1d_best.pt
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

The app has two modes — **Live** (continuous real-time prediction from webcam) and **Game** (tap to start, model locks in a prediction at the end of each shuffle round).

---

## Training From Scratch

Place your labeled videos in `data/raw_videos/` and update `data/labels.csv`, then run the pipeline in order:

```bash
# Extract keypoints from all videos
python -m src.extraction.process_all_videos

# Build features
python -m src.features.build_features

# Train with LOOCV
python run_training.py

# Evaluate
python run_evaluation.py
```

Results and model weights are written to `outputs/`.

---

## Results

| Model | LOOCV Accuracy | AUC |
|---|---|---|
| CNN1D | 84.2% | 0.967 |
| Bi-LSTM | 84.2% | 0.951 |
| Transformer | 78.9% | 0.923 |
| Random Forest (baseline) | 73.7% | 0.889 |

CNN1D selected as primary model based on lowest fold variance.