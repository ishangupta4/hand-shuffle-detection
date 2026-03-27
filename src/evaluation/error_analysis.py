"""Step 30 -- Error analysis on misclassified videos.

For each misclassified video, examines:
- Prediction confidence (how far from 0.5)
- Video characteristics: duration, detection quality
- Whether model fails on specific patterns (switched vs non-switched, hand)
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.augmentation.dataset import LABEL_MAP

INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def analyze_errors(
    loocv_results: dict,
    features_dir: str = "data/features",
    labels_path: str = "data/labels.csv",
    detection_report_path: str = "outputs/detection_quality_report.csv",
    output_dir: str = "outputs",
) -> dict:
    """Analyze misclassified videos.

    Returns a dict with error analysis findings.
    """
    os.makedirs(output_dir, exist_ok=True)

    labels_df = pd.read_csv(labels_path, dtype={"video_id": str})
    labels_df["switched"] = (labels_df["start_hand"] != labels_df["end_hand"]).astype(int)

    # Load detection quality if available
    det_df = None
    if os.path.exists(detection_report_path):
        det_df = pd.read_csv(detection_report_path, dtype={"video_id": str})
        # Normalize video IDs
        if "video_id" in det_df.columns:
            det_df["video_id"] = det_df["video_id"].apply(lambda x: f"{int(x):05d}" if str(x).isdigit() else x)

    folds = loocv_results["folds"]

    records = []
    for f in folds:
        vid = f["video_id"]
        true_label = f["true_label"]
        pred_label = f["predicted_label"]
        prob_right = f["probability_right"]
        prob_left = f["probability_left"]
        correct = f["correct"]

        # Confidence: distance from 0.5 for predicted class
        pred_prob = prob_right if pred_label == 1 else prob_left
        confidence = abs(pred_prob - 0.5)

        # Sequence length from features
        feat_path = os.path.join(features_dir, f"{vid}.npy")
        if os.path.exists(feat_path):
            seq = np.load(feat_path)
            duration_frames = seq.shape[0]
        else:
            duration_frames = -1

        # Label info
        row = labels_df[labels_df["video_id"] == vid]
        switched = bool(row["switched"].values[0]) if len(row) else None
        start_hand = row["start_hand"].values[0] if len(row) else "?"
        end_hand = row["end_hand"].values[0] if len(row) else "?"

        rec = {
            "video_id": vid,
            "true_label": INVERSE_LABEL_MAP[true_label],
            "predicted_label": INVERSE_LABEL_MAP[pred_label],
            "correct": correct,
            "prob_left": prob_left,
            "prob_right": prob_right,
            "predicted_confidence": pred_prob,
            "distance_from_boundary": confidence,
            "duration_frames": duration_frames,
            "start_hand": start_hand,
            "end_hand": end_hand,
            "switched": switched,
        }

        # Add detection quality info if available
        if det_df is not None and len(det_df[det_df["video_id"] == vid]):
            det_row = det_df[det_df["video_id"] == vid].iloc[0]
            for col in det_df.columns:
                if col != "video_id":
                    rec[f"det_{col}"] = det_row[col]

        records.append(rec)

    df = pd.DataFrame(records)

    # Separate correct/incorrect
    df_correct = df[df["correct"]]
    df_errors = df[~df["correct"]]

    analysis = {
        "total_videos": len(df),
        "n_correct": len(df_correct),
        "n_errors": len(df_errors),
        "accuracy": len(df_correct) / len(df),
        "error_videos": df_errors["video_id"].tolist(),
    }

    # Error patterns
    if len(df_errors) > 0:
        # Confidence analysis
        analysis["mean_error_confidence"] = float(df_errors["predicted_confidence"].mean())
        analysis["mean_correct_confidence"] = float(df_correct["predicted_confidence"].mean()) if len(df_correct) else 0

        confident_wrong = df_errors[df_errors["distance_from_boundary"] > 0.2]
        uncertain_wrong = df_errors[df_errors["distance_from_boundary"] <= 0.1]
        analysis["n_confident_errors"] = len(confident_wrong)
        analysis["n_uncertain_errors"] = len(uncertain_wrong)

        # Switched vs non-switched
        if "switched" in df.columns:
            switched_vids = df[df["switched"] == True]
            nonswitched_vids = df[df["switched"] == False]
            if len(switched_vids):
                analysis["accuracy_switched"] = float(switched_vids["correct"].mean())
            if len(nonswitched_vids):
                analysis["accuracy_nonswitched"] = float(nonswitched_vids["correct"].mean())

        # By end hand
        left_vids = df[df["true_label"] == "left"]
        right_vids = df[df["true_label"] == "right"]
        if len(left_vids):
            analysis["accuracy_end_left"] = float(left_vids["correct"].mean())
        if len(right_vids):
            analysis["accuracy_end_right"] = float(right_vids["correct"].mean())

        # By duration
        median_dur = df["duration_frames"].median()
        short = df[df["duration_frames"] <= median_dur]
        long = df[df["duration_frames"] > median_dur]
        if len(short):
            analysis["accuracy_short_videos"] = float(short["correct"].mean())
        if len(long):
            analysis["accuracy_long_videos"] = float(long["correct"].mean())

        # Per-error details
        error_details = []
        for _, row in df_errors.iterrows():
            detail = {
                "video_id": row["video_id"],
                "true": row["true_label"],
                "predicted": row["predicted_label"],
                "confidence": round(row["predicted_confidence"], 3),
                "switched": row["switched"],
                "duration_frames": row["duration_frames"],
            }
            if row["distance_from_boundary"] <= 0.1:
                detail["note"] = "uncertain (close to 0.5)"
            elif row["distance_from_boundary"] > 0.3:
                detail["note"] = "confidently wrong"
            else:
                detail["note"] = "moderately confident"
            error_details.append(detail)
        analysis["error_details"] = error_details
    else:
        analysis["error_details"] = []

    # Save detailed CSV
    csv_path = os.path.join(output_dir, f"{loocv_results['model']}_error_analysis.csv")
    df.to_csv(csv_path, index=False)
    analysis["csv_path"] = csv_path

    # Plot: confidence distribution for correct vs incorrect
    plot_path = _plot_confidence_distribution(df, output_dir, loocv_results["model"])
    analysis["confidence_plot_path"] = plot_path

    return analysis


def _plot_confidence_distribution(df, output_dir, model_name):
    """Plot predicted probability distribution, colored by correctness."""
    fig, ax = plt.subplots(figsize=(8, 4))

    correct = df[df["correct"]]
    errors = df[~df["correct"]]

    # Plot all predictions as a scatter along the probability axis
    for idx, row in correct.iterrows():
        ax.scatter(row["prob_right"], 0.6, c="#4CAF50", s=80,
                   edgecolors="black", linewidth=0.5, zorder=3)
        ax.annotate(row["video_id"], (row["prob_right"], 0.65),
                    fontsize=6, ha="center", rotation=45)

    for idx, row in errors.iterrows():
        ax.scatter(row["prob_right"], 0.4, c="#F44336", s=80,
                   edgecolors="black", linewidth=0.5, zorder=3, marker="X")
        ax.annotate(row["video_id"], (row["prob_right"], 0.32),
                    fontsize=6, ha="center", rotation=45)

    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Decision boundary")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.set_xlabel("P(right)")
    ax.set_yticks([0.4, 0.6])
    ax.set_yticklabels(["Errors", "Correct"])
    ax.set_title(f"{model_name} -- Prediction Confidence Distribution")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{model_name}_confidence_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path