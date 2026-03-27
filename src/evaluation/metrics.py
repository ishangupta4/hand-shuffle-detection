"""Step 29 -- Evaluation metrics and plots.

Computes accuracy, precision, recall, F1, ROC-AUC from LOOCV results.
Generates confusion matrix, ROC curve, and per-fold accuracy bar chart.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)


def extract_arrays(loocv_results: dict):
    """Pull parallel arrays from LOOCV results dict."""
    folds = loocv_results["folds"]
    true = np.array([f["true_label"] for f in folds])
    pred = np.array([f["predicted_label"] for f in folds])
    # probability of positive class (right=1)
    prob = np.array([f["probability_right"] for f in folds])
    vids = [f["video_id"] for f in folds]
    return true, pred, prob, vids


def compute_metrics(true, pred, prob) -> dict:
    """Compute all classification metrics."""
    acc = accuracy_score(true, pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        true, pred, labels=[0, 1], zero_division=0,
    )

    # ROC-AUC (handle edge cases where only one class present)
    try:
        auc = roc_auc_score(true, prob)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision_left": prec[0],
        "precision_right": prec[1],
        "recall_left": rec[0],
        "recall_right": rec[1],
        "f1_left": f1[0],
        "f1_right": f1[1],
        "support_left": int(sup[0]),
        "support_right": int(sup[1]),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "roc_auc": auc,
    }


def plot_confusion_matrix(true, pred, output_path: str, title: str = "Confusion Matrix"):
    """Save confusion matrix as a PNG."""
    cm = confusion_matrix(true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Left", "Right"])
    ax.set_yticklabels(["Left", "Right"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curve(true, prob, output_path: str, title: str = "ROC Curve"):
    """Save ROC curve as a PNG."""
    try:
        fpr, tpr, _ = roc_curve(true, prob)
        auc = roc_auc_score(true, prob)
    except ValueError:
        # Single class in true labels
        fpr, tpr, auc = [0, 1], [0, 1], float("nan")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_fold_accuracy(loocv_results: dict, output_path: str):
    """Bar chart of per-fold (per-video) correctness from LOOCV."""
    folds = loocv_results["folds"]
    vids = [f["video_id"] for f in folds]
    correct = [int(f["correct"]) for f in folds]
    probs_right = [f["probability_right"] for f in folds]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Correctness bar chart
    colors = ["#4CAF50" if c else "#F44336" for c in correct]
    axes[0].bar(range(len(vids)), correct, color=colors, edgecolor="white")
    axes[0].set_ylabel("Correct")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Wrong", "Correct"])
    axes[0].set_title("LOOCV Per-Video Predictions")

    # Probability bar chart
    true_labels = [f["true_label"] for f in folds]
    bar_colors = []
    for i, (tl, pr) in enumerate(zip(true_labels, probs_right)):
        pred = 1 if pr >= 0.5 else 0
        if pred == tl:
            bar_colors.append("#4CAF50")
        elif abs(pr - 0.5) < 0.1:
            bar_colors.append("#FF9800")  # uncertain and wrong
        else:
            bar_colors.append("#F44336")  # confident and wrong

    axes[1].bar(range(len(vids)), probs_right, color=bar_colors, edgecolor="white")
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("P(right)")
    axes[1].set_xlabel("Video ID")
    axes[1].set_xticks(range(len(vids)))
    axes[1].set_xticklabels(vids, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_metrics(loocv_results: dict, output_dir: str = "outputs") -> dict:
    """Compute all metrics and save plots. Returns metrics dict."""
    os.makedirs(output_dir, exist_ok=True)

    true, pred, prob, vids = extract_arrays(loocv_results)
    model_name = loocv_results.get("model", "model")

    metrics = compute_metrics(true, pred, prob)
    metrics["model"] = model_name

    # Plots
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(true, pred, cm_path, title=f"{model_name} Confusion Matrix (LOOCV)")
    metrics["confusion_matrix_path"] = cm_path

    roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    plot_roc_curve(true, prob, roc_path, title=f"{model_name} ROC Curve (LOOCV)")
    metrics["roc_curve_path"] = roc_path

    fold_path = os.path.join(output_dir, f"{model_name}_per_video_accuracy.png")
    plot_per_fold_accuracy(loocv_results, fold_path)
    metrics["per_video_path"] = fold_path

    return metrics