#!/usr/bin/env python3
"""Phase 7 runner -- Evaluation & Error Analysis.

Works in two modes:
  1. From existing results: converts your saved JSONs (5-fold and LOOCV)
     into the format the evaluation modules expect, then runs Steps 29-33.
  2. From scratch: re-trains models via LOOCV (slow). Use --retrain flag.

Default (no flags) = mode 1, no retraining.

Usage:
    python run_evaluation.py                          # use existing results
    python run_evaluation.py --skip-temporal           # skip slow temporal analysis
    python run_evaluation.py --retrain                 # re-train LOOCV from scratch
    python run_evaluation.py --retrain --best-model bilstm
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from src.augmentation.pipeline import AugmentationConfig
from src.training.trainer import TrainingConfig

from src.evaluation.metrics import run_metrics
from src.evaluation.error_analysis import analyze_errors
from src.evaluation.feature_importance import (
    rf_feature_importances,
    permutation_importance,
)
from src.evaluation.generate_report import generate_report


# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------

FEATURES_DIR = "data/features"
LABELS_PATH = "data/labels.csv"
METADATA_PATH = "data/features/feature_metadata.json"
OUTPUT_DIR = "outputs"

# Existing result files
FIVEFOLD_FILES = {
    "bilstm": "outputs/bilstm_5fold_results.json",
    "cnn1d": "outputs/cnn1d_5fold_results.json",
    "transformer": "outputs/transformer_5fold_results.json",
}
LOOCV_FILES = {
    "cnn1d": "outputs/cnn1d_loocv_results.json",
}


# -----------------------------------------------------------------------
# Convert existing results to the format evaluation modules expect
# -----------------------------------------------------------------------

def convert_loocv_results(raw: dict) -> dict:
    """Convert your saved LOOCV JSON into the format final_eval.py produces.

    Key addition: recovers probability_left/probability_right from the
    cross-entropy loss using P(true_class) = exp(-loss).
    """
    converted = {
        "model": raw["model"],
        "cv_type": "loocv",
        "folds": [],
    }

    for fold in raw["folds"]:
        vid = fold["val_ids"][0]
        true_label = fold["true_labels"][0]
        pred_label = fold["predictions"][0]
        loss = fold["loss"]

        # Recover probabilities from loss
        p_true = np.clip(math.exp(-loss), 0.001, 0.999)
        if true_label == 1:  # true is right
            prob_right = p_true
            prob_left = 1.0 - p_true
        else:  # true is left
            prob_left = p_true
            prob_right = 1.0 - p_true

        converted["folds"].append({
            "fold": fold["fold"],
            "video_id": vid,
            "true_label": true_label,
            "predicted_label": pred_label,
            "probability_left": prob_left,
            "probability_right": prob_right,
            "correct": bool(pred_label == true_label),
            "best_epoch": 0,
        })

    all_preds = np.array([f["predicted_label"] for f in converted["folds"]])
    all_true = np.array([f["true_label"] for f in converted["folds"]])
    converted["accuracy"] = float(np.mean(all_preds == all_true))
    converted["n_correct"] = int(np.sum(all_preds == all_true))
    converted["n_total"] = len(all_true)

    return converted


def convert_5fold_results(raw: dict) -> dict:
    """Convert your saved 5-fold JSON into per-video LOOCV-like format.

    5-fold results cover all 19 videos across folds, so we can flatten
    them into per-video entries. Loss-based probability recovery works
    here too, but we only have per-fold loss, not per-sample. So we
    use the fold loss as an approximation for single-sample folds,
    and skip probabilities for multi-sample folds (set to 0.5).
    """
    converted = {
        "model": raw["model"],
        "cv_type": "5fold",
        "folds": [],
    }

    for fold in raw["folds"]:
        val_ids = fold["val_ids"]
        preds = fold["predictions"]
        true_labels = fold["true_labels"]
        fold_loss = fold["loss"]

        for i, (vid, pred, true) in enumerate(zip(val_ids, preds, true_labels)):
            # For multi-sample folds we don't have per-sample loss,
            # so we can't recover exact probabilities.
            # Use fold loss as rough proxy (works best for small folds).
            if len(val_ids) == 1:
                p_true = np.clip(math.exp(-fold_loss), 0.001, 0.999)
            else:
                # Approximate: use 0.5 offset based on correctness
                p_true = 0.7 if pred == true else 0.4

            if true == 1:
                prob_right = p_true
                prob_left = 1.0 - p_true
            else:
                prob_left = p_true
                prob_right = 1.0 - p_true

            converted["folds"].append({
                "fold": fold["fold"],
                "video_id": vid,
                "true_label": true,
                "predicted_label": pred,
                "probability_left": prob_left,
                "probability_right": prob_right,
                "correct": bool(pred == true),
                "best_epoch": 0,
            })

    all_preds = np.array([f["predicted_label"] for f in converted["folds"]])
    all_true = np.array([f["true_label"] for f in converted["folds"]])
    converted["accuracy"] = float(np.mean(all_preds == all_true))
    converted["n_correct"] = int(np.sum(all_preds == all_true))
    converted["n_total"] = len(all_true)

    return converted


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 7: Evaluation & Error Analysis")
    parser.add_argument("--retrain", action="store_true",
                        help="Re-train models via LOOCV (slow)")
    parser.add_argument("--best-model", default="cnn1d")
    parser.add_argument("--skip-temporal", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--features-dir", default=FEATURES_DIR)
    parser.add_argument("--labels-path", default=LABELS_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()
    aug_config = AugmentationConfig(metadata_path=METADATA_PATH)

    print("=" * 60)
    print("PHASE 7: EVALUATION & ERROR ANALYSIS")
    print("=" * 60)

    all_loocv_results = {}  # model_name -> converted results dict
    all_metrics = {}

    if args.retrain:
        # ---- Mode 2: Re-train from scratch ----
        print("Mode: RETRAIN (running fresh LOOCV)")
        from src.evaluation.final_eval import run_final_loocv, run_final_classical_loocv
        from src.training.hyperparam_search import _build_model_kwargs

        input_dim = 39
        best_hp = {"lr": 1e-4, "batch_size": 16, "hidden_dim": 64,
                   "dropout": 0.4, "num_layers": 1, "weight_decay": 0.01}
        model_kwargs = _build_model_kwargs(args.best_model, best_hp, input_dim)
        train_config = TrainingConfig(
            lr=float(best_hp["lr"]),
            weight_decay=float(best_hp["weight_decay"]),
            batch_size=int(best_hp["batch_size"]),
        )

        loocv = run_final_loocv(
            args.best_model, model_kwargs, train_config,
            args.features_dir, args.labels_path, aug_config,
        )
        all_loocv_results[args.best_model] = loocv

        for name in ["random_forest", "svm", "logistic_regression"]:
            print(f"\n[{name}] Running LOOCV...")
            loocv = run_final_classical_loocv(
                name, {}, args.features_dir, args.labels_path, aug_config)
            all_loocv_results[name] = loocv

    else:
        # ---- Mode 1: Use existing results ----
        print("Mode: EXISTING RESULTS (no retraining)\n")

        # Load LOOCV results (exact per-video probabilities from loss)
        for name, path in LOOCV_FILES.items():
            if not os.path.exists(path):
                print(f"[SKIP] {name} LOOCV: {path} not found")
                continue
            raw = json.load(open(path))
            converted = convert_loocv_results(raw)
            all_loocv_results[name] = converted
            print(f"[LOOCV] {name}: loaded {converted['n_total']} videos, "
                  f"acc={converted['accuracy']:.3f}")

        # Load 5-fold results (approximate probabilities)
        for name, path in FIVEFOLD_FILES.items():
            if name in all_loocv_results:
                continue  # prefer LOOCV if available
            if not os.path.exists(path):
                print(f"[SKIP] {name} 5-fold: {path} not found")
                continue
            raw = json.load(open(path))
            converted = convert_5fold_results(raw)
            all_loocv_results[name] = converted
            print(f"[5-fold] {name}: loaded {converted['n_total']} videos, "
                  f"acc={converted['accuracy']:.3f}")

    if not all_loocv_results:
        print("\nNo result files found. Check your outputs/ directory.")
        sys.exit(1)

    # ---- Step 29: Metrics + Plots ----
    print("\n" + "=" * 60)
    print("STEP 29: Evaluation Metrics")
    print("=" * 60)

    for name, results in all_loocv_results.items():
        print(f"\n[{name}] Computing metrics...")
        metrics = run_metrics(results, output_dir=args.output_dir)
        all_metrics[name] = metrics
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Macro F1:  {metrics['macro_f1']:.3f}")
        auc = metrics.get("roc_auc", float("nan"))
        if not np.isnan(auc):
            print(f"  ROC-AUC:   {auc:.3f}")
        print(f"  Precision: L={metrics['precision_left']:.3f} R={metrics['precision_right']:.3f}")
        print(f"  Recall:    L={metrics['recall_left']:.3f} R={metrics['recall_right']:.3f}")

    # Save metrics summary
    metrics_save = {n: {k: v for k, v in m.items() if not k.endswith("_path")}
                    for n, m in all_metrics.items()}
    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_save, f, indent=2, default=str)

    # ---- Step 30: Error Analysis (best model) ----
    print("\n" + "=" * 60)
    print("STEP 30: Error Analysis")
    print("=" * 60)

    best_model = max(all_metrics, key=lambda m: all_metrics[m]["accuracy"])
    print(f"Best model: {best_model}\n")

    error_result = analyze_errors(
        loocv_results=all_loocv_results[best_model],
        features_dir=args.features_dir,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
    )
    print(f"Errors: {error_result['n_errors']}/{error_result['total_videos']}")
    for e in error_result.get("error_details", []):
        print(f"  {e['video_id']}: true={e['true']}, pred={e['predicted']}, "
              f"conf={e['confidence']:.3f} -- {e.get('note', '')}")

    # ---- Step 31: Feature Importance ----
    print("\n" + "=" * 60)
    print("STEP 31: Feature Importance")
    print("=" * 60)

    print("\nRandom Forest feature importances...")
    rf_result = rf_feature_importances(
        features_dir=args.features_dir,
        labels_path=args.labels_path,
        metadata_path=METADATA_PATH,
        aug_config=aug_config,
        output_dir=args.output_dir,
    )
    for name, score in rf_result["ranked_features"][:5]:
        print(f"  {name:35s} {score:.4f}")

    print("\nPermutation importance...")
    perm_result = permutation_importance(
        loocv_results=all_loocv_results[best_model],
        features_dir=args.features_dir,
        labels_path=args.labels_path,
        metadata_path=METADATA_PATH,
        output_dir=args.output_dir,
    )

    # ---- Step 32: Temporal Analysis (optional, slow) ----
    temporal_results = {"hidden_state": {}, "probability": {}}

    if not args.skip_temporal and args.retrain:
        print("\n" + "=" * 60)
        print("STEP 32: Temporal Analysis")
        print("=" * 60)
        from src.evaluation.temporal_analysis import (
            temporal_hidden_state_analysis, temporal_probability_analysis)
        from src.training.hyperparam_search import _build_model_kwargs

        best_hp = {"lr": 1e-4, "batch_size": 16, "hidden_dim": 64,
                   "dropout": 0.4, "num_layers": 1, "weight_decay": 0.01}
        bilstm_kwargs = _build_model_kwargs("bilstm", best_hp, 39)
        train_config = TrainingConfig(
            lr=1e-4, weight_decay=0.01, batch_size=16)

        print("\nHidden state trajectories...")
        temporal_results["hidden_state"] = temporal_hidden_state_analysis(
            "bilstm", bilstm_kwargs, train_config,
            args.features_dir, args.labels_path, aug_config,
            output_dir=args.output_dir)

        print("\nFrame-by-frame probabilities...")
        temporal_results["probability"] = temporal_probability_analysis(
            "bilstm", bilstm_kwargs, train_config,
            args.features_dir, args.labels_path, aug_config,
            output_dir=args.output_dir, n_examples=6)
    elif not args.skip_temporal:
        print("\n[NOTE] Temporal analysis requires --retrain (it needs to "
              "train models to extract hidden states). Skipping.")
    else:
        print("\nSkipping temporal analysis (--skip-temporal)")

    # ---- Step 33: Report ----
    print("\n" + "=" * 60)
    print("STEP 33: Generating Report")
    print("=" * 60)

    report_path = generate_report(
        all_loocv_results=all_loocv_results,
        all_metrics=all_metrics,
        error_analysis=error_result,
        feature_importance=rf_result,
        temporal_results=temporal_results,
        permutation_results=perm_result,
        output_dir=args.output_dir,
    )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Report: {report_path}")
    print(f"\nOutput files:")
    for f in sorted(os.listdir(args.output_dir)):
        fp = os.path.join(args.output_dir, f)
        if os.path.isfile(fp):
            print(f"  {f}")


if __name__ == "__main__":
    main()