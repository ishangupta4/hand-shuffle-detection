"""Step 28 -- Final LOOCV evaluation with best model + hyperparameters.

Trains from scratch on each LOOCV fold, records per-fold predictions
and probabilities. Since LOOCV gives exactly one prediction per video,
all 19 predictions are aggregated into a single results dict.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.augmentation.cv_splits import load_labels, leave_one_out_splits, load_splits, stratified_kfold_splits
from src.augmentation.dataset import HandShuffleDataset, LABEL_MAP
from src.augmentation.pipeline import AugmentationConfig
from src.models import get_dl_model, DL_MODEL_REGISTRY
from src.models.classical import aggregate_dataset, aggregate_sequence, get_classical_model
from src.training.trainer import TrainingConfig, train_model, evaluate


INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def _evaluate_with_probs(model, loader, device):
    """Evaluate and return predictions with probabilities."""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            mask = batch.get("mask")
            lengths = batch.get("length")

            if mask is not None:
                mask = mask.to(device)
            if lengths is not None:
                lengths = lengths.to(device)

            logits = model(features, lengths=lengths, mask=mask)
            probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        np.array(all_preds),
        np.concatenate(all_probs, axis=0),
        np.array(all_labels),
    )


def run_final_loocv(
    model_name: str,
    model_kwargs: dict,
    train_config: TrainingConfig,
    features_dir: str = "data/features",
    labels_path: str = "data/labels.csv",
    aug_config: AugmentationConfig | None = None,
    target_col: str = "end_hand",
    verbose: bool = True,
) -> dict:
    """Run full LOOCV with the best configuration.

    Returns a dict with per-video predictions, probabilities, and labels.
    """
    labels_df = load_labels(labels_path)
    video_ids = sorted(labels_df["video_id"].tolist())
    splits = leave_one_out_splits(video_ids)

    results = {
        "model": model_name,
        "cv_type": "loocv",
        "target_col": target_col,
        "model_kwargs": {k: v for k, v in model_kwargs.items()},
        "train_config": {
            "lr": train_config.lr,
            "weight_decay": train_config.weight_decay,
            "batch_size": train_config.batch_size,
            "max_epochs": train_config.max_epochs,
            "patience": train_config.patience,
        },
        "folds": [],
    }

    all_preds = []
    all_probs = []
    all_true = []
    all_vids = []

    start = time.time()

    for split in splits:
        fold = split["fold"]
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]
        vid = val_ids[0]

        if verbose:
            print(f"LOOCV fold {fold:2d}/{len(splits)-1} -- held out: {vid}")

        train_ds = HandShuffleDataset(
            video_ids=train_ids,
            features_dir=features_dir,
            labels_df=labels_df,
            augment=True,
            aug_config=aug_config,
            seq_mode="pad",
            target_col=target_col,
        )
        val_ds = HandShuffleDataset(
            video_ids=val_ids,
            features_dir=features_dir,
            labels_df=labels_df,
            augment=False,
            seq_mode="pad",
            target_length=train_ds.target_length,
            target_col=target_col,
        )

        train_loader = DataLoader(
            train_ds, batch_size=train_config.batch_size, shuffle=True,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        model = get_dl_model(model_name, **model_kwargs)
        device = torch.device(train_config.device)

        model, history = train_model(
            model, train_loader, val_loader, train_config, verbose=False,
        )

        preds, probs, true_labels = _evaluate_with_probs(model, val_loader, device)

        pred_label = int(preds[0])
        prob_vec = probs[0].tolist()
        true_label = int(true_labels[0])

        fold_result = {
            "fold": fold,
            "video_id": vid,
            "true_label": true_label,
            "predicted_label": pred_label,
            "probability_left": prob_vec[0],
            "probability_right": prob_vec[1],
            "correct": bool(pred_label == true_label),
            "best_epoch": history.best_epoch,
        }
        results["folds"].append(fold_result)

        all_preds.append(pred_label)
        all_probs.append(prob_vec)
        all_true.append(true_label)
        all_vids.append(vid)

        marker = "OK" if pred_label == true_label else "MISS"
        if verbose:
            print(f"  -> true={INVERSE_LABEL_MAP[true_label]}, "
                  f"pred={INVERSE_LABEL_MAP[pred_label]}, "
                  f"prob=[{prob_vec[0]:.3f}, {prob_vec[1]:.3f}] [{marker}]")

    elapsed = time.time() - start

    # Aggregate
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    accuracy = float(np.mean(all_preds == all_true))

    results["accuracy"] = accuracy
    results["n_correct"] = int(np.sum(all_preds == all_true))
    results["n_total"] = len(all_true)
    results["elapsed_sec"] = elapsed

    if verbose:
        print(f"\nLOOCV accuracy: {accuracy:.3f} ({results['n_correct']}/{results['n_total']})")
        print(f"Elapsed: {elapsed:.1f}s")

    return results


def run_final_kfold(
    model_name: str,
    model_kwargs: dict,
    train_config: TrainingConfig,
    features_dir: str = "data/features",
    labels_path: str = "data/labels.csv",
    splits_path: str = "data/splits_5fold.json",
    aug_config: AugmentationConfig | None = None,
    target_col: str = "end_hand",
    verbose: bool = True,
) -> dict:
    """Run 5-fold CV with probabilities for a deep learning model."""
    labels_df = load_labels(labels_path)
    splits = load_splits(splits_path)

    results = {
        "model": model_name,
        "cv_type": "5fold",
        "target_col": target_col,
        "folds": [],
    }

    for split in splits:
        fold = split["fold"]
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]

        if verbose:
            print(f"Fold {fold} -- val: {val_ids}")

        train_ds = HandShuffleDataset(
            video_ids=train_ids,
            features_dir=features_dir,
            labels_df=labels_df,
            augment=True,
            aug_config=aug_config,
            seq_mode="pad",
            target_col=target_col,
        )
        val_ds = HandShuffleDataset(
            video_ids=val_ids,
            features_dir=features_dir,
            labels_df=labels_df,
            augment=False,
            seq_mode="pad",
            target_length=train_ds.target_length,
            target_col=target_col,
        )

        train_loader = DataLoader(
            train_ds, batch_size=train_config.batch_size, shuffle=True,
        )
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        model = get_dl_model(model_name, **model_kwargs)
        device = torch.device(train_config.device)

        model, history = train_model(
            model, train_loader, val_loader, train_config, verbose=False,
        )

        preds, probs, true_labels = _evaluate_with_probs(model, val_loader, device)

        fold_result = {
            "fold": fold,
            "val_ids": val_ids,
            "true_labels": true_labels.tolist(),
            "predicted_labels": preds.tolist(),
            "probabilities": probs.tolist(),
            "accuracy": float(np.mean(preds == true_labels)),
        }
        results["folds"].append(fold_result)

        if verbose:
            print(f"  Fold {fold} acc: {fold_result['accuracy']:.3f}")

    accs = [f["accuracy"] for f in results["folds"]]
    results["mean_accuracy"] = float(np.mean(accs))
    results["std_accuracy"] = float(np.std(accs))
    return results


def run_final_classical_loocv(
    model_name: str,
    model_kwargs: dict,
    features_dir: str = "data/features",
    labels_path: str = "data/labels.csv",
    aug_config: AugmentationConfig | None = None,
    target_col: str = "end_hand",
    verbose: bool = True,
) -> dict:
    """Run LOOCV for a classical model, returning per-video probabilities."""
    from src.augmentation.pipeline import augment_single

    labels_df = load_labels(labels_path)
    video_ids = sorted(labels_df["video_id"].tolist())
    splits = leave_one_out_splits(video_ids)

    if aug_config is None:
        aug_config = AugmentationConfig()

    label_lookup = labels_df.set_index("video_id").to_dict("index")

    results = {
        "model": model_name,
        "cv_type": "loocv",
        "target_col": target_col,
        "folds": [],
    }

    for split in splits:
        fold = split["fold"]
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]
        vid = val_ids[0]

        rng = np.random.default_rng(42 + fold)

        # Augment training data
        train_seqs, train_labels = [], []
        for tid in train_ids:
            seq = np.load(os.path.join(features_dir, f"{tid}.npy")).astype(np.float32)
            seq = np.nan_to_num(seq, nan=0.0)
            row = label_lookup[tid]
            ld = {"start_hand": row["start_hand"], "end_hand": row["end_hand"]}
            variants = augment_single(seq, ld, aug_config, rng=rng)
            for v_seq, v_label, _ in variants:
                train_seqs.append(np.nan_to_num(v_seq, nan=0.0))
                train_labels.append(LABEL_MAP[v_label[target_col]])

        # Val data
        val_seq = np.load(os.path.join(features_dir, f"{vid}.npy")).astype(np.float32)
        val_seq = np.nan_to_num(val_seq, nan=0.0)
        val_label = LABEL_MAP[label_lookup[vid][target_col]]

        X_train = aggregate_dataset(train_seqs)
        y_train = np.array(train_labels)
        X_val = aggregate_sequence(val_seq).reshape(1, -1)

        model = get_classical_model(model_name, **model_kwargs)
        model.fit(X_train, y_train)

        pred = int(model.predict(X_val)[0])
        prob = model.predict_proba(X_val)[0].tolist()

        fold_result = {
            "fold": fold,
            "video_id": vid,
            "true_label": val_label,
            "predicted_label": pred,
            "probability_left": prob[0],
            "probability_right": prob[1],
            "correct": bool(pred == val_label),
        }
        results["folds"].append(fold_result)

        if verbose:
            marker = "OK" if pred == val_label else "MISS"
            print(f"Fold {fold:2d} ({vid}): true={INVERSE_LABEL_MAP[val_label]}, "
                  f"pred={INVERSE_LABEL_MAP[pred]}, prob=[{prob[0]:.3f}, {prob[1]:.3f}] [{marker}]")

    all_preds = np.array([f["predicted_label"] for f in results["folds"]])
    all_true = np.array([f["true_label"] for f in results["folds"]])
    results["accuracy"] = float(np.mean(all_preds == all_true))
    results["n_correct"] = int(np.sum(all_preds == all_true))
    results["n_total"] = len(all_true)

    if verbose:
        print(f"\nLOOCV accuracy: {results['accuracy']:.3f} "
              f"({results['n_correct']}/{results['n_total']})")

    return results