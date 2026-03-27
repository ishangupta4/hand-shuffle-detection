"""Cross-validation training wrapper.

Iterates over CV folds, augments only training data, trains from scratch
per fold, and collects per-fold metrics. Supports both deep learning
models (PyTorch) and classical ML baselines (scikit-learn).
"""

import os
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.augmentation.cv_splits import load_splits
from src.augmentation.dataset import HandShuffleDataset, LABEL_MAP
from src.augmentation.pipeline import AugmentationConfig
from src.models import get_dl_model, DL_MODEL_REGISTRY
from src.models.classical import aggregate_dataset, get_classical_model
from src.training.trainer import TrainingConfig, train_model, evaluate


@dataclass
class FoldResult:
    """Results from a single CV fold."""
    fold: int
    val_ids: list[str]
    val_accuracy: float
    val_loss: float
    predictions: list[int]
    true_labels: list[int]
    best_epoch: int = 0
    elapsed_sec: float = 0.0


@dataclass
class CVResult:
    """Aggregated cross-validation results."""
    model_name: str
    fold_results: list[FoldResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    @property
    def mean_accuracy(self) -> float:
        return np.mean([f.val_accuracy for f in self.fold_results])

    @property
    def std_accuracy(self) -> float:
        return np.std([f.val_accuracy for f in self.fold_results])

    @property
    def mean_loss(self) -> float:
        return np.mean([f.val_loss for f in self.fold_results])

    @property
    def all_predictions(self) -> np.ndarray:
        return np.concatenate([f.predictions for f in self.fold_results])

    @property
    def all_true_labels(self) -> np.ndarray:
        return np.concatenate([f.true_labels for f in self.fold_results])

    def summary(self) -> str:
        accs = [f.val_accuracy for f in self.fold_results]
        lines = [
            f"Model: {self.model_name}",
            f"Folds: {len(self.fold_results)}",
            f"Accuracy: {self.mean_accuracy:.3f} +/- {self.std_accuracy:.3f}",
            f"Per-fold: {[f'{a:.3f}' for a in accs]}",
            f"Mean loss: {self.mean_loss:.4f}",
        ]
        return "\n".join(lines)


def run_cv_deep_learning(
    model_name: str,
    model_kwargs: dict,
    splits: list[dict],
    features_dir: str,
    labels_df: pd.DataFrame,
    train_config: TrainingConfig,
    aug_config: AugmentationConfig | None = None,
    target_col: str = "end_hand",
    verbose: bool = True,
) -> CVResult:
    """Run cross-validation for a deep learning model.

    For each fold:
    1. Create train dataset with augmentation
    2. Create val dataset without augmentation
    3. Initialize fresh model
    4. Train with early stopping
    5. Evaluate on held-out val set
    """
    cv_result = CVResult(
        model_name=model_name,
        config={"model": model_kwargs, "training": vars(train_config)},
    )

    for split in splits:
        fold = split["fold"]
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]

        if verbose:
            print(f"\n--- Fold {fold} (val: {val_ids}) ---")

        # Create datasets
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
            train_ds, batch_size=train_config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=len(val_ds), shuffle=False
        )

        # Fresh model for each fold
        model = get_dl_model(model_name, **model_kwargs)

        model, history = train_model(
            model, train_loader, val_loader, train_config, verbose=verbose
        )

        # Final evaluation
        device = torch.device(train_config.device)
        criterion = torch.nn.CrossEntropyLoss()
        val_loss, val_acc, preds, labels = evaluate(
            model, val_loader, criterion, device
        )

        fold_result = FoldResult(
            fold=fold,
            val_ids=val_ids,
            val_accuracy=val_acc,
            val_loss=val_loss,
            predictions=preds.tolist(),
            true_labels=labels.tolist(),
            best_epoch=history.best_epoch,
            elapsed_sec=history.elapsed_sec,
        )
        cv_result.fold_results.append(fold_result)

        if verbose:
            print(f"  Fold {fold} result: acc={val_acc:.3f}, loss={val_loss:.4f}")

    if verbose:
        print(f"\n{cv_result.summary()}")

    return cv_result


def run_cv_classical(
    model_name: str,
    model_kwargs: dict,
    splits: list[dict],
    features_dir: str,
    labels_df: pd.DataFrame,
    aug_config: AugmentationConfig | None = None,
    target_col: str = "end_hand",
    verbose: bool = True,
) -> CVResult:
    """Run cross-validation for a classical ML model.

    Augmentation is applied to training data by generating augmented
    sequences, then aggregating each to a fixed-length vector.
    """
    from src.augmentation.pipeline import augment_single

    cv_result = CVResult(
        model_name=model_name,
        config={"model": model_kwargs},
    )

    if aug_config is None:
        aug_config = AugmentationConfig()

    label_lookup = labels_df.set_index("video_id").to_dict("index")

    # Helper to normalize video IDs
    def norm_vid(v):
        return f"{int(v):05d}"

    for split in splits:
        fold = split["fold"]
        train_ids = [norm_vid(v) for v in split["train_ids"]]
        val_ids = [norm_vid(v) for v in split["val_ids"]]

        if verbose:
            print(f"\n--- Fold {fold} (val: {val_ids}) ---")

        rng = np.random.default_rng(42 + fold)

        # Load and augment training data
        train_seqs = []
        train_labels = []
        for vid in train_ids:
            seq = np.load(os.path.join(features_dir, f"{vid}.npy")).astype(np.float32)
            seq = np.nan_to_num(seq, nan=0.0)
            row = label_lookup[vid]
            label_dict = {"start_hand": row["start_hand"], "end_hand": row["end_hand"]}

            variants = augment_single(seq, label_dict, aug_config, rng=rng)
            for v_seq, v_label, _ in variants:
                train_seqs.append(np.nan_to_num(v_seq, nan=0.0))
                if target_col == "switched":
                    train_labels.append(int(v_label["start_hand"] != v_label["end_hand"]))
                else:
                    train_labels.append(LABEL_MAP[v_label[target_col]])

        # Load val data (no augmentation)
        val_seqs = []
        val_labels = []
        for vid in val_ids:
            seq = np.load(os.path.join(features_dir, f"{vid}.npy")).astype(np.float32)
            seq = np.nan_to_num(seq, nan=0.0)
            val_seqs.append(seq)
            row = label_lookup[vid]
            if target_col == "switched":
                val_labels.append(int(row["start_hand"] != row["end_hand"]))
            else:
                val_labels.append(LABEL_MAP[row[target_col]])

        # Aggregate to fixed-length vectors
        X_train = aggregate_dataset(train_seqs)
        y_train = np.array(train_labels)
        X_val = aggregate_dataset(val_seqs)
        y_val = np.array(val_labels)

        # Train
        model = get_classical_model(model_name, **model_kwargs)
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)
        # Use negative log-likelihood as "loss" for comparability
        from sklearn.metrics import log_loss
        try:
            val_loss = log_loss(y_val, probs, labels=[0, 1])
        except Exception:
            val_loss = float("inf")

        val_acc = np.mean(preds == y_val)

        fold_result = FoldResult(
            fold=fold,
            val_ids=val_ids,
            val_accuracy=val_acc,
            val_loss=val_loss,
            predictions=preds.tolist(),
            true_labels=y_val.tolist(),
        )
        cv_result.fold_results.append(fold_result)

        if verbose:
            print(f"  Fold {fold}: acc={val_acc:.3f}, loss={val_loss:.4f}")

    if verbose:
        print(f"\n{cv_result.summary()}")

    return cv_result