"""Step 31 -- Feature importance analysis.

Three approaches:
1. Random Forest built-in importances (Gini / impurity-based)
2. Ablation study: remove one feature group, measure accuracy drop
3. Permutation importance: shuffle one feature, measure degradation
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.augmentation.cv_splits import load_labels, leave_one_out_splits
from src.augmentation.dataset import HandShuffleDataset, LABEL_MAP
from src.augmentation.pipeline import AugmentationConfig, augment_single
from src.models import get_dl_model
from src.models.classical import aggregate_dataset, aggregate_sequence, get_classical_model
from src.training.trainer import TrainingConfig, train_model


def load_feature_metadata(path: str = "data/features/feature_metadata.json") -> dict:
    with open(path) as f:
        return json.load(f)


# -----------------------------------------------------------------------
# 1. Random Forest feature importances
# -----------------------------------------------------------------------

def rf_feature_importances(
    features_dir: str = "data/features",
    labels_path: str = "data/labels.csv",
    metadata_path: str = "data/features/feature_metadata.json",
    aug_config: AugmentationConfig | None = None,
    target_col: str = "end_hand",
    output_dir: str = "outputs",
) -> dict:
    """Train RF on full dataset (with augmentation) and extract importances.

    Since aggregation creates 4 stats per feature (mean, std, min, max),
    we sum importances across these to get per-original-feature scores.
    """
    meta = load_feature_metadata(metadata_path)
    feature_names = meta["feature_names"]
    n_features = meta["num_features"]

    labels_df = load_labels(labels_path)
    video_ids = sorted(labels_df["video_id"].tolist())
    label_lookup = labels_df.set_index("video_id").to_dict("index")

    if aug_config is None:
        aug_config = AugmentationConfig()

    rng = np.random.default_rng(42)
    all_seqs, all_labels = [], []

    for vid in video_ids:
        seq = np.load(os.path.join(features_dir, f"{vid}.npy")).astype(np.float32)
        seq = np.nan_to_num(seq, nan=0.0)
        row = label_lookup[vid]
        ld = {"start_hand": row["start_hand"], "end_hand": row["end_hand"]}
        variants = augment_single(seq, ld, aug_config, rng=rng)
        for v_seq, v_label, _ in variants:
            all_seqs.append(np.nan_to_num(v_seq, nan=0.0))
            all_labels.append(LABEL_MAP[v_label[target_col]])

    X = aggregate_dataset(all_seqs)
    y = np.array(all_labels)

    rf = get_classical_model("random_forest", n_estimators=200)
    rf.fit(X, y)

    raw_importances = rf.named_steps["clf"].feature_importances_

    # Aggregate: 4 stats per feature
    agg_importances = np.zeros(n_features)
    agg_names = ["mean", "std", "min", "max"]
    for stat_idx in range(4):
        start = stat_idx * n_features
        end = start + n_features
        agg_importances += raw_importances[start:end]

    # Rank
    order = np.argsort(agg_importances)[::-1]
    ranked = [(feature_names[i], float(agg_importances[i])) for i in order]

    # Plot
    plot_path = os.path.join(output_dir, "rf_feature_importances.png")
    _plot_importances(ranked[:20], plot_path, "Random Forest Feature Importances (Top 20)")

    return {
        "ranked_features": ranked,
        "plot_path": plot_path,
        "raw_importances_shape": raw_importances.shape,
    }


# -----------------------------------------------------------------------
# 2. Ablation study: remove feature groups
# -----------------------------------------------------------------------

def ablation_study(
    model_name: str,
    model_kwargs: dict,
    train_config: TrainingConfig,
    features_dir: str = "data/features",
    labels_path: str = "data/labels.csv",
    metadata_path: str = "data/features/feature_metadata.json",
    aug_config: AugmentationConfig | None = None,
    target_col: str = "end_hand",
    output_dir: str = "outputs",
    verbose: bool = True,
) -> dict:
    """Remove one feature group at a time, re-run 5-fold CV, compare accuracy.

    Feature groups: left_hand, right_hand, inter_hand, dynamic, static.
    """
    meta = load_feature_metadata(metadata_path)
    feature_names = meta["feature_names"]
    n_features = meta["num_features"]

    # Define groups by category and type
    groups = {
        "left_hand": [],
        "right_hand": [],
        "inter_hand": [],
        "static_only": [],
        "dynamic_only": [],
    }
    for fname, finfo in meta["features"].items():
        idx = finfo["index"]
        cat = finfo["category"]
        ftype = finfo["type"]
        if cat == "left_hand":
            groups["left_hand"].append(idx)
        elif cat == "right_hand":
            groups["right_hand"].append(idx)
        elif cat == "inter_hand":
            groups["inter_hand"].append(idx)
        if ftype == "static":
            groups["static_only"].append(idx)
        elif ftype == "dynamic":
            groups["dynamic_only"].append(idx)

    labels_df = load_labels(labels_path)
    video_ids = sorted(labels_df["video_id"].tolist())

    # 5-fold splits (faster than LOOCV for ablation)
    from src.augmentation.cv_splits import stratified_kfold_splits
    splits = stratified_kfold_splits(video_ids, labels_df, n_folds=5)

    # Baseline accuracy (all features)
    baseline_acc = _run_quick_cv(
        model_name, model_kwargs, train_config, splits,
        features_dir, labels_df, aug_config, target_col,
        feature_mask=None, n_features=n_features, verbose=False,
    )
    if verbose:
        print(f"Baseline accuracy (all features): {baseline_acc:.3f}")

    results = {"baseline_accuracy": baseline_acc, "ablations": []}

    for group_name, indices in groups.items():
        if not indices:
            continue
        # Create mask: True = keep, False = drop
        mask = np.ones(n_features, dtype=bool)
        mask[indices] = False

        acc = _run_quick_cv(
            model_name, model_kwargs, train_config, splits,
            features_dir, labels_df, aug_config, target_col,
            feature_mask=mask, n_features=n_features, verbose=False,
        )
        drop = baseline_acc - acc
        results["ablations"].append({
            "removed_group": group_name,
            "n_features_removed": len(indices),
            "accuracy_without": acc,
            "accuracy_drop": drop,
        })
        if verbose:
            direction = "dropped" if drop > 0 else "improved" if drop < 0 else "unchanged"
            print(f"  Without {group_name:15s} ({len(indices):2d} feats): "
                  f"acc={acc:.3f} ({direction} by {abs(drop):.3f})")

    # Sort by importance (largest drop = most important)
    results["ablations"].sort(key=lambda x: x["accuracy_drop"], reverse=True)

    # Plot
    plot_path = os.path.join(output_dir, f"{model_name}_ablation_study.png")
    _plot_ablation(results, plot_path, model_name)
    results["plot_path"] = plot_path

    return results


def _run_quick_cv(
    model_name, model_kwargs, train_config, splits,
    features_dir, labels_df, aug_config, target_col,
    feature_mask, n_features, verbose=False,
):
    """Run quick 5-fold CV, optionally masking out features."""
    all_correct = 0
    all_total = 0

    # Adjust model input_dim if masking
    mkw = dict(model_kwargs)
    if feature_mask is not None:
        mkw["input_dim"] = int(feature_mask.sum())

    for split in splits:
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]

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

        # Apply feature mask if needed
        if feature_mask is not None:
            mask_t = torch.tensor(feature_mask)
            _apply_feature_mask(train_ds, mask_t)
            _apply_feature_mask(val_ds, mask_t)

        train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        model = get_dl_model(model_name, **mkw)
        model, _ = train_model(model, train_loader, val_loader, train_config, verbose=False)

        device = torch.device(train_config.device)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                if feature_mask is not None:
                    features = features[:, :, mask_t]
                labels = batch["label"]
                mask = batch.get("mask")
                lengths = batch.get("length")
                if mask is not None:
                    mask = mask.to(device)
                if lengths is not None:
                    lengths = lengths.to(device)

                logits = model(features, lengths=lengths, mask=mask)
                preds = logits.argmax(dim=1).cpu()
                all_correct += (preds == labels).sum().item()
                all_total += labels.size(0)

    return all_correct / max(all_total, 1)


def _apply_feature_mask(dataset, mask_tensor):
    """Apply feature mask to pre-loaded sequences in a dataset."""
    for i in range(len(dataset.sequences)):
        dataset.sequences[i] = dataset.sequences[i][:, mask_tensor.numpy().astype(bool)]
    # Also mask augmented cache
    new_cache = []
    for seq, label, desc in dataset._aug_cache:
        new_cache.append((seq[:, mask_tensor.numpy().astype(bool)], label, desc))
    dataset._aug_cache = new_cache


# -----------------------------------------------------------------------
# 3. Permutation importance
# -----------------------------------------------------------------------

def permutation_importance(
    loocv_results: dict,
    features_dir: str = "data/features",
    labels_path: str = "data/labels.csv",
    metadata_path: str = "data/features/feature_metadata.json",
    n_repeats: int = 10,
    output_dir: str = "outputs",
    verbose: bool = True,
) -> dict:
    """Permutation importance using classical RF on aggregated features.

    For each feature: shuffle its column, predict, measure accuracy drop.
    Repeat n_repeats times and average.
    """
    meta = load_feature_metadata(metadata_path)
    feature_names = meta["feature_names"]
    n_features = meta["num_features"]

    labels_df = load_labels(labels_path)
    video_ids = sorted(labels_df["video_id"].tolist())
    label_lookup = labels_df.set_index("video_id").to_dict("index")

    # Load all sequences and aggregate
    seqs = []
    labels = []
    for vid in video_ids:
        seq = np.load(os.path.join(features_dir, f"{vid}.npy")).astype(np.float32)
        seq = np.nan_to_num(seq, nan=0.0)
        seqs.append(seq)
        labels.append(LABEL_MAP[label_lookup[vid]["end_hand"]])

    X = aggregate_dataset(seqs)
    y = np.array(labels)

    # Train RF on full data
    rf = get_classical_model("random_forest", n_estimators=200)
    rf.fit(X, y)
    baseline_acc = float(np.mean(rf.predict(X) == y))

    rng = np.random.default_rng(42)

    # For each original feature, we shuffle all 4 aggregation columns (mean, std, min, max)
    importance_scores = {}
    for feat_idx in range(n_features):
        cols = [feat_idx + stat * n_features for stat in range(4)]
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            for col in cols:
                rng.shuffle(X_perm[:, col])
            acc_perm = float(np.mean(rf.predict(X_perm) == y))
            drops.append(baseline_acc - acc_perm)
        importance_scores[feature_names[feat_idx]] = {
            "mean_drop": float(np.mean(drops)),
            "std_drop": float(np.std(drops)),
        }

    # Rank by mean drop
    ranked = sorted(importance_scores.items(), key=lambda x: x[1]["mean_drop"], reverse=True)

    if verbose:
        print(f"Permutation importance (baseline acc: {baseline_acc:.3f}):")
        for name, scores in ranked[:10]:
            print(f"  {name:35s}  drop={scores['mean_drop']:+.4f} +/- {scores['std_drop']:.4f}")

    # Plot top 20
    plot_path = os.path.join(output_dir, "permutation_importance.png")
    top_20 = ranked[:20]
    names = [r[0] for r in top_20]
    means = [r[1]["mean_drop"] for r in top_20]
    stds = [r[1]["std_drop"] for r in top_20]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, xerr=stds, color="#2196F3", alpha=0.8,
            edgecolor="white", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Accuracy Drop")
    ax.set_title("Permutation Importance (Top 20 Features)")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "baseline_accuracy": baseline_acc,
        "ranked_features": [(name, scores) for name, scores in ranked],
        "plot_path": plot_path,
    }


# -----------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------

def _plot_importances(ranked_features, output_path, title):
    """Horizontal bar chart of feature importances."""
    names = [r[0] for r in ranked_features]
    values = [r[1] for r in ranked_features]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="#FF9800", alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ablation(results, output_path, model_name):
    """Bar chart of accuracy drop when removing feature groups."""
    ablations = results["ablations"]
    names = [a["removed_group"] for a in ablations]
    drops = [a["accuracy_drop"] for a in ablations]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#F44336" if d > 0 else "#4CAF50" for d in drops]
    ax.barh(range(len(names)), drops, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Accuracy Drop (positive = feature group matters)")
    ax.set_title(f"{model_name} Ablation Study -- Feature Group Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)