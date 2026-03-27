"""Train CNN1D with LOOCV and save the best fold's weights.

Usage:
    python train_and_save.py
    python train_and_save.py --features-dir data/features --labels data/labels.csv
    python train_and_save.py --output-dir outputs/models

The best fold model (highest val accuracy) is saved as:
    outputs/models/cnn1d_best.pt

The checkpoint includes model weights, architecture kwargs, and the
target_length used during training so the inference server can reconstruct
the model exactly.
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import torch

from src.augmentation.cv_splits import load_splits
from src.augmentation.pipeline import AugmentationConfig
from src.training.cv_trainer import run_cv_deep_learning
from src.training.trainer import TrainingConfig
from src.augmentation.dataset import HandShuffleDataset
from src.models import get_dl_model


def main():
    parser = argparse.ArgumentParser(description="Train CNN1D and save best weights.")
    parser.add_argument("--features-dir", default="../../data/features")
    parser.add_argument("--labels", default="../../data/labels.csv")
    parser.add_argument("--splits", default="../../data/splits_loocv.json")
    parser.add_argument("--output-dir", default="../../outputs/models")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels_df = pd.read_csv(args.labels)
    labels_df["video_id"] = labels_df["video_id"].apply(lambda v: f"{int(v):05d}")

    splits = load_splits(args.splits)

    model_kwargs = dict(input_dim=39, filters=[32, 64], kernel_sizes=[5, 3],
                        dropout_conv=0.3, dropout_fc=0.3)

    train_config = TrainingConfig(
        lr=1e-3, weight_decay=1e-3, batch_size=8,
        max_epochs=200, patience=20, device=args.device,
    )

    aug_config = AugmentationConfig()

    print("Running LOOCV training for CNN1D...")
    cv_result = run_cv_deep_learning(
        model_name="cnn1d",
        model_kwargs=model_kwargs,
        splits=splits,
        features_dir=args.features_dir,
        labels_df=labels_df,
        train_config=train_config,
        aug_config=aug_config,
        target_col="end_hand",
        verbose=True,
    )

    # Find best fold by val accuracy
    best_fold = max(cv_result.fold_results, key=lambda f: f.val_accuracy)
    print(f"\nBest fold: {best_fold.fold} (acc={best_fold.val_accuracy:.3f})")
    print(f"Mean LOOCV accuracy: {cv_result.mean_accuracy:.3f} +/- {cv_result.std_accuracy:.3f}")

    # Retrain on all-but-best-val to get a clean saved model
    # (Re-use the best fold's train split to produce the saved weights)
    best_split = next(s for s in splits if s["fold"] == best_fold.fold)
    train_ids = best_split["train_ids"]

    train_ds = HandShuffleDataset(
        video_ids=train_ids,
        features_dir=args.features_dir,
        labels_df=labels_df,
        augment=True,
        aug_config=aug_config,
        seq_mode="pad",
        target_col="end_hand",
    )
    target_length = train_ds.target_length

    from torch.utils.data import DataLoader
    from src.augmentation.dataset import HandShuffleDataset as DS

    # Full training set (all videos) for the final saved model
    all_ids = labels_df["video_id"].tolist()
    full_ds = DS(
        video_ids=all_ids,
        features_dir=args.features_dir,
        labels_df=labels_df,
        augment=True,
        aug_config=aug_config,
        seq_mode="pad",
        target_col="end_hand",
    )
    full_loader = DataLoader(full_ds, batch_size=train_config.batch_size, shuffle=True)

    # Use a small held-out set for early stopping (last 3 videos)
    val_ids = all_ids[-3:]
    train_ids_final = all_ids[:-3]
    train_ds_final = DS(
        video_ids=train_ids_final,
        features_dir=args.features_dir,
        labels_df=labels_df,
        augment=True,
        aug_config=aug_config,
        seq_mode="pad",
        target_col="end_hand",
    )
    val_ds_final = DS(
        video_ids=val_ids,
        features_dir=args.features_dir,
        labels_df=labels_df,
        augment=False,
        seq_mode="pad",
        target_length=train_ds_final.target_length,
        target_col="end_hand",
    )

    print("\nTraining final model on full dataset for deployment...")
    from src.training.trainer import train_model
    final_model = get_dl_model("cnn1d", **model_kwargs)
    t_loader = DataLoader(train_ds_final, batch_size=train_config.batch_size, shuffle=True)
    v_loader = DataLoader(val_ds_final, batch_size=len(val_ds_final), shuffle=False)
    final_model, history = train_model(final_model, t_loader, v_loader, train_config, verbose=True)

    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "model_kwargs": model_kwargs,
        "target_length": train_ds_final.target_length,
        "loocv_mean_accuracy": cv_result.mean_accuracy,
        "loocv_std_accuracy": cv_result.std_accuracy,
        "label_map": {"left": 0, "right": 1},
    }

    out_path = os.path.join(args.output_dir, "cnn1d_best.pt")
    torch.save(checkpoint, out_path)
    print(f"\nModel saved to {out_path}")

    # Also save metadata as JSON for the server to read without torch
    meta = {
        "model_kwargs": model_kwargs,
        "target_length": train_ds_final.target_length,
        "loocv_mean_accuracy": cv_result.mean_accuracy,
        "label_map": {"left": 0, "right": 1},
    }
    meta_path = os.path.join(args.output_dir, "cnn1d_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()