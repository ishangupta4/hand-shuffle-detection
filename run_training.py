#!/usr/bin/env python3
"""Main training runner for hand-shuffle AI.

Usage:
    # Single model cross-validation
    python run_training.py --model bilstm --cv-type 5fold

    # Hyperparameter search
    python run_training.py --model bilstm --search --n-configs 30

    # Classical baseline
    python run_training.py --model random_forest --cv-type 5fold

    # All baselines
    python run_training.py --all-baselines --cv-type 5fold

    # Use LOOCV instead of 5-fold
    python run_training.py --model bilstm --cv-type loocv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

from src.augmentation.cv_splits import load_splits, load_labels
from src.augmentation.pipeline import AugmentationConfig
from src.models import DL_MODEL_REGISTRY
from src.models.classical import CLASSICAL_MODELS
from src.training.trainer import TrainingConfig
from src.training.cv_trainer import run_cv_deep_learning, run_cv_classical
from src.training.hyperparam_search import (
    run_hyperparameter_search,
    load_search_space,
    rank_configs,
)


def load_project_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_input_dim(features_dir: str) -> int:
    """Infer input dimension from the first feature file."""
    for fname in sorted(os.listdir(features_dir)):
        if fname.endswith(".npy"):
            arr = np.load(os.path.join(features_dir, fname))
            return arr.shape[1]
    raise FileNotFoundError(f"No .npy files in {features_dir}")


def main():
    parser = argparse.ArgumentParser(description="Hand-shuffle AI training runner")
    parser.add_argument(
        "--model", type=str, default="bilstm",
        choices=list(DL_MODEL_REGISTRY) + list(CLASSICAL_MODELS),
        help="Model architecture to train",
    )
    parser.add_argument(
        "--cv-type", type=str, default="5fold",
        choices=["5fold", "loocv"],
        help="Cross-validation strategy",
    )
    parser.add_argument(
        "--search", action="store_true",
        help="Run hyperparameter search instead of single config",
    )
    parser.add_argument(
        "--n-configs", type=int, default=30,
        help="Number of hyperparameter configs to sample (with --search)",
    )
    parser.add_argument(
        "--all-baselines", action="store_true",
        help="Run all classical baselines",
    )
    parser.add_argument(
        "--target", type=str, default="end_hand",
        choices=["end_hand", "switched"],
        help="Prediction target",
    )
    parser.add_argument(
        "--features-dir", type=str, default="data/features",
        help="Path to feature .npy files",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Project config path",
    )
    parser.add_argument(
        "--search-space", type=str, default="configs/search_space.yaml",
        help="Hyperparameter search space config",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    # Load data
    labels_df = load_labels("data/labels.csv")
    input_dim = get_input_dim(args.features_dir)

    if verbose:
        print(f"Features dir: {args.features_dir}")
        print(f"Input dim: {input_dim}")
        print(f"Videos: {len(labels_df)}")
        print(f"Label distribution: {labels_df['end_hand'].value_counts().to_dict()}")

    # Load CV splits
    splits_path = f"data/splits_{'loocv' if args.cv_type == 'loocv' else '5fold'}.json"
    splits = load_splits(splits_path)
    if verbose:
        print(f"CV strategy: {args.cv_type} ({len(splits)} folds)")

    # Augmentation config
    aug_config = AugmentationConfig()

    # Run all classical baselines
    if args.all_baselines:
        print("\n" + "=" * 60)
        print("Running all classical baselines")
        print("=" * 60)

        results = []
        for name in CLASSICAL_MODELS:
            print(f"\n--- {name} ---")
            cv_result = run_cv_classical(
                model_name=name,
                model_kwargs={},
                splits=splits,
                features_dir=args.features_dir,
                labels_df=labels_df,
                aug_config=aug_config,
                target_col=args.target,
                verbose=verbose,
            )
            results.append({
                "model": name,
                "mean_acc": cv_result.mean_accuracy,
                "std_acc": cv_result.std_accuracy,
                "mean_loss": cv_result.mean_loss,
            })

        summary = pd.DataFrame(results).sort_values("mean_acc", ascending=False)
        print("\n" + "=" * 60)
        print("Classical Baseline Summary:")
        print(summary.to_string(index=False))
        os.makedirs("outputs", exist_ok=True)
        summary.to_csv("outputs/classical_baselines.csv", index=False)
        return

    # Hyperparameter search
    if args.search:
        search_config = load_search_space(args.search_space)
        is_classical = args.model in CLASSICAL_MODELS

        if is_classical:
            space = search_config.get(args.model, {})
        else:
            space = search_config.get("deep_learning", {})

        print(f"\nRunning hyperparameter search for {args.model}")
        print(f"Search space: {space}")
        print(f"Configs to try: {args.n_configs}")

        df = run_hyperparameter_search(
            model_name=args.model,
            search_space=space,
            splits=splits,
            features_dir=args.features_dir,
            labels_df=labels_df,
            input_dim=input_dim,
            n_configs=args.n_configs,
            aug_config=aug_config,
            target_col=args.target,
            output_dir="outputs/hyperparam_search",
            seed=args.seed,
            verbose=verbose,
        )

        ranked = rank_configs(df)
        print("\nFinal rankings:")
        print(ranked[["rank", "model", "mean_accuracy", "std_accuracy",
                       "high_variance"]].head(10).to_string())
        return

    # Single model training
    is_classical = args.model in CLASSICAL_MODELS

    if is_classical:
        cv_result = run_cv_classical(
            model_name=args.model,
            model_kwargs={},
            splits=splits,
            features_dir=args.features_dir,
            labels_df=labels_df,
            aug_config=aug_config,
            target_col=args.target,
            verbose=verbose,
        )
    else:
        # Build model kwargs from CLI args
        if args.model == "bilstm":
            dims = [args.hidden_dim] if args.num_layers == 1 else [args.hidden_dim, args.hidden_dim // 2]
            model_kwargs = {
                "input_dim": input_dim,
                "hidden_dims": dims,
                "dropout_lstm": args.dropout,
                "dropout_fc": args.dropout * 0.6,
            }
        elif args.model == "cnn1d":
            model_kwargs = {
                "input_dim": input_dim,
                "filters": [32, args.hidden_dim],
                "dropout_conv": args.dropout,
                "dropout_fc": args.dropout,
            }
        elif args.model == "transformer":
            model_kwargs = {
                "input_dim": input_dim,
                "nhead": 4,
                "dim_feedforward": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            }
        else:
            model_kwargs = {"input_dim": input_dim}

        train_config = TrainingConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
        )

        cv_result = run_cv_deep_learning(
            model_name=args.model,
            model_kwargs=model_kwargs,
            splits=splits,
            features_dir=args.features_dir,
            labels_df=labels_df,
            train_config=train_config,
            aug_config=aug_config,
            target_col=args.target,
            verbose=verbose,
        )

    # Save results
    os.makedirs("outputs", exist_ok=True)
    result_path = f"outputs/training_results/{args.model}_{args.cv_type}_results.json"
    with open(result_path, "w") as f:
        json.dump({
            "model": args.model,
            "cv_type": args.cv_type,
            "mean_accuracy": cv_result.mean_accuracy,
            "std_accuracy": cv_result.std_accuracy,
            "mean_loss": cv_result.mean_loss,
            "folds": [
                {
                    "fold": fr.fold,
                    "val_ids": fr.val_ids,
                    "accuracy": fr.val_accuracy,
                    "loss": fr.val_loss,
                    "predictions": fr.predictions,
                    "true_labels": fr.true_labels,
                }
                for fr in cv_result.fold_results
            ],
        }, f, indent=2)

    print(f"\nResults saved to {result_path}")
    print(f"\nFinal: {cv_result.model_name} accuracy = "
          f"{cv_result.mean_accuracy:.3f} +/- {cv_result.std_accuracy:.3f}")


if __name__ == "__main__":
    main()