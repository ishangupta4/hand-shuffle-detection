"""Hyperparameter search via random sampling over a config space.

Each sampled config runs full cross-validation. Results are logged
to CSV and ranked by mean CV accuracy.
"""

import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.augmentation.cv_splits import load_splits
from src.augmentation.pipeline import AugmentationConfig
from src.training.cv_trainer import (
    run_cv_deep_learning,
    run_cv_classical,
    CVResult,
)
from src.training.trainer import TrainingConfig


def load_search_space(path: str) -> dict:
    """Load hyperparameter search space from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def sample_config(search_space: dict, rng: np.random.Generator) -> dict:
    """Sample one hyperparameter configuration from the search space."""
    config = {}
    for param, values in search_space.items():
        if isinstance(values, list):
            config[param] = rng.choice(values)
        elif isinstance(values, dict):
            # Support min/max/step or other structured ranges
            if "min" in values and "max" in values:
                config[param] = rng.uniform(values["min"], values["max"])
            else:
                config[param] = values
        else:
            config[param] = values

    # Convert numpy types to native Python for YAML/JSON compat
    for k, v in config.items():
        if isinstance(v, (np.integer,)):
            config[k] = int(v)
        elif isinstance(v, (np.floating,)):
            config[k] = float(v)

    return config


def _build_model_kwargs(model_name: str, hp: dict, input_dim: int) -> dict:
    """Map sampled hyperparams to model constructor kwargs."""
    common = {"input_dim": input_dim, "num_classes": 2}

    if model_name == "bilstm":
        n_layers = int(hp.get("num_layers", 2))
        h_dim = int(hp.get("hidden_dim", 64))
        dims = [h_dim] if n_layers == 1 else [h_dim, h_dim // 2]
        return {
            **common,
            "hidden_dims": dims,
            "dropout_lstm": float(hp.get("dropout", 0.5)),
            "dropout_fc": float(hp.get("dropout", 0.5)) * 0.6,
        }
    elif model_name == "cnn1d":
        return {
            **common,
            "filters": [32, int(hp.get("hidden_dim", 64))],
            "dropout_conv": float(hp.get("dropout", 0.3)),
            "dropout_fc": float(hp.get("dropout", 0.3)),
        }
    elif model_name == "transformer":
        return {
            **common,
            "nhead": 4,
            "dim_feedforward": int(hp.get("hidden_dim", 64)),
            "num_layers": int(hp.get("num_layers", 2)),
            "dropout": float(hp.get("dropout", 0.3)),
        }
    else:
        return common


def run_hyperparameter_search(
    model_name: str,
    search_space: dict,
    splits: list[dict],
    features_dir: str,
    labels_df: pd.DataFrame,
    input_dim: int = 39,
    n_configs: int = 30,
    aug_config: AugmentationConfig | None = None,
    target_col: str = "end_hand",
    output_dir: str = "outputs/hyperparam_search",
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run random hyperparameter search.

    Args:
        model_name: One of 'bilstm', 'cnn1d', 'transformer', or a classical model name.
        search_space: Dict mapping param names to lists of values.
        splits: CV fold definitions.
        features_dir: Path to feature .npy files.
        labels_df: Labels DataFrame.
        input_dim: Number of input features.
        n_configs: Number of random configs to try.
        aug_config: Augmentation settings.
        target_col: Target label column.
        output_dir: Where to save results CSV.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        DataFrame with results sorted by mean accuracy (descending).
    """
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"{model_name}_search_results.csv")

    rng = np.random.default_rng(seed)
    is_classical = model_name in ("random_forest", "svm", "logistic_regression")

    all_results = []

    for i in range(n_configs):
        hp = sample_config(search_space, rng)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Config {i+1}/{n_configs}: {hp}")
            print(f"{'='*60}")

        try:
            if is_classical:
                cv_result = run_cv_classical(
                    model_name=model_name,
                    model_kwargs=hp,
                    splits=splits,
                    features_dir=features_dir,
                    labels_df=labels_df,
                    aug_config=aug_config,
                    target_col=target_col,
                    verbose=verbose,
                )
            else:
                model_kwargs = _build_model_kwargs(model_name, hp, input_dim)
                train_config = TrainingConfig(
                    lr=float(hp.get("lr", 1e-3)),
                    weight_decay=float(hp.get("weight_decay", 1e-3)),
                    batch_size=int(hp.get("batch_size", 8)),
                )

                cv_result = run_cv_deep_learning(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    splits=splits,
                    features_dir=features_dir,
                    labels_df=labels_df,
                    train_config=train_config,
                    aug_config=aug_config,
                    target_col=target_col,
                    verbose=verbose,
                )

            # Record results
            row = {
                "config_id": i,
                "model": model_name,
                "mean_accuracy": cv_result.mean_accuracy,
                "std_accuracy": cv_result.std_accuracy,
                "mean_loss": cv_result.mean_loss,
                "fold_variance": cv_result.std_accuracy ** 2,
                **{f"hp_{k}": v for k, v in hp.items()},
            }
            # Per-fold accuracies
            for fr in cv_result.fold_results:
                row[f"fold_{fr.fold}_acc"] = fr.val_accuracy

            all_results.append(row)

        except Exception as e:
            print(f"  Config {i+1} failed: {e}")
            all_results.append({
                "config_id": i,
                "model": model_name,
                "mean_accuracy": 0.0,
                "std_accuracy": 0.0,
                "mean_loss": float("inf"),
                "error": str(e),
                **{f"hp_{k}": v for k, v in hp.items()},
            })

        # Save incrementally
        df = pd.DataFrame(all_results)
        df.to_csv(results_path, index=False)

    # Final ranking
    df = pd.DataFrame(all_results)
    df = df.sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(results_path, index=False)

    if verbose:
        print(f"\n{'='*60}")
        print("SEARCH COMPLETE - Top 5 configs:")
        print(f"{'='*60}")
        print(rank_configs(df).head(5).to_string())

    return df


def rank_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Rank configs by mean accuracy, preferring simpler models when close.

    Flags configs with high fold-to-fold variance (>0.15 std).
    """
    df = df.copy()

    # Compute a complexity proxy from hyperparams
    complexity = np.zeros(len(df))
    if "hp_hidden_dim" in df.columns:
        complexity += df["hp_hidden_dim"].fillna(64).values / 128
    if "hp_num_layers" in df.columns:
        complexity += df["hp_num_layers"].fillna(1).values / 2
    if "hp_batch_size" in df.columns:
        # Smaller batch = more updates = effectively more complex training
        complexity += (1 - df["hp_batch_size"].fillna(8).values / 16) * 0.3

    df["complexity_score"] = complexity

    # Flag high variance
    df["high_variance"] = df["std_accuracy"] > 0.15

    # Sort: primary by accuracy (desc), secondary by complexity (asc) for ties
    df["rank_score"] = df["mean_accuracy"] - 0.01 * complexity
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df