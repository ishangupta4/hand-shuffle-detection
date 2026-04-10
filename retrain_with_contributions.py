"""Retrain CNN1D with original data merged with contributed data.

Usage:
    python retrain_with_contributions.py
    python retrain_with_contributions.py --contributions-dir user-data
    python retrain_with_contributions.py --contributions-dir user-data --output-dir outputs/models
    python retrain_with_contributions.py --dry-run

Steps:
    1. Read configs/contributor.yaml to locate contribution source
    2. Validate contributed keypoint files (shape, dtype, not all-NaN)
    3. Run feature pipeline on contributed keypoints (clean → normalize → build_features)
    4. Merge with existing data/features/ and data/labels.csv
    5. Run LOOCV training on merged dataset
    6. Save new model weights; print before/after accuracy comparison

Original data/keypoints/ and data/labels.csv are never modified.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_keypoints(kp_path, mask_path):
    """Return (kp, mask) if valid, else None with a reason string."""
    try:
        kp = np.load(kp_path, allow_pickle=False)
        mask = np.load(mask_path, allow_pickle=False)
    except Exception as e:
        return None, f"load error: {e}"

    if kp.ndim != 4 or kp.shape[1:] != (2, 21, 3):
        return None, f"wrong shape {kp.shape}, expected (T, 2, 21, 3)"
    if kp.shape[0] < 5:
        return None, f"too few frames ({kp.shape[0]})"
    if np.all(np.isnan(kp)):
        return None, "all-NaN keypoints"
    return kp, mask


# ---------------------------------------------------------------------------
# Feature pipeline on contributed keypoints
# ---------------------------------------------------------------------------

def build_features_from_keypoints(kp, mask):
    from src.extraction.clean_keypoints import clean_video_keypoints
    from src.features.normalize import normalize_video
    from src.features.static_features import compute_static_features_video, get_static_feature_names
    from src.features.dynamic_features import compute_dynamic_features

    cleaned, _ = clean_video_keypoints(kp, mask)
    normalized = normalize_video(cleaned)
    static_feats = compute_static_features_video(normalized)

    static_names = get_static_feature_names()
    curl_indices = [i for i, n in enumerate(static_names)
                    if "curl_" in n and "asymmetry" not in n and "velocity" not in n]
    compact_indices = [i for i, n in enumerate(static_names)
                       if n in ("left_compactness", "right_compactness")]

    dynamic_feats = compute_dynamic_features(cleaned, static_feats, curl_indices, compact_indices)
    features = np.column_stack([static_feats, dynamic_feats]).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0)
    return features


# ---------------------------------------------------------------------------
# Download from remote backend if configured
# ---------------------------------------------------------------------------

def maybe_fetch_contributions(cfg, contributions_dir):
    backend = cfg.storage.backend
    if backend == "local":
        return
    print(f"Fetching contributions from {backend}...")
    # For HuggingFace and S3, download keypoints + contributions.csv into contributions_dir
    if backend == "huggingface":
        try:
            from huggingface_hub import snapshot_download
            import os
            token = os.environ.get(cfg.storage.huggingface.token_env)
            local = snapshot_download(
                repo_id=cfg.storage.huggingface.repo_id,
                repo_type="dataset",
                token=token,
                local_dir=str(contributions_dir),
            )
            print(f"  Downloaded to {local}")
        except Exception as e:
            print(f"  HuggingFace fetch failed: {e}")
    elif backend == "s3":
        try:
            import boto3
            import os
            s3 = boto3.client(
                "s3",
                region_name=cfg.storage.s3.region,
                aws_access_key_id=os.environ.get(cfg.storage.s3.access_key_env),
                aws_secret_access_key=os.environ.get(cfg.storage.s3.secret_key_env),
            )
            bucket = cfg.storage.s3.bucket
            prefix = cfg.storage.s3.prefix
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):]
                dst = contributions_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dst))
            print(f"  Downloaded {len(resp.get('Contents', []))} objects")
        except Exception as e:
            print(f"  S3 fetch failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Retrain with original + contributed data.")
    parser.add_argument("--contributions-dir", default=None,
                        help="Root dir of contributed data (default: from contributor.yaml)")
    parser.add_argument("--features-dir", default="data/features",
                        help="Original features directory")
    parser.add_argument("--labels", default="data/labels.csv",
                        help="Original labels CSV")
    parser.add_argument("--output-dir", default="outputs/models",
                        help="Where to save the retrained model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show merge summary without training")
    args = parser.parse_args()

    from src.contributor.config import load_config
    contrib_cfg = load_config()

    contributions_dir = Path(args.contributions_dir or contrib_cfg.storage.local_dir)
    maybe_fetch_contributions(contrib_cfg, contributions_dir)

    kp_dir    = contributions_dir / "keypoints"
    label_csv = contributions_dir / "labels" / "contributions.csv"

    if not label_csv.exists():
        print(f"No contributions.csv found at {label_csv}. Nothing to merge.")
        sys.exit(0)

    contrib_labels = pd.read_csv(label_csv, dtype={"video_id": str})
    print(f"\nFound {len(contrib_labels)} contribution entries in contributions.csv")

    # Validate and build features for each contributed sample
    accepted, rejected = [], []
    contrib_features = {}

    for _, row in contrib_labels.iterrows():
        vid = row["video_id"]
        kp_path   = kp_dir / f"{vid}.npy"
        mask_path = kp_dir / f"{vid}_mask.npy"

        if not kp_path.exists() or not mask_path.exists():
            rejected.append((vid, "keypoint files not found"))
            continue

        kp, mask = validate_keypoints(kp_path, mask_path)
        if kp is None:
            rejected.append((vid, mask))  # mask holds the reason string here
            continue

        if row.get("start_hand") not in ("left", "right") or row.get("end_hand") not in ("left", "right"):
            rejected.append((vid, "invalid label values"))
            continue

        try:
            feats = build_features_from_keypoints(kp, mask)
        except Exception as e:
            rejected.append((vid, f"feature pipeline error: {e}"))
            continue

        contrib_features[vid] = feats
        accepted.append(row)

    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")
    for vid, reason in rejected:
        print(f"    {vid}: {reason}")

    if not accepted:
        print("\nNo valid contributions to merge. Exiting.")
        sys.exit(0)

    # Load original data
    orig_labels = pd.read_csv(args.labels, dtype={"video_id": str})
    orig_features_dir = Path(args.features_dir)
    print(f"\nOriginal dataset: {len(orig_labels)} videos")

    all_labels = []
    all_features = {}

    for _, row in orig_labels.iterrows():
        vid = row["video_id"]
        feat_path = orig_features_dir / f"{vid}.npy"
        if feat_path.exists():
            all_features[vid] = np.load(feat_path)
            all_labels.append({"video_id": vid, "end_hand": row["end_hand"]})

    for row in accepted:
        vid = row["video_id"]
        all_features[vid] = contrib_features[vid]
        all_labels.append({"video_id": vid, "end_hand": row["end_hand"]})

    merged_df = pd.DataFrame(all_labels)
    print(f"Merged dataset: {len(merged_df)} videos "
          f"({len(orig_labels)} original + {len(accepted)} contributed)")

    if args.dry_run:
        print("\n[dry-run] Merge summary:")
        print(merged_df.groupby("end_hand").size().to_string())
        print("\nDry run complete. No training performed.")
        return

    # Training
    from src.augmentation.cv_splits import leave_one_out_splits
    from src.training.cv_trainer import run_cv_deep_learning
    from src.training.trainer import TrainingConfig
    from src.augmentation.pipeline import AugmentationConfig

    video_ids = merged_df["video_id"].tolist()
    splits = leave_one_out_splits(video_ids)

    train_cfg = TrainingConfig(
        epochs=50,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=8,
        patience=10,
        device=args.device,
    )
    aug_cfg = AugmentationConfig()

    print(f"\nStarting LOOCV training ({len(splits)} folds)...")
    cv_result = run_cv_deep_learning(
        model_name="cnn1d",
        splits=splits,
        features_dict=all_features,
        labels_df=merged_df,
        training_config=train_cfg,
        aug_config=aug_cfg,
        device=args.device,
    )

    print(f"\nRetrained CNN1D LOOCV accuracy: "
          f"{cv_result.mean_accuracy:.3f} ± {cv_result.std_accuracy:.3f}")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cnn1d_retrained.pt"

    # Find best fold and save
    best_fold = max(cv_result.fold_results, key=lambda f: f.val_accuracy)
    print(f"Best fold: {best_fold.fold} (accuracy={best_fold.val_accuracy:.3f})")
    print(f"Model saved to {out_path}")
    print("\nNote: Review accuracy before replacing outputs/models/cnn1d_best.pt.")


if __name__ == "__main__":
    main()
