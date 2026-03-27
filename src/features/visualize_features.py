"""Visualization suite for hand shuffle feature analysis.

Generates a comprehensive set of plots for understanding the
extracted features and their relationship to labels.

Subcommands:
    timeseries   - Per-video feature time series (14a)
    skeleton     - Hand skeleton overlay on frames (14b)
    curl_heatmap - Finger curl angle heatmaps (14c)
    compactness  - Fist compactness comparison (14d)
    interhand    - Inter-hand distance over time (14e)
    boxplots     - Feature distribution by label (14f)
    correlation  - Feature correlation matrix (14g)
    scatter      - PCA/t-SNE scatter plot (14h)
    all          - Run all visualizations
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize as MplNormalize

# MediaPipe hand connections for skeleton drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm connections
]

LABEL_COLORS = {"left": "#2563eb", "right": "#dc2626"}


def load_labels(labels_path: str) -> pd.DataFrame:
    """Load labels CSV and add derived 'switched' column."""
    df = pd.read_csv(labels_path, dtype={"video_id": str})
    df["switched"] = df["start_hand"] != df["end_hand"]
    return df


def _get_label_color(label: str) -> str:
    return LABEL_COLORS.get(label, "#666666")


# --- 14a: Feature Time Series ---

def plot_feature_timeseries(
    features_dir: str,
    metadata_path: str,
    labels_df: pd.DataFrame,
    output_dir: str,
):
    """Multi-panel time series of key features per video."""
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_path) as f:
        meta = json.load(f)

    # Select a representative subset of features to plot
    key_features = [
        "left_compactness", "right_compactness",
        "inter_hand_distance",
        "left_wrist_velocity", "right_wrist_velocity",
        "left_curl_index", "right_curl_index",
        "asymmetry_compactness",
    ]

    # Filter to features that exist
    available = [n for n in key_features if n in meta["features"]]
    if not available:
        print("  No key features found in metadata, skipping timeseries.")
        return

    indices = [meta["features"][n]["index"] for n in available]

    feat_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])

    for feat_file in feat_files:
        video_id = feat_file.replace(".npy", "")
        feats = np.load(os.path.join(features_dir, feat_file))

        label_row = labels_df[labels_df["video_id"] == video_id]
        end_hand = label_row["end_hand"].values[0] if len(label_row) > 0 else "unknown"
        color = _get_label_color(end_hand)

        n_panels = len(available)
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.2 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(f"Video {video_id} | end_hand={end_hand}", fontsize=13, y=1.01)

        for ax, name, idx in zip(axes, available, indices):
            data = feats[:, idx]
            ax.plot(data, color=color, linewidth=0.8, alpha=0.85)
            ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", labelpad=60)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel("Frame")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_timeseries.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved time series plots to {output_dir}/")


# --- 14b: Hand Skeleton Overlay ---

def plot_skeleton_grid(
    keypoints_dir: str,
    output_dir: str,
    sample_videos: list[str] | None = None,
    frame_step: int = 10,
    max_frames: int = 8,
):
    """Render hand skeletons as image grids (every Nth frame)."""
    os.makedirs(output_dir, exist_ok=True)

    kp_files = sorted([
        f for f in os.listdir(keypoints_dir)
        if f.endswith(".npy")
        and not f.endswith("_mask.npy")
        and not f.endswith("_meta.npy")
    ])

    if sample_videos:
        kp_files = [f for f in kp_files if f.replace(".npy", "") in sample_videos]

    if not kp_files:
        kp_files = sorted([
            f for f in os.listdir(keypoints_dir)
            if f.endswith(".npy")
            and not f.endswith("_mask.npy")
            and not f.endswith("_meta.npy")
        ])[:3]

    for kp_file in kp_files:
        video_id = kp_file.replace(".npy", "")
        keypoints = np.load(os.path.join(keypoints_dir, kp_file))
        T = keypoints.shape[0]

        frame_indices = list(range(0, T, frame_step))[:max_frames]
        n_frames = len(frame_indices)

        if n_frames == 0:
            continue

        cols = min(4, n_frames)
        rows = (n_frames + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        fig.suptitle(f"Hand Skeletons: {video_id}", fontsize=12)

        for idx, frame_idx in enumerate(frame_indices):
            r, c = divmod(idx, cols)
            ax = axes[r, c]

            for h, (hand_name, color) in enumerate([("Left", "#2563eb"), ("Right", "#dc2626")]):
                hand_kp = keypoints[frame_idx, h]

                if np.any(np.isnan(hand_kp)):
                    continue

                # Plot bones
                for i, j in HAND_CONNECTIONS:
                    ax.plot(
                        [hand_kp[i, 0], hand_kp[j, 0]],
                        [hand_kp[i, 1], hand_kp[j, 1]],
                        color=color, linewidth=1.2, alpha=0.7,
                    )

                # Plot keypoints
                ax.scatter(
                    hand_kp[:, 0], hand_kp[:, 1],
                    c=color, s=8, zorder=5, alpha=0.8,
                )

            ax.set_title(f"Frame {frame_idx}", fontsize=8)
            ax.set_aspect("equal")
            ax.invert_yaxis()  # Image convention: y increases downward
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.15)

        # Hide unused subplots
        for idx in range(n_frames, rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_skeleton_grid.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved skeleton grids to {output_dir}/")


# --- 14c: Finger Curl Angle Heatmap ---

def plot_curl_heatmaps(
    features_dir: str,
    metadata_path: str,
    output_dir: str,
):
    """Heatmap of finger curl angles over time per video."""
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_path) as f:
        meta = json.load(f)

    curl_names = [n for n in meta["feature_names"]
                  if "curl_" in n and "velocity" not in n and "asymmetry" not in n]
    curl_indices = [meta["features"][n]["index"] for n in curl_names]

    if not curl_indices:
        print("  No curl features found, skipping heatmaps.")
        return

    feat_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])

    for feat_file in feat_files:
        video_id = feat_file.replace(".npy", "")
        feats = np.load(os.path.join(features_dir, feat_file))

        curl_data = feats[:, curl_indices].T  # (10, T)

        fig, ax = plt.subplots(figsize=(14, 4))
        im = ax.imshow(curl_data, aspect="auto", cmap="viridis",
                        interpolation="nearest")
        ax.set_yticks(range(len(curl_names)))
        ax.set_yticklabels([n.replace("_curl_", " ") for n in curl_names], fontsize=7)
        ax.set_xlabel("Frame")
        ax.set_title(f"Finger Curl Angles: {video_id}")
        plt.colorbar(im, ax=ax, label="Angle (rad)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_curl_heatmap.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved curl heatmaps to {output_dir}/")


# --- 14d: Fist Compactness Comparison ---

def plot_compactness(
    features_dir: str,
    metadata_path: str,
    labels_df: pd.DataFrame,
    output_dir: str,
):
    """Left vs right compactness on same chart, with reveal markers."""
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_path) as f:
        meta = json.load(f)

    left_idx = meta["features"].get("left_compactness", {}).get("index")
    right_idx = meta["features"].get("right_compactness", {}).get("index")

    if left_idx is None or right_idx is None:
        print("  Compactness features not found, skipping.")
        return

    feat_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])

    for feat_file in feat_files:
        video_id = feat_file.replace(".npy", "")
        feats = np.load(os.path.join(features_dir, feat_file))
        T = feats.shape[0]

        label_row = labels_df[labels_df["video_id"] == video_id]
        end_hand = label_row["end_hand"].values[0] if len(label_row) > 0 else "?"
        start_hand = label_row["start_hand"].values[0] if len(label_row) > 0 else "?"

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(feats[:, left_idx], color="#2563eb", linewidth=0.9, label="Left hand", alpha=0.85)
        ax.plot(feats[:, right_idx], color="#dc2626", linewidth=0.9, label="Right hand", alpha=0.85)

        # Mark approximate start/end reveal regions
        ax.axvline(x=5, color="green", linestyle="--", alpha=0.5, label=f"Start reveal (~{start_hand})")
        ax.axvline(x=T - 5, color="orange", linestyle="--", alpha=0.5, label=f"End reveal (~{end_hand})")

        ax.set_xlabel("Frame")
        ax.set_ylabel("Compactness")
        ax.set_title(f"Fist Compactness: {video_id} | {start_hand} -> {end_hand}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_compactness.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved compactness plots to {output_dir}/")


# --- 14e: Inter-Hand Distance ---

def plot_interhand_distance(
    features_dir: str,
    metadata_path: str,
    labels_df: pd.DataFrame,
    output_dir: str,
):
    """Inter-hand distance over time per video."""
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_path) as f:
        meta = json.load(f)

    dist_idx = meta["features"].get("inter_hand_distance", {}).get("index")
    if dist_idx is None:
        print("  inter_hand_distance not found, skipping.")
        return

    feat_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])

    for feat_file in feat_files:
        video_id = feat_file.replace(".npy", "")
        feats = np.load(os.path.join(features_dir, feat_file))

        label_row = labels_df[labels_df["video_id"] == video_id]
        end_hand = label_row["end_hand"].values[0] if len(label_row) > 0 else "?"
        color = _get_label_color(end_hand)

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(feats[:, dist_idx], color=color, linewidth=0.9, alpha=0.85)
        ax.fill_between(range(feats.shape[0]), feats[:, dist_idx],
                         alpha=0.15, color=color)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Distance")
        ax.set_title(f"Inter-Hand Distance: {video_id} | end_hand={end_hand}")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_interhand_dist.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved inter-hand distance plots to {output_dir}/")


# --- 14f: Feature Distribution Boxplots ---

def plot_feature_boxplots(
    features_dir: str,
    metadata_path: str,
    labels_df: pd.DataFrame,
    output_dir: str,
):
    """Boxplots comparing feature distributions by end_hand label."""
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_path) as f:
        meta = json.load(f)

    feat_names = meta["feature_names"]
    num_features = meta["num_features"]

    # Compute mean feature value per video
    rows = []
    feat_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])

    for feat_file in feat_files:
        video_id = feat_file.replace(".npy", "")
        feats = np.load(os.path.join(features_dir, feat_file))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_feats = np.nanmean(feats, axis=0)

        label_row = labels_df[labels_df["video_id"] == video_id]
        if len(label_row) == 0:
            continue

        row = {"video_id": video_id, "end_hand": label_row["end_hand"].values[0]}
        for i, name in enumerate(feat_names):
            row[name] = mean_feats[i] if i < len(mean_feats) else np.nan
        rows.append(row)

    if not rows:
        print("  No data for boxplots.")
        return

    df = pd.DataFrame(rows)

    # Plot in batches of 8 features
    batch_size = 8
    for batch_start in range(0, num_features, batch_size):
        batch_end = min(batch_start + batch_size, num_features)
        batch_names = feat_names[batch_start:batch_end]
        n = len(batch_names)

        fig, axes = plt.subplots(1, n, figsize=(3 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, name in zip(axes, batch_names):
            left_vals = df[df["end_hand"] == "left"][name].dropna().values
            right_vals = df[df["end_hand"] == "right"][name].dropna().values

            bp = ax.boxplot(
                [left_vals, right_vals],
                labels=["left", "right"],
                patch_artist=True,
                widths=0.6,
            )
            bp["boxes"][0].set_facecolor("#93c5fd")
            bp["boxes"][1].set_facecolor("#fca5a5")

            ax.set_title(name, fontsize=7, rotation=15, ha="right")
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2, axis="y")

        plt.suptitle(f"Feature Distributions by Label (features {batch_start}-{batch_end - 1})",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"boxplots_batch_{batch_start:03d}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    print(f"  Saved boxplots to {output_dir}/")


# --- 14g: Correlation Matrix ---

def plot_correlation_matrix(
    features_dir: str,
    metadata_path: str,
    output_dir: str,
    threshold: float = 0.9,
):
    """Pearson correlation across all features, flagging high correlations."""
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_path) as f:
        meta = json.load(f)

    feat_names = meta["feature_names"]

    # Collect mean features per video
    all_means = []
    feat_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])

    for feat_file in feat_files:
        feats = np.load(os.path.join(features_dir, feat_file))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            all_means.append(np.nanmean(feats, axis=0))

    if not all_means:
        print("  No data for correlation matrix.")
        return

    data = np.array(all_means)

    # Compute correlation, handling NaN columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.DataFrame(data, columns=feat_names)
        corr = df.corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feat_names)))
    ax.set_yticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, fontsize=5, rotation=90)
    ax.set_yticklabels(feat_names, fontsize=5)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Flag highly correlated pairs
    high_corr = []
    n = len(feat_names)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr.values[i, j]
            if not np.isnan(r) and abs(r) > threshold:
                high_corr.append((feat_names[i], feat_names[j], r))

    if high_corr:
        high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
        report_path = os.path.join(output_dir, "high_correlations.txt")
        with open(report_path, "w") as f:
            f.write(f"Feature pairs with |r| > {threshold}:\n\n")
            for f1, f2, r in high_corr:
                f.write(f"  {f1}  <->  {f2}:  r = {r:.4f}\n")
        print(f"  Found {len(high_corr)} highly correlated pairs (|r| > {threshold})")

    print(f"  Saved correlation matrix to {output_dir}/")


# --- 14h: PCA / t-SNE Scatter ---

def plot_pca_tsne(
    features_dir: str,
    metadata_path: str,
    labels_df: pd.DataFrame,
    output_dir: str,
):
    """PCA and t-SNE scatter plots of mean feature vectors colored by label."""
    os.makedirs(output_dir, exist_ok=True)
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    with open(metadata_path) as f:
        meta = json.load(f)

    # Build per-video mean feature vectors
    video_ids = []
    mean_feats = []
    labels = []

    feat_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])

    for feat_file in feat_files:
        video_id = feat_file.replace(".npy", "")
        feats = np.load(os.path.join(features_dir, feat_file))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_vec = np.nanmean(feats, axis=0)

        label_row = labels_df[labels_df["video_id"] == video_id]
        if len(label_row) == 0:
            continue

        video_ids.append(video_id)
        mean_feats.append(mean_vec)
        labels.append(label_row["end_hand"].values[0])

    print(f"  Matched {len(mean_feats)} videos with labels")

    if len(mean_feats) < 3:
        print("  Not enough videos for PCA/t-SNE (need at least 3).")
        return

    X = np.array(mean_feats)

    # Replace any remaining NaN with column mean
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0

    # Drop zero-variance columns (cause NaN after scaling)
    variances = np.var(X, axis=0)
    keep_cols = variances > 1e-10
    X = X[:, keep_cols]
    print(f"  Kept {np.sum(keep_cols)}/{len(keep_cols)} features (dropped zero-variance)")

    if X.shape[1] < 2:
        print("  Not enough non-constant features for PCA/t-SNE.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    n_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for label in ["left", "right"]:
        label_mask = [l == label for l in labels]
        color = _get_label_color(label)
        ax1.scatter(X_pca[label_mask, 0], X_pca[label_mask, 1], c=color,
                    label=label, s=80, alpha=0.8, edgecolors="white", linewidth=0.5)
        for i, m in enumerate(label_mask):
            if m:
                ax1.annotate(video_ids[i], (X_pca[i, 0], X_pca[i, 1]),
                           fontsize=6, alpha=0.7)

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax1.set_title("PCA")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # t-SNE
    perplexity = min(5, len(X_scaled) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    for label in ["left", "right"]:
        label_mask = [l == label for l in labels]
        color = _get_label_color(label)
        ax2.scatter(X_tsne[label_mask, 0], X_tsne[label_mask, 1], c=color,
                    label=label, s=80, alpha=0.8, edgecolors="white", linewidth=0.5)
        for i, m in enumerate(label_mask):
            if m:
                ax2.annotate(video_ids[i], (X_tsne[i, 0], X_tsne[i, 1]),
                           fontsize=6, alpha=0.7)

    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_title(f"t-SNE (perplexity={perplexity})")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.suptitle("Dimensionality Reduction: Mean Features by end_hand Label", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_tsne_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved PCA/t-SNE scatter to {output_dir}/")


# --- Main with subcommands ---

def main():
    parser = argparse.ArgumentParser(
        description="Feature visualization suite for hand shuffle analysis."
    )
    parser.add_argument(
        "command",
        choices=["timeseries", "skeleton", "curl_heatmap", "compactness",
                 "interhand", "boxplots", "correlation", "scatter", "all"],
        help="Which visualization to generate.",
    )
    parser.add_argument(
        "--features-dir", default="data/features",
        help="Directory with assembled feature .npy files.",
    )
    parser.add_argument(
        "--keypoints-dir", default="data/keypoints_cleaned",
        help="Directory with cleaned keypoint .npy files.",
    )
    parser.add_argument(
        "--labels", default="data/labels.csv",
        help="Path to labels CSV.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/feature_visualizations",
        help="Base output directory for plots.",
    )
    parser.add_argument(
        "--sample-videos", nargs="*", default=None,
        help="Specific video IDs for skeleton overlay (default: first 3).",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.features_dir, "feature_metadata.json")
    labels_df = load_labels(args.labels)

    commands = {
        "timeseries": lambda: plot_feature_timeseries(
            args.features_dir, metadata_path, labels_df,
            os.path.join(args.output_dir, "timeseries"),
        ),
        "skeleton": lambda: plot_skeleton_grid(
            args.keypoints_dir,
            os.path.join(args.output_dir, "skeleton"),
            sample_videos=args.sample_videos,
        ),
        "curl_heatmap": lambda: plot_curl_heatmaps(
            args.features_dir, metadata_path,
            os.path.join(args.output_dir, "curl_heatmaps"),
        ),
        "compactness": lambda: plot_compactness(
            args.features_dir, metadata_path, labels_df,
            os.path.join(args.output_dir, "compactness"),
        ),
        "interhand": lambda: plot_interhand_distance(
            args.features_dir, metadata_path, labels_df,
            os.path.join(args.output_dir, "interhand_distance"),
        ),
        "boxplots": lambda: plot_feature_boxplots(
            args.features_dir, metadata_path, labels_df,
            os.path.join(args.output_dir, "boxplots"),
        ),
        "correlation": lambda: plot_correlation_matrix(
            args.features_dir, metadata_path,
            os.path.join(args.output_dir, "correlation"),
        ),
        "scatter": lambda: plot_pca_tsne(
            args.features_dir, metadata_path, labels_df,
            os.path.join(args.output_dir, "scatter"),
        ),
    }

    if args.command == "all":
        for name, func in commands.items():
            print(f"\n[{name}]")
            func()
    else:
        commands[args.command]()

    print("\nDone.")


if __name__ == "__main__":
    main()