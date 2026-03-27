"""Step 32 -- Temporal analysis of model decisions.

Reveals *when* in the sequence the model makes its decision:
1. Extract LSTM hidden states per timestep, project via PCA/t-SNE
2. Plot predicted probability over time (frame-by-frame) per video
3. Identify whether the model uses shuffle dynamics or just the reveal
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as torchF
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.augmentation.cv_splits import load_labels, leave_one_out_splits
from src.augmentation.dataset import HandShuffleDataset, LABEL_MAP
from src.augmentation.pipeline import AugmentationConfig
from src.augmentation.sequence_utils import pad_sequence
from src.models import get_dl_model
from src.training.trainer import TrainingConfig, train_model


INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def _extract_bilstm_hidden_states(model, x, lengths=None, mask=None):
    """Extract per-timestep hidden states from a BiLSTM.

    Returns: (T, hidden_dim*2) array of hidden states for the input.
    """
    model.eval()
    with torch.no_grad():
        out = x
        for lstm, drop in zip(model.lstm_layers, model.dropout_layers):
            if model.use_packing and lengths is not None:
                from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
                lengths_cpu = lengths.cpu().clamp(min=1)
                packed = pack_padded_sequence(out, lengths_cpu, batch_first=True, enforce_sorted=False)
                packed_out, _ = lstm(packed)
                out, _ = pad_packed_sequence(packed_out, batch_first=True)
            else:
                out, _ = lstm(out)
            # Skip dropout for analysis

    # out: (1, T, hidden*2)
    return out.squeeze(0).cpu().numpy()


def _frame_by_frame_probs(model, x, lengths=None, mask=None):
    """Get classification probabilities at each timestep.

    Feeds progressively longer prefixes of the sequence through the model,
    using the last hidden state at each prefix length.
    """
    model.eval()
    T = x.size(1)
    actual_len = int(lengths[0]) if lengths is not None else T
    probs_over_time = []

    with torch.no_grad():
        for t in range(1, actual_len + 1):
            x_prefix = x[:, :t, :]
            prefix_len = torch.tensor([t], dtype=torch.long)
            prefix_mask = torch.zeros(1, t)
            prefix_mask[0, :t] = 1.0

            logits = model(x_prefix, lengths=prefix_len, mask=prefix_mask)
            prob = torchF.softmax(logits, dim=1).cpu().numpy()[0]
            probs_over_time.append(prob)

    return np.array(probs_over_time)  # (actual_len, 2)


def temporal_hidden_state_analysis(
    model_name: str,
    model_kwargs: dict,
    train_config: TrainingConfig,
    features_dir: str = "data/features",
    labels_path: str = "data/labels.csv",
    aug_config: AugmentationConfig | None = None,
    target_col: str = "end_hand",
    output_dir: str = "outputs",
    verbose: bool = True,
) -> dict:
    """Extract hidden states from LOOCV folds and visualize via PCA/t-SNE.

    For each video (held out in its fold), we train on the rest, then
    extract hidden states from the held-out video.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels_df = load_labels(labels_path)
    video_ids = sorted(labels_df["video_id"].tolist())
    splits = leave_one_out_splits(video_ids)

    all_hidden = []  # list of (T_i, D) arrays
    all_labels = []  # per-video label
    all_vids = []
    all_timesteps = []  # normalized timestep for each hidden state

    for split in splits:
        fold = split["fold"]
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]
        vid = val_ids[0]

        if verbose:
            print(f"  Hidden states: fold {fold} ({vid})")

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

        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        model = get_dl_model(model_name, **model_kwargs)
        model, _ = train_model(model, train_loader, val_loader, train_config, verbose=False)

        # Extract hidden states for the held-out video
        sample = val_ds[0]
        x = sample["features"].unsqueeze(0)
        mask = sample["mask"].unsqueeze(0) if "mask" in sample else None
        length = torch.tensor([sample["length"]]) if "length" in sample else None
        label = sample["label"].item()

        if hasattr(model, "lstm_layers"):
            hidden = _extract_bilstm_hidden_states(model, x, lengths=length, mask=mask)
            actual_len = int(length[0]) if length is not None else hidden.shape[0]
            hidden = hidden[:actual_len]
        else:
            # For non-LSTM models, skip hidden state extraction
            continue

        all_hidden.append(hidden)
        all_labels.append(label)
        all_vids.append(vid)

        # Normalized timesteps for this video
        T = hidden.shape[0]
        all_timesteps.append(np.linspace(0, 1, T))

    if not all_hidden:
        return {"error": "No hidden states extracted (model may not be LSTM-based)"}

    # Concatenate all hidden states
    all_h_concat = np.concatenate(all_hidden, axis=0)

    # PCA to 2D
    pca = PCA(n_components=2)
    h_pca = pca.fit_transform(all_h_concat)

    # t-SNE to 2D (if enough points)
    if all_h_concat.shape[0] > 5:
        perplexity = min(30, all_h_concat.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        h_tsne = tsne.fit_transform(all_h_concat)
    else:
        h_tsne = h_pca

    # Build per-video index ranges
    offsets = np.cumsum([0] + [h.shape[0] for h in all_hidden])

    # --- Plot PCA ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for proj, ax, title in [
        (h_pca, axes[0], "PCA of Hidden States"),
        (h_tsne, axes[1], "t-SNE of Hidden States"),
    ]:
        for i, (vid, label) in enumerate(zip(all_vids, all_labels)):
            start, end = offsets[i], offsets[i + 1]
            pts = proj[start:end]
            times = all_timesteps[i]
            color = "blue" if label == 0 else "red"

            # Color gradient by timestep
            scatter = ax.scatter(
                pts[:, 0], pts[:, 1],
                c=times, cmap="viridis", s=8, alpha=0.5,
                edgecolors="none",
            )
            # Mark start and end
            ax.plot(pts[0, 0], pts[0, 1], "o", color=color, markersize=5, alpha=0.8)
            ax.plot(pts[-1, 0], pts[-1, 1], "s", color=color, markersize=5, alpha=0.8)

        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="blue", label="Left (start)", markersize=6, linestyle="None"),
        Line2D([0], [0], marker="s", color="blue", label="Left (end)", markersize=6, linestyle="None"),
        Line2D([0], [0], marker="o", color="red", label="Right (start)", markersize=6, linestyle="None"),
        Line2D([0], [0], marker="s", color="red", label="Right (end)", markersize=6, linestyle="None"),
    ]
    axes[1].legend(handles=legend_elements, loc="best", fontsize=7)
    plt.colorbar(scatter, ax=axes[1], label="Normalized Time", shrink=0.8)

    fig.suptitle("LSTM Hidden State Trajectories (colored by timestep)", fontsize=12)
    fig.tight_layout()
    hidden_plot = os.path.join(output_dir, f"{model_name}_hidden_state_trajectories.png")
    fig.savefig(hidden_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "hidden_state_plot": hidden_plot,
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
        "n_videos_analyzed": len(all_vids),
    }


def temporal_probability_analysis(
    model_name: str,
    model_kwargs: dict,
    train_config: TrainingConfig,
    features_dir: str = "data/features",
    labels_path: str = "data/labels.csv",
    aug_config: AugmentationConfig | None = None,
    target_col: str = "end_hand",
    output_dir: str = "outputs",
    n_examples: int = 6,
    verbose: bool = True,
) -> dict:
    """Plot frame-by-frame predicted probability for selected videos.

    For each held-out video in LOOCV, feeds progressively longer prefixes
    through the model and records the evolving prediction.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels_df = load_labels(labels_path)
    video_ids = sorted(labels_df["video_id"].tolist())
    splits = leave_one_out_splits(video_ids)

    prob_traces = {}

    for split in splits[:n_examples]:
        fold = split["fold"]
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]
        vid = val_ids[0]

        if verbose:
            print(f"  Probability trace: fold {fold} ({vid})")

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

        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        model = get_dl_model(model_name, **model_kwargs)
        model, _ = train_model(model, train_loader, val_loader, train_config, verbose=False)

        sample = val_ds[0]
        x = sample["features"].unsqueeze(0)
        mask = sample["mask"].unsqueeze(0) if "mask" in sample else None
        length = torch.tensor([sample["length"]]) if "length" in sample else None
        label = sample["label"].item()

        probs = _frame_by_frame_probs(model, x, lengths=length, mask=mask)

        prob_traces[vid] = {
            "probs": probs,  # (T, 2)
            "true_label": label,
            "actual_length": int(length[0]) if length is not None else probs.shape[0],
        }

    # Plot grid
    n_plots = len(prob_traces)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, (vid, data) in enumerate(prob_traces.items()):
        ax = axes_flat[i]
        probs = data["probs"]
        T = probs.shape[0]
        true_label = data["true_label"]
        true_name = INVERSE_LABEL_MAP[true_label]

        frames = np.arange(T)
        ax.plot(frames, probs[:, 1], color="red", alpha=0.8, label="P(right)")
        ax.plot(frames, probs[:, 0], color="blue", alpha=0.8, label="P(left)")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(frames, 0.5, probs[:, true_label],
                         where=probs[:, true_label] > 0.5,
                         alpha=0.15, color="green")
        ax.set_title(f"Video {vid} (true: {true_name})", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Probability")
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    # Hide unused axes
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Frame-by-Frame Prediction Probability", fontsize=13)
    fig.tight_layout()
    prob_plot = os.path.join(output_dir, f"{model_name}_temporal_probabilities.png")
    fig.savefig(prob_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "probability_plot": prob_plot,
        "n_videos_plotted": n_plots,
        "traces": {vid: {"true_label": d["true_label"], "length": d["actual_length"]}
                   for vid, d in prob_traces.items()},
    }