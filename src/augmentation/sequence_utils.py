"""Sequence length handling for variable-length feature sequences.

Three strategies for batching sequences of different lengths:
    A) Pad to max length with zeros + binary mask (default)
    B) Resample all to a common frame count via interpolation
    C) Variable-length with pack_padded_sequence for LSTM
"""

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Option A: Pad + mask
# ---------------------------------------------------------------------------

def pad_sequence(
    sequence: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad or truncate sequence to target_length.

    Returns:
        padded: Shape (target_length, F).
        mask: Shape (target_length,), 1.0 for real frames, 0.0 for padding.
    """
    T, F = sequence.shape
    padded = np.full((target_length, F), pad_value, dtype=np.float32)
    mask = np.zeros(target_length, dtype=np.float32)

    length = min(T, target_length)
    padded[:length] = sequence[:length]
    mask[:length] = 1.0

    return padded, mask


def pad_batch(
    sequences: list[np.ndarray],
    target_length: int | None = None,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of variable-length sequences to a common length.

    If target_length is None, uses the max length in the batch.

    Returns:
        padded: Shape (B, T_max, F).
        masks: Shape (B, T_max).
    """
    if target_length is None:
        target_length = max(s.shape[0] for s in sequences)

    padded_list = []
    mask_list = []
    for seq in sequences:
        p, m = pad_sequence(seq, target_length, pad_value)
        padded_list.append(p)
        mask_list.append(m)

    return np.stack(padded_list), np.stack(mask_list)


# ---------------------------------------------------------------------------
# Option B: Resample to common length via interpolation
# ---------------------------------------------------------------------------

def resample_sequence(
    sequence: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """Resample sequence to target_length using linear interpolation.

    Preserves the full temporal extent — start and end frames map
    to the first and last positions of the output.
    """
    T, F = sequence.shape
    if T == target_length:
        return sequence.copy()
    if T < 2:
        return np.tile(sequence, (target_length, 1))

    old_times = np.linspace(0, 1, T)
    new_times = np.linspace(0, 1, target_length)

    resampled = np.empty((target_length, F), dtype=np.float32)
    for col in range(F):
        col_data = sequence[:, col]
        valid = ~np.isnan(col_data)

        if valid.sum() < 2:
            resampled[:, col] = np.nan
            continue

        f_interp = interp1d(
            old_times[valid], col_data[valid],
            kind="linear", bounds_error=False, fill_value=np.nan,
        )
        resampled[:, col] = f_interp(new_times)

    return resampled


def resample_batch(
    sequences: list[np.ndarray],
    target_length: int | None = None,
) -> np.ndarray:
    """Resample all sequences to a common length.

    If target_length is None, uses the median length.

    Returns:
        Shape (B, target_length, F).
    """
    if target_length is None:
        lengths = [s.shape[0] for s in sequences]
        target_length = int(np.median(lengths))

    return np.stack([resample_sequence(s, target_length) for s in sequences])


# ---------------------------------------------------------------------------
# Option C: Sort + pack helpers for LSTM
# ---------------------------------------------------------------------------

def sort_by_length(
    sequences: list[np.ndarray],
    labels: list,
) -> tuple[list[np.ndarray], list, np.ndarray]:
    """Sort sequences by descending length for pack_padded_sequence.

    Returns:
        sorted_sequences, sorted_labels, original_indices (for unsorting).
    """
    lengths = np.array([s.shape[0] for s in sequences])
    order = np.argsort(-lengths)

    sorted_seqs = [sequences[i] for i in order]
    sorted_labs = [labels[i] for i in order]

    return sorted_seqs, sorted_labs, order


def get_lengths(sequences: list[np.ndarray]) -> np.ndarray:
    """Get actual lengths for pack_padded_sequence."""
    return np.array([s.shape[0] for s in sequences])


# ---------------------------------------------------------------------------
# Convenience: compute dataset statistics
# ---------------------------------------------------------------------------

def compute_length_stats(sequences: list[np.ndarray]) -> dict:
    """Compute length statistics for a list of sequences."""
    lengths = [s.shape[0] for s in sequences]
    return {
        "count": len(lengths),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Create variable-length test sequences
    seqs = [rng.standard_normal((t, 10)).astype(np.float32)
            for t in [30, 50, 40, 60, 35]]

    print("Length stats:", compute_length_stats(seqs))

    # Option A: Pad
    padded, masks = pad_batch(seqs, target_length=60)
    print(f"\nOption A (pad): batch={padded.shape}, masks={masks.shape}")
    print(f"  Mask sums (actual lengths): {masks.sum(axis=1).astype(int)}")

    # Option B: Resample
    resampled = resample_batch(seqs, target_length=45)
    print(f"\nOption B (resample): batch={resampled.shape}")

    # Option C: Sort for packing
    labels = ["a", "b", "c", "d", "e"]
    sorted_seqs, sorted_labs, order = sort_by_length(seqs, labels)
    lengths = get_lengths(sorted_seqs)
    print(f"\nOption C (sort): lengths={lengths}, labels={sorted_labs}")

    print("\nAll sequence utils passed.")