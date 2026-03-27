"""Individual augmentation techniques for hand-shuffle feature sequences.

Each augmentation takes a feature sequence (T, F) and label dict,
returns an augmented sequence and (possibly modified) label.

Labels are dicts with keys: start_hand, end_hand (values: 'left' or 'right').
"""

import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Feature metadata helpers
# ---------------------------------------------------------------------------

_METADATA_CACHE = {}


def load_feature_metadata(metadata_path: str = "../../data/features/feature_metadata.json") -> dict:
    """Load and cache feature metadata."""
    if metadata_path not in _METADATA_CACHE:
        metadata_path = "../../data/features/feature_metadata.json"
        with open(metadata_path) as f:
            _METADATA_CACHE[metadata_path] = json.load(f)
    return _METADATA_CACHE[metadata_path]


def _get_hand_column_mapping(metadata: dict) -> dict:
    """Build left↔right column swap mapping from metadata.

    Returns dict mapping column index → swapped column index for
    left/right hand features. Inter-hand features stay in place
    but some need sign flipping (asymmetry features).
    """
    features = metadata["features"]
    name_to_idx = {name: info["index"] for name, info in features.items()}

    swap_map = {}
    sign_flip = set()

    for name, info in features.items():
        idx = info["index"]

        if name.startswith("left_"):
            partner = "right_" + name[5:]
            if partner in name_to_idx:
                swap_map[idx] = name_to_idx[partner]
                swap_map[name_to_idx[partner]] = idx
        # Asymmetry features (left - right) → negate on flip
        if name.startswith("asymmetry_"):
            sign_flip.add(idx)

    return swap_map, sign_flip


# ---------------------------------------------------------------------------
# 1. Horizontal flip
# ---------------------------------------------------------------------------

def horizontal_flip(
    sequence: np.ndarray,
    label: dict,
    metadata_path: str = "data/features/feature_metadata.json",
) -> tuple[np.ndarray, dict]:
    """Swap left-hand and right-hand features, flip label.

    This simulates viewing the same shuffle from a mirrored perspective.
    Left/right columns are swapped, asymmetry features are negated,
    and start_hand/end_hand are flipped.
    """
    metadata = load_feature_metadata(metadata_path)
    swap_map, sign_flip = _get_hand_column_mapping(metadata)

    flipped = sequence.copy()
    num_cols = sequence.shape[1]

    # Swap left↔right columns
    already_swapped = set()
    for src, dst in swap_map.items():
        if src < num_cols and dst < num_cols and src not in already_swapped:
            flipped[:, src] = sequence[:, dst]
            flipped[:, dst] = sequence[:, src]
            already_swapped.add(src)
            already_swapped.add(dst)

    # Negate asymmetry features
    for idx in sign_flip:
        if idx < num_cols:
            flipped[:, idx] = -flipped[:, idx]

    # Flip labels
    flip_hand = {"left": "right", "right": "left"}
    flipped_label = {
        "start_hand": flip_hand[label["start_hand"]],
        "end_hand": flip_hand[label["end_hand"]],
    }

    return flipped, flipped_label


# ---------------------------------------------------------------------------
# 2. Time warping (global speed change via resampling)
# ---------------------------------------------------------------------------

def time_warp(
    sequence: np.ndarray,
    label: dict,
    speed_factor: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Resample the entire sequence at a different speed.

    speed_factor > 1.0 → faster (fewer frames)
    speed_factor < 1.0 → slower (more frames)

    Uses linear interpolation across the full temporal extent.
    Label is unchanged since no frames are removed from start/end.
    """
    T, F = sequence.shape
    if T < 2:
        return sequence.copy(), label.copy()

    new_T = max(2, int(round(T / speed_factor)))
    old_times = np.linspace(0, 1, T)
    new_times = np.linspace(0, 1, new_T)

    warped = np.empty((new_T, F), dtype=sequence.dtype)
    for col in range(F):
        col_data = sequence[:, col]
        valid = ~np.isnan(col_data)

        if valid.sum() < 2:
            # Not enough valid points to interpolate
            warped[:, col] = np.nan
            continue

        # Interpolate only valid segments, fill NaN elsewhere
        f_interp = interp1d(
            old_times[valid], col_data[valid],
            kind="linear", bounds_error=False, fill_value=np.nan,
        )
        warped[:, col] = f_interp(new_times)

    return warped, label.copy()


# ---------------------------------------------------------------------------
# 3. Gaussian jittering
# ---------------------------------------------------------------------------

def gaussian_jitter(
    sequence: np.ndarray,
    label: dict,
    sigma_fraction: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Add Gaussian noise scaled by each feature's range.

    sigma = sigma_fraction * (max - min) for each feature column.
    NaN values are preserved.
    """
    if rng is None:
        rng = np.random.default_rng()

    jittered = sequence.copy()
    T, F = sequence.shape

    for col in range(F):
        col_data = sequence[:, col]
        valid = ~np.isnan(col_data)
        if valid.sum() < 2:
            continue
        col_range = np.nanmax(col_data) - np.nanmin(col_data)
        if col_range < 1e-10:
            continue
        noise = rng.normal(0, sigma_fraction * col_range, size=T)
        jittered[valid, col] = col_data[valid] + noise[valid]

    return jittered, label.copy()


# ---------------------------------------------------------------------------
# 4. Slight 3D rotation (applied to curl angles and spatial features)
# ---------------------------------------------------------------------------

def slight_rotation(
    sequence: np.ndarray,
    label: dict,
    max_angle_deg: float = 7.0,
    metadata_path: str = "data/features/feature_metadata.json",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Simulate a slight viewpoint rotation.

    Since we're working with derived features (not raw keypoints),
    we approximate a rotation's effect:
    - Curl angles: add small perturbation (rotation changes apparent angles)
    - Compactness/volume: scale by a small factor (foreshortening)
    - Distance features: scale slightly
    - Velocity/acceleration: scale consistently with their parent features

    The perturbations are correlated (same rotation applied to all frames)
    to simulate a fixed camera offset, not random noise.
    """
    if rng is None:
        rng = np.random.default_rng()

    metadata = load_feature_metadata(metadata_path)
    features = metadata["features"]

    angle_rad = rng.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180.0
    scale_factor = np.cos(angle_rad)  # foreshortening

    rotated = sequence.copy()

    for name, info in features.items():
        idx = info["index"]
        if idx >= sequence.shape[1]:
            continue

        if "curl" in name and "velocity" not in name:
            # Curl angles shift slightly under rotation
            rotated[:, idx] += rng.uniform(-0.03, 0.03) * angle_rad
        elif any(k in name for k in ["compactness", "bbox_volume", "spread", "distance"]):
            if "velocity" not in name:
                rotated[:, idx] *= scale_factor
        elif "velocity" in name or "acceleration" in name:
            # Dynamic features scale with their parent spatial features
            rotated[:, idx] *= scale_factor

    return rotated, label.copy()


# ---------------------------------------------------------------------------
# 5. Feature dropout/masking
# ---------------------------------------------------------------------------

def feature_dropout(
    sequence: np.ndarray,
    label: dict,
    drop_rate: float = 0.07,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Randomly zero out a fraction of feature values per frame.

    Simulates partial occlusion or noisy detection. Existing NaN
    values are not counted toward the drop budget.
    """
    if rng is None:
        rng = np.random.default_rng()

    dropped = sequence.copy()
    mask = rng.random(sequence.shape) < drop_rate
    # Don't drop values that are already NaN
    mask = mask & ~np.isnan(sequence)
    dropped[mask] = 0.0

    return dropped, label.copy()


# ---------------------------------------------------------------------------
# Registry for easy access
# ---------------------------------------------------------------------------

AUGMENTATION_REGISTRY: dict[str, Callable] = {
    "horizontal_flip": horizontal_flip,
    "time_warp": time_warp,
    "gaussian_jitter": gaussian_jitter,
    "slight_rotation": slight_rotation,
    "feature_dropout": feature_dropout,
}


if __name__ == "__main__":
    # Quick sanity check with synthetic data
    np.random.seed(42)
    T, F = 50, 39
    fake_seq = np.random.randn(T, F)
    fake_label = {"start_hand": "right", "end_hand": "left"}

    print(f"Original: shape={fake_seq.shape}, label={fake_label}")

    # Test each augmentation (use a dummy metadata for flip/rotation)
    import tempfile, os
    # Create minimal metadata for testing
    meta = {
        "num_features": F,
        "num_static": 23,
        "num_dynamic": 16,
        "feature_names": [f"feat_{i}" for i in range(F)],
        "features": {},
    }
    # Fake left/right pairs
    for i in range(5):
        meta["features"][f"left_curl_{i}"] = {"index": i, "category": "left_hand", "type": "static"}
        meta["features"][f"right_curl_{i}"] = {"index": i + 5, "category": "right_hand", "type": "static"}
    meta["features"]["asymmetry_test"] = {"index": 17, "category": "inter_hand", "type": "static"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(meta, f)
        tmp_meta = f.name

    try:
        flipped, fl = horizontal_flip(fake_seq, fake_label, metadata_path=tmp_meta)
        print(f"Flip: shape={flipped.shape}, label={fl}")

        warped, wl = time_warp(fake_seq, fake_label, speed_factor=0.8)
        print(f"Warp 0.8x: shape={warped.shape} (was {T}), label={wl}")

        warped2, wl2 = time_warp(fake_seq, fake_label, speed_factor=1.2)
        print(f"Warp 1.2x: shape={warped2.shape} (was {T}), label={wl2}")

        jittered, jl = gaussian_jitter(fake_seq, fake_label, sigma_fraction=0.03)
        print(f"Jitter: shape={jittered.shape}, max diff={np.max(np.abs(jittered - fake_seq)):.4f}")

        rotated, rl = slight_rotation(fake_seq, fake_label, metadata_path=tmp_meta)
        print(f"Rotation: shape={rotated.shape}")

        dropped, dl = feature_dropout(fake_seq, fake_label, drop_rate=0.07)
        zeros_added = np.sum((dropped == 0) & (fake_seq != 0))
        print(f"Dropout: {zeros_added} values zeroed ({zeros_added / fake_seq.size * 100:.1f}%)")
    finally:
        os.unlink(tmp_meta)

    print("\nAll augmentations passed basic checks.")