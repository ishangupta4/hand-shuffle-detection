"""Data augmentation and cross-validation for hand-shuffle detection."""

from src.augmentation.augmentations import (
    horizontal_flip,
    time_warp,
    gaussian_jitter,
    slight_rotation,
    feature_dropout,
    AUGMENTATION_REGISTRY,
)
from src.augmentation.pipeline import (
    AugmentationConfig,
    augment_single,
    augment_dataset,
    estimate_augmentation_factor,
)
from src.augmentation.sequence_utils import (
    pad_sequence,
    pad_batch,
    resample_sequence,
    resample_batch,
    sort_by_length,
    get_lengths,
)
from src.augmentation.cv_splits import (
    leave_one_out_splits,
    stratified_kfold_splits,
    save_splits,
    load_splits,
)
try:
    from src.augmentation.dataset import (
        HandShuffleDataset,
        collate_variable_length,
    )
except ImportError:
    pass  # torch not installed