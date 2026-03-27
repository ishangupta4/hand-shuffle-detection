"""Model architectures for hand-shuffle classification."""

from src.models.bilstm import BiLSTMClassifier
from src.models.cnn1d import CNN1DClassifier
from src.models.transformer import TransformerClassifier
from src.models.classical import (
    aggregate_sequence,
    aggregate_dataset,
    get_classical_model,
    CLASSICAL_MODELS,
)

# Registry for deep learning models
DL_MODEL_REGISTRY = {
    "bilstm": BiLSTMClassifier,
    "cnn1d": CNN1DClassifier,
    "transformer": TransformerClassifier,
}


def get_dl_model(name: str, **kwargs):
    """Create a deep learning model by name."""
    if name not in DL_MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(DL_MODEL_REGISTRY)}")
    return DL_MODEL_REGISTRY[name](**kwargs)