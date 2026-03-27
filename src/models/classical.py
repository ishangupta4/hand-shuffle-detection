"""Classical ML baselines for hand-shuffle classification.

These models operate on fixed-length feature vectors computed by
aggregating temporal sequences: mean, std, min, max of each feature
over time → one vector per video.

Models: Random Forest, SVM (RBF kernel), Logistic Regression.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def aggregate_sequence(sequence: np.ndarray) -> np.ndarray:
    """Collapse a (T, F) temporal sequence into a (4*F,) feature vector.

    Computes mean, std, min, max across time for each feature.
    NaN values are ignored in aggregation.
    """
    with np.errstate(all="ignore"):
        agg_mean = np.nanmean(sequence, axis=0)
        agg_std = np.nanstd(sequence, axis=0)
        agg_min = np.nanmin(sequence, axis=0)
        agg_max = np.nanmax(sequence, axis=0)

    vec = np.concatenate([agg_mean, agg_std, agg_min, agg_max])
    return np.nan_to_num(vec, nan=0.0).astype(np.float32)


def aggregate_dataset(
    sequences: list[np.ndarray],
) -> np.ndarray:
    """Aggregate a list of sequences into a (N, 4*F) feature matrix."""
    return np.stack([aggregate_sequence(s) for s in sequences])


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

CLASSICAL_MODELS = {
    "random_forest": lambda **kw: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=kw.get("n_estimators", 100),
            max_depth=kw.get("max_depth", None),
            random_state=kw.get("random_state", 42),
            class_weight="balanced",
        )),
    ]),
    "svm": lambda **kw: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=kw.get("C", 1.0),
            gamma=kw.get("gamma", "scale"),
            probability=True,
            class_weight="balanced",
            random_state=kw.get("random_state", 42),
        )),
    ]),
    "logistic_regression": lambda **kw: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=kw.get("C", 1.0),
            max_iter=1000,
            class_weight="balanced",
            random_state=kw.get("random_state", 42),
        )),
    ]),
}


def get_classical_model(name: str, **kwargs) -> Pipeline:
    """Create a classical ML pipeline by name."""
    if name not in CLASSICAL_MODELS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(CLASSICAL_MODELS)}")
    return CLASSICAL_MODELS[name](**kwargs)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Synthetic test
    seqs = [rng.standard_normal((t, 39)).astype(np.float32) for t in [30, 50, 40, 60, 35, 45, 55, 25]]
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    X = aggregate_dataset(seqs)
    print(f"Aggregated shape: {X.shape}")

    for name in CLASSICAL_MODELS:
        model = get_classical_model(name)
        model.fit(X[:6], labels[:6])
        preds = model.predict(X[6:])
        probs = model.predict_proba(X[6:])
        print(f"{name}: preds={preds}, probs={probs.round(3)}")