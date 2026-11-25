"""Training-only covariance features and participant-disjoint model selection."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from eeg_bci.model import EpochBandpass, make_model


# Fixed before viewing outer-fold results. Ties prefer the simpler CSP baseline.
CANDIDATES = ("csp4_lda", "logcov_C0.1", "logcov_C1")


class EpochCovariances(TransformerMixin, BaseEstimator):
    """Per-trial covariance, unit trace and fixed 10% identity shrinkage."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3 or X.shape[1] != 22 or not np.isfinite(X).all():
            raise ValueError("Expected finite EEG epochs with 22 channels.")
        centered = X - X.mean(axis=-1, keepdims=True)
        covariance = centered @ centered.transpose(0, 2, 1)
        trace = np.trace(covariance, axis1=1, axis2=2)
        if np.any(trace <= 0):
            raise ValueError("A trial has no EEG variance.")
        covariance /= trace[:, None, None]
        return 0.9 * covariance + 0.1 * np.eye(22)[None] / 22


class CovarianceLogFeatures(TransformerMixin, BaseEstimator):
    """Matrix-log features around a reference fitted exclusively to training data.

    The reference is the arithmetic mean of training covariances, not a
    Riemannian mean. No reference update or alignment uses test-person data.
    """

    def fit(self, X, y=None):
        values, vectors = np.linalg.eigh(np.mean(X, axis=0))
        if np.min(values) <= 0:
            raise ValueError("Training covariance reference is not positive definite.")
        self.reference_inverse_sqrt_ = (vectors * (1 / np.sqrt(values))) @ vectors.T
        return self

    def transform(self, X):
        check_is_fitted(self, "reference_inverse_sqrt_")
        normalized = self.reference_inverse_sqrt_[None] @ X @ self.reference_inverse_sqrt_[None]
        values, vectors = np.linalg.eigh(normalized)
        if np.any(values <= 0) or not np.isfinite(values).all():
            raise ValueError("Invalid covariance eigenvalues.")
        logarithm = (vectors * np.log(values)[:, None, :]) @ vectors.transpose(0, 2, 1)
        rows, cols = np.triu_indices(X.shape[1])
        weights = np.where(rows == cols, 1.0, np.sqrt(2.0))
        return logarithm[:, rows, cols] * weights


def estimator(name, sfreq=250.0):
    if name == "csp4_lda":
        return Pipeline(make_model(sfreq).steps[1:])
    if name not in CANDIDATES:
        raise ValueError(f"Unknown candidate: {name}")
    c = 0.1 if name == "logcov_C0.1" else 1.0
    return Pipeline([
        ("logcov", CovarianceLogFeatures()),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=42)),
    ])


def candidate_input(name, filtered, covariances):
    return filtered if name == "csp4_lda" else covariances


def held_out_indices(groups, person):
    groups = np.asarray(groups)
    train, test = np.flatnonzero(groups != person), np.flatnonzero(groups == person)
    if not len(train) or not len(test):
        raise ValueError("Need both training participants and the held-out participant.")
    if set(groups[train]) & set(groups[test]):
        raise RuntimeError("Participant leakage detected.")
    return train, test


def select_candidate(filtered, covariances, labels, groups):
    """Three inner folds, with no person shared between training and validation."""
    groups, labels = np.asarray(groups), np.asarray(labels)
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three training participants are required for inner selection.")
    splits = list(GroupKFold(n_splits=3).split(filtered, labels, groups))
    details = []
    for name in CANDIDATES:
        X = candidate_input(name, filtered, covariances)
        scores, folds = [], []
        for a, b in splits:
            if set(groups[a]) & set(groups[b]):
                raise RuntimeError("Inner participant leakage detected.")
            model = estimator(name).fit(X[a], labels[a])
            predicted = model.predict(X[b])
            person_scores = [float(balanced_accuracy_score(labels[b][groups[b] == person],
                                                         predicted[groups[b] == person]))
                             for person in np.unique(groups[b])]
            scores.extend(person_scores)
            folds.append({"training_participants": np.unique(groups[a]).tolist(),
                          "validation_participants": np.unique(groups[b]).tolist(),
                          "mean_person_balanced_accuracy": float(np.mean(person_scores))})
        details.append({"candidate": name, "inner_mean_person_balanced_accuracy": float(np.mean(scores)),
                        "folds": folds})
    best = max(range(len(details)), key=lambda i: details[i]["inner_mean_person_balanced_accuracy"])
    return CANDIDATES[best], details


def raw_pipeline(name, fitted, sfreq=250.0):
    steps = [("bandpass", EpochBandpass(sfreq))]
    if name != "csp4_lda":
        steps.append(("covariances", EpochCovariances()))
    return Pipeline(steps + fitted.steps)
