"""Serializable EEG preprocessing and a fixed CSP + shrinkage LDA baseline."""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP


class EpochBandpass(TransformerMixin, BaseEstimator):
    """Filter each 5-second padded epoch, retaining 0.5-3.5 s after the cue.

    Filtering never mixes trials. This zero-phase filter is for offline decoding;
    it uses future context and must be replaced/revalidated for streaming use.
    """

    def __init__(self, sfreq=250.0):
        self.sfreq = sfreq

    def fit(self, X, y=None):
        self._validate(X)
        return self

    def _validate(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3 or X.shape[1:] != (22, round(5 * self.sfreq)):
            raise ValueError("Expected epochs shaped (trials, 22 EEG channels, 5 seconds of samples).")
        if not np.isfinite(X).all():
            raise ValueError("Epochs contain non-finite values.")
        return X

    def transform(self, X):
        X = self._validate(X)
        sos = butter(4, [8, 30], btype="bandpass", fs=self.sfreq, output="sos")
        filtered = sosfiltfilt(sos, X, axis=-1)
        # Input begins at cue -0.5 s, so offsets 1-4 s give cue +0.5 to +3.5 s.
        return filtered[:, :, round(self.sfreq):round(4 * self.sfreq)].copy()


def make_model(sfreq=250.0):
    return Pipeline([
        ("bandpass", EpochBandpass(sfreq=sfreq)),
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True,
                    norm_trace=False, cov_est="concat")),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def run_split(runs):
    unique = np.unique(runs)
    if len(unique) != 6:
        raise ValueError(f"Expected six movement runs; found {len(unique)}.")
    return np.flatnonzero(np.isin(runs, unique[:4])), np.flatnonzero(np.isin(runs, unique[4:]))
