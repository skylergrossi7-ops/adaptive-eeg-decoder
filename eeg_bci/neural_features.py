"""Fixed per-trial features for the feedforward hand-imagery neural network."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin


class HandFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, raw):
        """Five-second, 22-channel epochs in montage order at 250 Hz, cue at 0.5 s."""
        if raw.shape[1:] != (22, 1250) or not np.isfinite(raw).all():
            raise ValueError('Expected finite (trials,22,1250) epochs')
        # Re-reference both acquisition systems to the mean of the shared electrodes.
        raw = raw - raw.mean(axis=1, keepdims=True)
        # Local central-electrode derivations suppress spatially broad common activity.
        central = []
        for center, adjacent in [(7,[1,6,8,13]),(9,[3,8,10,15]),(11,[5,10,12,17])]:
            central.append(raw[:,center] - raw[:,adjacent].mean(axis=1))
        raw = np.concatenate([raw, np.stack(central, axis=1)], axis=1)
        features = []
        for band in [(8,12),(12,16),(16,22),(22,30)]:
            sos = butter(4,band,fs=250,btype='bandpass',output='sos')
            signal = sosfiltfilt(sos,raw,axis=-1)[:,:,250:1000]
            power = np.maximum(signal.var(axis=-1),1e-24)
            # Within-trial relative power is independent of overall amplifier gain.
            logpower = np.log(power) - np.log(power[:,:22]).mean(axis=1,keepdims=True)
            features.extend([logpower, logpower[:,[7,22]] - logpower[:,[11,24]]])
        return np.concatenate(features,axis=1)
