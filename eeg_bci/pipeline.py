"""Load BCI Competition IV 2a recordings and select genuine EEG channels."""

from pathlib import Path

import mne
import numpy as np


EOG_CHANNELS = ("EOG-left", "EOG-central", "EOG-right")


def load_recording(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place A01T.gdf in the project's data folder, or use --data PATH."
        )
    raw = mne.io.read_raw_gdf(path, eog=list(EOG_CHANNELS), preload=True, verbose="ERROR")
    if not set(EOG_CHANNELS).issubset(raw.ch_names):
        raise ValueError("Expected the three eye channels in a BCI Competition IV 2a GDF file.")
    raw.pick("eeg")
    if len(raw.ch_names) != 22 or not np.isclose(raw.info["sfreq"], 250):
        raise ValueError("Expected Dataset 2a: 22 EEG channels sampled at 250 Hz.")
    return raw


def preview_segment(raw, channel: str, start: float, duration: float):
    """Return time in seconds and amplitude in microvolts after an 8-30 Hz filter."""
    if channel not in raw.ch_names:
        raise ValueError(f"Unknown EEG channel {channel!r}. Available: {', '.join(raw.ch_names)}")
    if not np.isfinite(start) or not np.isfinite(duration) or start < 0 or duration <= 0:
        raise ValueError("Start must be finite and nonnegative; duration must be finite and positive.")
    sfreq = raw.info["sfreq"]
    first = int(round(start * sfreq))
    stop = int(round((start + duration) * sfreq))
    if stop > raw.n_times or stop <= first:
        raise ValueError(f"Requested window is outside the {raw.n_times / sfreq:.2f}-second recording.")
    # Filter a copy of the full selected channel to avoid window-edge transients.
    signal = raw.copy().pick([channel])
    if not np.isfinite(signal.get_data()).all():
        raise ValueError("The selected channel contains non-finite samples; inspect the recording.")
    signal.filter(l_freq=8.0, h_freq=30.0, verbose="ERROR")
    data, times = signal[:, first:stop]
    return times, data[0] * 1e6
