"""Extract independent, padded left/right trials using original GDF event codes."""

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from eeg_bci.pipeline import load_recording


LABELS = {0: "left_hand", 1: "right_hand"}
EVENT_CODES = (768, 769, 770, 771, 772, 783, 1023, 32766)


@dataclass
class TrialSet:
    X: np.ndarray
    y: np.ndarray
    runs: np.ndarray
    trial_ids: np.ndarray
    channel_names: list
    sfreq: float
    audit: list


def extract_trials(raw, external_labels=None):
    events, _ = mne.events_from_annotations(
        raw, event_id={str(code): code for code in EVENT_CODES}, verbose=False
    )
    starts = events[events[:, 2] == 768, 0]
    boundaries = events[events[:, 2] == 32766, 0]
    if not len(starts) or not len(boundaries):
        raise ValueError("The recording must contain trial and run markers.")
    if external_labels is not None:
        external_labels = np.asarray(external_labels).reshape(-1)
        if len(external_labels) != len(starts) or not np.isin(external_labels, [1, 2, 3, 4]).all():
            raise ValueError("External labels must contain one class (1-4) per original trial, before exclusions.")
    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, 250):
        raise ValueError("Expected 250 Hz sampling.")
    # Calibration-only run markers precede the six movement runs in A01T.
    run_keys = np.searchsorted(boundaries, starts, side="right") - 1
    if np.any(run_keys < 0):
        raise ValueError("A trial appears before the first run marker.")
    run_numbers = {key: i + 1 for i, key in enumerate(np.unique(run_keys))}
    windows, labels, runs, ids, audit = [], [], [], [], []
    for i, (start, run_key) in enumerate(zip(starts, run_keys)):
        next_start = int(starts[i + 1]) if i + 1 < len(starts) else raw.n_times
        next_boundary = int(boundaries[run_key + 1]) if run_key + 1 < len(boundaries) else raw.n_times
        end = min(next_start, next_boundary)
        trial_events = events[(events[:, 0] >= start) & (events[:, 0] < end)]
        cues = trial_events[np.isin(trial_events[:, 2], [769, 770, 771, 772, 783])]
        row = {"trial_id": i + 1, "run": run_numbers[run_key],
               "trial_start_sample": int(start), "cue_sample": None,
               "label": "", "status": ""}
        if len(cues) != 1:
            raise ValueError(f"Trial {i + 1} has {len(cues)} labeled cues; expected one.")
        cue, code = int(cues[0, 0]), int(cues[0, 2])
        if code == 783:
            if external_labels is None:
                raise ValueError("Evaluation cues are unlabeled; provide the official classlabel MAT file.")
            code = 768 + int(external_labels[i])
        elif external_labels is not None and code != 768 + int(external_labels[i]):
            raise ValueError(f"External label does not match the GDF cue in trial {i + 1}.")
        row.update(cue_sample=cue, label={769: "left_hand", 770: "right_hand", 771: "feet", 772: "tongue"}[code])
        if code not in (769, 770):
            row["status"] = "excluded_other_class"
        elif np.any(trial_events[:, 2] == 1023):
            row["status"] = "excluded_expert_artifact"
        else:
            # Five seconds of context per trial, never across a run or trial boundary.
            first, stop = cue - round(0.5 * sfreq), cue + round(4.5 * sfreq)
            if first < start or stop > end:
                row["status"] = "excluded_incomplete_window"
            else:
                window = raw.get_data(start=first, stop=stop)
                if not np.isfinite(window).all():
                    row["status"] = "excluded_nonfinite"
                else:
                    windows.append(window)
                    labels.append(code - 769)
                    runs.append(row["run"])
                    ids.append(i + 1)
                    row["status"] = "kept"
        audit.append(row)
    if not windows or set(labels) != {0, 1}:
        raise ValueError("Not enough clean labeled left/right trials.")
    return TrialSet(np.stack(windows), np.asarray(labels), np.asarray(runs),
                    np.asarray(ids), list(raw.ch_names), sfreq, audit)


def load_trials(path: Path, labels_path: Path | None = None):
    labels = None
    if labels_path is not None:
        from scipy.io import loadmat

        contents = loadmat(labels_path)
        if "classlabel" not in contents:
            raise ValueError("The labels MAT file must contain classlabel.")
        labels = contents["classlabel"]
    return extract_trials(load_recording(path), external_labels=labels)
