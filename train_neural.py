"""Fixed neural network and CSP benchmark on matched, participant-held-out trials.

Run fetch_neural_data.py first. See NEURAL.md for data sources and protocol.
"""
import csv
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import mne
import numpy as np
from scipy.signal import resample_poly
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eeg_bci.model import EpochBandpass
from eeg_bci.multi import estimator
from eeg_bci.trials import LABELS, load_trials
from fetch_neural_data import FILES, checksums
from eeg_bci.neural_features import HandFeatures

ROOT = Path(__file__).resolve().parent
MONTAGE = 'Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz'.split()


def external_trials():
    registry = checksums()
    for name in FILES:
        path = ROOT / 'data' / 'physionet' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != registry[name]:
            raise ValueError(f'Wrong source file: {name}')
        with mne.io.read_raw_edf(path, preload=True, verbose=False) as raw:
            raw.rename_channels(lambda ch: ch.strip('.').lower())
            raw.pick([ch.lower() for ch in MONTAGE])
            assert raw.ch_names == [ch.lower() for ch in MONTAGE]
            rate = int(raw.info['sfreq'])
            if rate != 160:
                raise ValueError(f'Unexpected sample rate for {name}')
            windows, labels = [], []
            for onset, duration, mark in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
                if mark not in ('T1', 'T2') or duration < 3.5:
                    continue
                start = round((onset - .5) * rate)
                stop = start + 5 * rate
                if start < 0 or stop > raw.n_times:
                    continue
                window = raw.get_data(start=start, stop=stop)
                if not np.isfinite(window).all() or not np.all(window.std(axis=1) > 0):
                    continue
                windows.append(resample_poly(window, 25, 16, axis=-1))
                labels.append(int(mark == 'T2'))
            if set(labels) != {0, 1}:
                raise ValueError(f'Missing usable hand class for {name}')
            yield name, np.stack(windows), labels


def network():
    # Fixed epoch budget and initialization. No validation or early stopping on test people.
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,),
        activation='relu', solver='adam', alpha=0.1, batch_size=128, max_iter=30,
        learning_rate_init=.001, early_stopping=False, tol=0.0, n_iter_no_change=31, random_state=124))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    mne.set_log_level('ERROR')
    target = ROOT / 'outputs' / 'neural'
    target.mkdir(exist_ok=True, parents=True)
    reference = ROOT / 'outputs' / 'multi_participant' / 'held_out_predictions.csv'
    with reference.open(newline='', encoding='utf8') as f:
        prior = {(r['recording'], int(r['trial_id'])): r for r in csv.DictReader(f)}
    features, filtered, all_y, groups, keys, sources = [], [], [], [], [], []
    gdf_names = None

    def add(raw, y, person):
        features.append(HandFeatures().transform(raw))
        filtered.append(EpochBandpass().transform(raw).astype('float32'))
        all_y.extend(y)
        groups.extend([person] * len(y))

    for p in range(1, 10):
        for session in 'TE':
            stem = f'A{p:02d}{session}'
            data = ROOT / 'data'
            trials = load_trials(data / (stem + '.gdf'), data / (stem + '.mat'))
            if gdf_names is None:
                gdf_names = trials.channel_names
            if trials.channel_names != gdf_names or [gdf_names[i] for i in (0,7,9,11,19)] != ['EEG-Fz','EEG-C3','EEG-Cz','EEG-C4','EEG-Pz']:
                raise ValueError('Unexpected GDF channel order')
            add(trials.X, trials.y, p)
            keys.extend((stem, int(i)) for i in trials.trial_ids)
        print('Prepared', stem[:3], flush=True)
    if set(keys) != set(prior) or len(keys) != len(prior):
        raise ValueError('CSP and neural trial sets differ')
    csp = np.array([int(prior[key]['multi_csp_prediction'] == 'right_hand') for key in keys])
    assert all(prior[key]['true_label'] == LABELS[y] for key, y in zip(keys, all_y))
    for name, raw, y in external_trials():
        # External participant IDs use a distinct namespace.
        add(raw, y, 1000 + int(name[1:4]))
        sources.append(dict(file=name, trials=len(y), left=y.count(0), right=y.count(1)))
    X, F = np.concatenate(features), np.concatenate(filtered)
    del features, filtered
    y, groups = np.asarray(all_y), np.asarray(groups)
    local = groups < 1000
    print(f'{sum(local)} benchmark trials, {sum(~local)} additional training trials', flush=True)
    arms = ('csp_original','nn_original','csp_expanded','nn_expanded')
    preds = {arm: np.full(len(keys), -1, dtype=int) for arm in arms}
    preds['csp_original'] = csp
    folds = []
    for person in range(1,10):
        test = np.flatnonzero(groups == person)
        train = groups != person
        local_train = train & local
        for arm, mask in [('nn_original',local_train),('nn_expanded',train)]:
            fitted = network().fit(X[mask], y[mask])
            preds[arm][test] = fitted.predict(X[test])
            if arm == 'nn_expanded':
                bundle = dict(network=fitted, pipeline=make_pipeline(HandFeatures(), fitted), channels=gdf_names, electrode_order=MONTAGE,
                              training_people=np.unique(groups[mask]).tolist(), excluded=person)
                path = target / f'neural_excluding_A{person:02d}.joblib'
                joblib.dump(bundle, path)
                assert np.array_equal(joblib.load(path)['network'].predict(X[test]), preds[arm][test])
        # Give the original CSP architecture the same enlarged training set.
        fitted_csp = estimator('csp4_lda').fit(F[train].astype(float), y[train])
        preds['csp_expanded'][test] = fitted_csp.predict(F[test].astype(float))
        row = dict(person=person, trials=len(test), original_training_trials=int(local_train.sum()),
                   expanded_training_trials=int(train.sum()))
        row.update({arm:int(np.sum(preds[arm][test] == y[test])) for arm in arms})
        folds.append(row)
        print(row, flush=True)
    rows = []
    for i, key in enumerate(keys):
        rows.append(dict(recording=key[0], trial=key[1], person=int(groups[i]), truth=int(y[i]),
                         **{arm:int(preds[arm][i]) for arm in arms}))
    write_csv(target / 'trial_predictions.csv', rows)
    write_csv(target / 'person_results.csv', folds)
    write_csv(target / 'additional_training_data.csv', sources)
    def score(mask):
        return {arm:dict(correct=int(np.sum(preds[arm][mask] == y[local][mask])), trials=int(mask.sum()))
                for arm in arms}
    report = dict(all_nine=score(np.ones(len(keys), bool)), matched_eight=score(groups[local] != 1),
                  external_trials=int(sum(~local)), external_people=int(len(set(groups[~local]))),
                  folds=folds, reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  benchmark_trials=len(keys), trial_window_seconds=[.5,3.5], random_seed=124)
    final = network().fit(X,y)
    joblib.dump(dict(network=final, pipeline=make_pipeline(HandFeatures(), final), channels=gdf_names, electrode_order=MONTAGE,
                     training_people=np.unique(groups).tolist(), excluded=None),
                target / 'neural_all_people.joblib')
    (target / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report['all_nine']), flush=True)

if __name__ == '__main__':
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.neural_network')
        run()
