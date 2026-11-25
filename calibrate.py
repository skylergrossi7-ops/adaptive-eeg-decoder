"""Personalize each held-out neural network on T and score every budget on E."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from eeg_bci.trials import load_trials
from eeg_bci.neural_features import HandFeatures

ROOT = Path(__file__).resolve().parent
BUDGETS = (0, 10, 20, 40)


def inputs(bundle, trials):
    network = bundle["network"]
    if bundle["channels"] != trials.channel_names or trials.sfreq != 250:
        raise ValueError("Recording montage differs from training")
    features = HandFeatures().transform(trials.X)
    # Use the scaler fitted on the original cross-person training set.
    standardized = network.steps[0][1].transform(features)
    logp = network.predict_log_proba(features)
    return standardized, logp[:, 1] - logp[:, 0]


def personalize(features, base, labels):
    # Add a ridge-regularized correction to the frozen neural log odds.
    # Fixed penalty, including intercept. Its strength does not depend on E.
    design = np.column_stack((features, np.ones(len(labels))))
    def objective(weights):
        logits = base + design @ weights
        loss = np.sum(np.logaddexp(0, logits) - labels * logits)
        gradient = design.T @ (expit(logits) - labels)
        return loss + 8 * weights @ weights, gradient + 16 * weights
    fit = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def run():
    data = ROOT / "data"
    base_dir = ROOT / "outputs" / "neural"
    out = ROOT / "outputs" / "calibration"
    out.mkdir(exist_ok=True)
    with (base_dir / "trial_predictions.csv").open(newline="") as f:
        previous = {(r["recording"], int(r["trial"])): int(r["nn_expanded"]) for r in csv.DictReader(f)}
    predictions, summary, people = [], [], []
    for person in range(1, 10):
        stem = f"A{person:02d}"
        model = joblib.load(base_dir / f"neural_excluding_{stem}.joblib")
        assert model["excluded"] == person and person not in model["training_people"]
        train = load_trials(data / (stem + "T.gdf"), data / (stem + "T.mat"))
        features, logits = inputs(model, train)
        weights = {0: np.zeros(features.shape[1] + 1)}
        ids = {"0": []}
        for n in BUDGETS[1:]:
            # Nested, chronological first clean examples per hand.
            idx = np.sort(np.concatenate([np.flatnonzero(train.y == c)[:n//2] for c in (0, 1)]))
            assert len(idx) == n and len(set(idx)) == n
            ids[str(n)] = train.trial_ids[idx].tolist()
            weights[n] = personalize(features[idx], logits[idx], train.y[idx])
        test = load_trials(data / (stem + "E.gdf"), data / (stem + "E.mat"))
        features_e, logits_e = inputs(model, test)
        design = np.column_stack((features_e, np.ones(len(test.y))))
        row = dict(person=stem, trials=len(test.y))
        for n in BUDGETS:
            p = (logits_e + design @ weights[n] > 0).astype(int)
            if n == 0:
                assert p.tolist() == [previous[(stem+"E", int(t))] for t in test.trial_ids]
            row[str(n)] = int(np.sum(p == test.y))
            predictions.extend(dict(person=stem, recording=stem+"E", trial=int(t), budget=n,
                                    truth=int(y), prediction=int(pred)) for t,y,pred in zip(test.trial_ids,test.y,p))
        people.append(row)
        (out / (stem + "_calibration.json")).write_text(json.dumps({
            "source_session":stem+"T", "test_session":stem+"E", "training_trial_ids":ids,
            "base_model":f"neural_excluding_{stem}.joblib",
            "corrections":{str(n):w.tolist() for n,w in weights.items()}},indent=2))
        print(row, flush=True)
    for n in BUDGETS:
        correct = [r[str(n)] for r in people]
        rates = [k/r["trials"] for k,r in zip(correct, people)]
        summary.append(dict(budget=n, correct=sum(correct), trials=sum(r["trials"] for r in people),
                            mean_person_accuracy=float(np.mean(rates)), over_80=sum(x>.8 for x in rates)))
    for name,rows in [("predictions",predictions),("participants",people),("summary",summary)]:
        with (out/(name+".csv")).open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(summary,flush=True)

if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
