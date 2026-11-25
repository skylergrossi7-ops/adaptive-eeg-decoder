"""Matched five-shot comparison. Run with the existing decoder's Python environment."""
import argparse
import copy
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eeg_bci.eegnet import EEGNet
from eeg_bci.model import EpochBandpass
from eeg_bci.neural_features import HandFeatures
from eeg_bci.train_eegnet import as_microvolts, predict_proba
from eeg_bci.trials import load_trials
from train_neural import ROOT, external_trials


def source_train(model, x, labels, passes):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(passes):
        for batch in torch.randperm(len(labels)).split(32):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x[batch]), labels[batch])
            loss.backward()
            opt.step()
            model.constrain_weights()
    return model.eval()


def adapt(model, x, y):
    model = copy.deepcopy(model).eval()
    model.requires_grad_(False)
    model.classifier.requires_grad_(True)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    opt = torch.optim.Adam(model.classifier.parameters(), lr=.001)
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if not k.startswith("classifier.")):
        raise RuntimeError("A frozen parameter or normalization buffer changed")
    return model


def classify(model, x):
    with torch.inference_mode():
        return torch.cat([model(batch).argmax(1) for batch in x.split(32)]).numpy()


def run(history, out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261124)
    old = pd.read_csv(history/"predictions.csv")
    support = pd.read_csv(history/"support.csv")
    band = EpochBandpass()
    xx, yy = [], []
    for _, raw, y in external_trials():
        xx.append(as_microvolts(band.transform(raw)))
        yy.extend(y)
    if len(yy)!=1023:
        raise ValueError("External training cohort differs")
    base = source_train(EEGNet(),torch.cat(xx),torch.tensor(yy),4)
    del xx
    print("External pretraining complete",flush=True)
    data = {}
    names = None
    for p in range(1,10):
        for day in "TE":
            stem = ROOT/"data"/f"A{p:02d}{day}"
            t = load_trials(stem.with_suffix(".gdf"),stem.with_suffix(".mat"))
            if names is not None and names!=t.channel_names:
                raise ValueError("Channel order differs")
            names=t.channel_names
            data[p,day]=(as_microvolts(band.transform(t.X)),torch.tensor(t.y),t.trial_ids,HandFeatures().transform(t.X))
    summaries=[]
    for target in range(1,10):
        person=f"A{target:02d}"
        torch.manual_seed(20261124+target)
        sources=[item for (p,day),item in data.items() if p!=target]
        model=source_train(copy.deepcopy(base),torch.cat([i[0] for i in sources]),torch.cat([i[1] for i in sources]),2)
        xt,yt,tids,features=data[target,"T"]
        xe,ye,eids,features_e=data[target,"E"]
        chosen=support.query("person==@person and session=='T' and shots_per_hand==5").sort_values("trial")
        idx=pd.Index(tids).get_indexer(chosen.trial)
        if np.any(idx<0) or len(set(idx))!=10 or np.bincount(yt[idx],minlength=2).tolist()!=[5,5] or not np.array_equal(yt[idx],chosen.truth):
            raise ValueError("Support mismatch")
        target_only=make_pipeline(StandardScaler(),LogisticRegression(C=.1))
        target_only.fit(features[idx],yt[idx].numpy())
        adapted=adapt(model,xt[idx],yt[idx])
        rows=pd.DataFrame(dict(person=person,session="E",trial=eids,truth=ye.numpy(),
                               target_only=target_only.predict(features_e),
                               eegnet_zero=classify(model,xe),eegnet_five=classify(adapted,xe)))
        for k in (0,5):
            prior=old.query("person==@person and session=='E' and shots_per_hand==@k").set_index("trial")
            if not prior.index.is_unique or set(eids)!=set(prior.index) or not np.array_equal(rows.truth,prior.loc[eids,"truth"]):
                raise ValueError("Historical trial mismatch")
            rows["current_"+str(k)]=prior.loc[eids,"prediction"].to_numpy()
        rows.to_csv(out/(person+".csv"),index=False)
        summary={"person":person,"test_trials":len(rows)}
        for name in ["target_only","current_0","current_5","eegnet_zero","eegnet_five"]:
            summary[name]=balanced_accuracy_score(rows.truth,rows[name])
        summaries.append(summary)
        print(person,summary,flush=True)
    pd.DataFrame(summaries).to_csv(out/"participants.csv",index=False)
    support.query("shots_per_hand==5").to_csv(out/"support.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    with mne.use_log_level("ERROR"),threadpool_limits(4):
        run(args.history,args.output_dir)
