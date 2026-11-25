"""Repeated paired five-shot draws. Use the decoder repository's Python environment."""
import argparse
import copy
import gzip
import csv
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from calibrate import personalize
from experiments.five_hand import source_train
from eeg_bci.eegnet import EEGNet
from eeg_bci.model import EpochBandpass
from eeg_bci.neural_features import HandFeatures
from eeg_bci.train_eegnet import as_microvolts
from eeg_bci.trials import load_trials
from train_neural import ROOT, external_trials


@torch.no_grad()
def embeddings(model, x):
    # Reuse the exact frozen forward pass through the penultimate layer.
    parts = []
    hook = model.classifier.register_forward_pre_hook(lambda layer, args: parts.append(args[0].detach()))
    try:
        for batch in x.split(32):
            model(batch)
    finally:
        hook.remove()
    return torch.cat(parts)


def head_update(head, x, y, test):
    head = copy.deepcopy(head)
    opt = torch.optim.Adam(head.parameters(), lr=.001)
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(head(x), y).backward()
        opt.step()
    with torch.no_grad():
        return head(test).argmax(1).numpy()


def run(history, out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261124)
    band = EpochBandpass()
    xx, yy = [], []
    for _, raw, y in external_trials():
        xx.append(as_microvolts(band.transform(raw)))
        yy.extend(y)
    assert len(yy)==1023
    base = source_train(EEGNet(),torch.cat(xx),torch.tensor(yy),4)
    del xx
    print("External pretraining complete",flush=True)
    data = {}
    for p in range(1,10):
        for day in "TE":
            stem=ROOT/"data"/f"A{p:02d}{day}"
            t=load_trials(stem.with_suffix(".gdf"),stem.with_suffix(".mat"))
            data[p,day]=(as_microvolts(band.transform(t.X)),torch.tensor(t.y),t.trial_ids,HandFeatures().transform(t.X),t.channel_names)
    metrics=[]
    with gzip.open(out/"predictions.csv.gz","wt",newline="") as pf, (out/"support.csv").open("w",newline="") as sf:
        pred,support=csv.writer(pf),csv.writer(sf)
        pred.writerow(["person","draw","day_E_trial","truth","target_only","current_5","eegnet_five"])
        support.writerow(["person","draw","day_T_trial","truth"])
        for p in range(1,10):
            person=f"A{p:02d}"
            torch.manual_seed(20261124+p)
            sources=[v for (q,d),v in data.items() if q!=p]
            net=source_train(copy.deepcopy(base),torch.cat([v[0] for v in sources]),torch.cat([v[1] for v in sources]),2)
            xt,y,tids,ft,names=data[p,"T"]
            xe,truth,eids,fe,enames=data[p,"E"]
            bundle=joblib.load(ROOT/f"outputs/neural/neural_excluding_{person}.joblib")
            assert names==enames==bundle["channels"] and bundle["excluded"]==p and p not in bundle["training_people"]
            network=bundle["network"]
            def current(f):
                logp=network.predict_log_proba(f)
                return network.steps[0][1].transform(f),logp[:,1]-logp[:,0]
            st,lt=current(ft)
            se,le=current(fe)
            ht,he=embeddings(net,xt),embeddings(net,xe)
            ref=pd.read_csv(history/f"{person}.csv")
            np.testing.assert_array_equal(ref.trial,eids)
            np.testing.assert_array_equal(ref.truth,truth)
            zeros={"current_0":(le>0).astype(int),"eegnet_zero":net.classifier(he).argmax(1).numpy()}
            for name,h in zeros.items():
                np.testing.assert_array_equal(h,ref[name])
            pools=[np.flatnonzero(y.numpy()==c) for c in (0,1)]
            rng=np.random.default_rng([20261124,p])
            for draw in range(101):
                idx=np.sort(np.concatenate([pool[:5] if draw==0 else rng.choice(pool,5,replace=False) for pool in pools]))
                assert len(set(idx))==10 and np.bincount(y[idx]).tolist()==[5,5]
                support.writerows((person,draw,tids[i],y[i].item()) for i in idx)
                local=make_pipeline(StandardScaler(),LogisticRegression(C=.1)).fit(ft[idx],y[idx].numpy())
                w=personalize(st[idx],lt[idx],y[idx].numpy())
                h={"target_only":local.predict(fe),"current_5":(le+se@w[:-1]+w[-1]>0).astype(int),
                   "eegnet_five":head_update(net.classifier,ht[idx],y[idx],he)}
                if draw==0:
                    for name,decision in h.items():
                        np.testing.assert_array_equal(decision,ref[name])
                pred.writerows(zip([person]*len(eids),[draw]*len(eids),eids,truth.numpy(),*h.values()))
                row=dict(person=person,draw=draw)
                for name,decision in (h|zeros).items():
                    row[name]=balanced_accuracy_score(truth,decision)
                metrics.append(row)
            print("Verified original and scored 100 matched draws:",person,flush=True)
    pd.DataFrame(metrics).to_csv(out/"scores.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    with mne.use_log_level("ERROR"),threadpool_limits(4):
        run(args.history,args.output_dir)
