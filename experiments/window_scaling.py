"""Matched time-window and external-participant scaling study."""
import argparse
from pathlib import Path
import json

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from calibrate import personalize
from eeg_bci.neural_features import HandFeatures
from eeg_bci.trials import load_trials
from train_neural import ROOT, external_trials


def features(raw):
    raw = raw - raw.mean(axis=1, keepdims=True)
    local = [raw[:,c] - raw[:,adj].mean(axis=1) for c,adj in
             [(7,[1,6,8,13]),(9,[3,8,10,15]),(11,[5,10,12,17])]]
    raw = np.concatenate([raw,np.stack(local,axis=1)],axis=1)
    blocks = [[],[],[],[]]
    for band in [(8,12),(12,16),(16,22),(22,30)]:
        sos = butter(4,band,fs=250,btype="bandpass",output="sos")
        signal = sosfiltfilt(sos,raw,axis=-1)[:,:,250:1000]
        for dest,part in zip(blocks,[signal,*np.split(signal,3,axis=-1)]):
            logpower = np.log(np.maximum(part.var(axis=-1),1e-24))
            logpower -= logpower[:,:22].mean(axis=1,keepdims=True)
            dest.extend([logpower,logpower[:,[7,22]]-logpower[:,[11,24]]])
    return np.concatenate(blocks[0],axis=1),np.concatenate(sum(blocks[1:],[]),axis=1)


def recalls(pred,y):
    return (~pred[:,y==0]).mean(1),pred[:,y==1].mean(1)


def fit(x,y,mask,c):
    scaler = StandardScaler().fit(x[mask])
    z = scaler.transform(x)
    model = LogisticRegression(C=c,max_iter=500).fit(z[mask],y[mask])
    assert model.n_iter_.max()<500
    return z,model.decision_function(z)


def predict(z,logit,y,support,test):
    out = []
    for idx in support:
        w = personalize(z[idx],logit[idx],y[idx])
        out.append(logit[test]+z[test]@w[:-1]+w[-1]>0)
    return np.stack(out)


def run(history,reference,out):
    out.mkdir(parents=True,exist_ok=True)
    saved = pd.read_csv(history/"support.csv")
    blocks,labels,groups,supports,tests,votes,ids = [],[],[],{},{},{},{}
    off = 0
    for p in range(1,10):
        name = f"A{p:02d}"
        for day in "TE":
            stem = ROOT/"data"/(name+day)
            t = load_trials(stem.with_suffix(".gdf"),stem.with_suffix(".mat"))
            pair = features(t.X)
            np.testing.assert_allclose(pair[0][:2],HandFeatures().transform(t.X[:2]),atol=1e-10)
            blocks.append(pair)
            labels.extend(t.y)
            groups.extend([p]*len(t.y))
            if day=="T":
                draws = []
                for d in range(1,101):
                    s=saved.query("person==@name and draw==@d").sort_values("day_T_trial")
                    idx=pd.Index(t.trial_ids).get_indexer(s.day_T_trial)
                    assert min(idx)>=0 and len(idx)==len(set(idx))==10
                    np.testing.assert_array_equal(t.y[idx],s.truth)
                    np.testing.assert_array_equal(np.bincount(t.y[idx]),[5,5])
                    draws.append(idx+off)
                supports[p]=np.stack(draws)
            else:
                tests[p]=np.arange(len(t.y))+off
                with np.load(reference/f"{name}.npz") as old:
                    np.testing.assert_array_equal(old["trial"],t.trial_ids)
                    np.testing.assert_array_equal(old["truth"],t.y)
                    votes[p]=old["vote"]
                ids[p]=t.trial_ids
            off+=len(t.y)
    for name,raw,y_ext in external_trials():
        blocks.append(features(raw))
        labels.extend(y_ext)
        groups.extend([1000+int(name[1:4])]*len(y_ext))
    y,g=np.array(labels),np.array(groups)
    external=np.unique(g[g>1000])
    assert len(external)==24 and sum(g>1000)==1023
    subsets={"zero":[],"all":external.tolist()}
    for i in range(3):
        subsets[f"half{i}"]=sorted(np.random.default_rng(20261124+i).choice(external,12,replace=False).tolist())
    (out/"external-groups.json").write_text(json.dumps(subsets,indent=2))
    rows,tuning=[],[]
    for arm,parts in zip(["whole","windows"],zip(*blocks)):
        x=np.concatenate(parts)
        for variant,ext in subsets.items():
            extra=np.isin(g,ext)
            for p in range(1,10):
                rest=np.array([v for v in tests if v!=p])
                scores={c:[] for c in [.001,.01,.1]}
                for held in (rest[::2],rest[1::2]):
                    mask=np.isin(g,np.setdiff1d(rest,held))|extra
                    for c in scores:
                        z,logit=fit(x,y,mask,c)
                        for h in held:
                            pred=predict(z,logit,y,supports[h][:5],tests[h])
                            l,r=recalls(pred,y[tests[h]])
                            scores[c].append(((l+r)/2).mean())
                chosen=max(scores,key=lambda c:np.mean(scores[c]))
                for c,vals in scores.items():
                    tuning.append((arm,variant,p,c,np.mean(vals),c==chosen))
                z,logit=fit(x,y,np.isin(g,rest)|extra,chosen)
                te=tests[p]
                pred=predict(z,logit,y,supports[p],te)
                assert pred.shape==votes[p].shape
                l,r=recalls(pred,y[te])
                vl,vr=recalls(votes[p],y[te])
                np.savez_compressed(out/f"{arm}-{variant}-A{p:02d}.npz",pred=pred,trial=ids[p],truth=y[te])
                for d in range(100):
                    rows.append((arm,len(ext),variant,p,d+1,l[d],r[d],(l[d]+r[d])/2,(vl[d]+vr[d])/2))
            print(arm,variant,"complete",flush=True)
    draws=pd.DataFrame(rows,columns=["arm","external","subset","person","draw","left","right","ba","vote"])
    draws.to_csv(out/"draws.csv.gz",index=False)
    summary=draws.groupby(["arm","external","person"])[["left","right","ba","vote"]].mean()*100
    summary["delta_pp"]=summary.ba-summary.vote
    summary.to_csv(out/"participants.csv")
    pd.DataFrame(tuning,columns=["arm","subset","person","C","inner_ba","selected"]).to_csv(out/"tuning.csv",index=False)
    print(summary.groupby(["arm","external"]).mean().round(2))
    print(summary.ba.unstack(["arm","external"]).round(2))
    print(draws.groupby(["arm","subset"]).ba.mean()*100)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("history","reference","out"):
        ap.add_argument(arg,type=Path)
    args=ap.parse_args()
    with mne.use_log_level("ERROR"),threadpool_limits(1):
        run(args.history,args.reference,args.out)
