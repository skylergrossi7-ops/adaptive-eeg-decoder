"""Learn a cross-day calibration rule and audit five-label weak-hand indicators."""
import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from eeg_bci.trials import load_trials
from train_neural import ROOT, external_trials
from calibrate import personalize
from experiments.window_scaling import features, fit, recalls


def load(history):
    support=pd.read_csv(history/"support.csv")
    blocks,y,g,draws,tests,ids=[],[],[],{},{},{}
    for p in range(1,10):
        person=f"A{p:02d}"
        for day in "TE":
            stem=ROOT/"data"/(person+day)
            t=load_trials(stem.with_suffix(".gdf"),stem.with_suffix(".mat"))
            off=len(y)
            blocks.append(features(t.X)[1])
            y.extend(t.y)
            g.extend([p]*len(t.y))
            if day=="E":
                tests[p]=np.arange(off,len(y))
                ids[p]=t.trial_ids
            else:
                draws[p]=[]
                for d in range(1,101):
                    rows=support.query("person==@person and draw==@d").sort_values("day_T_trial")
                    idx=pd.Index(t.trial_ids).get_indexer(rows.day_T_trial)
                    assert len(set(idx))==len(idx)==10 and min(idx)>=0
                    np.testing.assert_array_equal(t.y[idx],rows.truth)
                    np.testing.assert_array_equal(np.bincount(t.y[idx]),[5,5])
                    draws[p].append(off+idx)
    for name,raw,labels in external_trials():
        blocks.append(features(raw)[1])
        y.extend(labels)
        g.extend([1000+int(name[1:4])]*len(labels))
    return np.concatenate(blocks),np.array(y),np.array(g),draws,tests,ids


def episodes(z,s,y,lat,draws,test):
    episodes,diagnostics=[],[]
    for idx in draws:
        w=personalize(z[idx],s[idx],y[idx])
        correction=z[test]@w[:-1]+w[-1]
        left,right=[idx[y[idx]==h] for h in (0,1)]
        margins=np.array([-s[left].mean(),s[right].mean()])
        spread=np.array([z[hand].std(axis=0,ddof=1).mean() for hand in (left,right)])
        lateral=np.array([lat[left].mean(),-lat[right].mean()])
        diagnostics.append(np.stack([margins,spread,lateral]))
        # These support-only quantities are available before day-two labeling.
        strength=margins.mean()
        columns=[s[test],correction,np.full(len(test),margins[0]-margins[1]),
                 np.full(len(test),spread[0]-spread[1]),np.full(len(test),lateral[0]-lateral[1]),
                 strength*s[test],strength*correction,lat[test]]
        episodes.append(np.column_stack(columns))
    return np.stack(episodes),np.stack(diagnostics)


def rule(x,y,weight):
    # Weight people and hands equally; repeated draws share the weight of a trial.
    mu=np.average(x,weights=weight,axis=0)
    sd=np.sqrt(np.average((x-mu)**2,weights=weight,axis=0)).clip(1e-6)
    model=LogisticRegression(C=.1,max_iter=500).fit((x-mu)/sd,y,sample_weight=weight)
    return lambda a: ((a-mu)/sd)@model.coef_[0]+model.intercept_[0]


def run(history,prior,out):
    out.mkdir(exist_ok=True,parents=True)
    x,y,g,draws,tests,ids=load(history)
    assert x.shape==(3369,324)
    # Raw C3-minus-C4 log power (the first contrast of each band and window).
    lat=x[:,np.arange(25,324,27)].mean(axis=1)
    tuning=pd.read_csv(prior/"tuning.csv").query("arm=='windows' and subset=='all' and selected")
    results,signals=[],[]
    for p,test in tests.items():
        source=np.array([h for h in tests if h!=p])
        c=tuning.loc[tuning.person==p,"C"].item()
        blocks,labels,weights=[],[],[]
        for h in source:
            z,s=fit(x,y,(g!=p)&(g!=h),c)
            t=tests[h]
            a,_=episodes(z,s,y,lat,draws[h][:20],t)
            blocks.append(a.reshape(-1,a.shape[-1]))
            target=np.tile(y[t],len(a))
            labels.extend(target)
            weights.extend(100/(2*np.bincount(target)[target]))
        bx=np.concatenate(blocks)
        by,bw=np.array(labels),np.array(weights)
        simple=rule(bx[:,:2],by,bw)
        full=rule(bx,by,bw)
        z,s=fit(x,y,g!=p,c)
        a,diag=episodes(z,s,y,lat,draws[p],test)
        scores={"baseline":a[:,:,0]+a[:,:,1],"episode":simple(a[:,:,:2]),"hand_aware":full(a)}
        with np.load(prior/f"windows-all-A{p:02d}.npz") as old:
            np.testing.assert_array_equal(ids[p],old["trial"])
            np.testing.assert_array_equal(y[test],old["truth"])
            np.testing.assert_array_equal(scores["baseline"]>0,old["pred"])
        np.savez_compressed(out/f"A{p:02d}.npz",truth=y[test],trial=ids[p],**scores)
        for name,score in scores.items():
            l,r=recalls(score>0,y[test])
            for d in range(100):
                results.append((p,d+1,name,l[d],r[d],(l[d]+r[d])/2))
                if name=="baseline":
                    for h in (0,1):
                        signals.append((p,d+1,h,*diag[d,:,h],(l[d],r[d])[h]))
        print(p,"verified",flush=True)
    res=pd.DataFrame(results,columns=["person","draw","model","left","right","ba"])
    res.to_csv(out/"scores.csv.gz",index=False)
    summary=res.groupby(["person","model"])[["left","right","ba"]].mean()*100
    summary.to_csv(out/"participants.csv")
    print(summary.ba.unstack().round(2))
    print(summary.groupby("model").mean().round(3))
    print(summary.loc[[2,5]].round(2))
    sig=pd.DataFrame(signals,columns=["person","draw","hand","margin","spread","laterality","recall"])
    sig.to_csv(out/"hand-signals.csv.gz",index=False)
    ph=sig.groupby(["person","hand"]).mean(numeric_only=True)
    ph.to_csv(out/"hand-summary.csv")
    print(ph[["margin","spread","laterality","recall"]].corr(method="spearman").recall)
    for col,direction in [("margin",1),("spread",-1),("laterality",1)]:
        pivot=sig.pivot(index=["person","draw"],columns="hand",values=[col,"recall"])
        difference=pivot.recall[1]-pivot.recall[0]
        mask=difference.abs()>1e-12
        correct=(np.sign((pivot[col][1]-pivot[col][0])*direction)==np.sign(difference))[mask]
        print(col,correct.sum(),len(correct),correct.groupby(level=0).mean().round(2).to_dict())


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("history","prior","out"):
        ap.add_argument(arg,type=Path)
    args=ap.parse_args()
    with mne.use_log_level("ERROR"),threadpool_limits(1):
        run(args.history,args.prior,args.out)
