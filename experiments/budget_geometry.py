"""Reproduce the three calibration budgets and extract trial-local filter-bank features."""
import argparse
from pathlib import Path
from unittest.mock import patch
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from threadpoolctl import threadpool_limits
from experiments import episode_hands
from experiments.window_scaling import features,fit,predict


def geometry(raw):
    car=raw-raw.mean(1,keepdims=True)
    sets=[]
    for lo,hi in [(8,12),(12,16),(16,22),(22,30),(8,30)]:
        v=sosfiltfilt(butter(4,[lo,hi],fs=250,btype="bandpass",output="sos"),car,axis=-1)
        cell=[]
        for a,b in [(250,500),(500,750),(750,1000),(250,1000)]:
            wave=v[:,:,a:b]
            wave=wave-wave.mean(-1,keepdims=True)
            cov=wave@wave.transpose(0,2,1)
            cov/=np.trace(cov,axis1=1,axis2=2)[:,None,None]/22
            cell.append(cov.astype("float32"))
        sets.append(np.stack(cell,axis=1))
    return np.stack(sets,axis=1)


def run(results,out,cache):
    out.mkdir(parents=True,exist_ok=True)
    matrices=[]
    def extract(raw):
        matrices.append(geometry(raw))
        return features(raw)
    with patch.object(episode_hands,"features",extract):
        x,y,g,draws,tests,ids=episode_hands.load(results/"repeated-five")
    cache.mkdir(exist_ok=True,parents=True)
    np.save(cache/"cov.npy",np.concatenate(matrices))
    np.savez(cache/"data.npz",x=x,y=y,g=g,**{f"d{p}":np.stack(draws[p]) for p in tests},
             **{f"t{p}":t for p,t in tests.items()})
    rows=[]
    for p,test in tests.items():
        day1=np.setdiff1d(np.flatnonzero(g==p),test)
        z,s=fit(x,y,g!=p,.001)
        predictions={"zero":(s[test]>0)[None],"five":predict(z,s,y,draws[p],test),
                     "full":predict(z,s,y,[day1],test)}
        with np.load(results/"window-scaling"/f"windows-all-A{p:02d}.npz") as ref:
            np.testing.assert_array_equal(ids[p],ref["trial"])
            np.testing.assert_array_equal(y[test],ref["truth"])
            np.testing.assert_array_equal(predictions["five"],ref["pred"])
        for arm,h in predictions.items():
            l,r=[(h[:,y[test]==k]==k).mean()*100 for k in (0,1)]
            rows.append((p,arm,{"zero":0,"five":10,"full":len(day1)}[arm],l,r,(l+r)/2,100*(h==y[test]).mean()))
        np.savez_compressed(out/f"A{p:02d}.npz",trial=ids[p],truth=y[test],**predictions)
    df=pd.DataFrame(rows,columns=["person","budget","labels","left","right","ba","accuracy"])
    df.to_csv(out/"budgets.csv",index=False)
    print(df.pivot(index="person",columns="budget",values=["left","right"]).round(1))
    print(df.groupby("budget").mean().round(3))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for k in ["results","out","cache"]:
        ap.add_argument(k,type=Path)
    args=ap.parse_args()
    with threadpool_limits(1),mne.use_log_level("ERROR"):
        run(args.results,args.out,args.cache)
