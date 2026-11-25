"""Support-only reference estimation and source-session whitening for few-shot decoding."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


def compress(covs):
    # Select four bands; keep a whole-window and early-window matrix for each.
    idx=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    return np.ascontiguousarray(covs[:,:4][:,:,[3,0]][:,:,:,idx][:,:,:,:,idx].reshape(-1,8,len(idx),len(idx)))


def encode(c,ref):
    n=c.shape[-1]
    val,vec=np.linalg.eigh(ref)
    root=(vec*(.9*val.clip(0)+.1)[:,None,:]**-.5)@vec.transpose(0,2,1)
    mat=root@(.9*c+.1*np.eye(n))@root
    vals,vectors=np.linalg.eigh(mat)
    logs=(vectors*np.log(vals.clip(1e-8))[...,None,:])@vectors.swapaxes(-2,-1)
    ij=np.triu_indices(n)
    pack=logs[...,ij[0],ij[1]]
    pack[...,ij[0]!=ij[1]]*=np.sqrt(2)
    return pack.reshape(len(c),-1)


def fit_model(x,y,mask):
    sd=x[mask].std(0).clip(.01)
    z=x/sd
    lr=LogisticRegression(C=.001,max_iter=500).fit(z[mask],y[mask])
    return sd,lr


def adapt(z,s,y,support_count=10,penalty=16):
    # Solve in the ten-dimensional span of the calibration examples.
    train=z[:support_count]
    gram=train@train.T+1
    alpha=np.zeros(len(train))
    for _ in range(40):
        logit=s[:support_count]+gram@alpha
        prob=expit(logit)
        res=y-prob-penalty*alpha
        if np.linalg.norm(res)<1e-7:
            break
        step=np.linalg.solve(penalty*np.eye(len(y))+(prob*(1-prob))[:,None]*gram,res)
        # Damped Newton on the positive-definite primal objective.
        descent=-(res@gram@step)
        loss=(np.logaddexp(0,logit)-y*logit).sum()+penalty/2*alpha@gram@alpha
        frac=1.
        while frac>1e-6:
            a=alpha+frac*step
            l=s[:support_count]+gram@a
            if (np.logaddexp(0,l)-y*l).sum()+penalty/2*a@gram@a<=loss+.0001*frac*descent+1e-12:
                break
            frac/=2
        alpha+=frac*step
    return s[support_count:]+(z[support_count:]@train.T+1)@alpha


def run(cache,out):
    d=np.load(cache/"data.npz"); y,g=d["y"],d["g"]
    cov=compress(np.load(cache/"cov.npy",mmap_mode="r"))
    session=g.copy()
    for p in range(1,10):
        session[d[f"t{p}"]]=p+100
    src=np.empty((len(y),8*153))
    for sess in np.unique(session):
        mask=session==sess
        src[mask]=encode(cov[mask],cov[mask].mean(0,dtype=float))
    rows=[]
    for p in range(1,10):
        test=d[f"t{p}"]; support=d[f"d{p}"]; mask=g!=p
        sd,lr=fit_model(src,y,mask)
        ref=cov[mask].mean(0,dtype=float)
        all_s=[]
        for idx in support:
            r=.9*cov[idx].mean(0,dtype=float)+.1*ref
            z=encode(cov[np.r_[idx,test]],r)/sd
            s=lr.decision_function(z)
            all_s.append(adapt(z,s,y[idx]))
        scores=np.stack(all_s)
        h=scores>0
        l,r=[100*(h[:,y[test]==k]==k).mean() for k in (0,1)]
        rows.append((p,l,r,(l+r)/2,100*(h==y[test]).mean()))
        print(p,rows[-1],flush=True)
        np.savez_compressed(out/f"aligned-A{p:02d}.npz",score=scores,truth=y[test])
    frame=pd.DataFrame(rows,columns=["person","left","right","ba","accuracy"])
    frame.to_csv(out/"aligned.csv",index=False)
    print(frame.mean())


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("cache",type=Path);ap.add_argument("out",type=Path)
    args=ap.parse_args()
    with threadpool_limits(1):
        run(args.cache,args.out)
