"""Use three time crops of each ten-label calibration set with shared trial weight."""
import argparse
from itertools import combinations,product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from paired_transfer import encode
from openbmi_features import paired_sessions


def describe(c):
    # Estimate one source-independent tangent space per trial crop. A robust
    # broad spatial reference is fixed across subjects (identity in CAR units).
    return np.stack([encode(c[:,:4,w,1:18,1:18],np.broadcast_to(np.eye(17),(4,17,17))) for w in (0,1,2,3)],axis=1)


def train(x,y,g,ses,exclude):
    mask=~np.isin(g,exclude)
    z=x[mask].copy()
    for s in np.unique(ses[mask]):
        m=ses[mask]==s
        z[m]-=z[m].mean(0,keepdims=True)
    sd=z.reshape(-1,z.shape[-1]).std(0).clip(.025)
    z/=sd
    # A full-trial example and three time-crops each share one trial's weight.
    models=[]
    for reg in [.001,.01]:
        lr=LogisticRegression(C=reg,max_iter=350).fit(z.reshape(-1,z.shape[-1]),np.repeat(y[mask],4),
                                                      sample_weight=np.full(4*len(z),.25))
        models.append((lr.coef_[0],lr.intercept_[0]))
    return sd,models


PARAMS=list(product([.001,.01],[False,True],[.08,.5],[0.,.15]))


def predict(x,model,draw,test,y):
    sd,models=model
    a=x[draw]/sd
    q=x[test]/sd
    mean=a.mean(0)
    fixed=q-mean
    sums=np.concatenate([np.zeros_like(q[:1]),np.cumsum(q[:-1],axis=0)])
    live=q-(15*mean+sums)/(15+np.arange(len(test))[:,None,None])
    a-=mean
    labels=2*(2*y[draw]-1)
    signals=[]
    for w,b in models:
        for augmented in [False,True]:
            sup=a.reshape(-1,a.shape[-1]) if augmented else a[:,3]
            truth=np.repeat(labels,4) if augmented else labels
            scale=4 if augmented else 1
            gram=sup@sup.T/sup.shape[-1]
            for penalty in [.08,.5]:
                alpha=np.linalg.solve(gram+scale*penalty*np.eye(len(sup)),truth-(sup@w+b))
                vec=w+sup.T@alpha/sup.shape[-1]
                # Same four views are pooled for all models, including controls.
                v=np.stack([(.5*(z[:,3]@vec)+.5*(z[:,:3]@vec).mean(1))+b for z in (fixed,live)])
                for cut in [0.,.15]:
                    signals.append(v-cut)
    return np.stack(signals)


def run(cache,out,extra=None,settings_from=None):
    out.mkdir(exist_ok=True,parents=True)
    d=np.load(cache/'data.npz')
    y,g=d['y'],d['g']
    c=np.load(cache/'cov.npy',mmap_mode='r')
    x=np.concatenate([describe(c[a:a+100]) for a in range(0,len(c),100)])
    ses=g.copy()
    people=range(1,10)
    for p in people:
        ses[d[f't{p}']]=100+p
    for person,session,cov,h in paired_sessions(extra):
        x=np.concatenate([x,describe(cov)])
        g=np.r_[g,np.full(len(h),person)]
        ses=np.r_[ses,np.full(len(h),session)]
        y=np.r_[y,h]
    inner={p:[] for p in people}
    frozen=pd.read_csv(settings_from).set_index(['person','model']) if settings_from else None
    for a,b in (combinations(people,2) if frozen is None else []):
        model=train(x,y,g,ses,[a,b])
        for outer,p in [(a,b),(b,a)]:
            t=d[f't{p}']
            inner[outer].append(np.mean([((predict(x,model,draw,t,y)>0)==y[t]).mean(-1)
                                         for draw in d[f'd{p}'][:8]],axis=0))
    rows=[]
    for p in people:
        model=train(x,y,g,ses,[p])
        if frozen is None:
            js=np.mean(inner[p],axis=0).argmax(0)
        else:
            js=[]
            for arm in ('independent','streaming'):
                row=frozen.loc[(p,arm),:]
                js.append(PARAMS.index((row.reg,row.augmented,row.penalty,row.cutoff)))
        t=d[f't{p}']
        scores=np.stack([predict(x,model,draw,t,y)[js,np.arange(2)] for draw in d[f'd{p}']],axis=1)
        for arm,name in enumerate(['independent','streaming']):
            pred=scores[arm]>0
            l,r=[(pred[:,y[t]==h]==h).mean()*100 for h in (0,1)]
            rows.append((p,name,*PARAMS[js[arm]],l,r,(l+r)/2,(pred==y[t]).mean()*100))
        np.savez_compressed(out/f'A{p:02d}.npz',score=scores,truth=y[t],test=t)
    f=pd.DataFrame(rows,columns='person model reg augmented penalty cutoff left right ba accuracy'.split())
    f.to_csv(out/'results.csv',index=False)
    print(f.groupby('model')[['left','right','ba','accuracy']].mean())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for a in ('cache','out'):
        ap.add_argument(a,type=Path)
    ap.add_argument('--extra',type=Path)
    ap.add_argument('--settings-from',type=Path)
    with threadpool_limits(2):
        run(**vars(ap.parse_args()))
