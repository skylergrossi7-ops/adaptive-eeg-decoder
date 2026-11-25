"""Nested source validation of independent and predict-then-update transfer."""
import argparse
from itertools import combinations,product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from aligned_fewshot import compress


def encode(c,ref):
    v,u=np.linalg.eigh(ref)
    w=(u*(.9*v.clip(0)+.1)[...,None,:]**-.5)@u.swapaxes(-1,-2)
    v,u=np.linalg.eigh(w@(.9*c+.1*np.eye(c.shape[-1]))@w)
    logs=(u*np.log(v.clip(1e-8))[...,None,:])@u.swapaxes(-1,-2)
    ij=np.triu_indices(c.shape[-1])
    z=logs[...,ij[0],ij[1]]
    z[...,ij[0]!=ij[1]]*=np.sqrt(2)
    return z.reshape(len(c),-1).astype(np.float32)


def features(c,draw,test):
    support=c[draw].mean(0,dtype=float)
    eye=np.eye(c.shape[-1])
    anchor=.95*support+.05*eye
    a=encode(c[draw],anchor)
    frozen=encode(c[test],anchor)
    # Prior at row i uses exactly the first i previous EEG trials.
    past=np.zeros((len(test),*support.shape))
    past[1:]=np.cumsum(c[test[:-1]],axis=0,dtype=float)
    count=np.arange(len(test))[:,None,None,None]
    running=.95*(10*support+past)/(10+count)+.05*eye
    online=encode(c[test],running)
    np.testing.assert_allclose(frozen[0],online[0],atol=1e-6)
    return a,frozen,online


def train(src,y,g,excluded):
    mask=~np.isin(g,excluded)
    sd=src[mask].std(0).clip(.02)
    x=src[mask]/sd
    models=[]
    for reg in [.001,.02]:
        lr=LogisticRegression(C=reg,max_iter=350).fit(x,y[mask])
        models.append((lr.coef_[0],lr.intercept_[0]))
    return sd,models


PARAMS=list(product([.001,.02],[.05,.4,2.],[0.,.2]))


def predict(model,episode,labels):
    sd,models=model
    a,frozen,online=[x/sd for x in episode]
    x=np.vstack([frozen,online])
    gram=a@a.T/len(sd)+1/len(sd)
    k=x@a.T/len(sd)+1/len(sd)
    rows=[[],[]]
    for w,b in models:
        prior_support=a@w+b
        base=x@w+b
        for penalty in [.05,.4,2.]:
            # Ridge regression fits only a residual to the ten support labels.
            alpha=solve(gram+penalty*np.eye(len(a)),2*(2*labels-1)-prior_support,assume_a='pos')
            fixed,stream=np.split(base+k@alpha,2)
            old_sum=np.r_[0.,np.cumsum(stream[:-1])]
            corr=old_sum/(15+np.arange(len(stream)))
            for cut in [0.,.2]:
                rows[0].append(fixed-cut)
                rows[1].append(stream-corr-cut)
    return np.array(rows)


def run(cache,extra,out):
    out.mkdir(exist_ok=True,parents=True)
    d=np.load(cache/'data.npz')
    y,g=d['y'],d['g']
    c=compress(np.load(cache/'cov.npy',mmap_mode='r'))
    people=range(1,10)
    ses=g.copy()
    for p in people:
        ses[d[f't{p}']]=100+p
    blocks=[c]
    paths=sorted(extra.glob('sess*_subj*_EEG_MI.npz'))
    paired={p for p in range(1,5) if all(
        (extra/f'sess{day:02d}_subj{p:02d}_EEG_MI.npz').is_file() for day in (1,2))}
    for path in paths:
        day=int(path.name[4:6])
        p=int(path.name[11:13])
        if p not in paired:
            continue
        with np.load(path) as ext:
            blocks.append(compress(ext['cov']))
            y=np.r_[y,ext['y']]
            g=np.r_[g,np.full(len(ext['y']),2000+p)]
            ses=np.r_[ses,np.full(len(ext['y']),2000+day*100+p)]
    c=np.concatenate(blocks)
    (out/'sources.txt').write_text('Paired OpenBMI participants: '+str(sorted(paired))+'\n')
    src=np.empty((len(c),1224),dtype=np.float32)
    for session in np.unique(ses):
        m=ses==session
        src[m]=encode(c[m],.95*c[m].mean(0,dtype=float)+.05*np.eye(c.shape[-1]))
    # Cache only source-validation draws; outer evaluation uses all 100.
    valid={p:[features(c,draw,d[f't{p}']) for draw in d[f'd{p}'][:8]] for p in people}
    inner={p:[] for p in people}
    for a,b in combinations(people,2):
        model=train(src,y,g,[a,b])
        for outer,p in [(a,b),(b,a)]:
            s=np.mean([((predict(model,episode,y[draw])>0)==y[d[f't{p}']]).mean(-1)
                       for episode,draw in zip(valid[p],d[f'd{p}'])],axis=0)
            inner[outer].append(s)
    rows=[]
    for p in people:
        model=train(src,y,g,[p])
        js=np.mean(inner[p],axis=0).argmax(1)
        test=d[f't{p}']
        draws=d[f'd{p}']
        assert np.all(np.diff(test)>0) and not np.intersect1d(draws,test).size
        assert np.all(g[draws]==p) and (y[draws].sum(1)==5).all()
        scores=[]
        for i,draw in enumerate(draws):
            ep=valid[p][i] if i<len(valid[p]) else features(c,draw,test)
            s=predict(model,ep,y[draw])
            scores.append(s[np.arange(2),js])
        scores=np.stack(scores,axis=1)
        for arm,name in enumerate(['independent','streaming']):
            score=scores[arm]
            l,r=[((score[:,y[test]==h]>0)==h).mean()*100 for h in (0,1)]
            rows.append((p,name,*PARAMS[js[arm]],l,r,(l+r)/2,((score>0)==y[test]).mean()*100))
        np.savez_compressed(out/f'A{p:02d}.npz',score=scores,truth=y[test],draws=draws,test=test)
        print('Verified fold',p,flush=True)
    f=pd.DataFrame(rows,columns='person model reg penalty cutoff left right ba accuracy'.split())
    f.to_csv(out/'results.csv',index=False)
    print(f.groupby('model')[['left','right','ba','accuracy']].mean())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for a in ('cache','extra','out'):
        ap.add_argument(a,type=Path)
    with threadpool_limits(2):
        run(**vars(ap.parse_args()))
