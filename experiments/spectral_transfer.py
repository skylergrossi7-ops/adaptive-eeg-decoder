"""Multitaper cross-spectral decoder, with frozen or chronological centering."""
import argparse
from itertools import combinations,product
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from scipy.signal.windows import dpss
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from eeg_bci.trials import load_trials
from train_neural import external_trials


def spectrum(raw):
    x=raw-raw.mean(1,keepdims=True)
    x=x[:,1:18]
    pieces=np.stack([x[:,:,i:i+250] for i in range(250,751,125)],axis=1)
    pieces-=pieces.mean(-1,keepdims=True)
    tapers=dpss(250,2,Kmax=3)
    amp=np.fft.rfft(pieces[:,:,None]*tapers[None,None,:,None,:],axis=-1)
    upper=np.triu_indices(17)
    off=np.triu_indices(17,1)
    powers,phases=[],[]
    for lo,hi in [(6,9),(9,12),(12,15),(15,18),(18,22),(22,26),(26,31),(31,36)]:
        a=amp[...,lo:hi].transpose(0,3,1,2,4).reshape(len(x),17,-1)
        cov=a@a.swapaxes(-1,-2).conj()
        cov/=cov.trace(axis1=1,axis2=2).real[:,None,None]/17
        v,u=np.linalg.eigh(cov)
        log=(u*np.log(.92*v.clip(0)+.08)[:,None,:])@u.swapaxes(-1,-2).conj()
        powers.append(log.real[:,upper[0],upper[1]])
        phases.append(log.imag[:,off[0],off[1]])
    return np.column_stack(powers+phases).astype('float32')


def train(src,y,g,session,excluded,first_dimensions=8*153):
    m=~np.isin(g,excluded)
    z=src.copy()
    for s in np.unique(session[m]):
        j=session==s
        z[j]-=z[j].mean(0)
    sd=z[m].std(0).clip(.025)
    models=[]
    for dims in [first_dimensions,z.shape[1]]:
        v=z[m,:dims]/sd[:dims]
        for C in [.0005,.005]:
            model=LogisticRegression(C=C,max_iter=400).fit(v,y[m])
            models.append((model.coef_[0],model.intercept_[0],sd[:dims]))
    return models


PARAMS=list(product(['real','complex'],[.0005,.005],[.05,.5],[0.,.15]))


def scores(src,models,y,draws,test):
    out=[]
    for w,b,sd in models:
        a=src[draws,:len(w)]/sd
        q=src[test,:len(w)]/sd
        mid=a.mean(1)
        a=a-mid[:,None]
        sup=a@w+b
        gram=a@a.swapaxes(-1,-2)/len(w)
        qfixed=q[None]-mid[:,None]
        sums=np.vstack([np.zeros_like(q[:1]),np.cumsum(q[:-1],axis=0)])
        qlive=q[None]-(12*mid[:,None]+sums)/(12+np.arange(len(q))[:,None])
        for pen in [.05,.5]:
            residual=2*(2*y[draws]-1)-sup
            coef=np.linalg.solve(gram+pen*np.eye(10),residual[...,None])
            vec=w+(a.swapaxes(-1,-2)@coef).squeeze(-1)/len(w)
            fixed=(qfixed@vec[...,None]).squeeze(-1)+b
            live=(qlive@vec[...,None]).squeeze(-1)+b
            for bias in [0.,.15]:
                out.append(np.stack([fixed-bias,live-bias]))
    return np.stack(out)


def run(root,cache,out):
    out.mkdir(exist_ok=True,parents=True)
    d=np.load(cache/'data.npz')
    y,g=d['y'],d['g']
    people=range(1,10)
    session=g.copy()
    blocks,labels=[],[]
    for p in people:
        session[d[f't{p}']]=100+p
        for day in 'TE':
            stem=root/'data'/f'A{p:02d}{day}'
            t=load_trials(stem.with_suffix('.gdf'),stem.with_suffix('.mat'))
            for a in range(0,len(t.y),16):
                blocks.append(spectrum(t.X[a:a+16]))
            labels.extend(t.y)
    for _,raw,truth in external_trials():
        blocks.append(spectrum(raw))
        labels.extend(truth)
    np.testing.assert_array_equal(labels,y)
    x=np.concatenate(blocks)
    valid={p:[] for p in people}
    for a,b in combinations(people,2):
        models=train(x,y,g,session,[a,b])
        for outer,p in [(a,b),(b,a)]:
            t=d[f't{p}']
            s=scores(x,models,y,d[f'd{p}'][:12],t)
            valid[outer].append(((s>0)==y[t]).mean((2,3)))
    rows=[]
    for p in people:
        models=train(x,y,g,session,[p])
        js=np.mean(valid[p],axis=0).argmax(0)
        test=d[f't{p}']
        draws=d[f'd{p}']
        batches=[]
        for a in range(0,100,10):
            s=scores(x,models,y,draws[a:a+10],test)
            batches.append(s[js,np.arange(2)])
        s=np.concatenate(batches,axis=1)
        for arm,name in enumerate(['independent','streaming']):
            pred=s[arm]>0
            l,r=[(pred[:,y[test]==h]==h).mean()*100 for h in (0,1)]
            rows.append((p,name,*PARAMS[js[arm]],l,r,(l+r)/2,(pred==y[test]).mean()*100))
        np.savez_compressed(out/f'A{p:02d}.npz',score=s,truth=y[test],test=test)
    f=pd.DataFrame(rows,columns='person model features reg penalty cutoff left right ba accuracy'.split())
    f.to_csv(out/'results.csv',index=False)
    print(f.groupby('model')[['left','right','ba','accuracy']].mean())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for a in ('root','cache','out'):
        ap.add_argument(a,type=Path)
    with threadpool_limits(2),mne.use_log_level('ERROR'):
        run(**vars(ap.parse_args()))
