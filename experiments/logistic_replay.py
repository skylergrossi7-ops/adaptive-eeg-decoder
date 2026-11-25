"""Re-estimate the verified logistic decoder's reference after each observed EEG trial."""
import argparse
from itertools import combinations,product
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from aligned_fewshot import compress,fit_model,adapt
from paired_transfer import encode
from openbmi_features import paired_sessions


def train(src,c,y,g,exclude):
    m=~np.isin(g,exclude)
    return (*fit_model(src,y,m),c[m].mean(0,dtype=float))


PARAMS=list(product(['fixed','cumulative','recent'],[0.,.75],[0.,.15]))


def predict(model,c,draw,test,labels):
    sd,lr,prior=model
    mean=c[draw].mean(0,dtype=float)
    ref=.9*mean+.1*prior
    a=encode(c[draw],ref)/sd
    hist=np.zeros((len(test),*mean.shape))
    hist[1:]=np.cumsum(c[test[:-1]],axis=0,dtype=float)
    cumulative=.9*(10*mean+hist)/(10+np.arange(len(test))[:,None,None,None])+.1*prior
    recent=np.empty_like(hist)
    running=mean.copy()
    for i,row in enumerate(test):
        recent[i]=.9*running+.1*prior
        running=running+(c[row]-running)/20
    scores=[]
    for reference in [ref,cumulative,recent]:
        z=np.vstack([a,encode(c[test],reference)/sd])
        s=adapt(z,lr.decision_function(z),labels)
        # Estimate any residual log-odds shift using exclusively previous scores.
        baseline=np.r_[0.,np.cumsum(s[:-1])]/(12+np.arange(len(test)))
        for amount,cut in product([0.,.75],[0.,.15]):
            scores.append(s-amount*baseline-cut)
    return np.array(scores)


def run(cache,baseline,out,extra=None,settings_from=None):
    out.mkdir(exist_ok=True,parents=True)
    d=np.load(cache/'data.npz')
    y,g=d['y'],d['g']
    c=compress(np.load(cache/'cov.npy',mmap_mode='r'))
    ses=g.copy()
    people=range(1,10)
    for p in people:
        ses[d[f't{p}']]=100+p
    for person,session,raw,h in paired_sessions(extra):
        c=np.concatenate([c,compress(raw)])
        g=np.r_[g,np.full(len(h),person)]
        ses=np.r_[ses,np.full(len(h),session)]
        y=np.r_[y,h]
    src=np.empty((len(c),1224))
    for s in np.unique(ses):
        mask=ses==s
        src[mask]=encode(c[mask],c[mask].mean(0,dtype=float))
    inner={p:[] for p in people}
    frozen=pd.read_csv(settings_from).set_index(['person','model']) if settings_from else None
    for a,b in (combinations(people,2) if frozen is None else []):
        model=train(src,c,y,g,[a,b])
        for outer,p in [(a,b),(b,a)]:
            t=d[f't{p}']
            draws=d[f'd{p}'][:6]
            s=np.stack([predict(model,c,draw,t,y[draw]) for draw in draws])
            inner[outer].append(((s>0)==y[t]).mean((0,2)))
    rows=[]
    for p in people:
        model=train(src,c,y,g,[p])
        if frozen is None:
            rate=np.mean(inner[p],axis=0)
            # Only fixed reference, zero historical correction for independent arm.
            js=[rate[:2].argmax(),rate.argmax()]
        else:
            js=[]
            for arm in ('independent','streaming'):
                row=frozen.loc[(p,arm),:]
                js.append(PARAMS.index((row.reference,row.offset,row.cutoff)))
        t=d[f't{p}']
        with np.load(baseline/f'aligned-A{p:02d}.npz') as f:
            np.testing.assert_equal(y[t],f['truth'])
            control=f['score']
        all_s=[]
        for i,draw in enumerate(d[f'd{p}']):
            s=predict(model,c,draw,t,y[draw])
            if extra is None:
                np.testing.assert_allclose(s[0],control[i],rtol=1e-4,atol=.0002)
            if i==0:
                np.testing.assert_allclose(predict(model,c,draw,t[:12],y[draw]),s[:,:12],atol=.0001)
            all_s.append(s[js])
        scores=np.stack(all_s,axis=1)
        for arm,name in enumerate(['independent','streaming']):
            pred=scores[arm]>0
            l,r=[(pred[:,y[t]==h]==h).mean()*100 for h in (0,1)]
            rows.append((p,name,*PARAMS[js[arm]],l,r,(l+r)/2,(pred==y[t]).mean()*100))
        np.savez_compressed(out/f'A{p:02d}.npz',score=scores,truth=y[t],test=t)
    f=pd.DataFrame(rows,columns='person model reference offset cutoff left right ba accuracy'.split())
    f.to_csv(out/'results.csv',index=False)
    print(f.groupby('model')[['left','right','ba','accuracy']].mean())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for a in ('cache','baseline','out'):
        ap.add_argument(a,type=Path)
    ap.add_argument('--extra',type=Path)
    ap.add_argument('--settings-from',type=Path)
    with threadpool_limits(2):
        run(**vars(ap.parse_args()))
