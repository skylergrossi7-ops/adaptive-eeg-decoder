"""Stack complementary EEG scores using only nested source-person predictions."""
import argparse
from itertools import combinations
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from eeg_bci.trials import load_trials
from train_neural import external_trials
from spectral_transfer import spectrum
from paired_transfer import encode,features
from aligned_fewshot import compress
import local_filters


def load_spectra(root,y,cache):
    path=cache/'spectra-v1.npz'
    if path.exists():
        with np.load(path) as saved:
            np.testing.assert_equal(saved['truth'],y)
            return saved['x']
    arrays,labels=[],[]
    for p in range(1,10):
        for day in 'TE':
            stem=root/'data'/f'A{p:02d}{day}'
            trials=load_trials(stem.with_suffix('.gdf'),stem.with_suffix('.mat'))
            arrays.extend(spectrum(trials.X[a:a+16]) for a in range(0,len(trials.y),16))
            labels.extend(trials.y)
    for _,raw,h in external_trials():
        arrays.append(spectrum(raw))
        labels.extend(h)
    np.testing.assert_equal(labels,y)
    x=np.concatenate(arrays)
    np.savez_compressed(path,x=x,truth=y)
    return x


def source_models(banks,y,g,exclude):
    models=[]
    for bank in banks:
        mask=~np.isin(g[:len(bank)],exclude)
        scale=bank[mask].std(0).clip(.025)
        fit=LogisticRegression(C=.002,max_iter=400).fit(bank[mask]/scale,y[:len(bank)][mask])
        models.append((scale,fit.coef_[0],fit.intercept_[0]))
    return models


def evidence(models,episode,labels):
    columns=[[],[]]
    for (scale,w,b),(support,*queries) in zip(models,episode):
        a=support/scale
        logits=a@w+b
        delta=(logits[labels==1].mean()-logits[labels==0].mean())/(1+logits.std())
        kernel=a@a.T/len(scale)
        residual=2*(2*labels-1)-logits
        dual=np.linalg.solve(kernel+.18*np.eye(10),residual)
        change=a.T@dual/len(scale)
        for i,q in enumerate(queries):
            q=q/scale
            base=q@w+b
            correction=q@change
            columns[i].extend([base,correction,delta*base,delta*correction])
            if i:
                # Stack can learn how quickly historical score offsets matter.
                columns[i].append(np.r_[0.,np.cumsum((base+correction)[:-1])]/(12+np.arange(len(q))))
    return [np.column_stack(v) for v in columns]


def fit_stack(blocks):
    x=np.concatenate([x for x,_,_ in blocks])
    y=np.concatenate([y for _,y,_ in blocks])
    wt=np.concatenate([w for _,_,w in blocks])
    mu=np.average(x,axis=0,weights=wt)
    sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=wt)).clip(.1)
    lr=LogisticRegression(C=.2,max_iter=400).fit((x-mu)/sd,y,sample_weight=wt)
    return mu,sd,lr


def run(root,cache,extra,out):
    out.mkdir(exist_ok=True,parents=True)
    d=np.load(cache/'data.npz')
    y,g=d['y'],d['g']
    raw=np.load(cache/'cov.npy',mmap_mode='r')
    c=compress(raw)
    people=range(1,10)
    ses=g.copy()
    for p in people:
        ses[d[f't{p}']]=p+100
    spec=load_spectra(root,y,cache)
    centered=spec.copy()
    for s in np.unique(ses):
        m=ses==s
        centered[m]-=centered[m].mean(0)
    cov_src=np.empty((len(c),1224),'float32')
    local_src=np.empty((len(c),256),'float32')
    for s in np.unique(ses):
        m=ses==s
        cov_src[m]=encode(c[m],.95*c[m].mean(0,dtype=float)+.05*np.eye(17))
        local_src[m]=local_filters.encode(raw[m],raw[m,:4,3].mean(0,dtype=float))
    for p in range(1,5):
        files=[extra/f'sess{s:02d}_subj{p:02d}_EEG_MI.npz' for s in (1,2)]
        if not all(path.exists() for path in files):
            continue
        for path in files:
            with np.load(path) as f:
                cr=compress(f['cov'])
                cov_src=np.vstack([cov_src,encode(cr,.95*cr.mean(0,dtype=float)+.05*np.eye(17))])
                local_src=np.vstack([local_src,local_filters.encode(f['cov'],f['cov'][:,:4,3].mean(0,dtype=float))])
                y=np.r_[y,f['y']]
                g=np.r_[g,np.full(len(cr),2000+p)]
    banks=[cov_src,local_src,centered]
    def episode(p,i):
        draw,test=d[f'd{p}'][i],d[f't{p}']
        a=spec[draw]
        q=spec[test]
        mid=a.mean(0)
        hist=(12*mid+np.vstack([np.zeros_like(q[:1]),np.cumsum(q[:-1],axis=0)]))/(12+np.arange(len(q))[:,None])
        return [features(c,draw,test),local_filters.episode(raw,draw,test),(a-mid,q-mid,q-hist)]
    valid={p:[episode(p,i) for i in range(8)] for p in people}
    inner={p:[[],[]] for p in people}
    for a,b in combinations(people,2):
        models=source_models(banks,y,g,[a,b])
        for outer,p in [(a,b),(b,a)]:
            test=d[f't{p}']
            truth=y[test]
            wt=100/(2*np.bincount(truth)[truth]*8)
            for i,ep in enumerate(valid[p]):
                e=evidence(models,ep,y[d[f'd{p}'][i]])
                for k in range(2):
                    inner[outer][k].append((e[k],truth,wt))
    rows=[]
    for p in people:
        models=source_models(banks,y,g,[p])
        stacks=[fit_stack(blocks) for blocks in inner[p]]
        test=d[f't{p}']
        scores=[]
        for i,draw in enumerate(d[f'd{p}']):
            ep=valid[p][i] if i<8 else episode(p,i)
            ev=evidence(models,ep,y[draw])
            scores.append([lr.decision_function((v-mu)/sd) for v,(mu,sd,lr) in zip(ev,stacks)])
        scores=np.stack(scores,axis=1)
        for k,name in enumerate(['independent','streaming']):
            pred=scores[k]>0
            l,r=[(pred[:,y[test]==h]==h).mean()*100 for h in (0,1)]
            rows.append((p,name,l,r,(l+r)/2,(pred==y[test]).mean()*100))
        np.savez_compressed(out/f'A{p:02d}.npz',score=scores,truth=y[test],test=test,draws=d[f'd{p}'])
        print('Stacked fold',p,flush=True)
    f=pd.DataFrame(rows,columns='person model left right ba accuracy'.split())
    f.to_csv(out/'results.csv',index=False)
    print(f.groupby('model')[['left','right','ba','accuracy']].mean())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for a in ('root','cache','extra','out'):
        ap.add_argument(a,type=Path)
    with threadpool_limits(2),mne.use_log_level('ERROR'):
        run(**vars(ap.parse_args()))
