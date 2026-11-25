"""Fixed, equal-weight prediction from four complementary EEG representations."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit


def combine(logits,stream):
    # logits: (view, draw, chronological trial). For independent trials, the
    # underlying source logits retain their scale. Streaming estimates scale and
    # offset after each prediction, with a twelve-trial source prior.
    if stream:
        past=np.concatenate([np.zeros_like(logits[:,:,:1]),np.cumsum(logits[:,:,:-1],axis=-1)],axis=-1)
        magnitude=np.concatenate([np.zeros_like(logits[:,:,:1]),np.cumsum(abs(logits[:,:,:-1]),axis=-1)],axis=-1)
        n=12+np.arange(logits.shape[-1])
        logits=(logits-.35*past/n)/((12+magnitude)/n).clip(.35)
    return expit(logits).mean(0)-.5


def run(results,out,covariance='logistic-replay',crop='crop-transfer'):
    out.mkdir(exist_ok=True,parents=True)
    families=[covariance,'spectral-transfer',crop,'burst-transfer']
    rows=[]
    for p in range(1,10):
        name=f'A{p:02d}.npz'
        parts=[]
        for study in families:
            with np.load(results/study/name) as f:
                if parts:
                    np.testing.assert_equal(f['test'],t)
                    np.testing.assert_equal(f['truth'],truth)
                t,truth=f['test'],f['truth']
                parts.append(f['score'])
        scores=[]
        for i,arm in enumerate(['independent','streaming']):
            logits=np.stack([s[i] for s in parts])
            s=combine(logits,i==1)
            if i:
                np.testing.assert_allclose(combine(logits[:,:,:12],True),s[:,:12])
            scores.append(s)
            pred=s>0
            l,r=[(pred[:,truth==h]==h).mean()*100 for h in (0,1)]
            rows.append((p,arm,l,r,(l+r)/2,(pred==truth).mean()*100))
        np.savez_compressed(out/name,score=scores,truth=truth,test=t)
    f=pd.DataFrame(rows,columns='person model left right ba accuracy'.split())
    f.to_csv(out/'results.csv',index=False)
    print(f.groupby('model')[['left','right','ba','accuracy']].mean())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for a in ('results','out'):
        ap.add_argument(a,type=Path)
    ap.add_argument('--covariance',default='logistic-replay')
    ap.add_argument('--crop',default='crop-transfer')
    run(**vars(ap.parse_args()))
