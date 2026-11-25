"""Test sensorimotor amplitude-envelope and burst features beyond mean power."""
import argparse
from pathlib import Path
from itertools import combinations
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter,hilbert,sosfiltfilt
from threadpoolctl import threadpool_limits
from spectral_transfer import train,scores
from eeg_bci.trials import load_trials
from train_neural import external_trials


def features(raw):
    x=raw-raw.mean(1,keepdims=True)
    # C3, C4 and immediate central/medial/parietal neighbours, with
    # both common-average and local Laplacian referencing at C3/C4.
    x=np.concatenate([x[:,[7,11,8,10,1,5,13,17]],
                      x[:,[7]]-x[:,[1,6,8,13]].mean(1,keepdims=True),
                      x[:,[11]]-x[:,[5,10,12,17]].mean(1,keepdims=True)],axis=1)
    powers,burst=[],[]
    for lo,hi in [(7,10),(10,13),(13,17),(17,22),(22,28),(28,36)]:
        wave=sosfiltfilt(butter(3,[lo,hi],fs=250,btype='bandpass',output='sos'),x,axis=-1)
        env=np.abs(hilbert(wave,axis=-1))**2
        # Nonoverlapping 100-ms bins. Compare sides on the same trial.
        vals=np.log(env[:,:,250:1000].reshape(len(x),10,30,25).mean(-1).clip(1e-30))
        for start,end in [(0,10),(10,20),(20,30),(0,30)]:
            v=vals[:,:,start:end]
            m=v.mean(-1)
            powers.extend([m,m[:,::2]-m[:,1::2]])
        q=np.quantile(vals,[.15,.5,.85],axis=-1)
        lateral=vals[:,::2]-vals[:,1::2]
        lag=vals[:,:,1:]-vals[:,:,:-1]
        # Burst intermittency, lateral occupancy and envelope coordination.
        burst.extend([q[2]-q[0],q[2]+q[0]-2*q[1],np.abs(lag).mean(-1),
                      (lateral>0).mean(-1),lateral.std(-1)])
        z=(vals-vals.mean(-1,keepdims=True))/vals.std(-1,keepdims=True).clip(.01)
        burst.append((z[:,::2]*z[:,1::2]).mean(-1))
    return np.column_stack(powers+burst).astype('float32'),sum(x.shape[1] for x in powers)


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
            for a in range(0,len(t.y),32):
                x,first=features(t.X[a:a+32])
                blocks.append(x)
            labels.extend(t.y)
    for _,raw,truth in external_trials():
        blocks.append(features(raw)[0])
        labels.extend(truth)
    np.testing.assert_equal(labels,y)
    x=np.concatenate(blocks)
    valid={p:[] for p in people}
    for a,b in combinations(people,2):
        models=train(x,y,g,session,[a,b],first)
        for outer,p in [(a,b),(b,a)]:
            t=d[f't{p}']
            s=scores(x,models,y,d[f'd{p}'][:8],t)
            valid[outer].append(((s>0)==y[t]).mean((2,3)))
    rows=[]
    for p in people:
        models=train(x,y,g,session,[p],first)
        js=np.mean(valid[p],axis=0).argmax(0)
        t=d[f't{p}']
        draws=d[f'd{p}']
        s=np.concatenate([scores(x,models,y,draws[a:a+10],t)[js,np.arange(2)]
                          for a in range(0,100,10)],axis=1)
        for arm,name in enumerate(['independent','streaming']):
            pred=s[arm]>0
            l,r=[(pred[:,y[t]==h]==h).mean()*100 for h in (0,1)]
            rows.append((p,name,js[arm],l,r,(l+r)/2,(pred==y[t]).mean()*100))
        np.savez_compressed(out/f'A{p:02d}.npz',score=s,truth=y[t],test=t)
    f=pd.DataFrame(rows,columns='person model setting left right ba accuracy'.split())
    f.to_csv(out/'results.csv',index=False)
    print(f.groupby('model')[['left','right','ba','accuracy']].mean())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for a in ('root','cache','out'):
        ap.add_argument(a,type=Path)
    with threadpool_limits(2),mne.use_log_level('ERROR'):
        run(**vars(ap.parse_args()))
