"""Participant-specific regularized spatial filters around homologous motor sites."""
import argparse
from itertools import combinations,product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


def encode(c,ref):
    # Two local montages, mirror-ordered. Templates emphasize central, medial,
    # lateral, anterior and posterior motor channels within each hemisphere.
    hemis=[[7,8,6,1,2,13,14,9,3,15],[11,10,12,5,4,17,16,9,3,15]]
    templates=np.eye(10)[:,:7]
    local=templates.copy()
    local[[1,2,3,5],0]=.33
    templates=np.column_stack([templates,local[:,0]])
    blocks=[]
    for ix in hemis:
        a=c[:,:4,:,:,:][:,:,:,ix][:,:,:,:,ix]
        r=ref[...,ix,:][...,:,ix]
        v,u=np.linalg.eigh(r)
        inv=(u/(.8*v.clip(0)+.2)[...,None,:])@u.swapaxes(-1,-2)
        w=inv@templates
        norm=np.maximum((w*(r@w)).sum(-2),1e-8)
        # Broadcasting supports one reference per session or per arriving trial.
        w=w/np.sqrt(norm)[...,None,:]
        w=np.broadcast_to(w,(len(c),*w.shape[-3:]))[:,:,None]
        power=((a@w)*w).sum(-2).clip(1e-8)
        blocks.append(np.log(power))
    delta=blocks[0]-blocks[1]
    return np.concatenate([delta,.5*(blocks[0]+blocks[1])],axis=-1).reshape(len(c),-1).astype('float32')


def episode(c,draw,t):
    ref=c[draw,:4,3].mean(0,dtype=float)
    prev=np.zeros((len(t),*ref.shape))
    prev[1:]=np.cumsum(c[t[:-1],:4,3],axis=0,dtype=float)
    hist=(12*ref+prev)/(12+np.arange(len(t))[:,None,None,None])
    return encode(c[draw],ref),encode(c[t],ref),encode(c[t],hist)


def fit(z,y,g,exclude):
    m=~np.isin(g,exclude)
    scale=z[m].std(0).clip(.025)
    models=[]
    for reg in [.002,.02]:
        lr=LogisticRegression(C=reg,max_iter=350).fit(z[m]/scale,y[m])
        models.append((lr.coef_[0],lr.intercept_[0]))
    return scale,models


PARAMS=list(product([.002,.02],[.1,.7],[0.,.15]))


def predict(model,ep,y):
    scale,models=model
    a,x,live=[z/scale for z in ep]
    views=np.concatenate([x,live])
    gram=a@a.T/len(scale)
    kernel=views@a.T/len(scale)
    rows=[]
    for w,b in models:
        residual=2*(2*y-1)-(a@w+b)
        for penalty in [.1,.7]:
            alpha=np.linalg.solve(gram+penalty*np.eye(10),residual)
            fixed,online=np.split(views@w+b+kernel@alpha,2)
            online-=np.r_[0.,np.cumsum(online[:-1])]/(15+np.arange(len(online)))
            for cut in [0.,.15]:
                rows.append(np.stack([fixed-cut,online-cut]))
    return np.stack(rows)


def run(cache,out):
    out.mkdir(exist_ok=True,parents=True)
    d=np.load(cache/'data.npz')
    y,g=d['y'],d['g']
    c=np.load(cache/'cov.npy',mmap_mode='r')
    people=range(1,10)
    session=g.copy()
    for p in people:
        session[d[f't{p}']]=100+p
    z=np.empty((len(y),256),'float32')
    for s in np.unique(session):
        m=session==s
        z[m]=encode(c[m],c[m,:4,3].mean(0,dtype=float))
    valid={p:[episode(c,draw,d[f't{p}']) for draw in d[f'd{p}'][:8]] for p in people}
    inner={p:[] for p in people}
    for a,b in combinations(people,2):
        model=fit(z,y,g,[a,b])
        for outer,p in [(a,b),(b,a)]:
            acc=[((predict(model,ep,y[draw])>0)==y[d[f't{p}']]).mean(-1)
                 for ep,draw in zip(valid[p],d[f'd{p}'])]
            inner[outer].append(np.mean(acc,axis=0))
    rows=[]
    for p in people:
        model=fit(z,y,g,[p])
        js=np.mean(inner[p],axis=0).argmax(0)
        t=d[f't{p}']
        all_s=[]
        for i,draw in enumerate(d[f'd{p}']):
            ep=valid[p][i] if i<8 else episode(c,draw,t)
            s=predict(model,ep,y[draw])
            if i==0:
                small=predict(model,episode(c,draw,t[:13]),y[draw])
                np.testing.assert_allclose(small,s[:,:,:13],atol=1e-5)
            all_s.append(s[js,np.arange(2)])
        scores=np.stack(all_s,axis=1)
        for arm,name in enumerate(['independent','streaming']):
            pred=scores[arm]>0
            l,r=[(pred[:,y[t]==h]==h).mean()*100 for h in (0,1)]
            rows.append((p,name,*PARAMS[js[arm]],l,r,(l+r)/2,(pred==y[t]).mean()*100))
        np.savez_compressed(out/f'A{p:02d}.npz',score=scores,truth=y[t],test=t)
    f=pd.DataFrame(rows,columns='person model reg penalty cutoff left right ba accuracy'.split())
    f.to_csv(out/'results.csv',index=False)
    print(f.groupby('model')[['left','right','ba','accuracy']].mean())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for a in ('cache','out'):
        ap.add_argument(a,type=Path)
    with threadpool_limits(2):
        run(**vars(ap.parse_args()))
