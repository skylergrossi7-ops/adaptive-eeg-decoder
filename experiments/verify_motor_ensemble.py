"""Independent sequential replay plus masked-label component prefix checks."""
import argparse
from pathlib import Path
import hashlib
import json
import mne
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from eeg_bci.trials import load_trials
from train_neural import external_trials
import burst_transfer
import crop_transfer
import logistic_replay
import spectral_transfer
from aligned_fewshot import compress
from paired_transfer import encode
from openbmi_features import paired_sessions
from stack_transfer import load_spectra


FAMILIES=['logistic-replay','spectral-transfer','crop-transfer','burst-transfer']


def summarize(pred,truth):
    l,r=[(pred[:,truth==h]==h).mean()*100 for h in (0,1)]
    return np.array([l,r,(l+r)/2,(pred==truth).mean()*100])


def run(root,cache,results,out,extra=None):
    out.mkdir(exist_ok=True,parents=True)
    d=np.load(cache/'data.npz')
    y,g=d['y'],d['g']
    people=range(1,10)
    support=pd.read_csv(results/'repeated-five'/'support.csv')
    raw_cov=np.load(cache/'cov.npy',mmap_mode='r')
    cov=compress(raw_cov)
    crop=np.concatenate([crop_transfer.describe(raw_cov[a:a+100]) for a in range(0,len(y),100)])
    spec=load_spectra(root,y,cache)
    burst,labels=[],[]
    session=g.copy()
    budget=[]
    for p in people:
        t=d[f't{p}']
        day1=np.setdiff1d(np.flatnonzero(g==p),t)
        session[t]=100+p
        name=f'A{p:02d}'
        for day in 'TE':
            stem=root/'data'/(name+day)
            trials=load_trials(stem.with_suffix('.gdf'),stem.with_suffix('.mat'))
            for a in range(0,len(trials.y),32):
                block,burst_first=burst_transfer.features(trials.X[a:a+32])
                burst.append(block)
            labels.extend(trials.y)
            if day=='T':
                np.testing.assert_equal(y[day1],trials.y)
                for i,draw in enumerate(d[f'd{p}']):
                    study=support[(support.person==name)&(support.draw==i+1)].sort_values('day_T_trial')
                    inds=pd.Index(trials.trial_ids).get_indexer(study.day_T_trial)
                    assert min(inds)>=0 and len(np.unique(inds))==len(inds)==10
                    np.testing.assert_equal(day1[inds],draw)
                    np.testing.assert_equal(np.bincount(y[draw]),[5,5])
            else:
                np.testing.assert_equal(y[t],trials.y)
                assert np.all(np.diff(trials.trial_ids)>0)
                with np.load(results/'budget-geometry'/(name+'.npz')) as base:
                    np.testing.assert_equal(base['trial'],trials.trial_ids)
        budget.append((p,len(day1),len(t)))
    for _,wave,h in external_trials():
        burst.append(burst_transfer.features(wave)[0]);labels.extend(h)
    np.testing.assert_equal(labels,y)
    burst=np.concatenate(burst)
    families=FAMILIES if extra is None else ['pilot-logistic','spectral-transfer','pilot-crops','burst-transfer']
    ensemble='multiview-replay' if extra is None else 'pilot-ensemble'
    choices={name:pd.read_csv(results/name/'results.csv').set_index(['person','model']) for name in families}
    final=pd.read_csv(results/ensemble/'results.csv').set_index(['person','model'])
    pooled_y,pooled_g,pooled_session=y.copy(),g.copy(),session.copy()
    for person,s,raw,h in paired_sessions(extra):
        cov=np.concatenate([cov,compress(raw)])
        crop=np.concatenate([crop,crop_transfer.describe(raw)])
        pooled_y=np.r_[pooled_y,h]
        pooled_g=np.r_[pooled_g,np.full(len(h),person)]
        pooled_session=np.r_[pooled_session,np.full(len(h),s)]
    rows,draw_rows=[],[]
    prefix_checks=0
    for p in people:
        name=f'A{p:02d}.npz'
        t=d[f't{p}'];truth=y[t];draws=d[f'd{p}']
        parts=[]
        for study in families:
            with np.load(results/study/name) as f:
                np.testing.assert_equal(f['test'],t)
                np.testing.assert_equal(f['truth'],truth)
                parts.append(f['score'])
        with np.load(results/ensemble/name) as f:
            expected=f['score']
        # Predict, then update. Each draw is a separate online deployment.
        preds=[]
        for k,arm in enumerate(['independent','streaming']):
            total=np.zeros((100,4));absolute=np.zeros_like(total)
            predictions=[]
            for i in range(len(t)):
                current=np.column_stack([s[k,:,i] for s in parts])
                value=current
                if k:
                    center=total/(12+i)
                    scale=np.maximum((12+absolute)/(12+i),.35)
                    value=(current-.35*center)/scale
                score=expit(value).sum(1)/4-.5
                np.testing.assert_allclose(score,expected[k,:,i],atol=1e-10)
                predictions.append(score>0)
                total+=current
                absolute+=abs(current)
            pred=np.stack(predictions,axis=1)
            stats=summarize(pred,truth)
            np.testing.assert_allclose(stats,final.loc[(p,arm),['left','right','ba','accuracy']].to_numpy(dtype=float))
            rows.append((p,arm,*stats))
            draw_rows.extend((p,arm,i,100*v) for i,v in enumerate((pred==truth).mean(1)))
            preds.append(pred)
        # Mask ALL held-out labels, including the rest of day one, during refit.
        masked=y.copy()
        masked[g==p]=77
        masked_pool=pooled_y.copy()
        masked_pool[pooled_g==p]=77
        src=np.zeros((len(cov),1224))
        for s in np.unique(pooled_session[pooled_g!=p]):
            m=(pooled_session==s)&(pooled_g!=p)
            src[m]=encode(cov[m],cov[m].mean(0,dtype=float))
        lm=logistic_replay.train(src,cov,masked_pool,pooled_g,[p])
        sm=spectral_transfer.train(spec,masked,g,session,[p])
        cm=crop_transfer.train(crop,masked_pool,pooled_g,pooled_session,[p])
        bm=spectral_transfer.train(burst,masked,g,session,[p],burst_first)
        for draw_id,prefix in [(0,1),(49,13),(99,len(t)//2)]:
            draw=draws[draw_id]
            allowed=masked.copy();allowed[draw]=y[draw]
            seq=t[:prefix]
            vals=[logistic_replay.predict(lm,cov,draw,seq,allowed[draw]),
                  spectral_transfer.scores(spec,sm,allowed,draw[None],seq)[:,:,0],
                  crop_transfer.predict(crop,cm,draw,seq,allowed),
                  spectral_transfer.scores(burst,bm,allowed,draw[None],seq)[:,:,0]]
            for k,arm in enumerate(['independent','streaming']):
                l,s,c,b=[choices[study].loc[(p,arm),:] for study in families]
                indices=[logistic_replay.PARAMS.index((l.reference,l.offset,l.cutoff)),
                         spectral_transfer.PARAMS.index((s.features,s.reg,s.penalty,s.cutoff)),
                         crop_transfer.PARAMS.index((c.reg,c.augmented,c.penalty,c.cutoff)),int(b.setting)]
                for j,idx in enumerate(indices):
                    actual=vals[j][idx] if j==0 else vals[j][idx,k]
                    np.testing.assert_allclose(actual,parts[j][k,draw_id,:prefix],atol=1e-4,rtol=1e-4)
                    prefix_checks+=1
    df=pd.DataFrame(rows,columns='person protocol left right ba accuracy'.split())
    df.to_csv(out/'ensemble-verified.csv',index=False)
    metric=df.pivot(index='person',columns='protocol',values='accuracy')
    rng=np.random.default_rng(20261125)
    boot=rng.integers(0,9,(20000,9))
    diff=metric.streaming-metric.independent
    draws=pd.DataFrame(draw_rows,columns='person protocol draw accuracy'.split())
    drawmeans=draws.groupby(['protocol','draw']).accuracy.mean().loc['streaming']
    checks=dict(individual_prefix_scores_checked=prefix_checks,participants=9,matched_draws=900,
                distinct_test_trials=sum(x[2] for x in budget),full_calibration_labels=sum(x[1] for x in budget),
                calibration_labels_per_person=10,streaming_improved_people=int((diff>0).sum()),
                delta_accuracy=float(diff.mean()),delta_bootstrap95=np.quantile(diff.to_numpy()[boot].mean(1),[.025,.975]).tolist(),
                streaming_accuracy_bootstrap95=np.quantile(metric.streaming.to_numpy()[boot].mean(1),[.025,.975]).tolist(),
                matched_draw_means_over_80=int((drawmeans>=80).sum()),draw_mean_range=[float(drawmeans.min()),float(drawmeans.max())],
                budget_cache_sha256=hashlib.sha256((cache/'data.npz').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(checks,indent=2))
    print(df.groupby('protocol').mean().round(3))
    print(checks)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('root','cache','results','out'):
        ap.add_argument(k,type=Path)
    ap.add_argument('--extra',type=Path)
    with threadpool_limits(2),mne.use_log_level('ERROR'):
        run(**vars(ap.parse_args()))
