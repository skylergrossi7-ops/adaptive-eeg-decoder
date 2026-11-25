"""Export real EEG and verified ensemble predictions to an offline HTML replay."""
import base64
import csv
import json
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from eeg_bci.trials import load_trials
from train_neural import ROOT


def build():
    out = Path('outputs/ten-label-search')
    verified = list(csv.DictReader((out/'ensemble-verified.csv').open()))
    data = {}
    for p in range(1, 10):
        code = f'A{p:02d}'
        stem = ROOT/'data'/(code+'E')
        trials = load_trials(stem.with_suffix('.gdf'), stem.with_suffix('.mat'))
        with np.load(f'outputs/multiview-replay/{code}.npz') as ensemble, np.load(f'outputs/budget-geometry/{code}.npz') as reference:
            np.testing.assert_array_equal(reference['trial'], trials.trial_ids)
            np.testing.assert_array_equal(ensemble['truth'], trials.y)
            score = ensemble['score']
            summary = []
            for k, protocol in enumerate(('independent','streaming')):
                row = next(r for r in verified if int(r['person'])==p and r['protocol']==protocol)
                pred = score[k]>0
                stats = [(pred==trials.y).mean()*100]+[(pred[:,trials.y==h]==h).mean()*100 for h in (0,1)]
                np.testing.assert_allclose(stats,[float(row[s]) for s in ('accuracy','left','right')])
                summary.append(stats)
            scores = score[:,0]
            clips = []
            inds = [trials.channel_names.index('EEG-'+n) for n in ('C3','Cz','C4')]
            for i, tid in enumerate(trials.trial_ids):
                uv = resample_poly(trials.X[i,inds]*1e6,2,5,axis=-1)
                scale = max(float(abs(uv).max())/32000, .01)
                packed = np.rint(uv/scale).astype('<i2')
                clips.append(dict(id=int(tid),run=int(trials.runs[i]),y=int(trials.y[i]),
                                  s=(scores[:,i]+.5).tolist(),scale=scale,
                                  wave=base64.b64encode(packed.tobytes()).decode()))
            data[code]=dict(clips=clips,stats=summary)
    template=Path(__file__).with_name('research_demo.html').read_text(encoding='utf8')
    (out/'EEG_Replay.html').write_text(template.replace('__DATA__',json.dumps(data,separators=(',',':'))),encoding='utf8')
    print('Exported',sum(len(v['clips']) for v in data.values()),'trials')


if __name__=='__main__':
    build()
