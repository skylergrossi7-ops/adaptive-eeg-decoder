"""Load complete OpenBMI source-session pairs from the checksum-verified pilot."""
from pathlib import Path
import numpy as np


def paired_sessions(folder):
    if folder is None:
        return
    for person in range(1,5):
        paths=[Path(folder)/f'sess{day:02d}_subj{person:02d}_EEG_MI.npz' for day in (1,2)]
        if all(path.is_file() for path in paths):
            for day,path in enumerate(paths,1):
                with np.load(path) as f:
                    np.testing.assert_array_equal(np.bincount(f['y']),[50,50])
                    yield 2000+person,2000+100*day+person,f['cov'],f['y']
