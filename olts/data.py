"""TRAIN-only scaling, causal fill and target masks (same preprocessing as the main study).

values  = model inputs: raw series, missing entries forward-filled using the past only
          (leading gaps get the TRAIN mean), then standardised with TRAIN mean/std.
targets = labels for loss and scoring: standardised raw series with NaN wherever the
          value was not observed. Losses and scores only count observed targets.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read, write, sha, now
from .datasets import REGISTRY, boundaries


def raw_array(name, data_dir):
    s = REGISTRY[name]
    p = Path(data_dir) / s.file
    if not p.is_file():
        raise FileNotFoundError(f'{name}: {p} not found (see data/README.md for where to get it)')
    if s.reader == 'h5':
        os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
        raw = pd.read_hdf(p).to_numpy(dtype='float64')
    elif s.reader == 'npz':
        raw = np.load(p)['data'][:, :, 0].astype('float64')
    elif s.reader == 'txt':
        raw = np.loadtxt(p, delimiter=',')
    else:
        df = pd.read_csv(p, encoding='utf-8', encoding_errors='replace')
        raw = df.drop(columns=['date']).to_numpy(dtype='float64')
    if raw.ndim != 2 or raw.shape[1] != s.channels or len(raw) < s.rows_used:
        raise ValueError(f'{name}: unexpected shape {raw.shape}, expected >= ({s.rows_used}, {s.channels})')
    mask = np.isfinite(raw)
    # Weather's documented missing-value sentinel and PEMS-BAY's zero speed are not observations.
    # PEMS08 flow zeros and Solar night zeros are kept as real values.
    if name == 'Weather':
        mask &= raw != -9999
    if name == 'PEMS-BAY':
        mask &= raw != 0
    return np.where(mask, raw, np.nan), p


def prepare(run, name, split, data_dir):
    """Write run/data/<name>_<split>/arrays.npz once; later calls reuse it."""
    out = Path(run) / 'data' / f'{name}_{split}'
    if (out / 'DATA.json').exists():
        meta = read(out / 'DATA.json')
        if sha(out / 'arrays.npz') != meta['arrays_sha256']:
            raise RuntimeError(f'{out}/arrays.npz changed after preparation')
        return meta
    raw, path = raw_array(name, data_dir)
    b = boundaries(name, split)
    nt, end = b[1], b[3]
    x = raw[:end].copy()
    mu = np.nanmean(x[:nt], axis=0)
    scale = np.nanstd(x[:nt], axis=0)
    if not np.isfinite(mu).all():
        raise ValueError(f'{name}: a variable has no observation in TRAIN')
    scale[scale < 1e-8] = 1
    targets = ((x - mu) / scale).astype('float32')
    # The fixed TRAIN mean fills only leading gaps; every later fill uses the past only.
    filled = pd.DataFrame(x).ffill().fillna(dict(enumerate(mu))).to_numpy()
    values = ((filled - mu) / scale).astype('float32')
    assert np.isfinite(values).all()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'arrays.npz', values=values, targets=targets, mean=mu, scale=scale)
    meta = dict(status='PASS', dataset=name, split_protocol=split, source=str(path), source_sha256=sha(path),
                shape=list(values.shape), boundaries=b, arrays_sha256=sha(out / 'arrays.npz'),
                observed_targets=int(np.isfinite(targets).sum()), missing=int(np.isnan(targets).sum()),
                preprocessing='TRAIN-only population mean/std; causal ffill; leading gap TRAIN mean; masked observed targets',
                created=now())
    write(out / 'DATA.json', meta)
    return meta


def load(run, name, split):
    root = Path(run) / 'data' / f'{name}_{split}'
    meta = read(root / 'DATA.json')
    assert meta['status'] == 'PASS'
    z = np.load(root / 'arrays.npz')
    return z['values'], z['targets'], z['mean'], z['scale'], meta
