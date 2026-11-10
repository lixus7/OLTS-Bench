"""Gather finished jobs into two flat tables: results/fits.csv and results/streams.csv.

    python -m olts.collect runs/etth1_quick
"""
import argparse
import csv
from pathlib import Path
from .common import read

FIT_COLS = ['job', 'dataset', 'model', 'split', 'H', 'seed', 'W', 'lr', 'common_W', 'track', 'architecture',
            'VALIDATION_MSE', 'chosen_epoch', 'epochs', 'train_seconds']
STREAM_COLS = ['job', 'evaluation', 'dataset', 'model', 'split', 'H', 'seed', 'method', 'W', 'lr', 'multiplier',
               'online_lr', 'track', 'MSE', 'MAE', 'raw_MSE', 'raw_MAE', 'origins', 'observed_targets',
               'response_ms_per_forecast', 'update_seconds_total', 'wall_seconds']


def collect(run):
    run = Path(run)
    fits, streams = [], []
    for done in sorted((run / 'jobs').glob('*/DONE.json')):
        r = read(done)
        c = r['configuration']
        if c['kind'] == 'fit':
            fits.append(dict(job=done.parent.name, **{k: c.get(k) for k in FIT_COLS if k in c},
                             VALIDATION_MSE=r.get('VALIDATION_MSE'), chosen_epoch=r.get('chosen_epoch'),
                             epochs=r.get('epochs'), train_seconds=r.get('train_seconds')))
        elif c['kind'] == 'stream':
            rc = r.get('response_cost') or {}
            streams.append(dict(job=done.parent.name, **{k: c.get(k) for k in STREAM_COLS if k in c},
                                MSE=r['MSE'], MAE=r['MAE'], raw_MSE=r['raw_MSE'], raw_MAE=r['raw_MAE'],
                                origins=r['origins'], observed_targets=r['observed_targets'],
                                response_ms_per_forecast=1000 * rc['total_seconds'] / rc['count'] if rc.get('count') else None,
                                update_seconds_total=(r.get('update_cost') or {}).get('total_seconds'),
                                wall_seconds=r.get('wall_seconds')))
    out = run / 'results'
    out.mkdir(parents=True, exist_ok=True)
    for name, rows, cols in (('fits.csv', fits, FIT_COLS), ('streams.csv', streams, STREAM_COLS)):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
    print(f'collected {len(fits)} fits and {len(streams)} streams into {out}', flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    collect(ap.parse_args().run)
