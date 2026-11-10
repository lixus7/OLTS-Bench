"""Pair online and frozen runs and summarise the gains (same definitions as the study's reports).

    python -m olts.analyze runs/etth1_quick                      # a local run (reads results/streams.csv)
    python -m olts.analyze --streams results/full_study/streams_main.csv --out results/full_study/check

Definitions
  pair       two TEST runs that share dataset, model, split, H and seed and are scored on the same
             forecast origins: one online run (method m, input window 96) and one reference run.
  gain       (reference MSE - online MSE) / reference MSE. Positive = the online run has lower error.
             +20% means 20% lower error; -20% means 20% higher error; -250% means 3.5 times the error.
  log ratio  ln(reference MSE / online MSE); gain = 1 - exp(-log ratio).
  fixed      reference = Frozen model with input window W (one table column per W).
  matched    reference = the online run's own copy frozen at the start of TEST ("<m>-frozen").
Medians are used throughout because a few diverging online runs make means meaningless.
"""
import argparse
import csv
import functools
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path
from .common import read, write

NUM = ('H', 'seed', 'W', 'lr', 'multiplier', 'online_lr', 'MSE', 'MAE', 'raw_MSE', 'raw_MAE', 'origins',
       'observed_targets', 'response_ms_per_forecast', 'update_seconds_total', 'wall_seconds')


def load_streams(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            for k in NUM:
                if r.get(k) not in (None, ''):
                    r[k] = float(r[k]) if k not in ('H', 'seed', 'W') else int(float(r[k]))
                else:
                    r[k] = None
            rows.append(r)
    return [r for r in rows if r.get('evaluation', 'TEST') == 'TEST']


def gain(ref, online):
    return (ref - online) / ref


def cell(values, pos=None):
    v = sorted(values)
    if not v:
        return None
    return dict(median=round(100 * st.median(v), 1), mean=round(100 * st.mean(v), 1), n=len(v),
                pos=sum(x > 0 for x in v) if pos is None else pos,
                min=round(100 * v[0], 1), max=round(100 * v[-1], 1))


@functools.lru_cache(maxsize=None)
def per_origin_blocks(run, job, blocks=10):
    """(SSE, count) of each of `blocks` contiguous time blocks of one stream (read once per stream)."""
    p = Path(run) / 'jobs' / job / 'per_origin.json'
    if not p.exists():
        return None
    recs = read(p)
    size = len(recs)
    out = []
    for b in range(blocks):
        part = recs[b * size // blocks:(b + 1) * size // blocks]
        out.append((sum(r['sse'] for r in part), sum(r['count'] for r in part)))
    return out


def block_gain(ref_blocks, onl_blocks, idx):
    rs = sum(ref_blocks[i][0] for i in idx); rc = sum(ref_blocks[i][1] for i in idx)
    os_ = sum(onl_blocks[i][0] for i in idx); oc = sum(onl_blocks[i][1] for i in idx)
    return gain(rs / rc, os_ / oc)


def bootstrap(by_dataset, B=2000, seed=7):
    """95% interval of the pooled median when whole datasets are resampled (needs >= 3 datasets)."""
    names = sorted(by_dataset)
    if len(names) < 3:
        return None
    rng = random.Random(seed)
    meds = sorted(st.median([g for d in (rng.choice(names) for _ in names) for g in by_dataset[d]]) for _ in range(B))
    return [round(100 * meds[int(0.025 * B)], 1), round(100 * meds[int(0.975 * B) - 1], 1)]


def analyze(run=None, streams_csv=None, out=None, online_window=96):
    run = Path(run) if run else None
    streams_csv = Path(streams_csv) if streams_csv else run / 'results' / 'streams.csv'
    out = Path(out) if out else run / 'results'
    rows = load_streams(streams_csv)
    key = lambda r: (r['dataset'], r['model'], r['split'], r['H'], r['seed'])
    frozen = defaultdict(dict)        # key -> W -> row
    matched = {}                      # (key, method) -> row
    online = []
    for r in rows:
        if r['method'] == 'Frozen':
            frozen[key(r)][r['W']] = r
        elif r['method'].endswith('-frozen'):
            if r['W'] == online_window:
                matched[(key(r), r['method'][:-len('-frozen')])] = r
        elif r['W'] == online_window:
            online.append(r)
    pairs = []
    for o in online:
        for W, f in sorted(frozen.get(key(o), {}).items()):
            pairs.append(dict(comparison='fixed', ref_W=W, online=o, ref=f))
        m = matched.get((key(o), o['method']))
        if m:
            pairs.append(dict(comparison='matched', ref_W=online_window, online=o, ref=m))
    for p in pairs:
        p['gain'] = gain(p['ref']['MSE'], p['online']['MSE'])
        p['log_ratio'] = math.log(p['ref']['MSE'] / p['online']['MSE'])
        if run is not None:
            rb, ob = per_origin_blocks(str(run), p['ref']['job']), per_origin_blocks(str(run), p['online']['job'])
            if rb and ob:
                p['first30'] = block_gain(rb, ob, (0, 1, 2))
                p['last30'] = block_gain(rb, ob, (7, 8, 9))

    groups = defaultdict(list)
    for p in pairs:
        o = p['online']
        if p['comparison'] == 'fixed':
            groups[('fixed', o['split'], o['method'], p['ref_W'])].append(p)
            groups[('fixed_by_dataset', o['split'], o['method'], p['ref_W'], o['dataset'])].append(p)
        else:
            groups[('matched', o['split'], o['method'])].append(p)
            groups[('matched_by_dataset', o['split'], o['method'], o['dataset'])].append(p)
    summary = defaultdict(dict)
    for g, ps in groups.items():
        c = cell([p['gain'] for p in ps])
        c['log_median'] = round(st.median(p['log_ratio'] for p in ps), 4)
        c['over10x'] = sum(p['online']['MSE'] > 10 * p['ref']['MSE'] for p in ps)
        if all('first30' in p for p in ps):
            c['first30'] = round(100 * st.median(p['first30'] for p in ps), 1)
            c['last30'] = round(100 * st.median(p['last30'] for p in ps), 1)
        if g[0] in ('fixed', 'matched'):
            by_d = defaultdict(list)
            for p in ps:
                by_d[p['online']['dataset']].append(p['gain'])
            c['datasets'] = len(by_d)
            c['datasets_online_better'] = sum(st.median(v) > 0 for v in by_d.values())
            ci = bootstrap(by_d)
            if ci:
                c['bootstrap95'] = ci
        summary[g[0]]['|'.join(str(x) for x in g[1:])] = c

    # cost: time per forecast, online / frozen with the same input window (same pairs as W96 column)
    cost = defaultdict(list)
    for p in pairs:
        if p['comparison'] == 'fixed' and p['ref_W'] == online_window:
            a, b = p['online']['response_ms_per_forecast'], p['ref']['response_ms_per_forecast']
            if a and b:
                cost[f"{p['online']['split']}|{p['online']['method']}"].append((a / b, a, b))
    summary['cost'] = {k: dict(ratio_median=round(st.median(x[0] for x in v), 2), online_ms=round(st.median(x[1] for x in v), 2),
                               frozen_ms=round(st.median(x[2] for x in v), 2), n=len(v)) for k, v in cost.items()}
    if run is not None and (run / 'results' / 'fits.csv').exists():
        tr = defaultdict(list)
        with open(run / 'results' / 'fits.csv', newline='') as f:
            for r in csv.DictReader(f):
                if r['architecture'] == 'Naive' and r['train_seconds']:
                    tr[f"{r['split']}|W{r['W']}"].append(float(r['train_seconds']))
        summary['train_seconds'] = {k: round(st.median(v), 1) for k, v in tr.items()}

    def uniq(k):
        return sorted({r[k] for r in rows})
    src = streams_csv.relative_to(run) if run is not None and streams_csv.is_relative_to(run) else Path(streams_csv.name)
    meta = dict(source=str(src), datasets=uniq('dataset'), models=uniq('model'), splits=uniq('split'),
                horizons=uniq('H'), seeds=uniq('seed'), online_window=online_window,
                methods=sorted({r['method'] for r in online}, key=lambda m: (['Naive', 'ER', 'PROCEED'] + [m]).index(m)),
                frozen_windows={sp: sorted({W for k, ws in frozen.items() if k[2] == sp for W in ws}) for sp in uniq('split')},
                pairs=len(pairs))
    if run is not None:
        for name in ('spec.json', 'selections/lr.json', 'selections/multipliers.json'):
            if (run / name).exists():
                meta[Path(name).stem] = read(run / name)
        if (run / 'data').exists():
            meta['boundaries'] = {p.name: read(p / 'DATA.json')['boundaries'] for p in sorted((run / 'data').iterdir()) if (p / 'DATA.json').exists()}
    # one concrete pair for the report's worked example
    ex = next((p for p in pairs if p['comparison'] == 'fixed' and p['ref_W'] != online_window), None)
    if ex:
        o = ex['online']
        meta['example_pair'] = dict(dataset=o['dataset'], model=o['model'], split=o['split'], H=o['H'], seed=o['seed'],
                                    method=o['method'], W=o['W'], ref_W=ex['ref_W'], online_MSE=round(o['MSE'], 4),
                                    ref_MSE=round(ex['ref']['MSE'], 4), gain=round(100 * ex['gain'], 1))
    summary['meta'] = meta
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'summary.json', dict(summary))
    with open(out / 'pairs.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['comparison', 'dataset', 'model', 'split', 'H', 'seed', 'method', 'W', 'ref_W', 'online_MSE', 'ref_MSE',
                    'gain_pct', 'log_ratio', 'first30_gain_pct', 'last30_gain_pct', 'online_job', 'ref_job'])
        for p in pairs:
            o = p['online']
            w.writerow([p['comparison'], o['dataset'], o['model'], o['split'], o['H'], o['seed'], o['method'], o['W'], p['ref_W'],
                        o['MSE'], p['ref']['MSE'], round(100 * p['gain'], 3), round(p['log_ratio'], 5),
                        round(100 * p['first30'], 3) if 'first30' in p else '', round(100 * p['last30'], 3) if 'last30' in p else '',
                        o['job'], p['ref']['job']])
    print(f'analysed {len(pairs)} pairs into {out}/summary.json and pairs.csv', flush=True)
    return out / 'summary.json'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', nargs='?')
    ap.add_argument('--streams', help='a streams.csv outside a run folder')
    ap.add_argument('--out', help='output folder (default RUN/results)')
    ap.add_argument('--online-window', type=int, default=96)
    a = ap.parse_args()
    if not a.run and not a.streams:
        ap.error('give a run folder or --streams')
    analyze(a.run, a.streams, a.out, a.online_window)
