#!/usr/bin/env python3
"""Compare a local run's TEST scores with the main study's released scores for the same configuration.

    python scripts/compare_with_study.py runs/etth1_quick

A configuration matches when dataset, model, split, H, seed, method, window, offline LR, online
multiplier and training track are all equal. Runs with fewer than 50 epochs (the smoke config) are
refused, and scenarios whose training origins (common_W) differ from the study's are left out.

What to expect: on one RTX A5000 with torch 2.2.1, configs/etth1_quick.json reproduced all 22 of the
study's TEST scores to within 1e-6 (relative), although the study ran torch 2.4.1 on other GPUs.
Differences of a few percent for every run mean the setup differs (data file, split, learning rate,
OnlineTSF commit).
"""
import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def key(r):
    lr = float(r['lr'])
    return (r['dataset'], r['model'], r['split'], int(float(r['H'])), int(float(r['seed'])), r['method'],
            int(float(r['W'])), f'{lr:.0e}', round(float(r['online_lr']) / lr, 6), r.get('track') or 'C')


def scenario_common_w(path):
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('common_W'):
                out.setdefault((r['dataset'], r['model'], r['split'], int(float(r['H']))), set()).add(int(float(r['common_W'])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--study', default=str(ROOT / 'results' / 'full_study' / 'streams_main.csv'))
    ap.add_argument('--study-fits', default=str(ROOT / 'results' / 'full_study' / 'fits_main.csv'))
    a = ap.parse_args()
    run = Path(a.run)
    if (run / 'spec.json').exists():
        spec = json.loads((run / 'spec.json').read_text())
        if int(spec.get('max_epochs', 50)) < 50:
            sys.exit(f'{run} trains only {spec["max_epochs"]} epochs (a smoke test): its scores are not comparable with the study')
    mine_cw = scenario_common_w(run / 'results' / 'fits.csv')
    study_cw = scenario_common_w(a.study_fits)
    differs = {s for s, v in mine_cw.items() if s in study_cw and v != study_cw[s]}
    for s in sorted(differs):
        print(f'left out {" · ".join(map(str, s))}: common_W {sorted(mine_cw[s])} here, {sorted(study_cw[s])} in the study')
    mine = {key(r): r for r in csv.DictReader(open(run / 'results' / 'streams.csv'))
            if r['evaluation'] == 'TEST' and (r['dataset'], r['model'], r['split'], int(float(r['H']))) not in differs}
    study = {key(r): r for r in csv.DictReader(open(a.study))}
    rows = []
    for k in sorted(mine):
        if k in study:
            m, s = float(mine[k]['MSE']), float(study[k]['MSE'])
            rows.append((k, m, s, 100 * (m - s) / s))
    if not rows:
        sys.exit('no configuration of this run appears in the released scores (other LR, multiplier, window or track?)')
    print(f'{"configuration":74} {"MSE here":>10} {"MSE study":>10} {"diff":>8}')
    for k, m, s, d in rows:
        name = ' · '.join(str(x) for x in k[:7]) + f' · lr {k[7]} · x{k[8]:g} · {k[9]}'
        print(f'{name:74} {m:10.4f} {s:10.4f} {d:+7.2f}%')
    diffs = [abs(d) for *_, d in rows]
    print(f'\n{len(rows)} of {len(mine)} TEST runs have a study counterpart; '
          f'|difference| median {st.median(diffs):.4f}%, largest {max(diffs):.4f}%')
    missing = [k for k in mine if k not in study]
    if missing:
        print(f'{len(missing)} runs have no counterpart (for example {" · ".join(map(str, missing[0]))})')


if __name__ == '__main__':
    main()
