"""Demo 1: the same series, cut two ways.

    python demos/demo1_splits.py                          # ETTh1, horizon 96
    python demos/demo1_splits.py --dataset ETTm1 --horizon 336

For the traditional split (T) and the online split (O) it prints
  - where TRAIN, VALIDATION and TEST begin and end;
  - the mean and standard deviation that the scaler takes from TRAIN;
  - how many forecasts each segment yields, and the longest input window that still
    leaves 1,024 training samples.
Needs numpy and pandas only; no GPU.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olts.data import raw_array
from olts.datasets import REGISTRY, boundaries

NAMES = ('TRAIN', 'VALIDATION', 'TEST')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='ETTh1', choices=sorted(REGISTRY))
    ap.add_argument('--horizon', type=int, default=96)
    ap.add_argument('--variable', type=int, default=-1, help='column whose scaler is printed (default: the last one)')
    ap.add_argument('--data-dir', default=str(Path(__file__).resolve().parents[1] / 'data'))
    a = ap.parse_args()
    spec, H = REGISTRY[a.dataset], a.horizon
    try:
        raw, path = raw_array(a.dataset, a.data_dir)
    except FileNotFoundError as exc:
        sys.exit(f'{exc}\nThe ETT files can be fetched with: bash scripts/download_data.sh ETTh1')
    dates, names = None, None
    if spec.reader == 'csv':
        dates = pd.read_csv(path, usecols=['date'])['date'].str.slice(0, 10).to_numpy()
        names = [c for c in pd.read_csv(path, nrows=0).columns if c != 'date']
    n = spec.rows_used
    print(f'{a.dataset}: {n:,} rows used, {spec.channels} variables, horizon H = {H}\n')

    print('1. Where the segments are (rows are counted from 0)')
    for split, label in (('T', 'traditional'), ('O', 'online')):
        b = boundaries(a.dataset, split)
        print(f'   {split} split ({label})')
        for k, name in enumerate(NAMES):
            lo, hi = b[k], b[k + 1] - 1
            when = f'   {dates[lo]} to {dates[hi]}' if dates is not None else ''
            print(f'      {name:<11}rows {lo:>6,} - {hi:>6,}   {hi - lo + 1:>6,} rows  {100 * (hi - lo + 1) / n:4.1f}%{when}')

    v = a.variable % spec.channels
    label = f'variable {v}' + (f', column {names[v]}' if names else '')
    print(f'\n2. What the scaler sees ({label}; mean and standard deviation of TRAIN only)')
    for split in 'TO':
        b = boundaries(a.dataset, split)
        mu, sd = np.nanmean(raw[:b[1], v]), np.nanstd(raw[:b[1], v])
        test = (raw[b[2]:b[3], v] - mu) / sd
        print(f'   {split} split: mean {mu:8.3f}, std {sd:7.3f}   ->  in these units, TEST has mean '
              f'{np.nanmean(test):+.2f} and std {np.nanstd(test):.2f}')
    print('   The two splits standardise the same values differently, so their errors are on different scales.')

    print(f'\n3. How many forecasts each segment yields at H = {H}')
    print('   A forecast issued at row t reads rows t-W+1 .. t and predicts rows t+1 .. t+H.')
    print('   It belongs to a segment when all of its targets lie in that segment; its input may reach back.')
    windows = (96, 512, 1024, 4096)
    print('   ' + ' ' * 9 + ''.join(f'{"TRAIN, W=" + str(W):>15}' for W in windows) + f'{"VALIDATION":>13}{"TEST":>9}')
    for split in 'TO':
        b = boundaries(a.dataset, split)
        cells = [b[1] - H - W + 1 for W in windows]
        row = ''.join(f'{c:>15,}' if c > 0 else f'{"none":>15}' for c in cells)
        print(f'   {split} split  {row}{b[2] - b[1] - H + 1:>13,}{b[3] - b[2] - H + 1:>9,}')
    print('\n4. The longest window that still leaves 1,024 training samples (the study\'s minimum)')
    for split in 'TO':
        b = boundaries(a.dataset, split)
        limit = b[1] - H - 1024 + 1
        usable = [W for W in (96, 512, 720, 1024, 2048, 4096) if W <= limit]
        best = f'up to W{max(usable)}' if usable else 'none'
        print(f'   {split} split: W <= {limit:,}  ->  of 96, 512, 720, 1024, 2048, 4096 the study can use {best}')
    print('\n   The benchmark gives every window of a scenario the same training samples: origins start at')
    print('   common_W - 1, where common_W is the longest window. A short window gets no extra samples.')


if __name__ == '__main__':
    main()
