"""Demo 2: what happens at each time step of an online stream.

    python demos/demo2_schedule.py                    # a toy stream with horizon 3, every step printed
    python demos/demo2_schedule.py --leaky            # the same stream under the leaky schedule
    python demos/demo2_schedule.py --dataset ETTh1 --split O --horizon 96    # the real boundaries, summarised

It only prints the order of events and counts them: no data, no model, no GPU.
The loop is the one in olts/worker.py (function `stream`) for Naive and ER. PROCEED has the same
schedule during TEST, but its warm-up is different: pairs of updates, starting H steps later.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olts.datasets import REGISTRY, boundaries


def rows(lo, hi):
    return f'{lo}-{hi}'


def events(bounds, W, H, leaky=False):
    """Yield (t, scored origin, update origin, forecast made?) for every step of the stream.

    Row t has just been observed. A forecast issued at t reads rows t-W+1 .. t and predicts t+1 .. t+H.
    """
    _, nt, ve, te = bounds
    for t in range(nt - 1, te):
        u = t - H
        scored = u if ve - 1 <= u < te - H else None        # 1. the forecast issued at u is complete now
        if t >= te - H:                                      # a forecast issued now would run past the end
            yield t, scored, None, False
            continue
        if leaky:
            update = t                                       # trains on rows t+1 .. t+H: not observed yet
        else:
            update = u if u >= W - 1 else None               # 2. newest sample whose targets have all arrived
        yield t, scored, update, t >= ve - 1                 # 3. forecasts are kept from the first TEST forecast on


def describe(t, scored, update, forecast, bounds, W, H, leaky):
    _, nt, ve, te = bounds
    seg = 'TRAIN' if t < nt else ('VALIDATION' if t < ve else 'TEST')
    c1 = f't={scored} (rows {rows(scored + 1, scored + H)})' if scored is not None else '-'
    if update is None:
        c2 = '-'
    else:
        c2 = f'sample {update}: {rows(update - W + 1, update)} -> {rows(update + 1, update + H)}'
        if leaky:
            c2 += '  <- NOT OBSERVED YET'
    c3 = f'rows {rows(t + 1, t + H)}' if forecast else '-'
    return seg, c1, c2, c3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=sorted(REGISTRY), help='use the real boundaries of this dataset')
    ap.add_argument('--split', default='O', choices=['O', 'T'])
    ap.add_argument('--window', type=int)
    ap.add_argument('--horizon', type=int)
    ap.add_argument('--leaky', action='store_true', help='show the schedule that updates with targets from the future')
    a = ap.parse_args()
    if a.dataset:
        bounds, W, H = boundaries(a.dataset, a.split), a.window or 96, a.horizon or 96
        title = f'{a.dataset}, {a.split} split'
    else:
        bounds, W, H = [0, 12, 16, 28], a.window or 4, a.horizon or 3
        title = 'toy stream'
    _, nt, ve, te = bounds
    print(f'{title}: TRAIN rows 0-{nt - 1} | VALIDATION {nt}-{ve - 1} | TEST {ve}-{te - 1}; window W = {W}, horizon H = {H}')
    print(f'Row t has just been observed. A forecast issued at t reads rows t-{W - 1} .. t and predicts rows t+1 .. t+{H}.')
    if a.leaky:
        print('LEAKY schedule: after forecasting at t, the model is trained at once on that forecast\'s own targets.')
    log = list(events(bounds, W, H, a.leaky))
    warm = sum(1 for t, s, u, f in log if u is not None and t < ve - 1)
    test = sum(1 for t, s, u, f in log if u is not None and t >= ve - 1)
    made = [t for t, s, u, f in log if f]
    scored = [t for t, s, u, f in log if s is not None]

    def show(steps):
        head = ['t', 'segment', '1 score the forecast issued at', '2 update with sample: input -> targets', '3 forecast']
        if a.leaky:                                          # the leaky schedule forecasts first, then updates
            head[3:] = ['2 forecast', '3 update at once with sample: input -> targets']
        table = []
        for t, s, u, f in steps:
            seg, c1, c2, c3 = describe(t, s, u, f, bounds, W, H, a.leaky)
            note = '  (a)' if t == ve - 1 else ('  (b)' if t == te - H - 1 else '')
            table.append([str(t), seg, c1, c3, c2 + note] if a.leaky else [str(t), seg, c1, c2, c3 + note])
        width = [max(len(r[k]) for r in table + [head]) for k in range(4)]
        line = lambda r: f'{r[0]:>{width[0]}}  {r[1]:<{width[1]}}  {r[2]:<{width[2]}}  {r[3]:<{width[3]}}  {r[4]}'.rstrip()
        print('\n' + line(head))
        last = None
        for r in table:
            if last is not None and int(r[0]) != last + 1:
                print(f'{"...":>{width[0]}}')
            print(line(r))
            last = int(r[0])
        print('(a) the first TEST forecast; a matched copy is frozen just before this step')
        print('(b) the last forecast: one issued later would need rows after the end of the data')

    if a.dataset:
        keep = {nt - 1, nt, ve - 2, ve - 1, ve, ve - 1 + H, ve + H, te - H - 1, te - H, te - 1}
        show([e for e in log if e[0] in keep])
    else:
        show(log)

    print(f'\nwarm-up, t = {nt - 1:,} .. {ve - 2:,}: {warm:,} updates, no forecast is kept; a matched copy is frozen after them')
    first = 'forecast' if a.leaky else 'update'
    print(f'TEST, t = {made[0]:,} .. {made[-1]:,}: {test:,} updates, {len(made):,} forecasts; at each step the {first} comes first')
    print(f'scoring: the forecast issued at t is scored at t + {H}, so scoring runs from t = {scored[0]:,} to {scored[-1]:,}')
    if a.leaky:
        print(f'\nUnder this schedule the update at t uses rows up to t + {H}.\n'
              f'One step later the model forecasts rows t+2 .. t+{H + 1}: {H - 1} of these {H} rows were already\n'
              f'used as training targets. The score is then not a forecast score.')
    else:
        print(f'\nThe update at t uses the sample issued at t - {H}: its input ends {H} rows before t,\n'
              f'its last target is row t. No update ever uses a row later than t,\n'
              f'and the newest sample the model has learned from is always {H} steps old.')


if __name__ == '__main__':
    main()
