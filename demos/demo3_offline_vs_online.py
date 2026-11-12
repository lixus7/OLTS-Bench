"""Demo 3: one model, trained once, used in three ways.

    python demos/demo3_offline_vs_online.py                 # ETTh1, O split, W = 96, H = 96; under a minute on a CPU
    python demos/demo3_offline_vs_online.py --split T
    python demos/demo3_offline_vs_online.py --horizon 24

A small linear model (one linear map from the last W values to the next H values, shared by all
variables) is trained offline on TRAIN. The trained model is then run through VALIDATION and TEST
three times, each time from the same starting point:

  frozen   never updated again                                    (traditional forecasting)
  online   at every step: update with the newest sample whose targets have all arrived,
           then forecast                                           (online forecasting, as in the study)
  leaky    at every step: forecast, then update at once with that forecast's own targets, although
           they would only arrive over the next H steps. NOT allowed. It is here to show how much
           it flatters the score.

This is a teaching script, not the benchmark. Its numbers must not go into a report. The benchmark's
engine is olts/engine.py and olts/worker.py; it uses DLinear and PatchTST from OnlineTSF, and it trains
every window of a scenario on the same samples (1,761 on ETTh1, O split), while this demo uses all
samples of its one window (2,689).
"""
import argparse
import copy
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olts.data import raw_array
from olts.datasets import REGISTRY, boundaries


class Linear(torch.nn.Module):
    """Forecast the next H values of every variable from its own last W values."""

    def __init__(self, W, H):
        super().__init__()
        self.map = torch.nn.Linear(W, H)

    def forward(self, x):                                       # x: [batch, W, variables]
        return self.map(x.transpose(1, 2)).transpose(1, 2)      # -> [batch, H, variables]


def sample(z, origins, W, H):
    """The sample issued at row t: input = rows t-W+1 .. t, targets = rows t+1 .. t+H."""
    x = torch.stack([z[t - W + 1:t + 1] for t in origins])
    y = torch.stack([z[t + 1:t + H + 1] for t in origins])
    return x, y


def gradient_step(model, optimizer, x, y):
    optimizer.zero_grad()
    loss = (model(x) - y).square().mean()
    loss.backward()
    optimizer.step()


def train_offline(model, optimizer, z, nt, W, H, epochs, rng):
    """Traditional training: every sample whose targets lie in TRAIN, shuffled, many passes."""
    origins = np.arange(W - 1, nt - H)
    steps = 0
    for _ in range(epochs):
        order = rng.permutation(origins)                        # time order is thrown away
        for k in range(0, len(order), 32):
            gradient_step(model, optimizer, *sample(z, order[k:k + 32], W, H))
            steps += 1
    return len(origins), steps


def stream(model, optimizer, z, bounds, W, H, mode):
    """Walk through VALIDATION and TEST one row at a time. Row t has just been observed."""
    _, nt, ve, te = bounds
    waiting = {}                                                # forecasts whose targets have not all arrived
    sse, count, updates, forecasts, seconds = 0.0, 0, 0, 0, 0.0
    for t in range(nt - 1, te):
        u = t - H
        if u in waiting:                                        # 1. score the forecast issued at u: complete now
            error = waiting.pop(u) - z[u + 1:u + H + 1]
            sse += float(error.square().sum())
            count += error.numel()
        if t >= te - H:
            continue                                            # a forecast issued now would run past the end
        begin = time.perf_counter()
        if mode == 'online':                                    # 2. update: the newest sample that is complete
            gradient_step(model, optimizer, *sample(z, [u], W, H))
            updates += 1
        if t >= ve - 1:                                         # 3. forecast rows t+1 .. t+H (kept for TEST only)
            with torch.no_grad():
                waiting[t] = model(sample(z, [t], W, H)[0])[0]
            forecasts += 1
        if mode == 'leaky':                                     # NOT allowed: rows t+1 .. t+H are in the future
            gradient_step(model, optimizer, *sample(z, [t], W, H))
            updates += 1
        if t >= ve - 1:
            seconds += time.perf_counter() - begin
    assert not waiting
    return sse / count, updates, forecasts, 1000 * seconds / forecasts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='ETTh1', choices=sorted(REGISTRY))
    ap.add_argument('--split', default='O', choices=['O', 'T'])
    ap.add_argument('--window', type=int, default=96)
    ap.add_argument('--horizon', type=int, default=96)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-4, help='used offline and online')
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--data-dir', default=str(Path(__file__).resolve().parents[1] / 'data'))
    a = ap.parse_args()
    W, H = a.window, a.horizon
    torch.set_num_threads(1)                                    # a tiny model: one thread is fastest

    try:
        raw, _ = raw_array(a.dataset, a.data_dir)
    except FileNotFoundError as exc:
        sys.exit(f'{exc}\nThe ETT files can be fetched with: bash scripts/download_data.sh ETTh1')
    if np.isnan(raw).any():
        sys.exit('this demo handles datasets without missing values only (the ETT files)')
    bounds = boundaries(a.dataset, a.split)
    _, nt, ve, te = bounds
    raw = raw[:te]
    mean, std = raw[:nt].mean(0), raw[:nt].std(0)               # the scaler sees TRAIN only
    z = torch.tensor((raw - mean) / std, dtype=torch.float32, device=a.device)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = Linear(W, H).to(a.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f'{a.dataset}, {a.split} split: TRAIN rows 0-{nt - 1} | VALIDATION {nt}-{ve - 1} | TEST {ve}-{te - 1}; W = {W}, H = {H}')
    begin = time.perf_counter()
    n, steps = train_offline(model, optimizer, z, nt, W, H, a.epochs, rng)
    print(f'offline training: all {n:,} samples of this window, {a.epochs} passes in batches of 32 = {steps:,} steps, '
          f'{time.perf_counter() - begin:.1f} s\n')
    start = copy.deepcopy((model.state_dict(), optimizer.state_dict()))

    print(f'{"mode":<8}{"updates":>9}{"TEST MSE":>11}{"MSE vs frozen":>15}{"ms per forecast":>17}')
    frozen = None
    for mode in ('frozen', 'online', 'leaky'):
        model.load_state_dict(start[0])                         # every mode starts from the same trained model
        optimizer.load_state_dict(copy.deepcopy(start[1]))      # a copy: Adam updates its state in place
        mse, updates, forecasts, ms = stream(model, optimizer, z, bounds, W, H, mode)
        frozen = mse if frozen is None else frozen
        change = '-' if mode == 'frozen' else f'{100 * (mse - frozen) / frozen:+.1f}%'
        note = '   <- not allowed: uses future targets' if mode == 'leaky' else ''
        print(f'{mode:<8}{updates:>9,}{mse:>11.4f}{change:>15}{ms:>17.2f}{note}')
    print(f'\n{forecasts:,} forecasts in every mode, scored on the same rows, in units of the TRAIN standard deviation.')
    print('MSE vs frozen: negative = lower error than the frozen model.')
    print('Timings differ from run to run; the MSE values do not.')


if __name__ == '__main__':
    main()
