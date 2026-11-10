"""Turn an experiment spec (configs/*.json) into worker jobs, in the main study's stages.

Stage 1  LR fits        seed `lr_selection_seed`, every window x every LR in `lr_grid`.
Stage 2  select LR      per scenario (dataset, model, split, H): rank the LRs inside each window by
                        VALIDATION MSE, average the ranks over windows, take the best (ties -> smaller LR).
Stage 3  seed fits      the other seeds at the chosen LR; plus PROCEED initialisation jobs.
Stage 4  multipliers    only when `online_multipliers` has more than one value: VALIDATION streams
                        (warm-up on the first half of VALIDATION, score the second half), seed
                        `multiplier_selection_seed`; lowest MSE wins (ties -> smaller multiplier).
Stage 5  TEST streams   Frozen at every frozen window; each online method at `online_window`;
                        optionally the matched copy "<method>-frozen" (same start, frozen at TEST start).
"""
from pathlib import Path
from .common import read, write
from .datasets import REGISTRY, boundaries

DEFAULTS = dict(
    datasets=['ETTh1'], models=['DLinear'], splits=['O', 'T'], horizons=[96], seeds=[2024],
    frozen_windows=[96, 512, 720, 1024], common_W='auto', online_window=96,
    online_methods=['Naive', 'ER', 'PROCEED'], matched_frozen=True,
    lr_grid=[1e-4, 3e-4, 1e-3], lr_selection_seed=2024, track='C',
    online_multipliers=[1.0], multiplier_selection_seed=2024,
    min_train_origins=1024, max_epochs=50, keep_predictions=False)
METHODS = ('Naive', 'ER', 'PROCEED')


def load_spec(path):
    spec = dict(DEFAULTS)
    spec.update({k: v for k, v in read(path).items() if not k.startswith('_')})
    spec.setdefault('name', Path(path).stem)
    for d in spec['datasets']:
        if d not in REGISTRY:
            raise ValueError(f'unknown dataset {d}; known: {sorted(REGISTRY)}')
    for m in spec['online_methods']:
        if m not in METHODS:
            raise ValueError(f'unknown online method {m}; use {METHODS}')
    if spec['track'] not in ('C', 'T'):
        raise ValueError('track must be C (50 epochs, keep last) or T (early stopping, keep best)')
    if spec['lr_selection_seed'] not in spec['seeds']:
        raise ValueError('lr_selection_seed must be one of seeds')
    return spec


def _per_split(value, split):
    return value[split] if isinstance(value, dict) else value


def frozen_windows(spec, split):
    return sorted(set(_per_split(spec['frozen_windows'], split)))


def all_windows(spec, split):
    return sorted(set(frozen_windows(spec, split)) | {spec['online_window']})


def common_W(spec, dataset, split, H):
    """Training origins start at common_W-1 for every window of a scenario, so all windows
    learn from the same targets; it must leave at least `min_train_origins` origins."""
    cw = _per_split(spec['common_W'], split)
    if cw == 'auto':
        cw = max(all_windows(spec, split))
    nt = boundaries(dataset, split)[1]
    limit = nt - H - spec['min_train_origins'] + 1
    if cw > limit:
        raise ValueError(f'{dataset} {split} H{H}: common_W {cw} leaves fewer than {spec["min_train_origins"]} '
                         f'training origins (largest allowed {limit}); drop the longer windows')
    if max(all_windows(spec, split)) > cw:
        raise ValueError(f'{dataset} {split}: a window is longer than common_W {cw}')
    return cw


def scenarios(spec):
    for d in spec['datasets']:
        for m in spec['models']:
            for sp in spec['splits']:
                for H in spec['horizons']:
                    yield d, m, sp, H


def skey(d, m, sp, H):
    return f'{d}|{m}|{sp}|H{H}'


def fmt_lr(lr):
    m, e = f'{lr:.0e}'.split('e')
    return f'{m}e{int(e)}'


def fmt_x(x):
    return f'{x:g}'


def fit_id(spec, d, m, sp, H, W, lr, seed, arch='Naive'):
    pre = 'fit' if arch == 'Naive' else 'init' + arch
    return f'{pre}-{d}-{m}-{sp}-H{H}-W{W}-lr{fmt_lr(lr)}-s{seed}-{spec["track"]}'


def fit_config(spec, d, m, sp, H, W, lr, seed, arch='Naive', initial_job=None):
    c = dict(kind='fit', dataset=d, model=m, split=sp, H=H, W=W, lr=lr, seed=seed, track=spec['track'],
             common_W=common_W(spec, d, sp, H), architecture=arch, batch_size=32, precision='FP32',
             max_epochs=spec['max_epochs'])
    if initial_job:
        c['initial_job'] = initial_job
    return c


def stream_id(spec, evaluation, d, m, sp, H, method, W, lr, x, seed):
    return f'stream-{evaluation}-{d}-{m}-{sp}-H{H}-{method}-W{W}-lr{fmt_lr(lr)}-x{fmt_x(x)}-s{seed}-{spec["track"]}'


def stream_config(spec, evaluation, d, m, sp, H, method, W, lr, x, seed, parent):
    return dict(kind='stream', evaluation=evaluation, dataset=d, model=m, split=sp, H=H, method=method, W=W,
                lr=lr, multiplier=x, online_lr=lr * x, seed=seed, track=spec['track'], initial_job=parent,
                common_W=common_W(spec, d, sp, H), batch_size=32, precision='FP32')


def online_parent(spec, d, m, sp, H, method, lr, seed):
    W = spec['online_window']
    if method.startswith('PROCEED'):
        return fit_id(spec, d, m, sp, H, W, lr, seed, 'PROCEED')
    return fit_id(spec, d, m, sp, H, W, lr, seed)


# ---------------------------------------------------------------- stages

def stage_lr_fits(spec):
    jobs = {}
    for d, m, sp, H in scenarios(spec):
        for lr in spec['lr_grid']:
            for W in all_windows(spec, sp):
                jobs[fit_id(spec, d, m, sp, H, W, lr, spec['lr_selection_seed'])] = \
                    fit_config(spec, d, m, sp, H, W, lr, spec['lr_selection_seed'])
    return jobs


def select_lr(spec, run):
    """Average rank of each LR over windows (rank inside a window by VALIDATION MSE)."""
    out = {}
    for d, m, sp, H in scenarios(spec):
        grid = sorted(spec['lr_grid'])
        table, usable = {}, []
        for W in all_windows(spec, sp):
            row = {}
            for lr in grid:
                done = Path(run) / 'jobs' / fit_id(spec, d, m, sp, H, W, lr, spec['lr_selection_seed']) / 'DONE.json'
                if done.exists():
                    row[fmt_lr(lr)] = read(done)['VALIDATION_MSE']
            table[str(W)] = row
            if len(row) == len(grid):
                usable.append(W)
        if not usable:
            raise RuntimeError(f'{skey(d, m, sp, H)}: no window has validation results for every LR')
        ranks = {fmt_lr(lr): [] for lr in grid}
        for W in usable:
            order = sorted(grid, key=lambda lr: (table[str(W)][fmt_lr(lr)], lr))
            for r, lr in enumerate(order, 1):
                ranks[fmt_lr(lr)].append(r)
        mean_rank = {k: sum(v) / len(v) for k, v in ranks.items()}
        chosen = min(grid, key=lambda lr: (mean_rank[fmt_lr(lr)], lr))
        out[skey(d, m, sp, H)] = dict(lr=chosen, mean_rank=mean_rank, validation_mse=table,
                                      windows_ranked=usable, seed=spec['lr_selection_seed'])
    return out


def stage_seed_fits(spec, lrs):
    jobs = {}
    for d, m, sp, H in scenarios(spec):
        lr = lrs[skey(d, m, sp, H)]['lr']
        for seed in spec['seeds']:
            for W in all_windows(spec, sp):
                jobs[fit_id(spec, d, m, sp, H, W, lr, seed)] = fit_config(spec, d, m, sp, H, W, lr, seed)
            if 'PROCEED' in spec['online_methods']:
                base = fit_id(spec, d, m, sp, H, spec['online_window'], lr, seed)
                jobs[fit_id(spec, d, m, sp, H, spec['online_window'], lr, seed, 'PROCEED')] = \
                    fit_config(spec, d, m, sp, H, spec['online_window'], lr, seed, 'PROCEED', initial_job=base)
    return jobs


def stage_multiplier_streams(spec, lrs):
    jobs = {}
    if len(spec['online_multipliers']) < 2:
        return jobs
    seed, W = spec['multiplier_selection_seed'], spec['online_window']
    for d, m, sp, H in scenarios(spec):
        lr = lrs[skey(d, m, sp, H)]['lr']
        for method in spec['online_methods']:
            for x in spec['online_multipliers']:
                jobs[stream_id(spec, 'VALIDATION', d, m, sp, H, method, W, lr, x, seed)] = stream_config(
                    spec, 'VALIDATION', d, m, sp, H, method, W, lr, x, seed, online_parent(spec, d, m, sp, H, method, lr, seed))
    return jobs


def select_multipliers(spec, run, lrs):
    out = {}
    seed, W = spec['multiplier_selection_seed'], spec['online_window']
    for d, m, sp, H in scenarios(spec):
        lr = lrs[skey(d, m, sp, H)]['lr']
        for method in spec['online_methods']:
            if len(spec['online_multipliers']) < 2:
                out[f'{skey(d, m, sp, H)}|{method}'] = dict(multiplier=spec['online_multipliers'][0], validation_mse={})
                continue
            scores = {}
            for x in spec['online_multipliers']:
                done = Path(run) / 'jobs' / stream_id(spec, 'VALIDATION', d, m, sp, H, method, W, lr, x, seed) / 'DONE.json'
                if done.exists():
                    scores[x] = read(done)['MSE']
            if not scores:
                raise RuntimeError(f'{skey(d, m, sp, H)} {method}: no VALIDATION stream finished')
            best = min(scores, key=lambda x: (scores[x], x))
            out[f'{skey(d, m, sp, H)}|{method}'] = dict(multiplier=best, validation_mse={fmt_x(k): v for k, v in scores.items()})
    return out


def stage_test_streams(spec, lrs, mults):
    jobs = {}
    for d, m, sp, H in scenarios(spec):
        lr = lrs[skey(d, m, sp, H)]['lr']
        for seed in spec['seeds']:
            for W in frozen_windows(spec, sp):
                jobs[stream_id(spec, 'TEST', d, m, sp, H, 'Frozen', W, lr, 1, seed)] = stream_config(
                    spec, 'TEST', d, m, sp, H, 'Frozen', W, lr, 1, seed, fit_id(spec, d, m, sp, H, W, lr, seed))
            W = spec['online_window']
            for method in spec['online_methods']:
                x = mults[f'{skey(d, m, sp, H)}|{method}']['multiplier']
                parent = online_parent(spec, d, m, sp, H, method, lr, seed)
                variants = [method, method + '-frozen'] if spec['matched_frozen'] else [method]
                for v in variants:
                    jobs[stream_id(spec, 'TEST', d, m, sp, H, v, W, lr, x, seed)] = stream_config(
                        spec, 'TEST', d, m, sp, H, v, W, lr, x, seed, parent)
    return jobs


def write_configs(run, jobs):
    """configs/<job>.json is the job's identity; refuse silent changes to an existing job."""
    for job, c in jobs.items():
        p = Path(run) / 'configs' / f'{job}.json'
        if p.exists():
            if read(p) != c:
                raise RuntimeError(f'config of existing job {job} differs; use a new run name')
        else:
            write(p, c)
