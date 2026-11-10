"""Run an experiment spec end to end on local GPUs.

    python -m olts.run configs/etth1_quick.json --gpus 0,1,2,3

Stages run in order (data -> LR fits -> LR choice -> seed fits -> multiplier choice -> TEST streams),
then results are collected, analysed and a draft report is written. Jobs that already finished
(DONE.json) are skipped, so the same command resumes an interrupted run.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, read, write, now, source_info
from .data import prepare
from . import plan


def job_state(run, job):
    d = Path(run) / 'jobs' / job
    if (d / 'DONE.json').exists():
        return 'done'
    if (d / 'FAILED.json').exists():
        return 'failed'
    return 'partial' if d.exists() else 'new'


def set_aside(run, job):
    """The worker never overwrites a job folder; move an unfinished one away before rerunning."""
    src = Path(run) / 'jobs' / job
    dst = Path(run) / 'jobs_attic' / f'{job}.{time.strftime("%Y%m%dT%H%M%S")}'
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def execute(run, jobs, gpus, per_gpu, retry_failed, keep_predictions, label):
    plan.write_configs(run, jobs)
    todo = []
    for job in jobs:
        state = job_state(run, job)
        if state == 'done':
            continue
        if state == 'failed' and not retry_failed:
            continue
        if state in ('failed', 'partial'):
            set_aside(run, job)
        todo.append(job)
    skipped_failed = [j for j in jobs if job_state(run, j) == 'failed']
    print(f'[{label}] {len(jobs)} jobs: {len(jobs) - len(todo) - len(skipped_failed)} already done, '
          f'{len(todo)} to run, {len(skipped_failed)} failed earlier (use --retry-failed)', flush=True)
    slots = [g for g in gpus for _ in range(per_gpu)]
    running = {}
    logs = Path(run) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''),
               CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8'))
    failed, started = [], time.time()
    last_note = started
    while todo or running:
        for i, gpu in enumerate(slots):
            if i in running or not todo:
                continue
            job = todo.pop(0)
            log = open(logs / f'{job}.log', 'w')
            p = subprocess.Popen([sys.executable, '-m', 'olts.worker', '--run', str(run), '--job', job],
                                 env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            running[i] = (p, job, log)
        time.sleep(0.5)
        for i in list(running):
            p, job, log = running[i]
            if p.poll() is None:
                continue
            log.close()
            del running[i]
            if job_state(run, job) != 'done':
                failed.append(job)
                print(f'  FAILED {job} (see {logs / (job + ".log")})', flush=True)
            elif not keep_predictions:
                for name in ('predictions.npy', 'deployment_start.pt', 'deployment_end.pt'):
                    f = Path(run) / 'jobs' / job / name
                    if f.exists():
                        f.unlink()
        if running and time.time() - last_note > 60:
            last_note = time.time()
            print(f'  {label}: {len(jobs) - len(todo) - len(running)}/{len(jobs)} finished', flush=True)
    print(f'[{label}] finished in {time.time() - started:.0f} s; {len(failed)} failed', flush=True)
    return failed


def visible_gpus(arg):
    """GPU ids for CUDA_VISIBLE_DEVICES. When a scheduler (SLURM, PBS, IONA) has already set
    CUDA_VISIBLE_DEVICES, indices refer to the devices it granted, never to physical GPUs."""
    granted = [g.strip() for g in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if g.strip()]
    if not granted:
        try:
            out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=30).stdout
            granted = [str(i) for i, line in enumerate(l for l in out.splitlines() if l.startswith('GPU '))]
        except (OSError, subprocess.SubprocessError):
            granted = []
    if not granted:
        sys.exit('no GPU visible: this benchmark needs CUDA GPUs')
    if arg == 'auto':
        return granted
    picked = []
    for x in arg.split(','):
        i = int(x)
        if i >= len(granted):
            sys.exit(f'--gpus {arg}: only {len(granted)} GPU(s) visible ({granted})')
        picked.append(granted[i])
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--gpus', default='auto', help='"auto" = every visible GPU, or indices such as 0,1,2,3')
    ap.add_argument('--jobs-per-gpu', type=int, default=1, help='small models (DLinear) can share a GPU')
    ap.add_argument('--data-dir', default=str(ROOT / 'data'))
    ap.add_argument('--runs-dir', default=str(ROOT / 'runs'))
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='print the job counts of the first stage and stop')
    ap.add_argument('--no-report', action='store_true')
    a = ap.parse_args()
    spec = plan.load_spec(a.config)
    run = Path(a.runs_dir) / spec['name']
    if (run / 'spec.json').exists() and read(run / 'spec.json') != spec:
        sys.exit(f'{run}/spec.json differs from {a.config}; use a new "name" for a changed experiment')
    write(run / 'spec.json', spec)
    info = source_info()
    if not info['onlinetsf_matches']:
        print(f'WARNING: OnlineTSF at {info["onlinetsf_commit"]}, the study used {info["onlinetsf_expected"]}', flush=True)
    gpus = visible_gpus(a.gpus)
    print(f'GPUs used: {gpus}', flush=True)
    for d in spec['datasets']:
        for sp in spec['splits']:
            meta = prepare(run, d, sp, a.data_dir)
            print(f'data {d} {sp}: rows {meta["boundaries"]} (TRAIN | VALIDATION | TEST ends)', flush=True)
    stage1 = plan.stage_lr_fits(spec)
    if a.dry_run:
        print(f'stage 1 would run {len(stage1)} fits'); return
    args = (gpus, a.jobs_per_gpu, a.retry_failed, spec['keep_predictions'])
    failed = execute(run, stage1, *args, 'stage 1: LR fits')
    lrs = plan.select_lr(spec, run)
    write(run / 'selections' / 'lr.json', lrs)
    for k, v in lrs.items():
        print(f'  chosen LR {k}: {v["lr"]:g} (mean ranks {v["mean_rank"]})', flush=True)
    failed += execute(run, plan.stage_seed_fits(spec, lrs), *args, 'stage 3: seed fits')
    mjobs = plan.stage_multiplier_streams(spec, lrs)
    if mjobs:
        failed += execute(run, mjobs, *args, 'stage 4: multiplier selection')
    mults = plan.select_multipliers(spec, run, lrs)
    write(run / 'selections' / 'multipliers.json', mults)
    failed += execute(run, plan.stage_test_streams(spec, lrs, mults), *args, 'stage 5: TEST streams')
    write(run / 'RUN.json', dict(finished=now(), failed=failed, source=info, gpus=gpus))
    from . import collect, analyze
    collect.collect(run)
    analyze.analyze(run, online_window=spec['online_window'])
    if not a.no_report:
        from . import report
        for lang in ('zh', 'en'):
            print('report:', report.build(run, lang), flush=True)
    if failed:
        print(f'{len(failed)} jobs failed; their logs are in {run / "logs"}', flush=True)


if __name__ == '__main__':
    main()
