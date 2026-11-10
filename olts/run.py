"""Run an experiment spec end to end on local GPUs.

    python -m olts.run configs/etth1_quick.json --gpus 0,1,2,3

Steps, in order:
  step 1  LR fits             every window x every LR in lr_grid, with seed lr_selection_seed
  step 2  LR choice           written to selections/lr.json
  step 3  more fits           the other seeds at the chosen LR, plus PROCEED initialisation
                              (fits already trained in step 1 are listed as "already done")
  step 4  multiplier choice   only when online_multipliers has several values (VALIDATION streams)
  step 5  TEST streams
Then the results are collected, analysed and a draft report is written.

Finished jobs (DONE.json) are skipped, so the same command resumes an interrupted run
(jobs stopped by Ctrl-C are rerun). If a job of steps 1-4 fails, the run stops before any later
step can use incomplete results: fix the cause, then rerun with --retry-failed.
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

INTERRUPTED = ('KeyboardInterrupt', 'SystemExit')


def job_state(run, job):
    d = Path(run) / 'jobs' / job
    if (d / 'DONE.json').exists():
        return 'done'
    if (d / 'FAILED.json').exists():
        try:
            kind = read(d / 'FAILED.json').get('type')
        except (OSError, ValueError):
            kind = None
        return 'partial' if kind in INTERRUPTED else 'failed'
    return 'partial' if d.exists() else 'new'


def set_aside(run, job):
    """The worker never overwrites a job folder; move an unfinished one away before rerunning."""
    src = Path(run) / 'jobs' / job
    dst = Path(run) / 'jobs_attic' / f'{job}.{time.strftime("%Y%m%dT%H%M%S")}'
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def worker_env():
    return dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8'))


def execute(run, jobs, gpus, per_gpu, retry_failed, keep_predictions, label):
    """Run the jobs that are not done yet; return the jobs that are failed afterwards."""
    plan.write_configs(run, jobs)
    todo = []
    for job in jobs:
        state = job_state(run, job)
        if state == 'done' or (state == 'failed' and not retry_failed):
            continue
        if state in ('failed', 'partial'):
            set_aside(run, job)
        todo.append(job)
    earlier = [j for j in jobs if job_state(run, j) == 'failed']
    print(f'[{label}] {len(jobs)} jobs: {len(jobs) - len(todo) - len(earlier)} already done, {len(todo)} to run'
          + (f', {len(earlier)} failed earlier (rerun them with --retry-failed)' if earlier else ''), flush=True)
    slots = [g for g in gpus for _ in range(per_gpu)]
    running = {}
    logs = Path(run) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = worker_env()
    started = last_note = time.time()
    try:
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
                    print(f'  FAILED {job} (see {logs / (job + ".log")})', flush=True)
                elif not keep_predictions:
                    for name in ('predictions.npy', 'deployment_start.pt', 'deployment_end.pt'):
                        f = Path(run) / 'jobs' / job / name
                        if f.exists():
                            f.unlink()
            if running and time.time() - last_note > 60:
                last_note = time.time()
                print(f'  {label}: {len(jobs) - len(todo) - len(running)}/{len(jobs)} finished', flush=True)
    except KeyboardInterrupt:
        for p, job, log in running.values():
            p.terminate()
        for p, job, log in running.values():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            log.close()
        sys.exit('\ninterrupted: run the same command again to continue (unfinished jobs will be rerun)')
    failed = [j for j in jobs if job_state(run, j) == 'failed']
    print(f'[{label}] finished in {time.time() - started:.0f} s; {len(failed)} failed', flush=True)
    return failed


def require(run, jobs, step):
    """Stop before the next step if a job of this step is missing."""
    missing = [j for j in jobs if job_state(run, j) != 'done']
    if missing:
        sys.exit(f'{step}: {len(missing)} job(s) did not finish, for example {missing[0]} '
                 f'(log: {Path(run) / "logs" / (missing[0] + ".log")}). The run stops here so that later steps '
                 f'do not use incomplete results. Fix the cause, then rerun with --retry-failed.')


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


def check_cuda(gpu):
    r = subprocess.run([sys.executable, '-c', 'import torch; print(torch.cuda.is_available())'],
                       capture_output=True, text=True, env=dict(worker_env(), CUDA_VISIBLE_DEVICES=str(gpu)))
    if r.stdout.strip() != 'True':
        sys.exit('PyTorch cannot use CUDA in this environment (torch missing, or a CPU-only build). '
                 'Install a CUDA build of PyTorch, see docs/GUIDE.md section 1.\n' + r.stderr[-500:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--gpus', default='auto', help='"auto" = every visible GPU, or indices such as 0,1,2,3')
    ap.add_argument('--jobs-per-gpu', type=int, default=1, help='small models (DLinear) can share a GPU')
    ap.add_argument('--data-dir', default=str(ROOT / 'data'))
    ap.add_argument('--runs-dir', default=str(ROOT / 'runs'))
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='prepare the data, print the size of step 1 and stop')
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
    step1 = plan.stage_lr_fits(spec)
    if a.dry_run:
        print(f'step 1 would run {len(step1)} fits'); return
    check_cuda(gpus[0])
    args = (gpus, a.jobs_per_gpu, a.retry_failed, spec['keep_predictions'])
    plan_jobs = list(step1)

    execute(run, step1, *args, 'step 1: LR fits')
    require(run, step1, 'step 1')
    lrs = plan.select_lr(spec, run)
    write(run / 'selections' / 'lr.json', lrs)
    for k, v in lrs.items():
        print(f'[step 2: LR choice] {k}: {v["lr"]:g} (mean rank of each LR across windows: {v["mean_rank"]})', flush=True)

    step3 = plan.stage_seed_fits(spec, lrs)
    plan_jobs += [j for j in step3 if j not in step1]
    execute(run, step3, *args, 'step 3: other seeds and PROCEED initialisation')
    require(run, step3, 'step 3')

    step4 = plan.stage_multiplier_streams(spec, lrs)
    if step4:
        plan_jobs += list(step4)
        execute(run, step4, *args, 'step 4: multiplier choice (VALIDATION streams)')
        require(run, step4, 'step 4')
    mults = plan.select_multipliers(spec, run, lrs)
    write(run / 'selections' / 'multipliers.json', mults)
    if step4:
        for k, v in mults.items():
            print(f'[step 4: multiplier choice] {k}: x{v["multiplier"]:g}', flush=True)

    step5 = plan.stage_test_streams(spec, lrs, mults)
    plan_jobs += list(step5)
    failed = execute(run, step5, *args, 'step 5: TEST streams')
    write(run / 'plan_jobs.json', plan_jobs)
    write(run / 'RUN.json', dict(finished=now(), failed=failed, source=info, gpus=gpus))
    from . import collect, analyze
    collect.collect(run)
    analyze.analyze(run, online_window=spec['online_window'])
    if not a.no_report:
        from . import report
        for lang in ('zh', 'en'):
            print('report:', report.build(run, lang), flush=True)
    if failed:
        print(f'WARNING: {len(failed)} TEST streams failed and are missing from the tables; logs in {run / "logs"}. '
              f'Fix the cause and rerun with --retry-failed.', flush=True)


if __name__ == '__main__':
    main()
