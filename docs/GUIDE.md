**English** | [简体中文](GUIDE.zh-CN.md)

# Getting started

This guide takes you through the study's protocol on your own GPUs: one dataset and a few fast models. Along the way you learn what every output means. At the end you write a report like the ones in [reports/full_study](../reports/full_study/). If you are new to the project, the [exercise plan](STUDENT_TASKS.md) says what to run and hand in; this guide explains how.

**You need:**
- Linux;
- 1–4 NVIDIA GPUs (an A100 40GB is plenty; any card with 8 GB or more works);
- Python 3.10 or 3.11;
- about 5 GB of disk.

**Timings** (measured on one RTX A5000 with `--jobs-per-gpu 4`, except the last, which is an estimate):
- installation, about 15 minutes;
- smoke test, about 4 minutes;
- ETTh1 quick run, about 14 minutes on one GPU;
- ETTh1 student run, roughly 1–2 hours on four A100s.

---

## 1. Install

```bash
git clone --recursive https://github.com/lixus7/OLTS-Bench.git
cd OLTS-Bench
conda create -n olts python=3.10 -y && conda activate olts
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121   # pick the build for your CUDA
pip install -r requirements.txt
python -c "import torch, olts.engine; print('OK, CUDA available:', torch.cuda.is_available())"
```

The last line must print `OK, CUDA available: True`. `OK` means OnlineTSF and all dependencies are in place. `True` means PyTorch can use your GPUs; if it says `False`, you installed a CPU-only PyTorch, and every job would fail.

If you cloned without `--recursive`, `third_party/OnlineTSF` is empty. Fix it with:

```bash
git submodule update --init
```

OnlineTSF is pinned to the study's commit `0a86461`. Do not change or update it, or your numbers will not match the study.

## 2. Data

```bash
bash scripts/download_data.sh ETTh1
```

This writes `data/ETTh1.csv` (17,420 rows). The benchmark uses the first 14,400 rows (12/4/4 months, the usual ETT convention). The two splits are:

| Split | TRAIN | VALIDATION | TEST |
|---|---|---|---|
| O (online, 20/5/75) | rows 0–2,879 | 2,880–3,599 | 3,600–14,399 |
| T (traditional, 60/20/20) | rows 0–8,639 | 8,640–11,519 | 11,520–14,399 |

Sources and file names for the other datasets (Weather, Electricity, Solar, PEMS-BAY, PEMS08) are in [data/README.md](../data/README.md).

## 3. Smoke test (about 4 minutes)

```bash
python -m olts.run configs/smoke.json --jobs-per-gpu 4
```

The smoke test runs the whole pipeline with DLinear on the O split of ETTh1:
- it trains frozen W96 and W512 models;
- it runs online Naive, ER and PROCEED, each with its matched frozen copy;
- it collects the results and writes a report.

**It trains for 2 epochs only, so the numbers mean nothing: it only checks the installation.** Normal output looks like this:

```
GPUs used: ['0']
data ETTh1 O: rows [0, 2880, 3600, 14400] (TRAIN | VALIDATION | TEST ends)
[step 1: LR fits] 2 jobs: 0 already done, 2 to run
[step 1: LR fits] finished in 10 s; 0 failed
[step 2: LR choice] ETTh1|DLinear|O|H96: 0.0001 (mean rank of each LR across windows: {'1e-4': 1.0})
[step 3: other seeds and PROCEED initialisation] 3 jobs: 2 already done, 1 to run
[step 3: other seeds and PROCEED initialisation] finished in 6 s; 0 failed
[step 5: TEST streams] 8 jobs: 0 already done, 8 to run
[step 5: TEST streams] finished in 209 s; 0 failed
collected 3 fits and 8 streams into <repository>/runs/smoke/results
analysed 9 pairs into <repository>/runs/smoke/results/summary.json and pairs.csv
report: <repository>/runs/smoke/results/report_zh.html
report: <repository>/runs/smoke/results/report_en.html
```

## 4. First real run: `etth1_quick` (about 14 minutes on one GPU)

```bash
python -m olts.run configs/etth1_quick.json --jobs-per-gpu 4
```

This is the main study's full protocol for ETTh1 with DLinear, horizon 96 and seed 2024, on both splits. The runner prints one line per step:

1. **Step 1, learning-rate fits (30).** Each LR in 1e-4, 3e-4 and 1e-3 is trained once at every window: W96/512/720/1024 for O, plus W2048/4096 for T. Every fit runs 50 epochs and keeps the last one.
2. **Step 2, LR choice.** Inside each window the three LRs are ranked by validation MSE. The LR with the best average rank wins (ties go to the smaller LR). It is then used for every window, for the online models' offline training and for the online updates. The choice is saved in `runs/etth1_quick/selections/lr.json`. On ETTh1 the study chose 1e-4 for both splits, and so should you.
3. **Step 3, other seeds and PROCEED initialisation.** The fits at the chosen LR for every seed, plus PROCEED's adapters initialised from the W96 model. With one seed, only the 2 PROCEED initialisations are new; the other fits were trained in step 1 and show as "already done".
4. **Step 4, multiplier choice.** Skipped here: this config only uses multiplier 1.
5. **Step 5, TEST streams (22).** There are:
   - one frozen model per window;
   - one stream per online method (W96);
   - one "matched frozen copy" per online method: the online model copied at the start of TEST and never updated afterwards.

Every model is scored on the same forecast origins. A forecast issued at time u is scored, and used for updating, only once all 96 of its targets have arrived (t ≥ u + 96).

Choose GPUs with `--gpus 0,1,2,3`; the default is every visible GPU. DLinear is tiny, so four DLinear jobs can share one A100 (`--jobs-per-gpu 4`); for PatchTST use 2. On one RTX A5000 with `--jobs-per-gpu 4`, step 1 took about 8 minutes and step 5 about 6 minutes.

If a run is interrupted (Ctrl-C, a lost session), **run the same command again**: finished jobs are skipped and unfinished ones are rerun. If a job of steps 1–4 fails, the run stops there, so that no later step uses incomplete results; fix the cause, then rerun with `--retry-failed`.

## 5. Read the outputs

Inside `runs/etth1_quick/`:

| File | What it is |
|---|---|
| `spec.json` | the complete configuration of this run (the reference) |
| `data/ETTh1_O/DATA.json` | split boundaries, scaling, number of observed targets |
| `configs/<job>.json` | each job's configuration; `jobs/<job>/DONE.json` is its result; `logs/<job>.log` its log |
| `selections/lr.json` | chosen LR per scenario, each LR's mean rank across windows, and the validation MSE of every fit |
| `results/fits.csv` | one row per offline fit: validation MSE, epochs, time |
| `results/streams.csv` | one row per stream: MSE, MAE, time per forecast (the multiplier-choice VALIDATION streams appear here too, marked `VALIDATION`) |
| `results/pairs.csv` | one row per pair: online MSE, reference MSE, gain, gains in the first and last 30% of TEST |
| `results/summary.json` | per table cell: median, number of pairs, how many favour the online model |
| `results/report_zh.html`, `report_en.html` | the auto-generated draft report |

**Gain** = (frozen MSE − online MSE) / frozen MSE.
- Positive means the online model is better. For example, frozen 0.50 and online 0.40 gives +20%.
- Negative means the frozen model is better. Online 0.60 gives −20%; online at 3.5 times the frozen error gives −250%.

**A pair** is two TEST runs with the same dataset, backbone, split, horizon and seed, each scored by MSE on the same origins; it yields one gain. The quick run has one backbone and one seed, so each table cell holds one pair. The student run pools two backbones and three seeds, so each cell holds 6.

**Check your numbers against the study:**

```bash
python scripts/compare_with_study.py runs/etth1_quick
```

This lines up each of your TEST streams with the study's score for the same configuration. On our RTX A5000 with torch 2.2.1, the quick run matched all 22 of the study's scores to within 0.0001% (relative), although the study ran torch 2.4.1 on other GPUs; [reports/example_etth1_quick/compare_with_study.txt](../reports/example_etth1_quick/compare_with_study.txt) shows the output. If your streams are off by a percent or more, your setup differs: check the data file, the OnlineTSF commit and the chosen LR. The script refuses the smoke run, which trains only 2 epochs.

## 6. Bigger runs

| Config | What | Size (rough) |
|---|---|---|
| `etth1_quick.json` (for comparison) | DLinear, 1 seed, both splits | 32 fits, 22 TEST streams |
| `etth1_student.json` | DLinear + PatchTST, 3 seeds, both splits: everything the main study ran on ETTh1 at H = 96 | 112 fits, 132 TEST streams; roughly 1–2 hours on four A100s (estimate) |
| `etth1_small_lr.json` | DLinear, 3 seeds; online LR = offline LR x multiplier, the multiplier picked on validation from 0.03, 0.1, 0.3, 1 (idea of supplementary round 1) | 56 fits, 24 VALIDATION and 66 TEST streams |
| `etth1_tuned_w1024.json` | DLinear, 3 seeds; long-window online W1024 against a properly tuned frozen W1024: 5 LRs, early stopping (idea of supplementary round 3) | 20 fits, 24 VALIDATION and 30 TEST streams |
| `etth2_quick.json`, `etth1_h336_quick.json` | like the quick run on ETTh2, or at horizon 336; the study ran both, so the comparison script checks them | 32 fits, 22 TEST streams each |
| `ettm1_quick.json` | like the quick run on ETTm1 (4x longer data, windows up to W4096 in both splits); the study ran it | 38 fits, 24 TEST streams; about 1 hour on one GPU (estimate) |

Fields you can set in your own spec (anything missing takes the default):

| Field | Meaning | Default |
|---|---|---|
| `name` | run name, also the folder under `runs/`; use a new name when you change a spec | file name |
| `datasets`, `models`, `splits`, `horizons`, `seeds` | what to run | ETTh1, DLinear, O and T, 96, 2024 |
| `frozen_windows` | input lengths of the frozen models; may differ per split | 96, 512, 720, 1024 |
| `common_W` | every window takes its training samples from the same first step, so short windows do not get extra samples; must leave at least 1,024 training samples | longest window |
| `online_window`, `online_methods` | input length and methods (Naive, ER, PROCEED) of the online models | 96; all three |
| `matched_frozen` | also run the matched frozen copies | true |
| `lr_grid`, `lr_selection_seed` | offline LR candidates and the seed used to choose | 1e-4, 3e-4, 1e-3; 2024 |
| `track` | C = 50 epochs, keep the last (main tables); T = up to 50 epochs, stop after 5 without improvement, keep the best | C |
| `online_multipliers` | online LR multipliers; with more than one, the choice is made on validation | 1 |
| `max_epochs` | for smoke tests only | 50 |
| `keep_predictions` | keep each stream's full prediction array (large) | false |

On SLURM or PBS clusters, run the same command inside your allocation. The runner only uses the GPUs the scheduler granted (it reads `CUDA_VISIBLE_DEVICES`).

## 7. Write the report

1. Open `results/report_en.html`. Every table, its number and a "who against whom, and is + better or worse" sentence for each cell are already filled in.
2. Write the verdict, the explanations and the "can say / cannot say yet" lists following [REPORTING.md](REPORTING.md).
3. Take every number from `summary.json`; never type numbers by hand.
4. Ask someone who did not do the work to read it.
5. Make the PDF: print to A4 from a browser, or run

```bash
bash scripts/html_to_pdf.sh runs/etth1_quick/results/report_en.html
```

Examples:
- the draft we produced is in [reports/example_etth1_quick](../reports/example_etth1_quick/);
- the finished reports of the full study are in [reports/full_study](../reports/full_study/).

## 8. Rules

- **Make every choice on validation data only.** This covers learning rate, multiplier and number of epochs. Changing a setting after seeing TEST scores invalidates the conclusion.
- **New spec, new name.** Existing jobs are never overwritten; reusing a name with a changed spec stops the runner.
- **Do not compare absolute errors across splits.** O and T differ in TEST period and training size, so compare online against frozen only within a split.
- **Report failed jobs.** Do not rerun only the runs whose results look bad. Use `--retry-failed` after fixing a real problem.
- **Report all seeds.** Do not pick seeds.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: OnlineTSF not found` | run `git submodule update --init` |
| `PyTorch cannot use CUDA in this environment` | you have a CPU-only PyTorch: install a CUDA build (section 1) |
| `CUDA out of memory` | lower `--jobs-per-gpu` |
| a job failed | read `runs/<name>/logs/<job>.log` and `jobs/<job>/FAILED.json`; after fixing, rerun with `--retry-failed`. A failure in steps 1–4 stops the run; a failed TEST stream is reported at the end and is missing from the tables |
| `spec.json differs` | the spec changed: give it a new `name` |
| interrupted run | run the same command again |
| disk use | after each stream the prediction array and state snapshots are deleted; `DONE.json` and per-origin errors (`per_origin.json`) stay. Set `keep_predictions` to keep everything |
| running one job by hand | `python -m olts.worker --run runs/<name> --job <job>`. Its config must be in `runs/<name>/configs/`, and `runs/<name>/jobs/<job>` must not exist: the worker never overwrites a job folder, so move an old one away first |
