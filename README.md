**English** | [简体中文](README.zh-CN.md)

# OLTS-Bench

**Can a short-window online learner beat a long-window frozen forecaster?**

An *online* model reads only the last 96 steps, but after deployment it keeps updating its parameters with the true values as they arrive. A *frozen* model reads a much longer history (512 to 4,096 steps) and is never changed after training. OLTS-Bench compares the two under one causal protocol: the same code ([OnlineTSF](https://github.com/SJTU-DMTai/OnlineTSF)), the same training budget, and no forecast is scored or used for an update before all of its targets have arrived.

This repository contains the study's code, its released scores and reports, and a short path for newcomers: run one dataset with a few fast models on your own GPUs, then turn the output into a report like ours.

**New here?**
1. Read the [primer](docs/CONCEPTS.md) first: how traditional and online forecasting differ in data, training and inference, with 8 figures and 3 small demos that need no GPU.
2. Then do the [exercise plan](docs/STUDENT_TASKS.md). Over about two weeks it takes you from installation to one small experiment of your own and a report, with what to hand in at each step.

## Results so far

Full study: 10 datasets, DLinear and PatchTST, horizons 24/96/336, 3 seeds, two splits. Each number is a median over pairs of runs: how much lower the online model's MSE is than the frozen model's (negative = the frozen model is better). Details and caveats: [docs/RESULTS.md](docs/RESULTS.md).

- **Traditional split T** (train/validation/test 60-70 / 10-20 / 20): the frozen long-window model wins all 15 comparisons (3 online methods x 5 frozen windows). The online MSE is 12% to 49% higher.
- **Online split O** (20 / 5 / 75): mixed.
  - Online loses against W512 and W720.
  - ER and PROCEED tie with W1024 and W2048 (−2.1% to +1.8%).
  - Online wins against W4096, mainly because the frozen W4096 model is itself worse than W2048.
- **Supplementary rounds** (mostly W1024 and H = 96):
  - With a smaller online learning rate, updating no longer hurts a long-window model.
  - When the frozen model is also tuned, updating still helps slightly: online W1024 has 3.7% (O) and 0.7% (T) lower error.
  - In the O split, a frozen model retrained once on more recent data beats the continuously updated online model: the online error is 8.0% higher.

## Quick start (one GPU, about 20 minutes of compute)

```bash
git clone --recursive https://github.com/lixus7/OLTS-Bench.git
cd OLTS-Bench
conda create -n olts python=3.10 -y && conda activate olts
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash scripts/download_data.sh ETTh1
python -m olts.run configs/smoke.json --jobs-per-gpu 4         # installation check: 2 epochs only, not a result
python -m olts.run configs/etth1_quick.json --jobs-per-gpu 4   # ETTh1, DLinear, H = 96, seed 2024, both splits
python scripts/compare_with_study.py runs/etth1_quick           # your scores against the study's
```

On one RTX A5000 the smoke test took about 4 minutes and the quick run about 14. The quick run reproduced all 22 of the study's TEST scores for this configuration (relative difference below 0.0001%).

Then open `runs/etth1_quick/results/report_en.html` (or `report_zh.html`). Step-by-step instructions, timings and troubleshooting: [docs/GUIDE.md](docs/GUIDE.md).

## What is in this repository

| Path | Contents |
|---|---|
| `olts/` | The benchmark: data preparation, engine (OnlineTSF kernels plus one causal scheduler), worker, planner, local multi-GPU runner, analysis, draft report |
| `third_party/OnlineTSF` | Upstream OnlineTSF as a git submodule, pinned to commit `0a86461`, the commit the study used |
| `configs/` | Experiment specs: `smoke`, `etth1_quick`, `etth1_student`; extensions `etth2_quick`, `ettm1_quick`, `etth1_h336_quick`, `etth1_small_lr`, `etth1_tuned_w1024` |
| `demos/` | Three small demos that need no GPU: the two splits; score, update and forecast step by step; frozen, online and leaky use of one model |
| `docs/` | [CONCEPTS](docs/CONCEPTS.md) (primer), [STUDENT_TASKS](docs/STUDENT_TASKS.md) (exercises for newcomers), [GUIDE](docs/GUIDE.md) (step by step), [PROTOCOL](docs/PROTOCOL.md) (exact rules), [REPORTING](docs/REPORTING.md) (how to write the report), [RESULTS](docs/RESULTS.md) (findings) |
| `reports/full_study/` | The study's reports (Chinese): the full brief and the two-page summary |
| `reports/example_etth1_quick/` | The draft report produced by `configs/etth1_quick.json` on our machine |
| `results/full_study/` | Per-run scores of the main study (CSV); they reproduce its Tables 1, 2, 4 and Figure 2 |
| `scripts/` | `download_data.sh` (ETT data), `compare_with_study.py` (your scores against the study's), `html_to_pdf.sh` (report to PDF) |

## Reproduce the main tables from the released scores

```bash
python -m olts.analyze --streams results/full_study/streams_main.csv --out runs/full_study_check
python -m olts.report --summary runs/full_study_check/summary.json --lang en
```

All 30 cells of Tables 1 and 2 of the full brief, Table 4 and Figure 2 come out unchanged.

## Acknowledgements and references

- OnlineTSF and PROCEED: L. Zhao and Y. Shen, *Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting*, KDD 2025. Code: <https://github.com/SJTU-DMTai/OnlineTSF>. This repository does not copy OnlineTSF; it uses it as a submodule.
- DLinear: A. Zeng et al., *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023.
- PatchTST: Y. Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023.
- Experience replay (ER): A. Chaudhry et al., *On Tiny Episodic Memories in Continual Learning*, 2019.
- ETT datasets: H. Zhou et al., *Informer*, AAAI 2021, <https://github.com/zhouhaoyi/ETDataset>.
