**English** | [简体中文](PROTOCOL.zh-CN.md)

# Protocol

These are the exact rules of the study. The code in `olts/` implements them; the engine, the worker and the preprocessing are the study's own files (see the last section).

## 1. Question and comparisons

Does an online model with a short input window (W = 96), updated continuously after deployment, beat a frozen model with a long input window? If it does, is that because the updates absorb more history than its window shows, or because updating itself helps (for example, by following drift)?

| Comparison | Online side | Reference | What it answers |
|---|---|---|---|
| 1. Fixed window (main tables) | online W96 | frozen model trained at W = 512, 720, 1,024, 2,048, 4,096; each W reported on its own | which deployment is more accurate, at a fixed training budget |
| 2. Matched copy | online W96 | the same online model copied at the start of TEST and frozen from then on | does continuing to update help, starting from an identical state |
| 3. Equal history | online W96 at time t | the shortest frozen window W ≥ K(t), where K(t) is the span of data the online model has used since its warm-up began; times without such a model are skipped | is the gain only "more history" |

Comparison 3 covers only a small part of TEST (median share 1.8% for O, 0% for T), so it could not answer its question. This repository runs comparisons 1 and 2. Each stream's `per_origin.json` records `history_span` = K(t), so comparison 3 can be computed from it.

## 2. Data and splits

| Dataset | File | Variables | Rows used | O: TRAIN / VALIDATION / TEST | T: TRAIN / VALIDATION / TEST |
|---|---|---|---|---|---|
| ETTh1 | `ETTh1.csv` | 7 | 14,400 | 2,880 / 720 / 10,800 | 8,640 / 2,880 / 2,880 |
| ETTh2 | `ETTh2.csv` | 7 | 14,400 | 2,880 / 720 / 10,800 | 8,640 / 2,880 / 2,880 |
| ETTm1 | `ETTm1.csv` | 7 | 57,600 | 11,520 / 2,880 / 43,200 | 34,560 / 11,520 / 11,520 |
| ETTm2 | `ETTm2.csv` | 7 | 57,600 | 11,520 / 2,880 / 43,200 | 34,560 / 11,520 / 11,520 |
| Weather | `weather.csv` | 21 | 52,696 | 10,539 / 2,635 / 39,522 | 36,887 / 5,270 / 10,539 |
| Electricity | `electricity.csv` | 321 | 26,304 | 5,260 / 1,316 / 19,728 | 18,412 / 2,632 / 5,260 |
| Solar | `solar_AL.txt` | 137 | 52,560 | 10,512 / 2,628 / 39,420 | 36,792 / 5,256 / 10,512 |
| PEMS-BAY | `pems-bay.h5` | 325 | 52,116 | 10,423 / 2,606 / 39,087 | 36,481 / 5,212 / 10,423 |
| PEMS08 | `PEMS08.npz` | 170 | 17,856 | 3,571 / 893 / 13,392 | 10,713 / 3,572 / 3,571 |

- **The split rules.**
  - O: TRAIN = int(0.2 N), TEST = int(0.75 N), VALIDATION = the rest.
  - T: TRAIN = int(r N) with r = 0.6 (ETT, PEMS08) or 0.7 (the others), TEST = int(0.2 N), VALIDATION = the rest.
  - These rules reproduce the number of TEST origins of all 60 (dataset, split, H) scenarios of the study.
- **The tenth dataset.** Beijing air quality was rebuilt from raw Quotsoft files, keeping the true missing-value pattern. It is not distributed here.
- **Scaling.** Each variable is standardised with the mean and (population) standard deviation of TRAIN only.
- **Missing values.**
  - Model inputs are forward-filled from the past only; leading gaps take the TRAIN mean.
  - Labels keep the gaps, and losses and scores use observed labels only.
  - Weather's −9999 and PEMS-BAY's zero speed count as missing.
  - Solar's night zeros and PEMS08's zero flow are real values.

## 3. Models and online methods

| | Setting |
|---|---|
| DLinear | OnlineTSF implementation, shared weights across variables (`individual=False`), moving-average kernel 25 |
| PatchTST | patch 16, stride 8, 3 layers, RevIN. ETTh1/ETTh2: d_model 16, 4 heads, d_ff 128, dropout 0.3. Weather, Electricity, ETTm1, ETTm2: d_model 128, 16 heads, d_ff 256, dropout 0.2. Others: d_model 64, 16 heads, d_ff 128, dropout 0.2 |
| Naive | after each new completed sample, one gradient step on it (batch size 1) |
| ER | as Naive, plus experience replay: buffer of 500 samples, 8 replayed per update, replay loss weight 0.2 |
| PROCEED | OnlineTSF's PROCEED (concept-drift estimate → generated adapters). Its paired (recent, current) updates during warm-up wait until all labels of the current sample have arrived |

The online learning rate equals the offline one in the main tables (multiplier 1).

## 4. Causal schedule

At every time step t, in this order:

1. Score each forecast whose H targets have now all arrived: the forecast issued at u = t − H.
2. Update the online model with the sample whose input ends at u and whose targets end at t. No label later than t is ever used.
3. Issue the forecast for t+1 … t+H.

The model, optimizer, replay buffer and random state carry over from step to step; nothing is reset per origin.

- **Warm-up.** Online models start updating during VALIDATION and continue through TEST. The matched copy follows the same warm-up and stops updating at the start of TEST. Frozen models never update.
- **Scored origins.** All forecasts issued during TEST whose targets lie inside TEST. Every stream of a scenario is scored on the same origins and the same observed targets. For example, ETTh1 · O · H96 has 10,705 origins.

## 5. Training budget (main tables, "track C")

- **Optimisation.** FP32, batch 32, Adam, MSE on observed targets, 50 epochs, no early stopping, last epoch kept. Deterministic algorithms are on.
- **Training origins.** All windows of a scenario (dataset, backbone, split, H) use the same training targets. Origins run from `common_W − 1` to the end of TRAIN minus H, where `common_W` is the longest window of the scenario. Short windows therefore get no extra samples. At least 1,024 origins are required, which is why ETTh1/ETTh2 stop at W1024 in the O split.

  Longest window (`common_W`) per scenario in the study, at H = 96:

  | Dataset | DLinear O | DLinear T | PatchTST O | PatchTST T |
  |---|---|---|---|---|
  | ETTh1, ETTh2 | 1024 | 4096 | 1024 | 4096 |
  | ETTm1, ETTm2 | 4096 | 4096 | 4096 | 4096 |
  | Weather | 4096 | 4096 | 2048 | 2048 |
  | Electricity | 4096 | 4096 | 720 | 720 |
  | Solar | 4096 | 4096 | 1024 | 1024 |
  | PEMS-BAY | 4096 | 4096 | 720 | 720 |
  | PEMS08 | 2048 | 4096 | 1024 | 1024 |

  PatchTST's shorter limits come from GPU memory: its cost grows with variables × input length. Batch size, precision and epochs were never reduced to make a window fit.
- **Learning rate.** One LR per scenario, chosen from 1e-4, 3e-4 and 1e-3 using seed 2024 on VALIDATION. The LRs are ranked inside each window, and the best average rank wins (ties go to the smaller LR). The chosen LR is used by every window, every seed, the online models' offline training and the online updates.
- **Seeds.** 2024, 2025 and 2026.

## 6. Scores and summaries

- **Per stream.** MSE and MAE over all observed targets of all scored origins, on the TRAIN-standardised scale (the raw scale is stored too).
- **Pair.** An online stream and a reference stream with the same dataset, backbone, split, H and seed.
  - Gain = (reference MSE − online MSE) / reference MSE.
  - Log ratio = ln(reference MSE / online MSE).
- **Table cell.** Median gain over its pairs, the number of pairs n, and how many favour the online model. Means are not used: one diverging online run of the study reached about 2 × 10^11 times the frozen model's error, and a single run like that dominates any mean.
- **Time blocks.** TEST is cut into 10 contiguous blocks, and gains are recomputed on the first three and the last three.
- **Robustness.** With 3 or more datasets, whole datasets are resampled 2,000 times; the 95% interval of the pooled median is reported, along with how many datasets favour the online model. This is descriptive, not a significance test: pairs from the same dataset share TEST data.

## 7. Supplementary rounds (Section 8 of the full brief)

| Round | Protocol | In this repository |
|---|---|---|
| 1–2 | online LR = offline LR × multiplier from {0.03, 0.1, 0.3, 1}. The multiplier is chosen with seed 2024 on VALIDATION: warm-up on the first half, score on the second half; ties go to the smaller one; then it is locked for all seeds. Frozen models are not retrained | `online_multipliers`; `configs/etth1_small_lr.json` |
| 3 | The frozen model is tuned too: LR from {1e-5, 3e-5, 1e-4, 3e-4, 1e-3} on VALIDATION, early stopping (up to 50 epochs, patience 5, best epoch). Online multipliers from {0.003, 0.01, 0.03, 0.1, 0.3, 1}; online starts from the same tuned model | `track: "T"`; `configs/etth1_tuned_w1024.json` |
| 4A | Memory cap: two staggered online replicas, each reset to the trained checkpoint every 832 steps, so no post-deployment data older than 1,024 steps is used | not included |
| 4B | O split: the frozen model is retrained on the first 75% (validation 75–80%) and scored on the last 20% against the online model | not included |
| 4C, 4D | PatchTST × PEMS-BAY with small LR; DLinear at H = 336 | set `models`, `datasets`, `horizons`, `online_multipliers` |

Pre-registered readings were written before each round ran, for example "median ≥ 0 means not worse; ≤ −5% means worse; in between, no conclusion". Write yours down before you run, too.

## 8. This code versus the study's cluster code

The study ran every job on a SLURM cluster, with sealed source manifests, resource probes, storage gates and a remote delivery pipeline. This repository runs the same jobs on local GPUs.

- `olts/engine.py` and `olts/worker.py` are the study's `engine.py` and `worker.py`, with these changes only:
  - the OnlineTSF path;
  - package imports;
  - provenance: the git commit is recorded instead of a sealed-manifest hash;
  - an optional `max_epochs` cap, used only by the smoke config.
- `olts/data.py` is the study's preprocessing.
- The planner, runner, collection, analysis and report are new.
- The released scores in `results/full_study/` were produced by the study's code with OnlineTSF `0a86461`, torch 2.4.1+cu118 and numpy 1.24.4.
