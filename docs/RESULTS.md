**English** | [简体中文](RESULTS.zh-CN.md)

# Results so far

The full write-up, every table and all caveats are in the Chinese report [reports/full_study/BUA_brief_full_zh.pdf](../reports/full_study/BUA_brief_full_zh.pdf) (below, "the full brief"). This page lists the conclusions and the key numbers.

**How to read the numbers:**
- Percentages are gains: how much lower the online model's MSE is than the reference's.
- Positive means the online model is better; negative means the reference (usually the frozen model) is better.
- Every number is a median over pairs of runs; "k/n" means the online model was better in k of n pairs.

## Main study

**Scope:**
- 10 datasets, DLinear and PatchTST;
- horizons 24/96/336, 3 seeds;
- two splits, one fixed training budget.

**Traditional split T: the frozen long-window model wins everywhere** (full brief, Table 2)
- 3 online methods x 5 frozen windows = 15 comparisons. The frozen model is better in all of them; the online MSE is 12% to 49% higher.
- For the 9 comparisons against W512–W1024, the dataset-resampling intervals lie below 0, and at most 1 of 10 datasets favours the online model.

**Online split O: mixed** (full brief, Table 1)
- Against W512 and W720 the online model loses: Naive 22.8% and 34.2% higher error, ER 10.3% and 9.9%, PROCEED 14.4% and 14.6%.
- Against W1024 and W2048, ER and PROCEED tie with the frozen model (from 2.1% higher to 1.8% lower error); Naive loses (12.0% and 19.9% higher).
- Against W4096 the online model wins (17.0%–32.1% lower error). This is mainly because the frozen W4096 model is itself worse: on the same pairs its MSE is 1.24 times that of W2048 (full brief, Table 3).
- With datasets resampled, all 15 O-split intervals cross 0.

**By dataset** (full brief, Table 5; one slice only, ER against W1024)
- The 7-variable datasets (ETTh1, ETTh2, ETTm1, ETTm2, Beijing) favour the online model in 9 of their 10 O-split cells.
- The wider datasets lose in all 8 cells that have data: Weather (21 variables), Solar (137), PEMS08 (170), Electricity (321), PEMS-BAY (325).

**Updating itself** (full brief, Figure 2)
- The online model is copied at the start of TEST and the copy is frozen. The updating copy ranges from 5.0% higher error (T, Naive) to 4.7% lower (O, PROCEED) against its frozen twin.
- A few runs diverge: 16 of the 1,079 short-window online runs have more than 10 times the frozen error, all with PatchTST. The worst reaches about 2 x 10^11 times.

**Same input length** (full brief, Table 4)
- Online W96 against frozen W96, T split: 14.1% (Naive), 8.1% (ER) and 7.0% (PROCEED) higher error.
- O split, roughly a tie: −6.7%, 0.0%, +2.0%.

**Cost** (full brief, Figure 4)
- Per forecast, the online model takes 3.9x (Naive), 6.5x (ER) and 7.2x (PROCEED) the time of a frozen W96.
- Offline training of the online model is 12%–21% cheaper than a frozen W1024 and 65%–87% cheaper than a frozen W4096.

## Supplementary rounds (full brief, Section 8; four rounds, mostly W1024 and H = 96)

| Section | Question | Result |
|---|---|---|
| 8.1 | Do updates hurt long-window models because the online LR is too large? | Most likely. With a smaller online LR (offline x 0.03–1, chosen on validation), pooled over 7 datasets, online W1024 has 7.4% lower error than frozen W1024 in O (46/58) and 0.7% higher in T (27/60). Electricity O is the exception (40% higher) |
| 8.2 | Why is the O split so much better for online models? | Not because of the time period: on the last ~20% of each series the long-window online model still has 13% lower error in O, while T is a tie |
| 8.6 | And when the frozen model is tuned too? | Online W1024 stays slightly better: 3.7% lower error in O (39/54), 0.7% in T (44/54) |
| 8.7 | Where does the O-split advantage come from? | The frozen model saw only the first 20% of the data. Retrained once on the first 75%, it beats the continuously updated online model on the last 20%: the online error is 8.0% higher (online better in only 12/54) |
| 8.8 | With the same 1,024 steps of history, who wins? | It depends on whether the frozen model is tuned. Against a tuned frozen W1024, a memory-capped online W96 has 11.4% (O) and 12.2% (T) higher error. Against the main-table frozen model the online side wins O by 22.0%. The online side's offline training was not tuned, so which use of history is better remains open |
| 8.9 | Are the PEMS-BAY blow-ups also an LR problem? | Yes: runs with more than 10x the frozen error drop from 3 of 12 to 0 of 12. Even so, the online error is still 28%–44% higher than the frozen W96's |
| 8.10 | And at horizon 336? | Same direction as at 96: 1.3% lower error in O, 3.1% higher in T |
| 8.11 | Why is Electricity O the exception? | The variable with column name 146 is almost constant in TRAIN; after scaling it makes up 98% of the frozen model's TEST error. The other 320 variables are also worse online. A rerun without it is still pending |

## Not settled yet

- Periodic retraining versus online updating at equal compute: retraining was done once, only for O, W1024 and H = 96.
- With equal history, whether reading it as input or absorbing it through updates is better: both sides need tuning first.
- The ranking after every method is fully tuned: the main tables share one LR across windows.
- How small the online step should be: the smallest multiplier tried (0.003) was chosen most often.
- Other horizons and backbones: no supplementary round at H = 24; H = 336 only with DLinear.
- Formal significance: there are only 7–10 datasets, and pairs from one dataset are correlated.
