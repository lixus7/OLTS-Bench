**English** | [简体中文](STUDENT_TASKS.zh-CN.md)

# Exercises for newcomers: a small-scale follow-up

Over about two weeks, this plan takes you through four things:

1. Run the study's protocol on your own GPUs and confirm that your scores match the study's.
2. Understand why each rule of the protocol is the way it is.
3. Do one small extension of your own: write down the reading rule first, then run.
4. Hand in a report written the way we write ours.

**You need:** 1–4 A100 40GB GPUs (one is enough; it is just slower).

**Read first (about two hours):**
- the [primer](CONCEPTS.md): how traditional and online forecasting differ. Run its three demos and answer the self-check questions of section 7;
- the [README](../README.md);
- the two-page summary [BUA_brief_onepage_zh.pdf](../reports/full_study/BUA_brief_onepage_zh.pdf) (in Chinese): what the study asks and what it has found so far;
- sections 1–5 of the [guide](GUIDE.md).

---

## Task 0: environment and smoke test (half a day)

Follow sections 1–3 of the [guide](GUIDE.md):

```bash
python -c "import torch, olts.engine; print('OK, CUDA available:', torch.cuda.is_available())"
bash scripts/download_data.sh ETTh1
python -m olts.run configs/smoke.json --jobs-per-gpu 4
```

**Hand in:**
- the output of the first command, which must be `OK, CUDA available: True`;
- the last lines of the smoke test, which must say `0 failed`.

## Task 1: reproduce the ETTh1 quick run (half a day; about 15 minutes on one GPU)

```bash
python -m olts.run configs/etth1_quick.json --jobs-per-gpu 4
python scripts/compare_with_study.py runs/etth1_quick
```

The last line of the comparison should read `22 of 22 TEST runs have a study counterpart; |difference| median 0.0000%, largest 0.0000%`: your 22 TEST streams match the study's scores. On our RTX A5000 the largest relative difference was 4 × 10⁻⁷. If yours differ by a percent or more, your environment or data differ; sort that out before going on.

**Hand in:** the comparison's last line, and one or two sentences for each of these five questions.

1. Which learning rate was chosen, and how? (See `runs/etth1_quick/selections/lr.json`.)
2. Why does the O split have no W2048 or W4096?
3. Neither "frozen W96" nor "Naive-frozen" updates during TEST. What is the difference between them?
4. At horizon 96, when is a forecast issued at time u scored, and when can it be used for an update?
5. Following rule 2 of [how to write the report](REPORTING.md), write the sentence for the cell "O split · ER vs frozen W1024" of your report's Table 1.

<details>
<summary>Reference answers (answer first, then look)</summary>

1. 1e-4 in both splits. Inside each window, the seed-2024 models rank 1e-4, 3e-4 and 1e-3 by validation MSE; the LR with the best mean rank across windows wins (ties go to the smaller). Every window, seed and online model then uses it.
2. Training must leave at least 1,024 training samples for the longest window. The O split's TRAIN has 2,880 rows, so the window W must satisfy 2,880 − 96 − W + 1 ≥ 1,024, that is W ≤ 1,761: the longest window is W1024.
3. Frozen W96 is only trained offline and never updated after deployment. Naive-frozen is the online model itself: it first warms up, that is, it is updated at every step of VALIDATION without forecasting, and at the start of TEST a copy is taken and frozen. Its starting point equals the online model's, so "online vs Naive-frozen" isolates the effect of continuing to update during TEST.
4. At t = u + 96, all 96 targets have arrived. The forecast is scored first, then this sample updates the online model, and then the next forecast is issued.
5. "O split · ER vs frozen W1024: the online error is 5.3% lower than the frozen model's (online better in 1 of 1 pair)." The quick run has a single pair per cell, so it supports a statement about that pair only, not "generally better".

</details>

## Task 2: the ETTh1 student run (1–2 days; about 1–2 hours on four A100s)

This runs everything the main study ran on ETTh1 at horizon 96: DLinear and PatchTST, 3 seeds, both splits.

```bash
python -m olts.run configs/etth1_student.json --gpus 0,1,2,3 --jobs-per-gpu 2
python scripts/compare_with_study.py runs/etth1_student
# the same tables from the study's scores, to compare with yours
python -m olts.analyze --streams results/full_study/streams_main.csv --where dataset=ETTh1 --where H=96 --out runs/study_etth1_h96
python -m olts.report --summary runs/study_etth1_h96/summary.json --lang en
```

PatchTST's online streams are much slower than DLinear's: with 4 PatchTST jobs on one A5000, our 8 O-split streams took about 20 minutes. Put at most 2 PatchTST jobs on a GPU and use more GPUs instead.

The study ran the same two backbones and three seeds on ETTh1 at horizon 96. In both reports every cell of Tables 1 and 2 therefore holds 6 pairs, and the tables should be nearly identical. We checked DLinear with seed 2024 only; it matched. For PatchTST we only checked that it runs, not its full 50-epoch scores. The other seeds and PatchTST may differ a little more. Report any difference as it is.

**Hand in:**
- your draft `runs/etth1_student/results/report_en.html`;
- the comparison's last line;
- answers to these four questions.

1. On ETTh1, do the O and T splits lead to the same conclusion? Give one cell from each, written as a sentence that follows the rules.
2. In the matched-copy table: does continuing to update during TEST help by itself? For which method most?
3. Did any run diverge (error above 10 times the frozen model's)? In which table do you see it?
4. How many times slower per forecast is the online model than the frozen W96?

   Note that timings depend on the GPU and on how many jobs share it. With 4 jobs on one A5000 we measured 5–13x; the study, with one job per GPU, measured 3.9–7.2x. To compare timings, run with `--jobs-per-gpu 1` and say in the report how you measured.

## Task 3: one small extension (2–3 days)

**Write the reading rule first and send it to your supervisor before you run.** Pick one option below.

Start with A or B: the study ran these exact configurations, so the comparison script can check your scores. If you have time left, do C or D: these are new questions that the study did not ask on ETTh1.

| | What | Config | Time (rough) | Checkable against the study? |
|---|---|---|---|---|
| A | another dataset: ETTh2 (same size as ETTh1) or ETTm1 (15-minute data, 4x longer) | `configs/etth2_quick.json`, `configs/ettm1_quick.json` | ETTh2 about 15 minutes; ETTm1 about 1 hour on one GPU (estimate) | yes |
| B | horizon 336 | `configs/etth1_h336_quick.json` | about 15 minutes | yes |
| C | a smaller online learning rate: multiplier picked on validation from 0.03, 0.1, 0.3, 1 | `configs/etth1_small_lr.json` | about 40 minutes on one GPU (estimate) | no |
| D | a properly tuned frozen model against long-window online W1024 | `configs/etth1_tuned_w1024.json` | about 30 minutes on one GPU (estimate) | no |

Write the config yourself first, then compare it with the reference version in `configs/`. Section 5 of the [protocol](PROTOCOL.md) has the table of windows and of the window that aligns training origins (`common_W`).

Put the reading rule in `runs/<name>/PREREG.md` **before running**, in this form:

```
Question: (one sentence)
Comparison: the online side is ..., the reference is ..., a pair consists of ...
Scope: datasets, backbones, splits, H, seeds
Reading rule (median gain; positive = lower online error):
  >= 0: means ...
  <= -5%: means ...
  in between: no conclusion
Guess before running (optional):
Written on: date and time
```

For A and B the question can be "does the ETTh1 conclusion hold here?". A rule can also count cells, for example "holds if the frozen model is better in at least 12 of the 15 T-split comparisons".

**Hand in:**
- before running: `PREREG.md`;
- after running: the draft report;
- for A or B, also the comparison's last line.

## Task 4: write the report (2–3 days)

Hand in two documents:
- a two-page summary;
- the full version (PDF).

They cover the results of Task 2 and the extension of Task 3: the pre-registered reading rule, the results, and whether the rule held. Follow [how to write the report](REPORTING.md). Before sending, go through its checklist and ask a classmate who was not involved to read it cold.

**Files to hand in:**
- the reports as PDF and HTML;
- for every run, `runs/<name>/spec.json` and the whole `results/` folder;
- the `PREREG.md` of Task 3.

Leave out `jobs/` and `data/`.

## Schedule (suggested)

| When | Tasks | Send to your supervisor |
|---|---|---|
| week 1, first half | Tasks 0 and 1 | the comparison's last line, answers to the five questions |
| week 1, second half | Task 2 | draft report, comparison with the study's tables, answers to the four questions |
| start of week 2 | Task 3 | the reading rule, **before running** |
| week 2 | Tasks 3 and 4 | the two-page summary and the full version |

## Rules

- **Make every choice on validation data only.** This covers learning rate, multiplier and number of epochs. Once you have seen TEST scores, do not change a setting.
- **New config, new name.** Report failed jobs as they are; never rerun only the runs whose results look bad.
- **Report all seeds.**
- **Do not compare absolute errors across splits.**
- **When something breaks,** read section 9 of the [guide](GUIDE.md) first, then ask. Attach the last few dozen lines of `runs/<name>/logs/<job>.log`.
