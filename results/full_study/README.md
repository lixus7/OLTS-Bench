[English](#english) | [中文](#中文)

## English

Per-run scores of the main study (track C: 50 epochs, last epoch kept; one LR per scenario; multiplier 1), exported from the study's final delivery. Only runs adopted into the main tables are included.

- `streams_main.csv`: one row per TEST stream (4,034 rows), with the same columns as a local `runs/<name>/results/streams.csv`. `dataset` = Beijing is the study's rebuilt Beijing air-quality data (not distributed). Online methods appear at W = 96. Matched frozen copies are named `<method>-frozen`.
- `fits_main.csv`: one row per offline fit (3,490 rows): dataset, model, split, H, seed, window, LR, `common_W`, epochs, training seconds.

Check: `python -m olts.analyze --streams results/full_study/streams_main.csv --out runs/full_study_check` reproduces the full brief's Tables 1, 2 and 4 and Figure 2. `python scripts/compare_with_study.py runs/<name>` compares a local run with these scores.

## 中文

主实验每次运行的分数（C 轨道：训练 50 轮取最后一轮；每个场景一个学习率；倍率 1），从研究的最终交付里导出，只收主表采用的运行。

- `streams_main.csv`：每条测试流一行（4,034 行），列和本地 `runs/<名字>/results/streams.csv` 一样。`dataset` 为 Beijing 的是研究重建的北京空气质量数据（不分发）。在线方法都是 W = 96。同起点冻结副本名为 `<方法>-frozen`。
- `fits_main.csv`：每个离线训练一行（3,490 行）：数据集、模型、划分、H、种子、窗口、学习率、`common_W`、训练轮数、训练秒数。

核对：`python -m olts.analyze --streams results/full_study/streams_main.csv --out runs/full_study_check` 能复现完整版的表 1、表 2、表 4 和图 2。`python scripts/compare_with_study.py runs/<名字>` 把本地运行和这些分数逐条对比。
