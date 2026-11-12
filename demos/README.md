[English](#english) | [中文](#中文)

## English

Three small scripts for the [primer](../docs/CONCEPTS.md). They run on a CPU, in a second to under a minute each, and need `data/ETTh1.csv`
(`bash scripts/download_data.sh ETTh1`); demo 2 needs no data at all.

| Script | What it shows |
|---|---|
| `demo1_splits.py` | The traditional split and the online split of one dataset: rows, dates, what the scaler sees, how many forecasts each segment yields |
| `demo2_schedule.py` | What an online stream does at every time step: score, update, forecast. `--leaky` shows the schedule that uses values from the future |
| `demo3_offline_vs_online.py` | One small linear model trained offline, then used frozen, online, and with the leaky schedule; prints the TEST MSE of each |

These are teaching scripts. Their numbers must not go into a report: the benchmark itself is `python -m olts.run`.

## 中文

配合[入门引导](../docs/CONCEPTS.zh-CN.md)的三个小脚本。都在 CPU 上跑，几秒到几十秒，需要 `data/ETTh1.csv`
（`bash scripts/download_data.sh ETTh1`）；演示 2 不需要数据。

| 脚本 | 演示什么 |
|---|---|
| `demo1_splits.py` | 同一个数据集的传统划分和在线划分：行号、日期、标准化的基准、每段能出多少个预测 |
| `demo2_schedule.py` | 在线测试流每个时刻做的事：评分、更新、预测。加 `--leaky` 看用到未来数据的那种顺序 |
| `demo3_offline_vs_online.py` | 一个小线性模型离线训练后，分别按冻结、在线、泄漏三种方式使用，打印各自的测试 MSE |

这些是教学脚本，数字不能写进报告。基准本身用 `python -m olts.run` 跑。
