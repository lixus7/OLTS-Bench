[English](GUIDE.md) | **简体中文**

# 上手指南

这份指南带你在自己的 GPU 上把研究的协议跑一遍：一个数据集、几个跑得快的模型。你要弄懂每个输出是什么意思，最后写出一份像 [reports/full_study](../reports/full_study/) 那样的报告。新同学先看[入门引导](CONCEPTS.zh-CN.md)和[练习任务书](STUDENT_TASKS.zh-CN.md)：前者讲两种预测设定差在哪，后者说跑什么、交什么；这份指南讲怎么跑。

**需要：**
- Linux；
- 1–4 张 NVIDIA GPU（A100 40G 足够，8 GB 显存以上的卡都行）；
- Python 3.10 或 3.11；
- 约 5 GB 磁盘。

**用时**（在一张 RTX A5000 上用 `--jobs-per-gpu 4` 实测；最后一项是估计）：
- 安装约 15 分钟；
- 冒烟测试约 4 分钟；
- ETTh1 快速版约 14 分钟（一张卡）；
- ETTh1 学生版大约 1–2 小时（四张 A100）。

---

## 1. 安装

```bash
git clone --recursive https://github.com/lixus7/OLTS-Bench.git
cd OLTS-Bench
conda create -n olts python=3.10 -y && conda activate olts
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121   # 按你的 CUDA 选版本
pip install -r requirements.txt
python -c "import torch, olts.engine; print('OK, CUDA available:', torch.cuda.is_available())"
```

最后一行必须打印 `OK, CUDA available: True`。`OK` 表示 OnlineTSF 和依赖都齐了；`True` 表示 PyTorch 能用 GPU。如果是 `False`，说明装的是只能用 CPU 的 PyTorch，所有作业都会失败。

如果克隆时忘了加 `--recursive`，`third_party/OnlineTSF` 是空的，补一句：

```bash
git submodule update --init
```

OnlineTSF 固定在研究用的提交 `0a86461`。不要改它，也不要换成别的版本，否则结果和研究对不上。

## 2. 数据

```bash
bash scripts/download_data.sh ETTh1
```

文件会放在 `data/ETTh1.csv`，共 17,420 行。基准只用前 14,400 行（12/4/4 个月，ETT 的通行做法），两种划分的边界是：

| 划分 | 训练 | 验证 | 测试 |
|---|---|---|---|
| O（在线式，20/5/75） | 第 0–2,879 行 | 2,880–3,599 | 3,600–14,399 |
| T（传统，60/20/20） | 第 0–8,639 行 | 8,640–11,519 | 11,520–14,399 |

其他数据集（Weather、Electricity、Solar、PEMS-BAY、PEMS08）的来源和文件名见 [data/README.md](../data/README.md)。

## 3. 冒烟测试（约 4 分钟）

```bash
python -m olts.run configs/smoke.json --jobs-per-gpu 4
```

它在 ETTh1 的 O 划分上，用 DLinear 跑完整个流程：
- 训练冻结 W96、W512 两个模型；
- 跑在线 Naive、ER、PROCEED 和它们的同起点副本；
- 汇总，出报告。

**只训练 2 轮，数字没有意义，只用来确认装好了。** 正常的输出像这样：

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

## 4. 第一个真实实验：`etth1_quick`（一张卡约 14 分钟）

```bash
python -m olts.run configs/etth1_quick.json --jobs-per-gpu 4
```

这一步就是主实验在 ETTh1 · DLinear · 预测 96 步 · 种子 2024 上的完整协议，两种划分都跑。运行器每一步打印一行：

1. **第 1 步，选学习率的训练**（30 个）：学习率 1e-4、3e-4、1e-3 在每档窗口各训练一次（O 划分 W96/512/720/1024，T 划分再加 W2048/4096），每个模型训练 50 轮、取最后一轮。
2. **第 2 步，选学习率**：在每档窗口里按验证集 MSE 给三个学习率排名，取平均名次最好的（平局取小）。所有窗口、在线模型的离线训练和在线更新都用这一个。结果写在 `runs/etth1_quick/selections/lr.json`。ETTh1 上研究两种划分选的都是 1e-4，你的也应该是。
3. **第 3 步，其他种子和 PROCEED 初始化**：选中学习率下每个种子的训练，加上从 W96 模型出发初始化 PROCEED 的适配器。快速版只有一个种子，新的只有 2 个 PROCEED 初始化；其余训练第 1 步已经做过，会显示“already done”。
4. **第 4 步，选倍率**：这个配置只用倍率 1，跳过。
5. **第 5 步，测试流**（22 条）：
   - 每档窗口一条冻结模型；
   - 每种在线方法（W96）一条；
   - 每种在线方法再加一条“同起点副本”：测试开始那一刻复制在线模型，之后不再更新。

所有模型都在同一批测试时刻上评分。在 u 时刻发出的预测，要等它预测的 96 个真实值全部到齐（时刻 t ≥ u + 96）才评分、才拿来更新。

用几张卡就写几张：`--gpus 0,1,2,3`；默认用所有能看到的卡。DLinear 很小，一张 A100 上可以同时跑 4 个作业（`--jobs-per-gpu 4`），PatchTST 建议 2 个。在一张 RTX A5000 上用 `--jobs-per-gpu 4`，第 1 步约 8 分钟，第 5 步约 6 分钟。

中途断了（按了 Ctrl-C、连接断开），**原样再运行一遍同一条命令**：跑完的作业会跳过，没跑完的会重跑。第 1–4 步里有作业失败，运行会停在那一步，免得后面的步骤用到不完整的结果；修好原因后加 `--retry-failed` 重跑。

## 5. 看懂输出

`runs/etth1_quick/` 下面：

| 文件 | 是什么 |
|---|---|
| `spec.json` | 这次实验的完整配置（以它为准） |
| `data/ETTh1_O/DATA.json` | 数据边界、标准化方式、观测值个数 |
| `configs/<作业>.json` | 每个作业的配置；`jobs/<作业>/DONE.json` 是它的结果；`logs/<作业>.log` 是它的日志 |
| `selections/lr.json` | 每个场景选中的学习率、每个学习率跨窗口的平均名次、每次训练的验证 MSE |
| `results/fits.csv` | 每个离线训练一行：验证 MSE、训练轮数、用时 |
| `results/streams.csv` | 每条流一行：MSE、MAE、每次预测用时（选倍率时的验证流也在这里，标为 `VALIDATION`） |
| `results/pairs.csv` | 每组配对一行：在线 MSE、参照 MSE、收益、测试期前后 30% 的收益 |
| `results/summary.json` | 每个表格格子的中位数、组数、在线更好的组数 |
| `results/report_zh.html`、`report_en.html` | 自动生成的报告初稿 |

**收益怎么算：** 收益 = (冻结 MSE − 在线 MSE) ÷ 冻结 MSE。
- 正数表示在线更好。例如冻结 0.50、在线 0.40，收益 +20%。
- 负数表示冻结更好。在线 0.60 时收益 −20%；在线是冻结的 3.5 倍时收益 −250%。

**一组配对**是两次测试运行：数据集、骨干、划分、预测长度、种子都相同，在同一批时刻上各算一个 MSE，得到一个收益。快速版只有一个骨干、一个种子，所以每个格子只有 1 组；学生版把两个骨干、三个种子合在一起，每格 6 组。

**和研究的数字对一下：**

```bash
python scripts/compare_with_study.py runs/etth1_quick
```

它把你跑出来的每条测试流，和研究里同一配置的分数逐条对比。我们在 RTX A5000、torch 2.2.1 上跑的快速版，22 条全部和研究的分数一致，相对差小于 0.0001%（研究用的是 torch 2.4.1 和别的 GPU）。输出见 [reports/example_etth1_quick/compare_with_study.txt](../reports/example_etth1_quick/compare_with_study.txt)。如果你的结果差到百分之一以上，说明设置不一样，检查数据文件、OnlineTSF 版本和选中的学习率。冒烟测试只训练 2 轮，脚本会拒绝比对。

## 6. 更大的实验

| 配置 | 内容 | 规模（参考） |
|---|---|---|
| `etth1_quick.json`（对照） | DLinear，1 个种子，两种划分 | 32 个训练、22 条测试流 |
| `etth1_student.json` | DLinear + PatchTST，3 个种子，两种划分；主实验在 ETTh1、H = 96 上的全部内容 | 112 个训练、132 条测试流；四张 A100 大约 1–2 小时（估计） |
| `etth1_small_lr.json` | DLinear，3 个种子；在线学习率 = 离线学习率 × 倍率，倍率从 0.03、0.1、0.3、1 里在验证集上挑（补充实验第一轮的思路） | 56 个训练、24 条验证流、66 条测试流 |
| `etth1_tuned_w1024.json` | DLinear，3 个种子；长窗口在线 W1024 对认真调过的冻结 W1024：5 个学习率、早停（补充实验第三轮的思路） | 20 个训练、24 条验证流、30 条测试流 |
| `etth2_quick.json`、`etth1_h336_quick.json` | 和快速版一样，换成 ETTh2，或预测 336 步；研究都跑过，对照脚本能核对 | 各 32 个训练、22 条测试流 |
| `ettm1_quick.json` | 和快速版一样，换成 ETTm1（数据长 4 倍，两种划分都到 W4096）；研究跑过 | 38 个训练、24 条测试流；一张卡约 1 小时（估计） |

自己写配置时，可用的字段如下（没写的用默认值）：

| 字段 | 意思 | 默认 |
|---|---|---|
| `name` | 实验名，也是 `runs/` 下的文件夹名。改了配置就换个名字 | 配置文件名 |
| `datasets`、`models`、`splits`、`horizons`、`seeds` | 要跑的组合 | ETTh1、DLinear、O 和 T、96、2024 |
| `frozen_windows` | 冻结模型的输入长度；可以按划分分别给 | 96、512、720、1024 |
| `common_W` | 训练起点对齐的窗口：所有窗口都从这一步开始取训练样本，短窗口不会多拿样本。必须给训练段留至少 1,024 个样本 | 最长的窗口 |
| `online_window`、`online_methods` | 在线模型的输入长度和方法（Naive、ER、PROCEED） | 96；三种都跑 |
| `matched_frozen` | 是否同时跑同起点副本 | true |
| `lr_grid`、`lr_selection_seed` | 离线学习率候选、用哪个种子选 | 1e-4、3e-4、1e-3；2024 |
| `track` | C = 训练 50 轮取最后一轮（主表）；T = 最多 50 轮、5 轮不降就停、取最好一轮 | C |
| `online_multipliers` | 在线学习率倍率候选；多于一个时在验证集上挑 | 1 |
| `max_epochs` | 只给冒烟测试用 | 50 |
| `keep_predictions` | 保留每条流的完整预测数组（占空间） | false |

在 SLURM 等集群上：在分到的资源里运行同一条命令即可。运行器只用调度器分给你的卡（读 `CUDA_VISIBLE_DEVICES`）。

## 7. 写报告

1. 打开 `results/report_zh.html`。所有表格、编号、“跟谁比、正负是好是坏”的句子都已经写好了。
2. 按 [REPORTING.zh-CN.md](REPORTING.zh-CN.md) 写结论、解释、“能说 / 还不能说”。
3. 每个数字都从 `summary.json` 来，不要手抄。
4. 找一个没参与的人通读一遍。
5. 导出 PDF：浏览器打印成 A4，或者

```bash
bash scripts/html_to_pdf.sh runs/etth1_quick/results/report_zh.html
```

样例：
- 我们跑出来的初稿在 [reports/example_etth1_quick](../reports/example_etth1_quick/)；
- 写完的完整报告在 [reports/full_study](../reports/full_study/)。

## 8. 规矩

- **只在验证集上做选择。** 学习率、倍率、训练轮数都只看验证集。看过测试集的结果再回头改设置，结论就不算数了。
- **配置变了就换名字。** 已有的作业不会被覆盖；改了配置还用旧名字，运行器会报错。
- **不跨划分比绝对误差。** O 和 T 的测试期、训练量都不同，只比同一划分里在线和冻结的相对差。
- **失败的作业要报告。** 不要只把结果不好的重跑一遍。`--retry-failed` 用于修好了真实的问题以后。
- **所有种子都报告**，不挑种子。

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| `ImportError: OnlineTSF not found` | 先运行 `git submodule update --init` |
| `PyTorch cannot use CUDA in this environment` | 装的是只能用 CPU 的 PyTorch：按第 1 节装带 CUDA 的版本 |
| `CUDA out of memory` | 调小 `--jobs-per-gpu` |
| 有作业失败 | 看 `runs/<名字>/logs/<作业>.log` 和 `jobs/<作业>/FAILED.json`。修好以后加 `--retry-failed` 重跑。第 1–4 步失败会停下；测试流失败会在最后报告，表格里缺这一条 |
| `spec.json differs` | 配置改过了，换一个 `name` |
| 跑到一半断了 | 原样再运行同一条命令 |
| 磁盘占用 | 每条流跑完会删掉预测数组和中间快照，只留 `DONE.json` 和逐时刻误差 `per_origin.json`。要保留就设 `keep_predictions` |
| 直接运行单个作业 | `python -m olts.worker --run runs/<名字> --job <作业>`。配置要先在 `runs/<名字>/configs/` 里，而且 `runs/<名字>/jobs/<作业>` 不能已经存在：作业文件夹从不覆盖，旧的要先挪走 |
