[English](README.md) | **简体中文**

# OLTS-Bench

**短窗口在线学习，能打败长窗口冻结模型吗？**

在线模型每次只读最近 96 步，但部署之后会用陆续到达的真实值不停更新参数。冻结模型读更长的历史（512 到 4,096 步），训练完就不再改。OLTS-Bench 在同一套因果协议下比较这两种做法：
- 同一份代码（[OnlineTSF](https://github.com/SJTU-DMTai/OnlineTSF)）；
- 同一套训练预算；
- 任何一个预测都要等它的真实值全部到齐，才评分、才拿来更新。

这个仓库放的是：
- 这项研究的代码、已发布的分数和报告；
- 一条给新同学的入门路线：在自己的 GPU 上用一个数据集、几个跑得快的模型跑一遍，再把输出写成我们这样的报告。

## 目前的结果

完整研究的范围：
- 10 个数据集；
- DLinear、PatchTST 两种骨干；
- 预测长度 24/96/336；
- 3 个种子；
- 两种划分。

下面每个数都是多组配对运行的中位数，意思是在线模型的 MSE 比冻结模型低多少；负数表示冻结更好。细节和注意事项见 [docs/RESULTS.zh-CN.md](docs/RESULTS.zh-CN.md)。

- **传统划分 T**（训练/验证/测试 = 60–70 / 10–20 / 20）：15 个比较（3 种在线方法 × 5 档冻结窗口）全是冻结长窗口赢，在线的 MSE 高 12%–49%。
- **在线式划分 O**（20 / 5 / 75）：有输有赢。
  - 对 W512、W720，在线输。
  - 对 W1024、W2048，ER、PROCEED 与冻结打平（−2.1% 到 +1.8%）。
  - 对 W4096，在线赢，但主要是因为冻结 W4096 本身比 W2048 差。
- **补充实验**（大多是 W1024、H = 96）：
  - 在线学习率调小后，长窗口模型加更新不再变差。
  - 冻结模型也调参以后，加更新仍略好：在线 W1024 的误差在 O 划分低 3.7%、T 划分低 0.7%。
  - 在 O 划分，把冻结模型用更新的数据重训一次，它就比一路更新的在线还好：在线误差高 8.0%。

## 快速上手（一张 GPU，计算约 20 分钟）

```bash
git clone --recursive https://github.com/lixus7/OLTS-Bench.git
cd OLTS-Bench
conda create -n olts python=3.10 -y && conda activate olts
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash scripts/download_data.sh ETTh1
python -m olts.run configs/smoke.json --jobs-per-gpu 4         # 检查安装：只训练 2 轮，不是结果
python -m olts.run configs/etth1_quick.json --jobs-per-gpu 4   # ETTh1、DLinear、H = 96、种子 2024、两种划分
python scripts/compare_with_study.py runs/etth1_quick           # 你的分数和研究的分数逐条对比
```

在一张 RTX A5000 上，冒烟测试约 4 分钟，快速版约 14 分钟。快速版这个配置的 22 条测试流，分数和研究的完全一致（相对差小于 0.0001%）。

跑完打开 `runs/etth1_quick/results/report_zh.html`（英文版是 `report_en.html`）。逐步说明、用时和常见问题见 [docs/GUIDE.zh-CN.md](docs/GUIDE.zh-CN.md)。

## 仓库里有什么

| 路径 | 内容 |
|---|---|
| `olts/` | 基准代码：数据准备、引擎（OnlineTSF 的算子加一个因果调度器）、单个作业、实验规划、本机多卡运行器、分析、报告初稿 |
| `third_party/OnlineTSF` | OnlineTSF 原仓库（git submodule），固定在研究用的提交 `0a86461` |
| `configs/` | 实验配置：`smoke`、`etth1_quick`、`etth1_student`、`etth1_small_lr`、`etth1_tuned_w1024` |
| `docs/` | [GUIDE](docs/GUIDE.zh-CN.md)（逐步上手）、[PROTOCOL](docs/PROTOCOL.zh-CN.md)（协议细节）、[REPORTING](docs/REPORTING.zh-CN.md)（怎么写报告）、[RESULTS](docs/RESULTS.zh-CN.md)（结果） |
| `reports/full_study/` | 这项研究的报告（中文）：完整版和两页摘要 |
| `reports/example_etth1_quick/` | 我们用 `configs/etth1_quick.json` 跑出来的报告初稿 |
| `results/full_study/` | 主实验每次运行的分数（CSV），能复现完整版的表 1、表 2、表 4 和图 2 |
| `scripts/` | `download_data.sh`（下载 ETT 数据）、`compare_with_study.py`（你的分数和研究的分数对比）、`html_to_pdf.sh`（报告转 PDF） |

## 用已发布的分数复现主表

```bash
python -m olts.analyze --streams results/full_study/streams_main.csv --out runs/full_study_check
python -m olts.report --summary runs/full_study_check/summary.json --lang zh
```

完整版表 1、表 2 的 30 格，以及表 4 和图 2，数字都和报告一致。

## 致谢与参考文献

- OnlineTSF 与 PROCEED：L. Zhao and Y. Shen, *Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting*, KDD 2025。代码：<https://github.com/SJTU-DMTai/OnlineTSF>。本仓库没有拷贝 OnlineTSF 的代码，而是以 submodule 引用。
- DLinear：A. Zeng et al., *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023。
- PatchTST：Y. Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023。
- 经验回放（ER）：A. Chaudhry et al., *On Tiny Episodic Memories in Continual Learning*, 2019。
- ETT 数据集：H. Zhou et al., *Informer*, AAAI 2021，<https://github.com/zhouhaoyi/ETDataset>。
