[English](#english) | [中文](#中文)

## English

Put the dataset files in this folder under the names below (the code looks for exactly these names).
`bash scripts/download_data.sh` fetches the four ETT files; the others must be downloaded by hand.

| Dataset | File | Shape used | Source |
|---|---|---|---|
| ETTh1, ETTh2 | `ETTh1.csv`, `ETTh2.csv` | first 14,400 rows × 7 | <https://github.com/zhouhaoyi/ETDataset> (`ETT-small/`) |
| ETTm1, ETTm2 | `ETTm1.csv`, `ETTm2.csv` | first 57,600 rows × 7 | same |
| Weather | `weather.csv` | 52,696 × 21 | the Autoformer / Time-Series-Library dataset collection (<https://github.com/thuml/Time-Series-Library>, dataset link in its README) |
| Electricity | `electricity.csv` | 26,304 × 321 | same collection |
| Solar | `solar_AL.txt` | 52,560 × 137 | <https://github.com/laiguokun/multivariate-time-series-data> (`solar-energy/`, unzip) |
| PEMS-BAY | `pems-bay.h5` | 52,116 × 325 | DCRNN data, <https://github.com/liyaguang/DCRNN> (needs `pip install tables`) |
| PEMS08 | `PEMS08.npz` | 17,856 × 170 (flow channel) | the ASTGCN / STSGCN PEMS release, e.g. <https://github.com/guoshnBJTU/ASTGNN> |

CSV files must have a `date` column followed by the variables. Data files are not committed to git (see `.gitignore`).

## 中文

把数据文件按下表的文件名放在这个文件夹里（代码只认这些名字）。
`bash scripts/download_data.sh` 会下载四个 ETT 文件，其余的需要手动下载。

| 数据集 | 文件 | 使用的形状 | 来源 |
|---|---|---|---|
| ETTh1、ETTh2 | `ETTh1.csv`、`ETTh2.csv` | 前 14,400 行 × 7 | <https://github.com/zhouhaoyi/ETDataset>（`ETT-small/`） |
| ETTm1、ETTm2 | `ETTm1.csv`、`ETTm2.csv` | 前 57,600 行 × 7 | 同上 |
| Weather | `weather.csv` | 52,696 × 21 | Autoformer / Time-Series-Library 的数据合集（<https://github.com/thuml/Time-Series-Library>，链接在它的 README 里） |
| Electricity | `electricity.csv` | 26,304 × 321 | 同上 |
| Solar | `solar_AL.txt` | 52,560 × 137 | <https://github.com/laiguokun/multivariate-time-series-data>（`solar-energy/`，解压） |
| PEMS-BAY | `pems-bay.h5` | 52,116 × 325 | DCRNN 的数据，<https://github.com/liyaguang/DCRNN>（需要 `pip install tables`） |
| PEMS08 | `PEMS08.npz` | 17,856 × 170（流量通道） | ASTGCN / STSGCN 的 PEMS 数据，例如 <https://github.com/guoshnBJTU/ASTGNN> |

CSV 文件第一列是 `date`，后面是各个变量。数据文件不提交到 git（见 `.gitignore`）。
