#!/usr/bin/env bash
# Download the ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2) into data/ from the official ETDataset repository.
# Usage: bash scripts/download_data.sh [ETTh1 ETTh2 ...]      (default: all four)
# The other datasets (Weather, Electricity, Solar, PEMS-BAY, PEMS08) are not downloaded here:
# see data/README.md for file names and sources.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
names=("$@")
[ ${#names[@]} -eq 0 ] && names=(ETTh1 ETTh2 ETTm1 ETTm2)
for n in "${names[@]}"; do
  case "$n" in ETTh1|ETTh2|ETTm1|ETTm2) ;; *) echo "only ETT datasets can be downloaded by this script: $n"; exit 1;; esac
  if [ -s "data/$n.csv" ]; then echo "data/$n.csv exists"; continue; fi
  url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/$n.csv"
  echo "downloading $url"
  curl -fsSL --retry 3 -o "data/$n.csv.part" "$url"
  mv "data/$n.csv.part" "data/$n.csv"
done
python3 - "${names[@]}" <<'EOF'
import sys, hashlib
rows = {'ETTh1': 17420, 'ETTh2': 17420, 'ETTm1': 69680, 'ETTm2': 69680}
for n in sys.argv[1:]:
    p = f'data/{n}.csv'
    lines = sum(1 for _ in open(p)) - 1
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    print(f'{p}: {lines} rows (expected {rows[n]}), sha256 {h[:16]}...', 'OK' if lines == rows[n] else 'UNEXPECTED ROW COUNT')
EOF
