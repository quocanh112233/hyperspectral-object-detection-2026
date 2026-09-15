#!/usr/bin/env bash
# Tai du lieu HODC2026. Yeu cau: ~/.kaggle/kaggle.json va da Join Competition.
set -euo pipefail
COMP=hyperspectral-object-detection-challenge-2026
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
KG="$ROOT/.venv/bin/kaggle"
mkdir -p "$RAW"

echo "== Danh sach file cua cuoc thi =="
"$KG" competitions files -c "$COMP" -v

echo "== Tai cac file nho (class.txt, sample_submission.csv, pseudo_rgb_demo.py) =="
for f in class.txt sample_submission.csv pseudo_rgb_demo.py; do
    "$KG" competitions download -c "$COMP" -f "$f" -p "$RAW" || echo "  bo qua: $f"
done

echo "== Xong. data_train.zip (11GB) tai rieng bang scripts/download_train.sh =="
