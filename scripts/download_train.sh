#!/usr/bin/env bash
# Tai data_train.zip (11GB) roi giai nen. Chay nen (background) vi lau.
set -euo pipefail
COMP=hyperspectral-object-detection-challenge-2026
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
KG="$ROOT/.venv/bin/kaggle"
mkdir -p "$RAW"
"$KG" competitions download -c "$COMP" -f data_train.zip -p "$RAW"
echo "== Tai xong, giai nen =="
unzip -q -o "$RAW/data_train.zip" -d "$ROOT/data/train_full"
echo "== Xong =="
