# Hyperspectral Object Detection Challenge 2026

Track 1 — object detection tren anh hyperspectral 16 bang pho (460-600 nm), 18 lop.
Tom tat luat/du lieu/deadline: [docs/HODC2026_reference.md](docs/HODC2026_reference.md).
Nhat ky thi nghiem: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Rang buoc quan trong
- **Chi mot model duy nhat, cam ensemble.**
- Diem chung cuoc chi tinh submission trong **Phase 2 (23-25/09/2026)**, va file CSV
  phai chua ca test set lan ranking set.
- Bbox theo he toa do **sau demosaic**, khong phai pixel cua PNG goc.

## Compute
May local khong co GPU NVIDIA -> **train tren Kaggle Notebooks**.
Local chi dung de: EDA, viet code, ghep/kiem tra file CSV.

## Cau truc
```
configs/           cau hinh train (yaml), class.txt
data/              du lieu (git-ignored)
  raw/             file tai ve tu Kaggle
  annotations/     VOC XML
  sample/          mau nho de EDA local
docs/              tai lieu tham chieu + nhat ky thi nghiem
experiments/       config + metrics cua tung lan chay (weight bi ignore)
kaggle_notebooks/  notebook chay tren Kaggle (train + inference)
notebooks/         notebook EDA chay local
scripts/           script tai du lieu, tien xu ly
src/               thu vien dung chung
submissions/       file CSV da nop
```

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
bash scripts/download_data.sh
```
