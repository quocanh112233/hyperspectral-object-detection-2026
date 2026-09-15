"""HODC2026 - EDA chay tren Kaggle (du lieu mount san, khong can tai ve).

Xuat ra /kaggle/working:
  annotations.zip     toan bo 3000 nhan VOC XML (~5MB) de lam viec o local
  band_stats.csv      thong ke tung bang pho tren toan bo anh
  box_spectra.csv     pho trung binh cua TUNG bounding box -> tach cap that/gia
  image_meta.csv      kich thuoc anh + so vat the
  samples.npz         vai cube mau de xem truc quan o local
"""
import os, glob, zipfile, time
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image

IN = "/kaggle/input/hyperspectral-object-detection-challenge-2026"
TRAIN_IMG = f"{IN}/data_train/data_train/VIS"
TRAIN_ANN = f"{IN}/data_train/data_train/Annotations/VIS"
TEST_IMG = f"{IN}/data_test/data_test/VIS"
OUT = "/kaggle/working"


def X2Cube(img, cellSize=4):
    B = [cellSize, cellSize]; skip = [cellSize, cellSize]
    M, N = img.shape
    col_extent = N - B[1] + 1; row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    return np.transpose(out).reshape(M // cellSize, N // cellSize, cellSize * cellSize)


# ---------- 1. Nen toan bo nhan ----------
xmls = sorted(glob.glob(f"{TRAIN_ANN}/*.xml"))
print(f"So file nhan: {len(xmls)}", flush=True)
with zipfile.ZipFile(f"{OUT}/annotations.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for x in xmls:
        z.write(x, os.path.basename(x))
print("Da nen annotations.zip: %.1f MB" % (os.path.getsize(f"{OUT}/annotations.zip") / 1e6), flush=True)


# ---------- 2. Doc nhan vao bo nho ----------
ann = {}
for x in xmls:
    root = ET.parse(x).getroot()
    size = root.find("size")
    ann[os.path.splitext(os.path.basename(x))[0]] = {
        "w": int(size.find("width").text),
        "h": int(size.find("height").text),
        "objs": [(o.find("name").text,
                  int(float(o.find("bndbox/xmin").text)), int(float(o.find("bndbox/ymin").text)),
                  int(float(o.find("bndbox/xmax").text)), int(float(o.find("bndbox/ymax").text)))
                 for o in root.findall("object")],
    }

# ---------- 3. Duyet anh: thong ke bang pho + pho tung bbox ----------
img_rows, box_rows = [], []
band_sum = np.zeros(16, np.float64); band_sqsum = np.zeros(16, np.float64)
band_min = np.full(16, np.inf); band_max = np.full(16, -np.inf); npix = 0
samples = {}
SAMPLE_IDS = set()

t0 = time.time()
paths = sorted(glob.glob(f"{TRAIN_IMG}/*.png"))
for i, p in enumerate(paths):
    img_id = os.path.splitext(os.path.basename(p))[0]
    raw = np.array(Image.open(p))
    cube = X2Cube(raw).astype(np.float32)
    H, W, _ = cube.shape
    a = ann.get(img_id)

    img_rows.append({"image_id": img_id, "png_h": raw.shape[0], "png_w": raw.shape[1],
                     "cube_h": H, "cube_w": W,
                     "xml_h": a["h"] if a else -1, "xml_w": a["w"] if a else -1,
                     "n_obj": len(a["objs"]) if a else 0,
                     "vmin": float(raw.min()), "vmax": float(raw.max())})

    flat = cube.reshape(-1, 16)
    band_sum += flat.sum(0); band_sqsum += (flat.astype(np.float64) ** 2).sum(0)
    band_min = np.minimum(band_min, flat.min(0)); band_max = np.maximum(band_max, flat.max(0))
    npix += flat.shape[0]

    if a:
        for name, x1, y1, x2, y2 in a["objs"]:
            xa, xb = max(0, min(x1, W - 1)), min(W, max(x2, x1 + 1))
            ya, yb = max(0, min(y1, H - 1)), min(H, max(y2, y1 + 1))
            if xb <= xa or yb <= ya:
                continue
            patch = cube[ya:yb, xa:xb, :].reshape(-1, 16)
            row = {"image_id": img_id, "name": name,
                   "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                   "n_pix": patch.shape[0]}
            m = patch.mean(0)
            for b in range(16):
                row[f"b{b:02d}"] = float(m[b])
            box_rows.append(row)

    if len(samples) < 8 and a and len(a["objs"]) >= 2:
        samples[img_id] = cube[:, :, :].astype(np.uint16)
        SAMPLE_IDS.add(img_id)

    if (i + 1) % 250 == 0:
        print(f"  {i+1}/{len(paths)}  ({time.time()-t0:.0f}s)", flush=True)

# ---------- 4. Kich thuoc anh test (khong co nhan) ----------
for p in sorted(glob.glob(f"{TEST_IMG}/*.png")):
    raw = np.array(Image.open(p))
    img_rows.append({"image_id": os.path.splitext(os.path.basename(p))[0],
                     "png_h": raw.shape[0], "png_w": raw.shape[1],
                     "cube_h": raw.shape[0] // 4, "cube_w": raw.shape[1] // 4,
                     "xml_h": -1, "xml_w": -1, "n_obj": -1,
                     "vmin": float(raw.min()), "vmax": float(raw.max())})

# ---------- 5. Ghi ket qua ----------
mean = band_sum / npix
std = np.sqrt(band_sqsum / npix - mean ** 2)
pd.DataFrame({"band": range(16), "mean": mean, "std": std,
              "min": band_min, "max": band_max}).to_csv(f"{OUT}/band_stats.csv", index=False)
pd.DataFrame(img_rows).to_csv(f"{OUT}/image_meta.csv", index=False)
pd.DataFrame(box_rows).to_csv(f"{OUT}/box_spectra.csv", index=False)
np.savez_compressed(f"{OUT}/samples.npz", **samples)

print("\n=== band_stats ===")
print(pd.read_csv(f"{OUT}/band_stats.csv").round(1).to_string(index=False))
print(f"\nTong: {len(img_rows)} anh, {len(box_rows)} bbox, {npix/1e6:.1f}M pixel")
print("File xuat ra:", sorted(os.listdir(OUT)))
