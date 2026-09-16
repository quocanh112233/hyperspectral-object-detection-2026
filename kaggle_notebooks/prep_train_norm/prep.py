"""HODC2026 - chuan bi du lieu train: PNG mosaic -> TIFF 16 trang + nhan YOLO.

Chay tren Kaggle CPU (khong ton quota GPU). Xuat ra /kaggle/working/hodc_train_norm:
  images/<id>.tif   cube 16 bang, uint16, TIFF 16 trang (Ultralytics doc native)
  labels/<id>.txt   nhan YOLO da lam sach
  meta.csv          kich thuoc anh + so vat the
  band_stats.csv    mean/std/min/max tung bang -> dung de chuan hoa
  box_spectra.csv   pho trung binh tung bbox -> phan tich cap that/gia
KHONG chia train/val o day: viec chia lam o notebook train de doi duoc ma khong
phai tao lai 12GB du lieu.
"""
import os, glob, time
import numpy as np
import pandas as pd
import tifffile
import xml.etree.ElementTree as ET
from PIL import Image

SRC = "/kaggle/input/datasets/moderantnoukoussi/hyperspectral-object-detection-challenge-2026"
IMG_DIR = f"{SRC}/data_train/data_train/VIS"
ANN_DIR = f"{SRC}/data_train/data_train/Annotations/VIS"
OUT = "/kaggle/working/hodc_train_norm"
# Khuech dai toan cuc: dua mean anh ve 114 = gia tri nen Ultralytics dung de dem,
# va = 0.447 sau khi chia 255, khop thong ke anh COCO cua trong so pretrain.
# Nhan toan cuc nen ti le giua 16 bang giu nguyen tuyet doi.
GAIN = 114.0 / 44.1
# CHUAN HOA PHO MEM theo tung pixel.
# Muc tieu: hai vat cung chat lieu o hai muc chieu sang khac nhau phai giong nhau.
# Chuan hoa CUNG (chia cho chuan L2) se xoa sach thong tin do sang -> mat canh,
# hong dinh vi. Dang mem: he so = (114+C)/(mean_pixel+C).
#   pixel sang (mean>>C) -> dua ve cung muc
#   pixel toi  (mean<<C) -> he so tien toi 2.0, khong thoi phong nhieu qua muc
C_SOFT = 114.0

def soft_spectral_norm(cube):
    m = cube.mean(axis=2, keepdims=True)
    return cube * ((114.0 + C_SOFT) / (m + C_SOFT))
# 3 anh co kich thuoc XML lech cube -> nhan ve tren phien ban anh khac, bo di.
SKIP_IDS = {"1227", "1836", "1855"}
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)

CLASSES = [l.strip() for l in open(f"{SRC}/class.txt") if l.strip()]
CID = {c: i for i, c in enumerate(CLASSES)}
print("So lop:", len(CLASSES), CLASSES[:4], "...")


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


# ---------- 0. Kiem tra toan ven truoc khi lam gi ----------
pngs = sorted(glob.glob(f"{IMG_DIR}/*.png"))
xmls = sorted(glob.glob(f"{ANN_DIR}/*.xml"))
png_ids = {os.path.splitext(os.path.basename(p))[0] for p in pngs}
xml_ids = {os.path.splitext(os.path.basename(p))[0] for p in xmls}
print(f"anh train: {len(pngs)} | nhan: {len(xmls)}")
print(f"anh thieu nhan: {len(png_ids - xml_ids)} | nhan thieu anh: {len(xml_ids - png_ids)}")
assert len(pngs) == 3000 and len(xmls) == 3000, "SO LUONG KHONG DUNG - dung lai"
assert png_ids == xml_ids, "ID anh va nhan khong khop - dung lai"
print("Toan ven: OK\n")

# ---------- 1. Chuyen doi ----------
meta, box_rows = [], []
band_sum = np.zeros(16, np.float64); band_sq = np.zeros(16, np.float64)
band_min = np.full(16, np.inf); band_max = np.full(16, -np.inf); npix = 0
n_drop_degenerate = n_drop_clip = n_unknown = 0

t0 = time.time()
for i, p in enumerate(pngs):
    img_id = os.path.splitext(os.path.basename(p))[0]
    if img_id in SKIP_IDS:
        continue
    cube = X2Cube(np.array(Image.open(p)))          # (H, W, 16) uint16
    cube = soft_spectral_norm(cube.astype(np.float32) * GAIN)
    cube = np.clip(cube, 0, 65535).astype(np.uint16)
    H, W, _ = cube.shape

    tifffile.imwrite(f"{OUT}/images/{img_id}.tif", cube.transpose(2, 0, 1))

    flat = cube.reshape(-1, 16).astype(np.float64)
    band_sum += flat.sum(0); band_sq += (flat ** 2).sum(0)
    band_min = np.minimum(band_min, flat.min(0)); band_max = np.maximum(band_max, flat.max(0))
    npix += flat.shape[0]

    root = ET.parse(f"{ANN_DIR}/{img_id}.xml").getroot()
    sz = root.find("size")
    xml_w, xml_h = int(sz.find("width").text), int(sz.find("height").text)

    lines = []
    for o in root.findall("object"):
        name = o.find("name").text
        if name not in CID:
            n_unknown += 1
            continue
        b = o.find("bndbox")
        x1, y1 = float(b.find("xmin").text), float(b.find("ymin").text)
        x2, y2 = float(b.find("xmax").text), float(b.find("ymax").text)
        # kep vao trong anh (toa do theo he SAU demosaic)
        cx1, cy1 = max(0.0, min(x1, W)), max(0.0, min(y1, H))
        cx2, cy2 = max(0.0, min(x2, W)), max(0.0, min(y2, H))
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            n_drop_clip += 1
        if cx2 - cx1 < 1.0 or cy2 - cy1 < 1.0:        # bbox suy bien -> bo
            n_drop_degenerate += 1
            continue
        lines.append(f"{CID[name]} {(cx1+cx2)/2/W:.6f} {(cy1+cy2)/2/H:.6f} "
                     f"{(cx2-cx1)/W:.6f} {(cy2-cy1)/H:.6f}")

        patch = cube[int(cy1):max(int(cy2), int(cy1)+1),
                     int(cx1):max(int(cx2), int(cx1)+1), :].reshape(-1, 16)
        row = {"image_id": img_id, "name": name, "n_pix": patch.shape[0],
               "w": cx2-cx1, "h": cy2-cy1}
        m = patch.mean(0)
        for b_ in range(16):
            row[f"b{b_:02d}"] = float(m[b_])
        box_rows.append(row)

    open(f"{OUT}/labels/{img_id}.txt", "w").write("\n".join(lines))
    meta.append({"image_id": img_id, "cube_w": W, "cube_h": H,
                 "xml_w": xml_w, "xml_h": xml_h, "n_obj": len(lines),
                 "size_khop": int(W == xml_w and H == xml_h)})

    if (i + 1) % 250 == 0:
        el = time.time() - t0
        print(f"  {i+1}/3000  {el:.0f}s  (con ~{el/(i+1)*(3000-i-1):.0f}s)", flush=True)

# ---------- 2. Ghi thong ke ----------
mean = band_sum / npix
std = np.sqrt(np.maximum(band_sq / npix - mean ** 2, 0))
pd.DataFrame({"band": range(16), "mean": mean, "std": std,
              "min": band_min, "max": band_max}).to_csv(f"{OUT}/band_stats.csv", index=False)
dfm = pd.DataFrame(meta); dfm.to_csv(f"{OUT}/meta.csv", index=False)
pd.DataFrame(box_rows).to_csv(f"{OUT}/box_spectra.csv", index=False)
open(f"{OUT}/classes.txt", "w").write("\n".join(CLASSES))

print("\n=== TONG KET ===")
print(f"anh chuyen doi : {len(meta)}  (da bo {len(SKIP_IDS)} anh nhan lech khung)")
print(f"GAIN da ap dung : {GAIN:.4f}")
print(f"bbox giu lai   : {len(box_rows)}")
print(f"bbox suy bien bi bo : {n_drop_degenerate}")
print(f"bbox phai kep bien  : {n_drop_clip}")
print(f"nhan la khong co trong class.txt : {n_unknown}")
print(f"anh co kich thuoc XML != cube    : {(dfm.size_khop == 0).sum()}  <-- phai bang 0")
print(f"\nkich thuoc cube: w {dfm.cube_w.min()}-{dfm.cube_w.max()}, h {dfm.cube_h.min()}-{dfm.cube_h.max()}")
print("\n=== band_stats ===")
print(pd.read_csv(f"{OUT}/band_stats.csv").round(1).to_string(index=False))
sz_gb = sum(os.path.getsize(f) for f in glob.glob(f"{OUT}/images/*.tif")) / 1e9
print(f"\nTong dung luong TIFF: {sz_gb:.2f} GB")
