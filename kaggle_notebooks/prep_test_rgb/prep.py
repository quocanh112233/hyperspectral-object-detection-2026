"""HODC2026 - chuan bi tap TEST 3 BANG (phai khop het voi prep_train_rgb).

Model hien tai train tren 3 bang [0, 8, 15] xuat uint8 bang min-max tung kenh.
Bo test phai duoc chuyen doi Y HET, neu khac mot buoc nao la lech phan phoi
train/test va diem tut ma khong bao loi gi.

Xuat /kaggle/working/hodc_test_rgb/images/<id>.tif + meta.csv
"""
import os, glob, time
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

SRC = "/kaggle/input/datasets/moderantnoukoussi/hyperspectral-object-detection-challenge-2026"
IMG_DIR = f"{SRC}/data_test/data_test/VIS"
OUT = "/kaggle/working/hodc_test_rgb"
BANDS = [0, 8, 15]    # PHAI giong het prep_train_rgb
# GAIN thuc ra la ma chet o nhanh 3 bang: to_rgb8 lam min-max SAU do, ma min-max
# bat bien voi moi phep nhan duong. Da do: nhan x1.7 chi doi dau ra toi da 1 don vi.
# Van giu de doi xung tuyet doi voi prep_train_rgb.
GAIN = 114.0 / 44.1


def to_rgb8(cube, bands):
    out = cube[:, :, bands].astype(np.float32)
    for c in range(len(bands)):
        ch = out[:, :, c]
        lo, hi = ch.min(), ch.max()
        out[:, :, c] = ((ch - lo) / (hi - lo) * 255) if hi > lo else 0
    return out.astype(np.uint8)
os.makedirs(f"{OUT}/images", exist_ok=True)


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


pngs = sorted(glob.glob(f"{IMG_DIR}/*.png"))
print("anh test:", len(pngs))
assert len(pngs) == 1000, "SO LUONG TEST KHONG DUNG"

meta = []
t0 = time.time()
for i, p in enumerate(pngs):
    img_id = os.path.splitext(os.path.basename(p))[0]
    cube = X2Cube(np.array(Image.open(p)))
    cube = np.clip(cube.astype(np.float32) * GAIN, 0, 65535).astype(np.uint16)
    H, W, _ = cube.shape
    tifffile.imwrite(f"{OUT}/images/{img_id}.tif", to_rgb8(cube, BANDS).transpose(2, 0, 1))
    # LUU KICH THUOC GOC: bat buoc de quy doi bbox ve dung he toa do khi nop
    meta.append({"image_id": img_id, "cube_w": W, "cube_h": H})
    if (i + 1) % 200 == 0:
        el = time.time() - t0
        print(f"  {i+1}/1000  {el:.0f}s (con ~{el/(i+1)*(1000-i-1):.0f}s)", flush=True)

df = pd.DataFrame(meta)
df.to_csv(f"{OUT}/meta.csv", index=False)
print("\nkich thuoc cube: w %d-%d, h %d-%d" % (df.cube_w.min(), df.cube_w.max(),
                                              df.cube_h.min(), df.cube_h.max()))
sz = sum(os.path.getsize(f) for f in glob.glob(f"{OUT}/images/*.tif")) / 1e9
print(f"Tong dung luong TIFF: {sz:.2f} GB")
