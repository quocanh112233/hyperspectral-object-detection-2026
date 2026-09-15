"""HODC2026 - chuan bi tap TEST: PNG mosaic -> TIFF 16 trang (khong co nhan).

Xuat /kaggle/working/hodc_ranking/images/<id>.tif + meta.csv
"""
import os, glob, time
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

SRC = "/kaggle/input/datasets/moderantnoukoussi/hyperspectral-object-detection-challenge-2026"
# Ranking set mo ngay 23/09, chua biet chinh xac cau truc thu muc -> do tim.
import glob as _g
cands = [d for d in _g.glob(f"{SRC}/**/", recursive=True) if "ranking" in d.lower()]
print("thu muc ung vien:", cands[:5])
_pngs = _g.glob(f"{SRC}/**/ranking*/**/*.png", recursive=True) or \
        _g.glob(f"{SRC}/**/*ranking*/*.png", recursive=True)
assert _pngs, "khong tim thay anh ranking - kiem tra lai cau truc /kaggle/input"
IMG_DIR = os.path.dirname(sorted(_pngs)[0])
print("dung IMG_DIR =", IMG_DIR)
OUT = "/kaggle/working/hodc_ranking"
GAIN = 114.0 / 44.1   # PHAI giong het prep_train, neu khac la lech phan phoi train/test
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
print("anh ranking:", len(pngs))
assert len(pngs) == 1000, f"SO LUONG RANKING KHONG DUNG: {len(pngs)}"

meta = []
t0 = time.time()
for i, p in enumerate(pngs):
    img_id = os.path.splitext(os.path.basename(p))[0]
    cube = X2Cube(np.array(Image.open(p)))
    cube = np.clip(cube.astype(np.float32) * GAIN, 0, 65535).astype(np.uint16)
    H, W, _ = cube.shape
    tifffile.imwrite(f"{OUT}/images/{img_id}.tif", cube.transpose(2, 0, 1))
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
