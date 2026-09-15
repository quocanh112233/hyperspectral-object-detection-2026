"""Thong ke nhan: phan bo lop, kich thuoc anh, kich thuoc bbox, so vat/anh."""
import os, sys, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.voc import parse_voc

CLASSES = [l.strip() for l in open(os.path.join(ROOT, "configs", "class.txt")) if l.strip()]
rows = []
sizes = []
for xml in sorted(glob.glob(os.path.join(ROOT, "data", "annotations", "*.xml"))):
    a = parse_voc(xml)
    img_id = os.path.splitext(os.path.basename(xml))[0]
    sizes.append((img_id, a["width"], a["height"], len(a["objects"])))
    for o in a["objects"]:
        rows.append({
            "image_id": img_id, "name": o["name"],
            "w": o["xmax"] - o["xmin"], "h": o["ymax"] - o["ymin"],
            "xmin": o["xmin"], "ymin": o["ymin"], "xmax": o["xmax"], "ymax": o["ymax"],
            "img_w": a["width"], "img_h": a["height"],
        })

df = pd.DataFrame(rows)
sz = pd.DataFrame(sizes, columns=["image_id", "img_w", "img_h", "n_obj"])
print(f"=== {len(sz)} anh, {len(df)} bounding box ===\n")

print("--- Kich thuoc anh (sau demosaic) ---")
print(sz[["img_w", "img_h"]].describe().round(1).to_string())
print("so cap (w,h) khac nhau:", sz.groupby(["img_w", "img_h"]).ngroups, "/", len(sz), "anh")
print("ty le khung hinh w/h: min %.2f  max %.2f  trung binh %.2f"
      % ((sz.img_w/sz.img_h).min(), (sz.img_w/sz.img_h).max(), (sz.img_w/sz.img_h).mean()))
print()

print("--- So vat the moi anh ---")
print(sz.n_obj.describe().to_string())
print("phan bo:", sz.n_obj.value_counts().sort_index().to_dict())
print()

print("--- Phan bo lop ---")
vc = df.name.value_counts()
unknown = set(df.name.unique()) - set(CLASSES)
tbl = pd.DataFrame({"so_bbox": [vc.get(c, 0) for c in CLASSES],
                    "so_anh": [df[df.name == c].image_id.nunique() for c in CLASSES]},
                   index=[f"{i:2d} {c}" for i, c in enumerate(CLASSES)])
tbl["ty_le_%"] = (tbl.so_bbox / tbl.so_bbox.sum() * 100).round(2)
print(tbl.to_string())
print("MAT CAN BANG: lop nhieu nhat / it nhat = %.1fx" % (tbl.so_bbox.max() / max(tbl.so_bbox.min(), 1)))
if unknown:
    print("!! LOP LA khong co trong class.txt:", unknown)
print()

print("--- Kich thuoc bbox (pixel, he sau demosaic) ---")
df["area"] = df.w * df.h
df["scale"] = np.sqrt(df.area)
print(df[["w", "h", "area", "scale"]].describe().round(1).to_string())
coco = pd.cut(df.area, [0, 32**2, 96**2, np.inf], labels=["small(<32²)", "medium", "large(>96²)"])
print("\nPhan loai theo chuan COCO:", coco.value_counts().sort_index().to_dict())
print()

print("--- Kich thuoc bbox trung binh theo lop (sap xep tang dan) ---")
g = df.groupby("name").agg(scale_tb=("scale", "mean"), w_tb=("w", "mean"), h_tb=("h", "mean")).round(1)
print(g.sort_values("scale_tb").to_string())
print()

print("--- Cac cap that/gia: co xuat hien cung anh khong? ---")
pairs = [("apple", "apple_plastic"), ("banana", "banana_plastic"),
         ("egg", "egg_plastic"), ("egg", "egg_wood"), ("egg_plastic", "egg_wood"),
         ("car", "car_toy"), ("orange", "orange_plastic")]
for a, b in pairs:
    sa = set(df[df.name == a].image_id); sb = set(df[df.name == b].image_id)
    print(f"  {a:14s} vs {b:16s}: chung {len(sa & sb):4d} anh  (rieng {len(sa-sb):4d} / {len(sb-sa):4d})")

df.to_csv(os.path.join(ROOT, "data", "annotations_flat.csv"), index=False)
sz.to_csv(os.path.join(ROOT, "data", "image_sizes.csv"), index=False)
print("\nDa luu data/annotations_flat.csv va data/image_sizes.csv")
