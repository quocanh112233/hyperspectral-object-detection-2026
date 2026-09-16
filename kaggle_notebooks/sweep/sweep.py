"""HODC2026 - quet tham so LUC SUY LUAN (khong train lai, gan nhu mien phi).

Tu truoc den gio toi dung iou=0.7 / max_det=300 - deu la mac dinh cua Ultralytics,
chua he kiem chung tren bai nay. Voi mAP@[.5:.95] thi nguong NMS anh huong that.
"""
import os, sys, glob, subprocess, importlib, itertools
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.152", "tifffile"], check=True)
importlib.invalidate_caches()
import torch
from ultralytics import YOLO

W = glob.glob("/kaggle/input/**/weights/best.pt", recursive=True)
assert W, "khong thay best.pt"
ROOT = os.path.dirname(glob.glob("/kaggle/input/**/hodc_train/images", recursive=True)[0])
CLASSES = [l.strip() for l in open(f"{ROOT}/classes.txt") if l.strip()]
val_ids = [l.strip() for l in open(glob.glob("/kaggle/input/**/val_balanced.txt", recursive=True)[0]) if l.strip()]

WORK = "/kaggle/working"
with open(f"{WORK}/val.txt", "w") as f:
    f.write("\n".join(f"{ROOT}/images/{i}.tif" for i in val_ids))
Y = f"{WORK}/data.yaml"
with open(Y, "w") as f:
    f.write(f"path: {WORK}\ntrain: val.txt\nval: val.txt\nchannels: 16\nnames:\n")
    for i, c in enumerate(CLASSES): f.write(f"  {i}: {c}\n")

ck = torch.load(W[0], weights_only=False, map_location="cpu")
IMGSZ = int((ck.get("train_args") or {}).get("imgsz", 768)); del ck
print(f"weights: {W[0]}\nimgsz: {IMGSZ}\nval: {len(val_ids)} anh\n", flush=True)

model = YOLO(W[0])
print(f"{'iou':>5s} {'conf':>8s} {'max_det':>8s} | {'mAP50-95':>9s} {'mAP50':>7s} {'recall':>7s}")
best = None
for iou, conf, md in itertools.product([0.5, 0.6, 0.7, 0.8], [0.001], [300]):
    m = model.val(data=Y, imgsz=IMGSZ, batch=8, device=0, verbose=False, plots=False,
                  iou=iou, conf=conf, max_det=md, project=f"{WORK}/s",
                  name=f"i{iou}", exist_ok=True)
    print(f"{iou:5.2f} {conf:8.4f} {md:8d} | {m.box.map:9.4f} {m.box.map50:7.4f} {m.box.mr:7.4f}", flush=True)
    if best is None or m.box.map > best[0]: best = (m.box.map, iou, conf, md)

# quanh nguong iou tot nhat, thu them max_det va conf
iou_b = best[1]
for conf, md in itertools.product([0.0001, 0.001, 0.01], [300, 600]):
    if (conf, md) == (0.001, 300): continue
    m = model.val(data=Y, imgsz=IMGSZ, batch=8, device=0, verbose=False, plots=False,
                  iou=iou_b, conf=conf, max_det=md, project=f"{WORK}/s",
                  name=f"c{conf}_{md}", exist_ok=True)
    print(f"{iou_b:5.2f} {conf:8.4f} {md:8d} | {m.box.map:9.4f} {m.box.map50:7.4f} {m.box.mr:7.4f}", flush=True)
    if m.box.map > best[0]: best = (m.box.map, iou_b, conf, md)

print(f"\n=== TOT NHAT: mAP50-95={best[0]:.4f} voi iou={best[1]}, conf={best[2]}, max_det={best[3]} ===")
print(f"(mac dinh dang dung: iou=0.7, conf=0.0001, max_det=300)")
