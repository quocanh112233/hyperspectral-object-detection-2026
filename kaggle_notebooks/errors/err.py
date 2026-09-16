"""HODC2026 - phan tich loi theo tung lop: nham lan giua cac lop, FP/FN, theo kich thuoc."""
import os, sys, glob, subprocess, importlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.152", "tifffile"], check=True)
importlib.invalidate_caches()
import numpy as np, torch
from ultralytics import YOLO

W = glob.glob("/kaggle/input/**/weights/best.pt", recursive=True)[0]
ROOT = os.path.dirname(glob.glob("/kaggle/input/**/hodc_train*/images", recursive=True)[0])
CLASSES = [l.strip() for l in open(f"{ROOT}/classes.txt") if l.strip()]
NC = len(CLASSES)
val_ids = [l.strip() for l in open(glob.glob("/kaggle/input/**/val_balanced.txt", recursive=True)[0]) if l.strip()]
WORK = "/kaggle/working"
with open(f"{WORK}/val.txt", "w") as f:
    f.write("\n".join(f"{ROOT}/images/{i}.tif" for i in val_ids))
Y = f"{WORK}/data.yaml"
with open(Y, "w") as f:
    f.write(f"path: {WORK}\ntrain: val.txt\nval: val.txt\nchannels: 16\nnames:\n")
    for i, c in enumerate(CLASSES): f.write(f"  {i}: {c}\n")
ck = torch.load(W, weights_only=False, map_location="cpu")
IMGSZ = int((ck.get("train_args") or {}).get("imgsz", 768)); del ck
print(f"weights={W}\ndata={ROOT}\nimgsz={IMGSZ}\n", flush=True)

model = YOLO(W)
RES = {}
for CONF in [0.001, 0.25]:
    mm = model.val(data=Y, imgsz=IMGSZ, batch=8, device=0, verbose=False, plots=True,
                   conf=CONF, iou=0.7, max_det=300, project=f"{WORK}/v",
                   name=f"e{CONF}", exist_ok=True)
    RES[CONF] = mm.confusion_matrix.matrix.copy()
    fp = sum(RES[CONF][j, NC] for j in range(NC))
    miss = sum(RES[CONF][NC, j] for j in range(NC))
    print(f"[conf={CONF}] FP tu nen = {int(fp)}, bo sot = {int(miss)}", flush=True)
print()
print("=== FP tu nen theo lop, o hai nguong ===")
print(f"{'lop':16s} {'conf>=0.001':>12s} {'conf>=0.25':>11s}")
for j, c in enumerate(CLASSES):
    print(f"{c:16s} {int(RES[0.001][j, NC]):12d} {int(RES[0.25][j, NC]):11d}")
print()
m = mm
cm = RES[0.001]   # (NC+1, NC+1): cot = that, hang = du doan; chi so cuoi = background
print("=== PHAN TICH THEO LOP ===")
print(f"{'lop':16s} {'GT':>5s} {'dung':>5s} {'sot':>5s} {'FP nen':>7s} {'nham thanh'}")
tot_miss = tot_fp = 0
for j, c in enumerate(CLASSES):
    gt = cm[:, j].sum()
    hit = cm[j, j]
    miss_bg = cm[NC, j]                      # that nhung bi coi la nen -> BO SOT
    conf_to = [(CLASSES[i], int(cm[i, j])) for i in range(NC) if i != j and cm[i, j] > 0]
    fp_bg = cm[j, NC]                        # nen nhung doan thanh lop nay
    tot_miss += miss_bg; tot_fp += fp_bg
    s = ", ".join(f"{n}:{v}" for n, v in sorted(conf_to, key=lambda x: -x[1])[:3]) or "-"
    print(f"{c:16s} {int(gt):5d} {int(hit):5d} {int(miss_bg):5d} {int(fp_bg):7d}  {s}")
print(f"\nTONG bo sot (that->nen): {int(tot_miss)} | FP tu nen: {int(tot_fp)}")

print("\n=== CAC CAP THAT/GIA CO NHAM LAN NHAU KHONG? ===")
pairs = [("apple","apple_plastic"),("banana","banana_plastic"),("orange","orange_plastic"),
         ("egg","egg_plastic"),("egg","egg_wood"),("car","car_toy")]
for a, b in pairs:
    ia, ib = CLASSES.index(a), CLASSES.index(b)
    print(f"  {a:14s} <-> {b:16s}: {a}->{b} = {int(cm[ib, ia])}, {b}->{a} = {int(cm[ia, ib])}"
          f"  (GT {a}={int(cm[:,ia].sum())}, {b}={int(cm[:,ib].sum())})")

print("\n=== mAP THEO KICH THUOC VAT THE ===")
sizes = {"small(<32px)": [], "medium(32-96)": [], "large(>96)": []}
for i in val_ids:
    f = f"{ROOT}/labels/{i}.txt"
    if not os.path.exists(f): continue
    import tifffile
    a = tifffile.imread(f"{ROOT}/images/{i}.tif")
    H, Wd = (a.shape[1], a.shape[2]) if a.shape[0] == 16 else (a.shape[0], a.shape[1])
    for ln in open(f):
        if not ln.strip(): continue
        _, _, _, bw, bh = map(float, ln.split())
        s = (bw * Wd * bh * H) ** 0.5
        k = "small(<32px)" if s < 32 else ("medium(32-96)" if s < 96 else "large(>96)")
        sizes[k].append(s)
for k, v in sizes.items():
    print(f"  {k:16s}: {len(v):5d} bbox ({100*len(v)/sum(len(x) for x in sizes.values()):.1f}%)")
