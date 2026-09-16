"""HODC2026 - overfitting hay thieu nang luc?

Do mAP cua CUNG mot model tren tap TRAIN va tap VAL.
  train >> val  -> overfitting  -> tang augmentation se giup
  train ~= val  -> thieu nang luc -> tang augmentation se lam TE HON
"""
import os, sys, glob, random, subprocess, importlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.152", "tifffile"], check=True)
importlib.invalidate_caches()
from ultralytics import YOLO

W = glob.glob("/kaggle/input/**/weights/best.pt", recursive=True)
assert W, "khong thay best.pt"
ROOT = os.path.dirname(glob.glob("/kaggle/input/**/hodc_train/images", recursive=True)[0])
CLASSES = [l.strip() for l in open(f"{ROOT}/classes.txt") if l.strip()]
vs = (glob.glob("/kaggle/input/**/val_balanced.txt", recursive=True))[0]
val_ids = [l.strip() for l in open(vs) if l.strip()]
all_ids = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{ROOT}/images/*.tif"))
train_ids = [i for i in all_ids if i not in set(val_ids)]

# lay mau train BANG KICH THUOC val de so sanh cong bang
random.Random(0).shuffle(train_ids)
sub = sorted(train_ids[:len(val_ids)])
print(f"val: {len(val_ids)} anh | mau train: {len(sub)} anh")

WORK = "/kaggle/working"
for name, lst in [("val", val_ids), ("trainsub", sub)]:
    with open(f"{WORK}/{name}.txt", "w") as f:
        f.write("\n".join(f"{ROOT}/images/{i}.tif" for i in lst))

model = YOLO(W[0])
import torch
_ck = torch.load(W[0], weights_only=False, map_location="cpu")
IMGSZ = int((_ck.get("train_args") or {}).get("imgsz", 768))
print(f"imgsz tu checkpoint: {IMGSZ}")
del _ck

res = {}
for name in ["val", "trainsub"]:
    y = f"{WORK}/data_{name}.yaml"
    with open(y, "w") as f:
        f.write(f"path: {WORK}\ntrain: {name}.txt\nval: {name}.txt\nchannels: 16\nnames:\n")
        for i, c in enumerate(CLASSES): f.write(f"  {i}: {c}\n")
    m = model.val(data=y, imgsz=IMGSZ, batch=8, device=0, verbose=False, plots=False,
                  project=f"{WORK}/v", name=name, exist_ok=True)
    res[name] = (m.box.map, m.box.map50, m.box.mr)
    print(f"\n[{name}] mAP50-95={m.box.map:.4f}  mAP50={m.box.map50:.4f}  recall={m.box.mr:.4f}", flush=True)

gap = res["trainsub"][0] - res["val"][0]
print("\n" + "="*56)
print(f"mAP50-95 tren TRAIN : {res['trainsub'][0]:.4f}")
print(f"mAP50-95 tren VAL   : {res['val'][0]:.4f}")
print(f"KHOANG CACH         : {gap:+.4f}")
print("="*56)
if gap > 0.15:
    print("=> OVERFITTING RO RANG. Tang augmentation / regularization la dung huong.")
elif gap > 0.06:
    print("=> Overfitting vua phai. Augmentation co the giup nhung khong nhieu.")
else:
    print("=> KHONG overfitting. Model dang THIEU NANG LUC hoac du lieu qua kho.")
    print("   Tang augmentation se lam TE HON. Nen di huong suc chua / du lieu / loss.")
