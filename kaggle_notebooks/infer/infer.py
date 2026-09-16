"""HODC2026 - inference + sinh file nop.

Chay tren anh TIFF da chuyen doi (hodc_test, va hodc_ranking o Phase 2).
Toa do tra ve da nam san trong he SAU demosaic vi TIFF chinh la cube.
"""
import os, sys, glob, importlib.util, pathlib, subprocess

if not os.environ.get("HODC_LOCAL"):
    def pip(*a): return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a]).returncode
    assert pip("ultralytics==8.4.152", "tifffile") == 0, "khong cai duoc ultralytics"
    importlib.invalidate_caches()

import numpy as np
import pandas as pd
import torch
import tifffile
from ultralytics import YOLO

# nguong thap + nhieu detection: mAP thuong loi khi giu ca cac du doan do tin cay thap,
# mien la chung duoc xep hang dung. Mac dinh conf=0.25 se mat diem.
CONF = float(os.environ.get("HODC_CONF", 0.0001))
MAX_DET = int(os.environ.get("HODC_MAXDET", 300))
IMGSZ = None   # doc tu checkpoint o duoi, tranh lech do phan giai train/inference
BATCH = int(os.environ.get("HODC_BATCH", 16))

LOCAL = os.environ.get("HODC_LOCAL")
if LOCAL:
    WEIGHTS = os.environ["HODC_WEIGHTS"]
    SETS = [("test", os.path.join(LOCAL, "images"))]
    OUT = os.path.join(LOCAL, "sub.csv")
    DEVICE = "cpu"
else:
    w = glob.glob("/kaggle/input/**/weights/best.pt", recursive=True)
    assert w, "khong thay best.pt - da gan output cua kernel train chua?"
    WEIGHTS = w[0]
    SETS = []
    for tag in ["test", "ranking"]:
        hits = glob.glob(f"/kaggle/input/**/hodc_{tag}*/images", recursive=True)
        if hits:
            SETS.append((tag, hits[0]))
    assert SETS, "khong thay thu muc anh nao"
    OUT = "/kaggle/working/submission.csv"
    DEVICE = 0

print("weights :", WEIGHTS)
print("cac bo  :", [(t, len(glob.glob(f'{d}/*.tif'))) for t, d in SETS])

model = YOLO(WEIGHTS)
names = model.names
CH = model.model.yaml.get("channels", 3)

# Doc imgsz DA DUNG LUC TRAIN tu checkpoint. Chay inference o do phan giai khac
# luc train lam mAP tut ma khong bao loi gi.
import torch as _t
_ck = _t.load(WEIGHTS, weights_only=False, map_location="cpu")
_ta = _ck.get("train_args") or {}
_train_imgsz = _ta.get("imgsz")
if os.environ.get("HODC_IMGSZ"):
    IMGSZ = int(os.environ["HODC_IMGSZ"])
    print(f"imgsz = {IMGSZ} (dat thu cong; luc train dung {_train_imgsz})")
else:
    assert _train_imgsz, "checkpoint khong ghi imgsz -> phai dat HODC_IMGSZ thu cong"
    IMGSZ = int(_train_imgsz)
    print(f"imgsz = {IMGSZ} (lay tu checkpoint, khop voi luc train)")
del _ck
print("so lop cua model:", len(names), "| so kenh:", CH)
assert CH == 16, f"model nay chi co {CH} kenh - sai checkpoint"

# AutoBackend khong mang theo thuoc tinh 'channels', nen predictor mac dinh ve 3 kenh
# va doc anh TIFF thanh RGB -> sai hoan toan. Gan lai sau khi AutoBackend duoc tao.
from ultralytics.engine.predictor import BasePredictor
_orig_setup = BasePredictor.setup_model
def _setup_with_channels(self, model=None, verbose=True):
    _orig_setup(self, model, verbose)
    self.model.channels = CH
BasePredictor.setup_model = _setup_with_channels

rows = []
n_degen = [0]
all_ids = []
for tag, d in SETS:
    files = sorted(glob.glob(f"{d}/*.tif"))
    all_ids += [os.path.splitext(os.path.basename(f))[0] for f in files]
    for i in range(0, len(files), BATCH):
        chunk = files[i:i+BATCH]
        # Loader theo duong dan file cua Ultralytics chi ho tro 1 hoac 3 kenh
        # (cv2.IMREAD_COLOR). Phai tu doc cube roi truyen mang numpy vao.
        imgs = []
        for f in chunk:
            a = tifffile.imread(f)                 # (16, H, W) vi luc ghi da transpose
            if a.ndim == 3 and a.shape[0] == 16:
                a = np.transpose(a, (1, 2, 0))     # -> (H, W, 16)
            assert a.shape[2] == 16, f"{f}: {a.shape} khong phai 16 kenh"
            imgs.append(np.ascontiguousarray(a))
        res = model.predict(imgs, imgsz=IMGSZ, conf=CONF, max_det=MAX_DET,
                            device=DEVICE, verbose=False, augment=False)
        for f, r in zip(chunk, res):
            img_id = os.path.splitext(os.path.basename(f))[0]
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            cf = b.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, s in zip(xyxy, cls, cf):
                # Model doi khi tra box suy bien khi vat the cham mep anh.
                # Box rong 0 pixel khong the khop IoU voi bat cu gi -> bo di.
                if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                    n_degen[0] += 1
                    continue
                rows.append((img_id, int(c), float(s), float(x1), float(y1), float(x2), float(y2)))
        if (i // BATCH) % 10 == 0:
            print(f"  [{tag}] {min(i+BATCH, len(files))}/{len(files)}", flush=True)

# Anh khong co du doan nao se mat TRANG diem cua anh do. Quet lai rieng chung
# voi nguong gan bang 0 truoc khi ket luan la model that su khong thay gi.
have = {r[0] for r in rows}
missing0 = [i for i in all_ids if i not in have]
if missing0:
    print(f"quet lai {len(missing0)} anh chua co du doan voi conf=1e-7...", flush=True)
    id2path = {os.path.splitext(os.path.basename(f))[0]: f
               for _, d in SETS for f in glob.glob(f"{d}/*.tif")}
    for i0 in range(0, len(missing0), BATCH):
        ch = [id2path[m] for m in missing0[i0:i0+BATCH]]
        ims = []
        for f in ch:
            a = tifffile.imread(f)
            if a.ndim == 3 and a.shape[0] == 16: a = np.transpose(a, (1, 2, 0))
            ims.append(np.ascontiguousarray(a))
        for f, r in zip(ch, model.predict(ims, imgsz=IMGSZ, conf=1e-7, max_det=MAX_DET,
                                          device=DEVICE, verbose=False)):
            iid = os.path.splitext(os.path.basename(f))[0]
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            for (x1, y1, x2, y2), c, sc in zip(b.xyxy.cpu().numpy(),
                                               b.cls.cpu().numpy().astype(int),
                                               b.conf.cpu().numpy()):
                if (x2 - x1) >= 1.0 and (y2 - y1) >= 1.0:
                    rows.append((iid, int(c), float(sc), float(x1), float(y1), float(x2), float(y2)))
    still = [i for i in missing0 if i not in {r[0] for r in rows}]
    print(f"  sau khi quet lai, con {len(still)} anh trang: {still[:10]}")

rows.sort(key=lambda r: (r[0], -r[2]))
df = pd.DataFrame(rows, columns=["image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
df.insert(0, "id", range(len(df)))
df.to_csv(OUT, index=False)
print(f"\nda ghi {OUT}: {len(df)} dong, {df.image_id.nunique()} anh co du doan")
print(f"da bo {n_degen[0]} box suy bien (rong hoac cao < 1 pixel)")
missing = sorted(set(all_ids) - set(df.image_id.astype(str)))
if missing:
    print(f"!! {len(missing)} anh KHONG co du doan nao:", missing[:20])

# ================= KIEM TRA FILE NOP =================
print("\n=== KIEM TRA ===")
n_img_expected = len(all_ids)
checks = [
    ("id lien tuc tu 0",        list(df.id) == list(range(len(df)))),
    ("id khong trung",          df.id.is_unique),
    ("class_id trong 0..17",    df.class_id.between(0, 17).all()),
    ("confidence trong [0,1]",  df.confidence.between(0, 1).all()),
    ("x1 < x2",                 (df.x1 < df.x2).all()),
    ("y1 < y2",                 (df.y1 < df.y2).all()),
    ("toa do khong am",         (df[["x1","y1"]] >= 0).all().all()),
    ("moi anh deu co du doan",  len(missing) == 0),
]
for name, ok in checks:
    print(f"  [{'OK ' if ok else 'LOI'}] {name}")
if df.image_id.nunique() != n_img_expected:
    print(f"      -> {n_img_expected - df.image_id.nunique()} anh KHONG co du doan nao. "
          f"Cac anh nay se bi 0 diem. Can ha CONF hoac kiem tra model.")
print("\nso du doan / anh: %.1f (trung binh)" % (len(df) / max(df.image_id.nunique(), 1)))
print("phan bo lop:", df.class_id.value_counts().sort_index().to_dict())
print(df.head(3).to_string(index=False))
