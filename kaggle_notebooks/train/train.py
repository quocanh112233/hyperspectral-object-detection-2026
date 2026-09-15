"""HODC2026 - train YOLO11 16 kenh.

Chay duoc ca tren Kaggle (mac dinh) lan local de kiem thu (dat HODC_LOCAL=<thu muc>).
"""
import os, sys, glob, json, random, importlib.util, pathlib, subprocess

# ============ 0a. CAI DAT (Kaggle khong co san ultralytics) ============
if not os.environ.get("HODC_LOCAL"):
    def pip(*args):
        return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args]).returncode
    if pip("ultralytics==8.4.152", "tifffile") != 0:
        whl = glob.glob("/kaggle/input/**/ultralytics-*.whl", recursive=True)
        assert whl, "khong cai duoc ultralytics: het mang va khong co banh xe offline"
        print("[cai dat] PyPI that bai -> dung banh xe offline")
        assert pip("--no-index", f"--find-links={os.path.dirname(whl[0])}",
                   "ultralytics==8.4.152", "tifffile") == 0, "cai offline cung that bai"
    else:
        print("[cai dat] xong tu PyPI")
    importlib.invalidate_caches()

# ============ 0. VA LOI MOSAIC PHA HONG DU LIEU 16-BIT ============
# Mosaic tao canvas bang dtype=np.uint8 roi chep anh uint16 vao -> cat vong gia tri.
# Phai va TRUOC khi import ultralytics.
def patch_mosaic_dtype():
    """Mosaic tao canvas bang dtype=np.uint8 roi chep anh uint16 vao -> cat vong gia tri
    (max 431 -> 255). Loi im lang: train van chay, loss van giam, nhung pho da hong.
    Kiem tra TRANG THAI CUOI chu khong phai so lan thay, de chay lai khong bao loi nham."""
    spec = importlib.util.find_spec("ultralytics")
    assert spec is not None, "ultralytics chua duoc cai"
    p = pathlib.Path(spec.origin).parent / "data" / "augment.py"
    src = p.read_text()
    for mult in (2, 3):
        head = f'np.full((self.imgsz * {mult}, self.imgsz * {mult}, labels["img"].shape[2]), 114, '
        src = src.replace(head + "dtype=np.uint8)", head + 'dtype=labels["img"].dtype)')
    p.write_text(src)
    n_ok = src.count('114, dtype=labels["img"].dtype)')
    n_bad = src.count('114, dtype=np.uint8)')
    assert n_ok == 2 and n_bad == 0, (
        f"VA THAT BAI: {n_ok}/2 cho da va, con {n_bad} cho dung uint8. "
        f"Phien ban ultralytics khac ban da kiem thu -> mosaic se pha hong du lieu 16-bit.")
    print(f"[va mosaic] OK: {n_ok}/2 cho dung dtype cua anh")

patch_mosaic_dtype()

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

P2_YAML = """# YOLO11 co them dau P2 (stride 4) cho vat the nho.
# Ly do: 71% vat the trong HODC2026 nho hon 32x32 px, trung vi 24x24.
# YOLO11 goc bat dau tu P3 (stride 8) -> vat 24px chi con 3 pixel tren feature map.
# Backbone giu NGUYEN so voi yolo11.yaml de con nap duoc trong so COCO.
nc: 80
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]            # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]           # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]    # 2       <- dac trung P2/4
  - [-1, 1, Conv, [256, 3, 2]]           # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]    # 4       <- dac trung P3/8
  - [-1, 1, Conv, [512, 3, 2]]           # 5-P4/16
  - [-1, 2, C3k2, [512, True]]           # 6       <- dac trung P4/16
  - [-1, 1, Conv, [1024, 3, 2]]          # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]          # 8
  - [-1, 1, SPPF, [1024, 5]]             # 9
  - [-1, 2, C2PSA, [1024]]               # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]            # cat backbone P4
  - [-1, 2, C3k2, [512, False]]          # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]            # cat backbone P3
  - [-1, 2, C3k2, [256, False]]          # 16 (P3/8)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]            # cat backbone P2
  - [-1, 2, C3k2, [128, False]]          # 19 (P2/4-xsmall)  <-- MOI

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]           # cat head P3
  - [-1, 2, C3k2, [256, False]]          # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]           # cat head P4
  - [-1, 2, C3k2, [512, False]]          # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]           # cat head P5
  - [-1, 2, C3k2, [1024, True]]          # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
"""

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

LOCAL = os.environ.get("HODC_LOCAL")
if LOCAL:
    DATA_ROOT, WORK = LOCAL, os.path.join(LOCAL, "work")
    MODEL_YAML = os.environ.get("HODC_MODEL", "yolo11n.yaml")
    EPOCHS, IMGSZ, BATCH, DEVICE = 2, 256, 2, "cpu"
    PRETRAINED = None
else:
    hits = glob.glob("/kaggle/input/**/hodc_train/images", recursive=True)
    assert hits, "khong thay hodc_train/images trong /kaggle/input"
    DATA_ROOT = str(pathlib.Path(hits[0]).parent)
    WORK = "/kaggle/working"
    MODEL_YAML = os.environ.get("HODC_MODEL", "p2")
    EPOCHS = int(os.environ.get("HODC_EPOCHS", 40))
    IMGSZ = int(os.environ.get("HODC_IMGSZ", 768))
    BATCH = int(os.environ.get("HODC_BATCH", 12))
    DEVICE = 0
    local_w = glob.glob("/kaggle/input/**/yolo11s.pt", recursive=True)
    PRETRAINED = local_w[0] if local_w else "yolo11s.pt"

IMG_DIR = os.path.join(DATA_ROOT, "images")
LBL_DIR = os.path.join(DATA_ROOT, "labels")
CLASSES = [l.strip() for l in open(os.path.join(DATA_ROOT, "classes.txt")) if l.strip()] \
    if os.path.exists(os.path.join(DATA_ROOT, "classes.txt")) else ["a", "b"]
NC = len(CLASSES)
os.makedirs(WORK, exist_ok=True)
if MODEL_YAML == "p2":
    MODEL_YAML = os.path.join(WORK, "yolo11s-p2.yaml")
    open(MODEL_YAML, "w").write(P2_YAML)
    print("dung kien truc P2-P5 (them dau stride 4 cho vat the nho)")

print(f"DATA_ROOT={DATA_ROOT}  so lop={NC}  model={MODEL_YAML}  imgsz={IMGSZ}  epochs={EPOCHS}")

# ============ 1. CHIA TRAIN / VAL ============
ids = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{IMG_DIR}/*.tif"))
print("tong so anh:", len(ids))
# Chia ngau nhien bi RO RI nang: 66% anh val co ban sao gan giong trong train
# (do tuong dong >= 0.95), khien val = 0.730 trong khi LB chi 0.604.
# Uu tien danh sach chia theo NHOM CANH do kernel dupcheck sinh ra.
gs = glob.glob("/kaggle/input/**/val_group_0.99.txt", recursive=True)
if gs:
    val_ids = set(l.strip() for l in open(gs[0]) if l.strip()) & set(ids)
    train_ids = [i for i in ids if i not in val_ids]
    print(f"chia theo NHOM CANH tu {gs[0]}: {len(val_ids)} anh val (ro ri = 0 o nguong 0.99)")
else:
    rng = random.Random(SEED); shuffled = ids[:]; rng.shuffle(shuffled)
    n_val = max(1, int(round(len(ids) * 0.10)))
    val_ids, train_ids = set(shuffled[:n_val]), shuffled[n_val:]
    print("CANH BAO: khong thay danh sach nhom canh -> chia ngau nhien (SE BI RO RI)")

def cls_count(id_list):
    c = np.zeros(NC, int)
    for i in id_list:
        f = f"{LBL_DIR}/{i}.txt"
        if os.path.exists(f):
            for ln in open(f):
                if ln.strip(): c[int(ln.split()[0])] += 1
    return c

ct, cv = cls_count(train_ids), cls_count(sorted(val_ids))
print(f"train {len(train_ids)} anh / val {len(val_ids)} anh")
print(f"{'lop':18s} {'train':>7s} {'val':>6s}")
for i, name in enumerate(CLASSES):
    print(f"{name:18s} {ct[i]:7d} {cv[i]:6d}" + ("   <-- VAL RONG" if cv[i] == 0 else ""))
assert (cv > 0).all(), "co lop khong xuat hien trong val - phai chia lai"

os.makedirs(WORK, exist_ok=True)
for split, lst in [("train", train_ids), ("val", sorted(val_ids))]:
    with open(f"{WORK}/{split}.txt", "w") as f:
        f.write("\n".join(f"{IMG_DIR}/{i}.tif" for i in lst))

yaml_path = f"{WORK}/data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"path: {WORK}\ntrain: train.txt\nval: val.txt\nchannels: 16\nnames:\n")
    for i, n in enumerate(CLASSES):
        f.write(f"  {i}: {n}\n")
print("da ghi", yaml_path)

# ============ 2. MODEL 16 KENH + THOI PHONG TRONG SO COCO ============
def build_16ch(cfg, nc, pretrained):
    m = DetectionModel(cfg=cfg, ch=16, nc=nc, verbose=False)
    if not pretrained:
        print("[model] khong nap pretrain (che do kiem thu)")
        return m
    src = YOLO(pretrained).model.state_dict()
    dst = m.state_dict()
    n_ok = n_inf = n_skip = 0
    for k, v in src.items():
        if k not in dst:
            continue
        if dst[k].shape == v.shape:
            dst[k] = v; n_ok += 1
        elif k == "model.0.conv.weight":
            # trung binh RGB -> lap ra 16 kenh, chia 3/16 de giu do lon kich hoat
            dst[k] = v.mean(1, keepdim=True).repeat(1, 16, 1, 1) * (3.0 / 16.0)
            n_inf += 1
        else:
            n_skip += 1
    m.load_state_dict(dst)
    print(f"[model] nap COCO: {n_ok} khop + {n_inf} thoi phong conv dau, "
          f"{n_skip} bo qua (head doi so lop)")
    return m

model_16 = build_16ch(MODEL_YAML, NC, PRETRAINED)
ckpt = f"{WORK}/init_16ch.pt"
torch.save({"model": model_16, "date": "", "version": "", "train_args": {}}, ckpt)
print("da luu model khoi tao:", ckpt)

model = YOLO(ckpt)
info = model.model.info(detailed=False, verbose=True)   # in ra so tham so + FLOPs

# ============ 3. TRAIN ============
results = model.train(
    data=yaml_path, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
    workers=os.cpu_count() if not LOCAL else 0,
    # TAT augmentation mau: no pha huy chu ky pho dung de phan biet that/gia
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    # chi giu bien doi hinh hoc
    mosaic=1.0, close_mosaic=10, fliplr=0.5, flipud=0.0,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    erasing=0.0,
    seed=SEED, deterministic=False, patience=50,
    save_period=10, plots=False, val=True, cache=False,
    project=f"{WORK}/runs", name="y11_16ch", exist_ok=True,
)
print("\n=== KET QUA ===")
print("mAP50-95:", results.box.map)
print("mAP50   :", results.box.map50)
print("theo lop:", dict(zip(CLASSES, [round(float(x), 4) for x in results.box.maps])))
json.dump({"map50_95": float(results.box.map), "map50": float(results.box.map50),
           "per_class": {c: float(x) for c, x in zip(CLASSES, results.box.maps)},
           "imgsz": IMGSZ, "epochs": EPOCHS, "model": MODEL_YAML, "params_flops": str(info)},
          open(f"{WORK}/metrics.json", "w"), indent=2)
