"""HODC2026 - train YOLO11 16 kenh.

Chay duoc ca tren Kaggle (mac dinh) lan local de kiem thu (dat HODC_LOCAL=<thu muc>).
"""
import os, sys, glob, json, random, importlib.util, pathlib, subprocess

# ============ CAU HINH LAN CHAY (sua o day roi day kernel len) ============
# Moi lan chay chi doi DUNG MOT muc so voi lan truoc - bai hoc tu lan 2 (doi 4 thu
# cung luc nen khong biet thu nao gay ra ket qua).
RUN_CFG = {
    "HODC_DFL": "4.0",   # giu cai da chung minh la tot hon (lan 11: +0.005)
    "HODC_SEED": "2",   # <== model doc lap thu 2; gop hop chi an khi cac model SAI KHAC NHAU
}
for _k, _v in RUN_CFG.items():
    os.environ.setdefault(_k, _v)
print("[cau hinh lan chay]", RUN_CFG)

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
def patch_uint8_assumptions():
    """Ultralytics gia dinh anh la uint8 o nhieu cho. Voi du lieu 16-bit cua ta,
    cac cho do CAT VONG gia tri (max 1123 -> 255) ma KHONG bao loi -> pho bi pha huy.

      1. Mosaic: canvas tao bang dtype=np.uint8
      2. MixUp : ket qua tron ep ve .astype(np.uint8)

    Kiem tra TRANG THAI CUOI chu khong phai so lan thay, de chay lai khong bao loi nham.
    (copy_paste cung co loi tuong tu nhung no can mask phan vung ma ta khong co -> khong dung.)"""
    spec = importlib.util.find_spec("ultralytics")
    assert spec is not None, "ultralytics chua duoc cai"
    p = pathlib.Path(spec.origin).parent / "data" / "augment.py"
    src = p.read_text()

    for mult in (2, 3):
        head = f'np.full((self.imgsz * {mult}, self.imgsz * {mult}, labels["img"].shape[2]), 114, '
        src = src.replace(head + "dtype=np.uint8)", head + 'dtype=labels["img"].dtype)')
    mix_old = '(labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)'
    mix_new = '(labels["img"] * r + labels2["img"] * (1 - r)).astype(labels["img"].dtype)'
    src = src.replace(mix_old, mix_new)
    p.write_text(src)

    n_mosaic = src.count('114, dtype=labels["img"].dtype)')
    n_mix = src.count(mix_new)
    bad = src.count('114, dtype=np.uint8)') + src.count(mix_old)
    assert n_mosaic == 2 and n_mix == 1 and bad == 0, (
        f"VA THAT BAI: mosaic {n_mosaic}/2, mixup {n_mix}/1, con {bad} cho dung uint8. "
        f"Phien ban ultralytics khac ban da kiem thu -> du lieu 16-bit se bi pha hong.")
    print(f"[va uint8] OK: mosaic {n_mosaic}/2, mixup {n_mix}/1")

patch_uint8_assumptions()

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


def patch_spectral_gain(jitter):
    """Augmentation DAC THU CHO ANH PHO, cam vao cho trong ma RandomHSV de lai.

    RandomHSV cua Ultralytics tu thoat khi anh khong co dung 3 kenh
    (`if img.shape[-1] != 3: return`), nen voi anh 16 bang no von KHONG chay.
    Ta thay no bang DO LOI TOAN CUC: nhan CA cac bang voi CUNG mot he so ngau nhien.

    Vi sao chi phep nay hop le: chu ky pho de phan biet that/gia nam o TI LE
    giua cac bang. Nhan chung mot he so giu ti le do nguyen ven tuyet doi, trong khi
    van mo phong dung thu bien thien co that ngoai doi (cuong do sang, thoi gian phoi
    sang, khoang cach den nguon). Bat ky phep nao lech giua cac bang deu pha dac trung."""
    from ultralytics.data import augment as A
    if not hasattr(A.RandomHSV, "_hodc_orig"):
        A.RandomHSV._hodc_orig = A.RandomHSV.__call__

    def __call__(self, labels):
        if jitter <= 0:
            return A.RandomHSV._hodc_orig(self, labels)
        img = labels["img"]
        g = random.uniform(1.0 - jitter, 1.0 + jitter)
        out = img.astype(np.float32) * g
        if np.issubdtype(img.dtype, np.integer):
            ii = np.iinfo(img.dtype)
            out = np.clip(out, ii.min, ii.max)
        labels["img"] = out.astype(img.dtype)
        return labels

    A.RandomHSV.__call__ = __call__
    print(f"[augment pho] do loi toan cuc +/-{jitter:.0%}" if jitter > 0
          else "[augment pho] tat (giu RandomHSV goc)")

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

# Hat giong doi duoc: de train nhieu model DOC LAP roi gop hop.
# Gop chi giam nhieu toa do khi cac model sai KHAC NHAU; cung hat giong thi
# chung sai giong het nhau va trung binh khong giam duoc gi.
SEED = int(os.environ.get("HODC_SEED", 0))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[hat giong] {SEED}")

LOCAL = os.environ.get("HODC_LOCAL")
if LOCAL:
    DATA_ROOT, WORK = LOCAL, os.path.join(LOCAL, "work")
    MODEL_YAML = os.environ.get("HODC_MODEL", "yolo11n.yaml")
    EPOCHS, IMGSZ, BATCH, DEVICE = 2, 256, 2, "cpu"
    PRETRAINED = None
else:
    # nhan ca bo goc (hodc_train) lan bo da chuan hoa pho (hodc_train_norm);
    # bo nao duoc gan qua kernel_sources thi dung bo do
    hits = glob.glob("/kaggle/input/**/hodc_train*/images", recursive=True)
    assert hits, "khong thay hodc_train*/images trong /kaggle/input"
    assert len(hits) == 1, f"gan nham NHIEU bo du lieu: {hits}"
    DATA_ROOT = str(pathlib.Path(hits[0]).parent)
    WORK = "/kaggle/working"
    MODEL_YAML = os.environ.get("HODC_MODEL", "yolo11s.yaml")
    EPOCHS = int(os.environ.get("HODC_EPOCHS", 40))
    IMGSZ = int(os.environ.get("HODC_IMGSZ", 768))
    BATCH = int(os.environ.get("HODC_BATCH", 8))
    DEVICE = 0
    # trong so pretrain phai KHOP kich co model, neu khong se bo qua gan het tensor
    _base = "yolo11s.pt" if MODEL_YAML == "p2" else \
            os.path.basename(str(MODEL_YAML)).replace(".yaml", ".pt")
    local_w = glob.glob(f"/kaggle/input/**/{_base}", recursive=True)
    PRETRAINED = local_w[0] if local_w else _base

IMG_DIR = os.path.join(DATA_ROOT, "images")
LBL_DIR = os.path.join(DATA_ROOT, "labels")
import tifffile as _tf
_probe = _tf.imread(sorted(glob.glob(f"{IMG_DIR}/*.tif"))[0]) if glob.glob(f"{IMG_DIR}/*.tif") else None
NCH = int(_probe.shape[0]) if (_probe is not None and _probe.ndim == 3) else 16
print(f"so kenh doc duoc tu du lieu: {NCH}")

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
gs = (glob.glob("/kaggle/input/**/val_balanced.txt", recursive=True)
      or glob.glob("/kaggle/input/**/val_group_0.99.txt", recursive=True))
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
    f.write(f"path: {WORK}\ntrain: train.txt\nval: val.txt\nchannels: {NCH}\nnames:\n")
    for i, n in enumerate(CLASSES):
        f.write(f"  {i}: {n}\n")
print("da ghi", yaml_path)

# ============ 2. MODEL 16 KENH + THOI PHONG TRONG SO COCO ============
# Them nhanh P2 lam CHI SO cac lop trong head dich di, nen khop trong so theo TEN
# bi truot hang loat (lan 2 chi nap duoc 296/499 thay vi 492). Anh xa thu cong:
# da kiem chung 90 tensor nay TRUNG shape tuyet doi -> 296 + 90 = 386.
P2_REMAP = {17: 23, 19: 25, 20: 26, 22: 28}


def build_16ch(cfg, nc, pretrained, ch=16):
    m = DetectionModel(cfg=cfg, ch=ch, nc=nc, verbose=False)
    if not pretrained:
        print("[model] khong nap pretrain (che do kiem thu)")
        return m
    src = YOLO(pretrained).model.state_dict()
    dst = m.state_dict()
    is_p2 = "p2" in str(cfg).lower()
    n_ok = n_inf = n_map = n_skip = 0
    for k, v in src.items():
        tgt = k if (k in dst and dst[k].shape == v.shape) else None
        if tgt is None and is_p2:
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) in P2_REMAP:
                nk = ".".join([parts[0], str(P2_REMAP[int(parts[1])])] + parts[2:])
                if nk in dst and dst[nk].shape == v.shape:
                    tgt, n_map = nk, n_map + 1
        if tgt is not None:
            dst[tgt] = v
            if tgt == k: n_ok += 1
        elif k == "model.0.conv.weight":
            dst[k] = v.mean(1, keepdim=True).repeat(1, ch, 1, 1) * (3.0 / ch)
            n_inf += 1
        else:
            n_skip += 1
    m.load_state_dict(dst)
    total = n_ok + n_map + n_inf
    print(f"[model] nap COCO: {n_ok} khop ten + {n_map} anh xa P2 + {n_inf} thoi phong "
          f"conv dau = {total} tensor ({n_skip} bo qua)")
    if total < (370 if is_p2 else 450):
        print(f"!! CANH BAO: chi nap duoc {total} tensor (ky vong "
              f"{"~378 voi P2" if is_p2 else "~492"}). Ket qua kem di co the do MAT "
              f"PRETRAIN chu khong phai do kien truc.")
    return m


model_16 = build_16ch(MODEL_YAML, NC, PRETRAINED, ch=NCH)
ckpt = f"{WORK}/init_16ch.pt"
torch.save({"model": model_16, "date": "", "version": "", "train_args": {}}, ckpt)
print("da luu model khoi tao:", ckpt)

model = YOLO(ckpt)
info = model.model.info(detailed=False, verbose=True)   # in ra so tham so + FLOPs

# ============ 3. TRAIN ============
patch_spectral_gain(float(os.environ.get("HODC_GAIN", 0.0)))
results = model.train(
    data=yaml_path, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
    workers=os.cpu_count() if not LOCAL else 0,
    # TAT augmentation mau: no pha huy chu ky pho dung de phan biet that/gia
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    # chi giu bien doi hinh hoc
    mosaic=1.0, close_mosaic=10, fliplr=0.5, flipud=0.0,
    degrees=0.0, shear=0.0, perspective=0.0, erasing=0.0,
    # Tang regularization: chan doan cho thay chenh train-val = 0.094,
    # va recall tren val tut manh o vai lop (e-bike 0.41) du train dat 0.98.
    translate=float(os.environ.get("HODC_TRANSLATE", 0.1)),
    scale=float(os.environ.get("HODC_SCALE", 0.5)),
    mixup=float(os.environ.get("HODC_MIXUP", 0.0)),
    copy_paste=float(os.environ.get("HODC_COPYPASTE", 0.0)),
    # Can bang loss. Chan doan: mAP@0.5 = 0.94 nhung mAP@[.5:.95] = 0.67
    # -> phat hien + phan loai gan nhu xong, nut that la DO KHIT CUA HOP.
    # Nen doi trong so tu cls sang box/dfl (dfl chinh la nhanh hoi quy canh hop).
    box=float(os.environ.get("HODC_BOX", 7.5)),
    cls=float(os.environ.get("HODC_CLS", 0.5)),
    dfl=float(os.environ.get("HODC_DFL", 1.5)),
    optimizer=os.environ.get("HODC_OPT", "auto"),
    lr0=float(os.environ.get("HODC_LR0", 0.01)),
    multi_scale=os.environ.get("HODC_MULTISCALE", "0") == "1",
    seed=SEED, deterministic=False, patience=50,
    save_period=10, plots=False, val=True, cache=False,
    project=f"{WORK}/runs", name="y11_16ch", exist_ok=True,
)
# TTA: lat ngang + da ti le roi gop hop lai. Nham thang vao nut that DO KHIT CUA HOP.
# Chay tren chinh best.pt vua train xong -> chi ton vai phut, khong ton them lan train.
tta_map = None
try:
    _t = model.val(data=yaml_path, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                   augment=True, plots=False, verbose=False)
    tta_map = float(_t.box.map)
    print(f"\n[TTA] mAP50-95 co TTA = {tta_map:.4f}  (khong TTA = {float(results.box.map):.4f})")
except Exception as e:
    print(f"\n[TTA] bo qua, loi: {type(e).__name__}: {e}")

print("\n=== KET QUA ===")
print("mAP50-95:", results.box.map)
print("mAP50   :", results.box.map50)
print("theo lop:", dict(zip(CLASSES, [round(float(x), 4) for x in results.box.maps])))
json.dump({"map50_95": float(results.box.map), "map50": float(results.box.map50),
           "per_class": {c: float(x) for c, x in zip(CLASSES, results.box.maps)},
           "map50_95_tta": tta_map,
           "box": float(os.environ.get("HODC_BOX", 7.5)),
           "dfl": float(os.environ.get("HODC_DFL", 1.5)),
           "gain_jitter": float(os.environ.get("HODC_GAIN", 0.0)),
           "imgsz": IMGSZ, "epochs": EPOCHS, "model": MODEL_YAML, "params_flops": str(info)},
          open(f"{WORK}/metrics.json", "w"), indent=2)
