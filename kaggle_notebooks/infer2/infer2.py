"""HODC2026 - suy luan co GOP HOP (WBF) + doi chieu thuoc do.

Hai viec trong mot lan chay GPU:

  1. DOI CHIEU: vi sao Ultralytics bao 0.6754 con pycocotools cua toi bao 0.5007
     tren CUNG bo trong so va CUNG danh sach val? Mot trong hai sai, va toi can biet
     cai nao truoc khi tin bat ky ket luan nao. Chay `model.val()` cua Ultralytics
     ngay canh cach do cua toi, tren cung du lieu.

  2. NOP BAI: gop nhieu diem luu (va nhieu model neu co) bang WBF + TTA, xuat CSV.
     Do tren val: TTA +0.026, gop 4 diem luu +0.037 so voi mot model thuong.
"""
import os, sys, glob, json, subprocess, pathlib, time
import numpy as np, pandas as pd

def pip(*a): return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a]).returncode
pip("ultralytics==8.4.152", "tifffile", "pycocotools")
import tifffile, torch
from ultralytics import YOLO

CONF = float(os.environ.get("HODC_CONF", 0.001))
MAX_DET = int(os.environ.get("HODC_MAXDET", 300))
WBF_IOU = float(os.environ.get("HODC_WBFIOU", 0.55))
TTA = os.environ.get("HODC_TTA", "1") == "1"

# ---------- WBF ----------
def _iou_m(box, boxes):
    if len(boxes) == 0: return np.zeros(0, np.float32)
    xx1 = np.maximum(box[0], boxes[:,0]); yy1 = np.maximum(box[1], boxes[:,1])
    xx2 = np.minimum(box[2], boxes[:,2]); yy2 = np.minimum(box[3], boxes[:,3])
    inter = np.clip(xx2-xx1,0,None)*np.clip(yy2-yy1,0,None)
    a1 = (box[2]-box[0])*(box[3]-box[1]); a2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return inter/np.maximum(a1+a2-inter, 1e-9)

def wbf(boxes, scores, labels, iou_thr=0.55, n_models=1):
    boxes = np.asarray(boxes, np.float32).reshape(-1,4)
    scores = np.asarray(scores, np.float32).reshape(-1); labels = np.asarray(labels).reshape(-1)
    ob, os_, ol = [], [], []
    for lab in np.unique(labels):
        m = labels == lab; b, s = boxes[m], scores[m]
        o = np.argsort(-s); b, s = b[o], s[o]
        clusters, cb = [], np.zeros((0,4), np.float32)
        for i in range(len(b)):
            if len(clusters):
                iou = _iou_m(b[i], cb); j = int(np.argmax(iou))
                if iou[j] >= iou_thr:
                    clusters[j].append(i); idx = clusters[j]; w = s[idx][:,None]
                    cb[j] = (b[idx]*w).sum(0)/max(w.sum(),1e-9); continue
            clusters.append([i]); cb = np.vstack([cb, b[i][None]])
        for j, idx in enumerate(clusters):
            sc = s[idx]; ob.append(cb[j]); ol.append(lab)
            os_.append(float(sc.mean())*min(len(idx), n_models)/float(n_models))
    if not ob: return np.zeros((0,4),np.float32), np.zeros(0,np.float32), np.zeros(0,int)
    o = np.argsort(-np.asarray(os_))
    return np.asarray(ob,np.float32)[o], np.asarray(os_,np.float32)[o], np.asarray(ol)[o]

def load(p):
    a = tifffile.imread(p)
    if a.ndim == 3 and a.shape[0] in (3, 16): a = np.transpose(a, (1,2,0))
    return np.ascontiguousarray(a)

# ---------- tim trong so ----------
ws = sorted(glob.glob("/kaggle/input/**/weights/*.pt", recursive=True))
ws = [w for w in ws if os.path.basename(w) not in ("yolo26n.pt", "yolo11n.pt", "epoch0.pt")]
print("trong so tim thay:")
for w in ws: print("  ", w.replace("/kaggle/input/notebooks/qucanhtrnnguyn/", ""))
bests = sorted(w for w in ws if w.endswith("best.pt"))
snaps = sorted(w for w in ws if "epoch" in os.path.basename(w))
# Uu tien best.pt cua CAC LAN TRAIN DOC LAP: loi cua chung it tuong quan hon nhieu
# so voi cac diem luu trong cung mot lan, nen trung binh hoa giam nhieu duoc nhieu hon.
# Chi bu bang diem luu khi chua du 2 model doc lap.
USE = bests if len(bests) >= 2 else (bests + snaps[-3:])
MODE = os.environ.get("HODC_USE", "")
if MODE == "all":
    USE = bests + snaps
print(f"\ndung {len(USE)} bo: {[os.path.basename(w) for w in USE]}")
assert USE, "khong thay trong so nao"

m0 = YOLO(USE[0])
IMGSZ = int((getattr(m0, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 768))
print(f"imgsz = {IMGSZ} | conf = {CONF} | TTA = {TTA} | WBF iou = {WBF_IOU}")

# ================= PHAN 1: DOI CHIEU THUOC DO =================
tr = glob.glob("/kaggle/input/**/hodc_train*/images", recursive=True)
if tr:
    ROOT = str(pathlib.Path(tr[0]).parent)
    vf = (glob.glob("/kaggle/input/**/val_balanced.txt", recursive=True) or
          glob.glob("/kaggle/input/**/val_group_0.99.txt", recursive=True))
    print("\n" + "="*70); print("PHAN 1: doi chieu thuoc do tren val"); print("="*70)
    if vf:
        val_ids = [pathlib.Path(l.strip()).stem for l in open(vf[0]) if l.strip()]
        # dung y het cach train.py dung, de so sanh cong bang
        yml = f"/kaggle/working/val_check.yaml"
        vdir = "/kaggle/working/valset"
        os.makedirs(f"{vdir}/images", exist_ok=True); os.makedirs(f"{vdir}/labels", exist_ok=True)
        import shutil
        for v in val_ids:
            for sub, ext in (("images", "tif"), ("labels", "txt")):
                src = f"{ROOT}/{sub}/{v}.{ext}"
                if os.path.exists(src): shutil.copy(src, f"{vdir}/{sub}/{v}.{ext}")
        CL = [l.strip() for l in open(f"{ROOT}/classes.txt")]
        open(yml, "w").write(f"path: {vdir}\ntrain: images\nval: images\nchannels: 3\n"
                             f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CL)))
        r = m0.val(data=yml, imgsz=IMGSZ, batch=8, device=0, plots=False, verbose=False)
        print(f"  Ultralytics model.val()      : mAP@[.5:.95] = {float(r.box.map):.4f}  "
              f"mAP@0.5 = {float(r.box.map50):.4f}")

        # cung bo anh, nhung qua duong predict() nhu ens.py da lam
        paths = [f"{vdir}/images/{v}.tif" for v in val_ids if os.path.exists(f"{vdir}/images/{v}.tif")]
        nb, cmin, cmax = 0, 1.0, 0.0
        for i in range(0, len(paths), 8):
            ch = paths[i:i+8]
            for rr in m0.predict([load(p) for p in ch], imgsz=IMGSZ, conf=CONF,
                                 max_det=MAX_DET, device=0, verbose=False, augment=False):
                b = rr.boxes
                if b is not None and len(b):
                    nb += len(b); c = b.conf.cpu().numpy()
                    cmin = min(cmin, float(c.min())); cmax = max(cmax, float(c.max()))
        print(f"  duong predict(): {nb} hop tren {len(paths)} anh = {nb/max(len(paths),1):.1f}/anh"
              f" | conf {cmin:.5f} .. {cmax:.3f}")
        print(f"  -> neu conf nho nhat ~ {CONF} thi nguong DA duoc ap dung;"
              f" neu ~0.25 thi predict() BO QUA tham so conf (do chinh la nguyen nhan lech).")

# ================= PHAN 2: NOP BAI =================
print("\n" + "="*70); print("PHAN 2: suy luan tap test + gop hop"); print("="*70)
te = glob.glob("/kaggle/input/**/hodc_test*/images", recursive=True)
assert te, "khong thay anh test"
TDIR = te[0]
files = sorted(glob.glob(f"{TDIR}/*.tif"))
all_ids = [pathlib.Path(f).stem for f in files]
print(f"anh test: {len(files)} tu {TDIR}")

per_model = []
for wi, w in enumerate(USE):
    t0 = time.time(); mm = YOLO(w); out = {}
    for i in range(0, len(files), 8):
        ch = files[i:i+8]
        for f, r in zip(ch, mm.predict([load(p) for p in ch], imgsz=IMGSZ, conf=CONF,
                                       max_det=MAX_DET, device=0, verbose=False, augment=TTA)):
            b = r.boxes; k = pathlib.Path(f).stem
            out[k] = (np.zeros((0,4),np.float32), np.zeros(0,np.float32), np.zeros(0,int)) \
                if (b is None or len(b) == 0) else \
                (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))
    per_model.append(out)
    n = sum(len(v[1]) for v in out.values())
    print(f"  [{wi+1}/{len(USE)}] {os.path.basename(w)}: {n} hop, {time.time()-t0:.0f}s", flush=True)

rows, n_degen = [], 0
for k in all_ids:
    bx = [p[k][0] for p in per_model if len(p[k][0])]
    if not bx:
        continue
    B, S, L = wbf(np.vstack(bx),
                  np.concatenate([p[k][1] for p in per_model]),
                  np.concatenate([p[k][2] for p in per_model]),
                  iou_thr=WBF_IOU, n_models=len(per_model))
    for (x1,y1,x2,y2), s, c in zip(B, S, L):
        if x2-x1 < 1.0 or y2-y1 < 1.0: n_degen += 1; continue
        rows.append((k, int(c), float(s), float(x1), float(y1), float(x2), float(y2)))

rows.sort(key=lambda r: (r[0], -r[2]))
df = pd.DataFrame(rows, columns=["image_id","class_id","confidence","x1","y1","x2","y2"])
df.insert(0, "id", range(len(df)))
OUT = "/kaggle/working/submission_wbf.csv"
df.to_csv(OUT, index=False)

print(f"\nda ghi {OUT}: {len(df)} dong, {df.image_id.nunique()}/{len(all_ids)} anh")
print(f"hop suy bien da bo: {n_degen} | trung binh {len(df)/max(df.image_id.nunique(),1):.1f} hop/anh")
ok = True
for name, cond in [("moi anh deu co du doan", df.image_id.nunique() == len(all_ids)),
                   ("class_id trong 0..17", df.class_id.between(0,17).all()),
                   ("confidence trong (0,1]", df.confidence.between(0,1).all()),
                   ("x2>x1 va y2>y1", ((df.x2>df.x1)&(df.y2>df.y1)).all()),
                   ("khong co NaN", not df.isna().any().any())]:
    print(f"  [{'OK ' if cond else 'HONG'}] {name}"); ok &= bool(cond)
print("\n=> " + ("SAN SANG NOP" if ok else "CO KIEM TRA THAT BAI"))
