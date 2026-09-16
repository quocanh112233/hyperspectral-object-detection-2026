"""HODC2026 - do 3 cach suy luan tren CUNG mot bo trong so, bang THUOC DO DOC LAP.

Ba cau hoi, mot lan chay GPU ~10 phut:
  A. mot model, suy luan thuong      (moc so sanh)
  B. mot model, co TTA               (lat + da ti le roi gop)
  C. gop nhieu diem luu cua CUNG lan train (epoch 20/30/40 + best) bang WBF

Vi sao do ba cai nay: chan doan noi nut that la DO KHIT CUA HOP (mAP@0.5 = 0.94 nhung
mAP@[.5:.95] = 0.67). Ca B lan C deu lam mot viec: lay trung binh nhieu du doan doc lap
de giam nhieu toa do. Neu ca hai deu khong an thi gia thuyet "trung binh hoa giup ich"
sai, va khong nen tieu quota vao ensemble that su.

Dung pycocotools de cham diem, KHONG dung so cua Ultralytics — de doi chieu doc lap.
"""
import os, sys, glob, json, subprocess, pathlib, time
import numpy as np

def pip(*a): return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a]).returncode
pip("ultralytics==8.4.152", "tifffile", "pycocotools")

import tifffile, torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---------- gop hop co trong so (ban sao cua src/utils/fusion.py) ----------
def _iou_m(box, boxes):
    if len(boxes) == 0: return np.zeros(0, np.float32)
    xx1 = np.maximum(box[0], boxes[:,0]); yy1 = np.maximum(box[1], boxes[:,1])
    xx2 = np.minimum(box[2], boxes[:,2]); yy2 = np.minimum(box[3], boxes[:,3])
    inter = np.clip(xx2-xx1,0,None) * np.clip(yy2-yy1,0,None)
    a1 = (box[2]-box[0])*(box[3]-box[1])
    a2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return inter / np.maximum(a1+a2-inter, 1e-9)

def wbf(boxes, scores, labels, iou_thr=0.55, n_models=1):
    boxes = np.asarray(boxes, np.float32).reshape(-1,4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    ob, os_, ol = [], [], []
    for lab in np.unique(labels):
        m = labels == lab
        b, s = boxes[m], scores[m]
        o = np.argsort(-s); b, s = b[o], s[o]
        clusters, cb = [], np.zeros((0,4), np.float32)
        for i in range(len(b)):
            if len(clusters):
                iou = _iou_m(b[i], cb); j = int(np.argmax(iou))
                if iou[j] >= iou_thr:
                    clusters[j].append(i); idx = clusters[j]
                    w = s[idx][:,None]
                    cb[j] = (b[idx]*w).sum(0)/max(w.sum(),1e-9)
                    continue
            clusters.append([i]); cb = np.vstack([cb, b[i][None]])
        for j, idx in enumerate(clusters):
            sc = s[idx]
            ob.append(cb[j]); ol.append(lab)
            os_.append(float(sc.mean()) * min(len(idx), n_models)/float(n_models))
    if not ob: return np.zeros((0,4),np.float32), np.zeros(0,np.float32), np.zeros(0,int)
    o = np.argsort(-np.asarray(os_))
    return np.asarray(ob,np.float32)[o], np.asarray(os_,np.float32)[o], np.asarray(ol)[o]

# ---------- tim du lieu ----------
hits = glob.glob("/kaggle/input/**/hodc_train*/images", recursive=True)
assert len(hits) == 1, f"gan nham bo du lieu: {hits}"
ROOT = str(pathlib.Path(hits[0]).parent)
IMG, LBL = f"{ROOT}/images", f"{ROOT}/labels"

vf = (glob.glob("/kaggle/input/**/val_balanced.txt", recursive=True) or
      glob.glob("/kaggle/input/**/val_group_0.99.txt", recursive=True))
assert vf, "khong thay danh sach val"
val_ids = [l.strip() for l in open(vf[0]) if l.strip()]
val_ids = [pathlib.Path(v).stem for v in val_ids]
print(f"anh val: {len(val_ids)} (tu {os.path.basename(vf[0])})")

ws = sorted(glob.glob("/kaggle/input/**/weights/*.pt", recursive=True))
print("diem luu tim thay:", [os.path.basename(w) for w in ws])
best = [w for w in ws if w.endswith("best.pt")]
assert best, "khong thay best.pt"
epochs = sorted([w for w in ws if "epoch" in os.path.basename(w)])
print(f"best = {best[0]}")

CLASSES = [l.strip() for l in open(f"{ROOT}/classes.txt")] if os.path.exists(f"{ROOT}/classes.txt") \
          else [str(i) for i in range(18)]

# ---------- dung COCO ground truth ----------
images, anns, aid = [], [], 1
for k, vid in enumerate(val_ids):
    p = f"{IMG}/{vid}.tif"
    if not os.path.exists(p): continue
    a = tifffile.imread(p)
    if a.ndim == 3 and a.shape[0] in (3, 16): a = np.transpose(a, (1,2,0))
    H, W = a.shape[:2]
    images.append({"id": k, "file_name": f"{vid}.tif", "width": W, "height": H})
    for line in open(f"{LBL}/{vid}.txt"):
        if not line.strip(): continue
        c, cx, cy, w_, h_ = line.split()
        c = int(c); cx, cy, w_, h_ = float(cx)*W, float(cy)*H, float(w_)*W, float(h_)*H
        anns.append({"id": aid, "image_id": k, "category_id": c,
                     "bbox": [cx-w_/2, cy-h_/2, w_, h_], "area": w_*h_, "iscrowd": 0})
        aid += 1
gt = {"images": images, "annotations": anns,
      "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}
json.dump(gt, open("/kaggle/working/gt.json", "w"))
print(f"COCO GT: {len(images)} anh, {len(anns)} vat the")
coco_gt = COCO("/kaggle/working/gt.json")

id2k = {im["file_name"][:-4]: im["id"] for im in images}
paths = [f"{IMG}/{im['file_name'][:-4]}.tif" for im in images]

def load(p):
    a = tifffile.imread(p)
    if a.ndim == 3 and a.shape[0] in (3, 16): a = np.transpose(a, (1,2,0))
    return np.ascontiguousarray(a)

def predict(model, imgsz, tta, conf=0.001, bs=8):
    """-> dict: ten anh -> (boxes xyxy, scores, labels)"""
    out = {}
    for i in range(0, len(paths), bs):
        ch = paths[i:i+bs]
        res = model.predict([load(p) for p in ch], imgsz=imgsz, conf=conf, max_det=300,
                            device=0, verbose=False, augment=tta)
        for p, r in zip(ch, res):
            b = r.boxes
            k = os.path.basename(p)[:-4]
            if b is None or len(b) == 0:
                out[k] = (np.zeros((0,4),np.float32), np.zeros(0,np.float32), np.zeros(0,int))
            else:
                out[k] = (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                          b.cls.cpu().numpy().astype(int))
    return out

def score(preds, tag):
    dets = []
    for k, (bx, sc, lb) in preds.items():
        for (x1,y1,x2,y2), s, c in zip(bx, sc, lb):
            if x2-x1 < 1 or y2-y1 < 1: continue
            dets.append({"image_id": id2k[k], "category_id": int(c), "score": float(s),
                         "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)]})
    if not dets:
        print(f"{tag}: KHONG CO DU DOAN"); return None
    json.dump(dets, open("/kaggle/working/dt.json", "w"))
    e = COCOeval(coco_gt, coco_gt.loadRes("/kaggle/working/dt.json"), "bbox")
    e.evaluate(); e.accumulate()
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        e.summarize()
    print(f"{tag:38s} mAP@[.5:.95] = {e.stats[0]:.4f}   mAP@0.5 = {e.stats[1]:.4f}   "
          f"({len(dets)} du doan)")
    return e.stats[0]

# ---------- chay ----------
m = YOLO(best[0])
IMGSZ = int((getattr(m, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 768))
print(f"\nimgsz = {IMGSZ} (doc tu checkpoint)\n" + "="*72)

t0 = time.time(); pA = predict(m, IMGSZ, tta=False)
sA = score(pA, "A. mot model, thuong"); print(f"   ({time.time()-t0:.0f}s)")

t0 = time.time(); pB = predict(m, IMGSZ, tta=True)
sB = score(pB, "B. mot model + TTA"); print(f"   ({time.time()-t0:.0f}s)")

# C: gop cac diem luu cua chinh lan train nay
snaps = ([best[0]] + epochs[-3:])[:4]
print(f"\nC dung {len(snaps)} diem luu: {[os.path.basename(x) for x in snaps]}")
if len(snaps) >= 2:
    allp = [pA] + [predict(YOLO(w), IMGSZ, tta=False) for w in snaps[1:]]
    fused = {}
    for k in pA:
        bx = np.vstack([p[k][0] for p in allp]) if any(len(p[k][0]) for p in allp) else np.zeros((0,4),np.float32)
        sc = np.concatenate([p[k][1] for p in allp])
        lb = np.concatenate([p[k][2] for p in allp])
        fused[k] = wbf(bx, sc, lb, iou_thr=0.55, n_models=len(allp))
    sC = score(fused, f"C. gop {len(allp)} diem luu (WBF)")
else:
    sC = None; print("   khong du diem luu de gop")

# ---------- D. SAI SO LA THIEN LECH HAY NHIEU? ----------
# TTA hau nhu khong doi gi (0.6757 vs 0.6754) -> cac goc nhin khac nhau cho hop gan
# y het. Neu vay thi 1 px sai lech co the la THIEN LECH HE THONG chu khong phai nhieu.
# Trung binh hoa chi giam NHIEU. Thien lech thi phai HIEU CHINH - va do la viec re hon
# nhieu. Do truc tiep: ghep du doan voi nhan that, xem sai so canh co lech ve mot phia.
print("\n" + "="*72)
print("D. Sai so canh: thien lech he thong hay nhieu ngau nhien?")
print("="*72)
gt_by_img = {}
for an in anns:
    gt_by_img.setdefault(an["image_id"], []).append(an)

de, dn, dw, dh, dcx, dcy = [], [], [], [], [], []
for k, (bx, sc, lb) in pA.items():
    iid = id2k[k]
    g = gt_by_img.get(iid, [])
    if not g or len(bx) == 0:
        continue
    keep = sc >= 0.25
    bx2, lb2 = bx[keep], lb[keep]
    for an in g:
        gx, gy, gw, gh = an["bbox"]
        gbox = np.array([gx, gy, gx+gw, gy+gh], np.float32)
        m = lb2 == an["category_id"]
        if not m.any():
            continue
        cand = bx2[m]
        iou = _iou_m(gbox, cand)
        j = int(np.argmax(iou))
        if iou[j] < 0.5:
            continue
        px1, py1, px2, py2 = cand[j]
        de += [px1-gbox[0], py1-gbox[1]]      # canh trai/tren: am = du doan ra ngoai
        dn += [px2-gbox[2], py2-gbox[3]]      # canh phai/duoi: duong = du doan ra ngoai
        dw.append((px2-px1) - gw); dh.append((py2-py1) - gh)
        dcx.append(((px1+px2)-(gbox[0]+gbox[2]))/2); dcy.append(((py1+py2)-(gbox[1]+gbox[3]))/2)

if dw:
    dw = np.array(dw); dh = np.array(dh)
    dcx = np.array(dcx); dcy = np.array(dcy)
    de = np.array(de); dn = np.array(dn)
    print(f"  so cap ghep duoc (IoU>=0.5, conf>=0.25): {len(dw)}")
    print(f"  chieu rong : lech trung binh {dw.mean():+.3f} px, do lech chuan {dw.std():.3f} px")
    print(f"  chieu cao  : lech trung binh {dh.mean():+.3f} px, do lech chuan {dh.std():.3f} px")
    print(f"  tam ngang  : lech trung binh {dcx.mean():+.3f} px, do lech chuan {dcx.std():.3f} px")
    print(f"  tam doc    : lech trung binh {dcy.mean():+.3f} px, do lech chuan {dcy.std():.3f} px")
    r = abs(dw.mean())/max(dw.std(), 1e-9)
    print(f"\n  ti le |thien lech| / nhieu (chieu rong) = {r:.3f}")
    print("  -> " + ("THIEN LECH chiem uu the: hieu chinh se an, trung binh hoa thi khong"
                     if r > 0.3 else
                     "NHIEU chiem uu the: trung binh hoa (ensemble/WBF) moi la duong dung"))

    # Thu hieu chinh: phong to/thu nho moi hop quanh tam de trieu tieu thien lech
    print("\n  Thu hieu chinh hop (bu thien lech kich thuoc):")
    best_s, best_m = None, sA
    for scale in (0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06):
        adj = {}
        for k, (bx, sc, lb) in pA.items():
            if len(bx) == 0:
                adj[k] = (bx, sc, lb); continue
            cx = (bx[:,0]+bx[:,2])/2; cy = (bx[:,1]+bx[:,3])/2
            w = (bx[:,2]-bx[:,0])*scale; h = (bx[:,3]-bx[:,1])*scale
            adj[k] = (np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1), sc, lb)
        m_ = score(adj, f"   he so {scale:.2f}")
        if m_ is not None and (best_m is None or m_ > best_m):
            best_m, best_s = m_, scale
    print(f"\n  he so tot nhat: {best_s} -> {best_m:.4f} (moc {sA:.4f}, "
          f"{'+' if best_m>sA else ''}{best_m-sA:.4f})")
else:
    print("  khong ghep duoc cap nao")

print("\n" + "="*72)
print("=== KET LUAN ===")
if sA:
    for tag, s in [("TTA", sB), ("gop diem luu", sC)]:
        if s is not None:
            d = s - sA
            print(f"{tag:16s}: {d:+.4f} so voi moc  -> {'CO AN' if d > 0.002 else 'khong an'}")
    print("\nNeu ca hai deu khong an: gia thuyet 'trung binh hoa giup hop khit hon' SAI,")
    print("dung tieu quota vao ensemble that su.")
json.dump({"A_plain": sA, "B_tta": sB, "C_snapshot_wbf": sC},
          open("/kaggle/working/ens_metrics.json", "w"), indent=2)
