"""HODC2026 - kiem tra trung lap canh giua train/val.

Gia thuyet can kiem chung: val = 0.730 nhung LB = 0.604. Neu du lieu la cac khung
hinh lien tiep cua cung mot canh thi anh gan giong nhau roi vao ca train lan val,
lam val de hon that -> moi so sanh thi nghiem deu lech.

Xuat ra /kaggle/working: splits/*.txt (cach chia theo nhom canh, khong ro ri)
"""
import os, sys, glob, random, subprocess, importlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tifffile"], check=True)
importlib.invalidate_caches()
import numpy as np
import cv2
import tifffile

hits = glob.glob("/kaggle/input/**/hodc_train/images", recursive=True)
assert hits, "khong thay hodc_train/images"
ROOT = os.path.dirname(hits[0])
IMG, LBL = f"{ROOT}/images", f"{ROOT}/labels"
CLASSES = [l.strip() for l in open(f"{ROOT}/classes.txt") if l.strip()]
NC = len(CLASSES)

files = sorted(glob.glob(f"{IMG}/*.tif"))
ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
print(f"{len(files)} anh, {NC} lop\n", flush=True)

# ---------- 1. Chu ky: luoi 8x16 khong gian x 16 bang ----------
GH, GW = 8, 16
sig = np.zeros((len(files), GH * GW * 16), np.float32)
for i, f in enumerate(files):
    a = tifffile.imread(f)                       # (16, H, W)
    if a.ndim == 3 and a.shape[0] == 16:
        a = np.transpose(a, (1, 2, 0))
    small = np.stack([cv2.resize(a[:, :, b].astype(np.float32), (GW, GH),
                                 interpolation=cv2.INTER_AREA) for b in range(16)], axis=2)
    v = small.reshape(-1)
    sig[i] = v / (np.linalg.norm(v) + 1e-9)      # chuan hoa -> do sang khong chi phoi
    if (i + 1) % 500 == 0:
        print(f"  chu ky {i+1}/{len(files)}", flush=True)

# ---------- 2. Ma tran tuong dong ----------
S = sig @ sig.T
np.fill_diagonal(S, -1.0)
print("\n=== Phan bo do tuong dong cua cap GIONG NHAU NHAT voi moi anh ===")
best = S.max(1)
for q in [50, 75, 90, 95, 99]:
    print(f"  phan vi {q:2d}%: {np.percentile(best, q):.4f}")
print(f"  lon nhat  : {best.max():.4f}")
print()
for t in [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]:
    n_img = int((best >= t).sum())
    n_pair = int((S >= t).sum() // 2)
    print(f"  nguong {t:.3f}: {n_img:4d} anh co it nhat 1 anh giong ({100*n_img/len(files):5.1f}%), {n_pair:6d} cap")

# ---------- 3. Cac cap giong nhat: ID co lien tiep khong? ----------
print("\n=== 10 cap giong nhau nhat ===")
iu = np.triu_indices(len(files), k=1)
vals = S[iu]
top = np.argsort(vals)[-10:][::-1]
for k in top:
    i, j = iu[0][k], iu[1][k]
    print(f"  {ids[i]:>6s} <-> {ids[j]:>6s}   cos={vals[k]:.5f}   |hieu ID|={abs(int(ids[i])-int(ids[j]))}")

# ---------- 4. Cach chia HIEN TAI ro ri bao nhieu? ----------
SEED = 0
rng = random.Random(SEED); sh = ids[:]; rng.shuffle(sh)
n_val = max(1, int(round(len(ids) * 0.10)))
val_now = set(sh[:n_val])
vi = np.array([k for k, x in enumerate(ids) if x in val_now])
ti = np.array([k for k, x in enumerate(ids) if x not in val_now])
cross = S[np.ix_(vi, ti)]
print(f"\n=== RO RI cua cach chia hien tai (seed 0, {len(vi)} anh val) ===")
for t in [0.95, 0.98, 0.99, 0.995]:
    n = int((cross.max(1) >= t).sum())
    print(f"  nguong {t:.3f}: {n:3d}/{len(vi)} anh val ({100*n/len(vi):5.1f}%) co anh RAT GIONG trong train")

# ---------- 5. Chia lai theo nhom canh ----------
def components(thresh):
    parent = list(range(len(files)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    ii, jj = np.where(np.triu(S >= thresh, k=1))
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    groups = {}
    for k in range(len(files)):
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())

# tinh truoc 1 lan, tranh doc lai file trong vong lap
CLS = np.zeros((len(files), NC), int)
for k, _id in enumerate(ids):
    f = f"{LBL}/{_id}.txt"
    if os.path.exists(f):
        for ln in open(f):
            if ln.strip(): CLS[k, int(ln.split()[0])] += 1
def cls_of(idx):
    return CLS[idx]

os.makedirs("/kaggle/working/splits", exist_ok=True)
for T in [0.98, 0.99]:
    comps = components(T)
    sizes = sorted((len(c) for c in comps), reverse=True)
    print(f"\n=== Nhom canh o nguong {T} ===")
    print(f"  {len(comps)} nhom / {len(files)} anh | nhom lon nhat: {sizes[:8]}")
    print(f"  so nhom chi co 1 anh: {sum(1 for s in sizes if s == 1)}")

    # gan nhom vao val theo thu tu uu tien lop hiem, den khi du ~10%
    print(f"  phan bo kich thuoc nhom: >100 anh: {sum(1 for x in sizes if x>100)}, "
          f"11-100: {sum(1 for x in sizes if 11<=x<=100)}, 2-10: {sum(1 for x in sizes if 2<=x<=10)}")
    rnd = random.Random(SEED); rnd.shuffle(comps)
    # uu tien nhom chua nhieu lop khac nhau -> de phu du 18 lop trong val
    comps.sort(key=lambda c: int((CLS[c].sum(0) > 0).sum()), reverse=True)
    target = int(round(len(files) * 0.10))
    val_idx, val_c = [], np.zeros(NC, int)
    for c in comps:
        if len(val_idx) >= target and (val_c > 0).all():
            break
        need_rare = not (val_c > 0).all() and any(cls_of(k)[val_c == 0].sum() for k in c)
        if len(val_idx) < target or need_rare:
            val_idx += c
            for k in c: val_c += cls_of(k)
    vset0 = set(val_idx)
    print(f"  val: {len(val_idx)} anh ({100*len(val_idx)/len(files):.1f}%)")
    print(f"  lop vang mat trong val: {[CLASSES[i] for i in range(NC) if val_c[i]==0] or 'khong co'}")
    vset = set(val_idx)
    print(f"  bbox trong val: {int(CLS[sorted(vset)].sum())} / tong {int(CLS.sum())}")
    cross2 = S[np.ix_(np.array(sorted(vset)), np.array([k for k in range(len(files)) if k not in vset]))]
    print(f"  ro ri con lai o nguong {T}: {int((cross2.max(1) >= T).sum())} anh (phai bang 0)")
    with open(f"/kaggle/working/splits/val_group_{T}.txt", "w") as f:
        f.write("\n".join(ids[k] for k in sorted(vset)))
print("\nda ghi /kaggle/working/splits/")
