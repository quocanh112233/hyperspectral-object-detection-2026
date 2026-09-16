"""HODC2026 - chia val vua THEO NHOM CANH vua CAN BANG LOP.

Bo chia truoc chi lo nhom canh -> het ro ri nhung val co lop 1 mau, lop khac 41%.
Lan nay toi uu truc tiep: chon cac nhom sao cho moi lop co ~10% so bbox nam trong val.
"""
import os, sys, glob, random, subprocess, importlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tifffile"], check=True)
importlib.invalidate_caches()
import numpy as np, cv2, tifffile

TARGET = 0.10
THRESH = 0.99

hits = glob.glob("/kaggle/input/**/hodc_train/images", recursive=True)
ROOT = os.path.dirname(hits[0]); IMG, LBL = f"{ROOT}/images", f"{ROOT}/labels"
CLASSES = [l.strip() for l in open(f"{ROOT}/classes.txt") if l.strip()]; NC = len(CLASSES)
files = sorted(glob.glob(f"{IMG}/*.tif"))
ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
N = len(files)

CLS = np.zeros((N, NC), int)
for k, _id in enumerate(ids):
    f = f"{LBL}/{_id}.txt"
    if os.path.exists(f):
        for ln in open(f):
            if ln.strip(): CLS[k, int(ln.split()[0])] += 1
TOT = CLS.sum(0)
print(f"{N} anh, {int(TOT.sum())} bbox\n", flush=True)

# --- chu ky + nhom canh ---
GH, GW = 8, 16
sig = np.zeros((N, GH*GW*16), np.float32)
for i, f in enumerate(files):
    a = tifffile.imread(f)
    if a.ndim == 3 and a.shape[0] == 16: a = np.transpose(a, (1,2,0))
    small = np.stack([cv2.resize(a[:,:,b].astype(np.float32), (GW,GH), interpolation=cv2.INTER_AREA)
                      for b in range(16)], axis=2)
    v = small.reshape(-1); sig[i] = v / (np.linalg.norm(v)+1e-9)
    if (i+1) % 750 == 0: print(f"  chu ky {i+1}/{N}", flush=True)
S = sig @ sig.T; np.fill_diagonal(S, -1.0)

parent = list(range(N))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
ii, jj = np.where(np.triu(S >= THRESH, k=1))
for a, b in zip(ii, jj):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
groups = {}
for k in range(N): groups.setdefault(find(k), []).append(k)
comps = list(groups.values())
CC = np.stack([CLS[c].sum(0) for c in comps])      # (n_comp, NC) bbox moi nhom
CSZ = np.array([len(c) for c in comps])
print(f"{len(comps)} nhom canh (nguong {THRESH}), nhom lon nhat {CSZ.max()} anh\n", flush=True)

# --- chon nhom tham lam: moi vong lay nhom giam do lech nhieu nhat ---
rng = np.random.default_rng(0)
cur = np.zeros(NC, float)
chosen = np.zeros(len(comps), bool)
target_bbox = TOT * TARGET

def dev(v):   # do lech tuong doi, chuan hoa theo so bbox cua tung lop
    return np.sum(((v - target_bbox) / np.maximum(TOT, 1)) ** 2)

n_img = 0
while n_img < N * TARGET * 1.15:
    cand = np.where(~chosen)[0]
    if len(cand) == 0: break
    scores = np.array([dev(cur + CC[c]) for c in cand])
    best = cand[scores.argmin()]
    if scores.min() >= dev(cur) and (cur > 0).all():
        break                                   # khong cai thien duoc nua
    chosen[best] = True; cur += CC[best]; n_img += CSZ[best]

val_idx = sorted(k for c_i in np.where(chosen)[0] for k in comps[c_i])
vset = set(val_idx)

print(f"=== Bo chia moi: {len(val_idx)} anh val ({100*len(val_idx)/N:.1f}%) ===")
print(f"{'lop':18s} {'tong':>6s} {'val':>6s} {'% val':>7s}")
ok = True
for i, c in enumerate(CLASSES):
    pct = 100*cur[i]/max(TOT[i],1)
    flag = "" if 4 <= pct <= 22 else "  <-- LECH"
    if flag: ok = False
    print(f"{c:18s} {TOT[i]:6d} {int(cur[i]):6d} {pct:6.1f}%{flag}")
print(f"\ntong bbox val: {int(cur.sum())}/{int(TOT.sum())} = {100*cur.sum()/TOT.sum():.1f}%")
print("moi lop deu trong khoang 4-22%:", "CO" if ok else "KHONG")

tr = np.array([k for k in range(N) if k not in vset])
va = np.array(val_idx)
leak = int((S[np.ix_(va, tr)].max(1) >= THRESH).sum())
print(f"ro ri o nguong {THRESH}: {leak} anh (phai = 0)")
for t in [0.95, 0.98]:
    print(f"  (tham khao) ro ri o nguong {t}: {int((S[np.ix_(va,tr)].max(1) >= t).sum())} anh")

os.makedirs("/kaggle/working/splits", exist_ok=True)
with open("/kaggle/working/splits/val_balanced.txt", "w") as f:
    f.write("\n".join(ids[k] for k in val_idx))
print("\nda ghi /kaggle/working/splits/val_balanced.txt")
