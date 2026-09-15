"""HODC2026 - cau hoi quyet dinh: pho co tach duoc cap that/gia khong, va bao nhieu?

So sanh truc tiep: dung 16 bang vs chi dung 3 bang (pseudo-RGB) de dinh luong
cai gia phai tra neu bo thong tin pho.
"""
import glob, os
import numpy as np
import pandas as pd

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"khong thay {name}"
    return hits[0]

box = pd.read_csv(find("box_spectra.csv"))
meta = pd.read_csv(find("meta.csv"))
bands = pd.read_csv(find("band_stats.csv"))
B = [f"b{i:02d}" for i in range(16)]
print(f"box_spectra: {len(box)} bbox | meta: {len(meta)} anh\n")

# ---------- 1. Anh co kich thuoc XML lech cube ----------
bad = meta[meta.size_khop == 0]
print("=== Anh co kich thuoc XML != cube ===")
print(bad[["image_id", "cube_w", "cube_h", "xml_w", "xml_h"]].to_string(index=False))
if len(bad):
    print("do lech:", (bad.cube_w - bad.xml_w).tolist(), (bad.cube_h - bad.xml_h).tolist())
print()

# ---------- 2. Dai gia tri thuc te ----------
print("=== Dai gia tri (anh huong toi chuan hoa) ===")
print("PNG la 16-bit nhung gia tri thuc te chi den:", int(bands["max"].max()))
print("mean toan cuc %.1f | std %.1f" % (bands["mean"].mean(), bands["std"].mean()))
print("-> Ultralytics chia cho 255: dau vao se co mean ~%.3f, std ~%.3f"
      % (bands["mean"].mean()/255, bands["std"].mean()/255))
print("   (anh tu nhien COCO: mean ~0.45, std ~0.25)")
print()

# ---------- 3. Hinh dang pho cua tung lop ----------
X = box[B].values.astype(np.float64)
keep = X.sum(1) > 0
box, X = box[keep].reset_index(drop=True), X[keep]
Xn = X / X.sum(1, keepdims=True)          # chuan hoa do sang -> chi con HINH DANG pho
print("=== Hinh dang pho trung binh (da chuan hoa tong = 1, nhan 100) ===")
shape = pd.DataFrame(Xn, columns=B).assign(name=box.name).groupby("name").mean() * 100
pairs = [("apple","apple_plastic"), ("banana","banana_plastic"),
         ("orange","orange_plastic"), ("egg","egg_plastic"), ("egg","egg_wood")]
for a, b in pairs:
    if a in shape.index and b in shape.index:
        d = (shape.loc[a] - shape.loc[b]).abs()
        print(f"{a:14s} vs {b:16s}  lech lon nhat o bang {d.idxmax()} ({d.max():.2f}), "
              f"tong lech {d.sum():.2f}")
print()

# ---------- 4. Dinh luong: 16 bang vs 3 bang ----------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("=== Do chinh xac phan biet that/gia CHI tu pho trung binh cua bbox ===")
print("(chia fold theo anh de khong ro ri; 0.5 = doan mo)\n")
print(f"{'cap':34s} {'n':>5s}  {'16 bang':>8s}  {'3 bang':>8s}  {'chenh':>7s}")
for a, b in pairs:
    sub = box[box.name.isin([a, b])]
    if len(sub) < 40: continue
    y = (sub.name == b).astype(int).values
    g = sub.image_id.values
    Xs = sub[B].values.astype(np.float64)
    Xs = Xs / np.maximum(Xs.sum(1, keepdims=True), 1e-9)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    s16 = cross_val_score(mk(), Xs, y, groups=g, cv=cv).mean()
    s3  = cross_val_score(mk(), Xs[:, :3], y, groups=g, cv=cv).mean()
    print(f"{a+' vs '+b:34s} {len(sub):5d}  {s16:8.3f}  {s3:8.3f}  {s16-s3:+7.3f}")

# ---------- 5. Toan bo 18 lop ----------
print("\n=== Phan loai ca 18 lop chi tu pho trung binh ===")
y = box.name.values; g = box.image_id.values
Xall = X / np.maximum(X.sum(1, keepdims=True), 1e-9)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
print("16 bang: %.3f" % cross_val_score(mk(), Xall, y, groups=g, cv=cv).mean())
print(" 3 bang: %.3f" % cross_val_score(mk(), Xall[:, :3], y, groups=g, cv=cv).mean())
print("doan lop pho bien nhat: %.3f" % (pd.Series(y).value_counts().iloc[0] / len(y)))
