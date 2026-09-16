"""HODC2026 - probe v3: XAC NHAN tai duoc du lieu cuoc thi tu ben trong Kaggle.

v2 da chung minh `competition_list_files` CHAY DUOC tu ben trong Kaggle (khong 429) —
no chi vo o cho toi goi len() tren object phan hoi. Gio tai that de chot.

Neu duong nay thong thi Phase 2 an toan: 23/09 chi can mot kernel CPU tai bo xep hang
ngay tren Kaggle, roi chay suy luan, roi tai ve MOT file CSV vai tram KB.
"""
import os, glob, time, traceback
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "hyperspectral-object-detection-challenge-2026"
api = KaggleApi(); api.authenticate()
print("xac thuc: OK")

print("\n=== 1. Liet ke file (xu ly phan hoi cho dung) ===")
resp = api.competition_list_files(COMP)
files = getattr(resp, "files", None) or list(resp)
print(f"  tong so file: {len(files)}")
names = [getattr(f, "name", str(f)) for f in files[:5]]
print(f"  5 cai dau: {names}")
tot = sum(int(getattr(f, "total_bytes", 0) or getattr(f, "totalBytes", 0) or 0) for f in files)
print(f"  tong dung luong trang nay: {tot/1e6:.0f} MB")

print("\n=== 2. Tai mot file nho ===")
try:
    api.competition_download_file(COMP, "class.txt", path="/kaggle/working/comp", force=True)
    g = glob.glob("/kaggle/working/comp/*")
    print("  OK ->", g, "|", open(g[0]).read()[:90].replace("\n", " ") if g else "")
except Exception:
    traceback.print_exc()

print("\n=== 3. Tai mot anh PNG (do toc do thuc) ===")
try:
    t0 = time.time()
    api.competition_download_file(COMP, "data_test/data_test/VIS/1009.png",
                                  path="/kaggle/working/comp", force=True)
    dt = time.time() - t0
    pngs = glob.glob("/kaggle/working/comp/**/*.png", recursive=True) + \
           glob.glob("/kaggle/working/comp/*.png")
    sz = sum(os.path.getsize(p) for p in pngs) if pngs else 0
    print(f"  OK -> {pngs} ({sz/1e6:.1f} MB trong {dt:.1f}s = {sz/1e6/max(dt,0.01):.1f} MB/s)")
    if sz:
        print(f"  uoc tinh 600 anh x 4 MB: {2400/(sz/1e6/max(dt,0.01))/60:.0f} phut")
except Exception:
    traceback.print_exc()

print("\n=== 4. Tai ca thu muc data_test (mot luot goi) - chi do 60s dau ===")
# Khong tai het o day, chi xem no co khoi dong duoc khong.
try:
    import threading
    done = []
    def _dl():
        try:
            api.competition_download_files(COMP, path="/kaggle/working/bulk", quiet=True)
            done.append("xong")
        except Exception as e:
            done.append(f"loi: {type(e).__name__}: {str(e)[:150]}")
    th = threading.Thread(target=_dl, daemon=True); th.start()
    for _ in range(12):
        time.sleep(5)
        cur = sum(os.path.getsize(f) for f in glob.glob("/kaggle/working/bulk/**/*", recursive=True)
                  if os.path.isfile(f))
        print(f"  {_*5+5}s: {cur/1e6:.0f} MB", flush=True)
        if done: break
    print("  ket qua:", done or "van dang chay (tuc la KHONG bi chan)")
except Exception:
    traceback.print_exc()

print("\n=== KET LUAN ===")
print("Neu 2 va 3 deu OK -> Phase 2 AN TOAN: 23/09 tai bo xep hang ngay tren Kaggle.")
