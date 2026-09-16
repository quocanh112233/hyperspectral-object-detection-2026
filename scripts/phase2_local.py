"""HODC2026 Phase 2 - chay TOAN BO tai may, khong can Kaggle GPU.

    python scripts/phase2_local.py --png-dir <thu_muc_png> --weights best.pt \
                                   --out submissions/phase2.csv

Vi sao lam local: bo xep hang ra ngay 23/09 se KHONG co trong ban sao cong khai nao
tren Kaggle, ma tai 4GB len Kaggle o ~200 KB/s la khong kha thi. Do duoc: suy luan
CPU chi 0.41 s/anh (i7-1165G7, 8 luong) -> 600 anh ~ 4 phut. Nen duong ra an toan
nhat la: tai PNG ve -> chuyen doi + suy luan tai cho -> nop CSV.

Khong ghi TIFF trung gian: chuyen doi va suy luan trong MOT luot, dua thang mang
numpy vao predict. Vua nhanh vua khong ton 0.4 GB dia.
"""
import argparse, glob, os, sys, time
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BANDS = [0, 8, 15]          # trai deu pho; PHAI khop voi luc train
GAIN = 114.0 / 44.1         # dua mean anh ve 114 = gia tri nen cua Ultralytics


def X2Cube(img, cellSize=4):
    """Giai kham mosaic 4x4 -> cube (H/4, W/4, 16). Sao y script mau ban to chuc."""
    B = [cellSize, cellSize]; skip = [cellSize, cellSize]
    M, N = img.shape
    col_extent = N - B[1] + 1; row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    return np.transpose(out).reshape(M // cellSize, N // cellSize, cellSize * cellSize)


def to_rgb8(cube, bands):
    """Min-max tung kenh tung anh -> uint8. Giong het luc chuan bi du lieu train."""
    out = cube[:, :, bands].astype(np.float32)
    for c in range(len(bands)):
        ch = out[:, :, c]
        lo, hi = ch.min(), ch.max()
        out[:, :, c] = ((ch - lo) / (hi - lo) * 255) if hi > lo else 0
    return out.astype(np.uint8)


def png_to_input(path):
    cube = X2Cube(np.array(Image.open(path)))
    cube = np.clip(cube.astype(np.float32) * GAIN, 0, 65535).astype(np.uint16)
    return np.ascontiguousarray(to_rgb8(cube, BANDS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.0001)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=0, help="0 = lay tu checkpoint")
    ap.add_argument("--tta", action="store_true", help="lat + da ti le roi gop hop")
    ap.add_argument("--limit", type=int, default=0, help="chi chay N anh dau, de thu")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(os.cpu_count() or 8)
    from ultralytics import YOLO

    pngs = sorted(glob.glob(os.path.join(a.png_dir, "**", "*.png"), recursive=True))
    assert pngs, f"khong thay PNG nao trong {a.png_dir}"
    if a.limit:
        pngs = pngs[:a.limit]
    print(f"anh dau vao: {len(pngs)}")

    model = YOLO(a.weights)
    # Lay imgsz TU CHECKPOINT: train mot do phan giai ma suy luan o do phan giai
    # khac la loi im lang, khong bao gi ca, chi tut diem.
    imgsz = a.imgsz
    if not imgsz:
        ta = (getattr(model, "ckpt", None) or {}).get("train_args") or {}
        imgsz = int(ta.get("imgsz", 768))
    print(f"imgsz = {imgsz} ({'chi dinh tay' if a.imgsz else 'doc tu checkpoint'})"
          f" | conf = {a.conf} | TTA = {a.tta}")

    rows, n_degen, t0 = [], 0, time.time()
    for i in range(0, len(pngs), a.batch):
        chunk = pngs[i:i + a.batch]
        imgs = [png_to_input(p) for p in chunk]
        res = model.predict(imgs, imgsz=imgsz, conf=a.conf, max_det=a.max_det,
                            device="cpu", verbose=False, augment=a.tta)
        for f, r in zip(chunk, res):
            iid = os.path.splitext(os.path.basename(f))[0]
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            for (x1, y1, x2, y2), c, s in zip(b.xyxy.cpu().numpy(),
                                              b.cls.cpu().numpy().astype(int),
                                              b.conf.cpu().numpy()):
                if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                    n_degen += 1
                    continue
                rows.append((iid, int(c), float(s), float(x1), float(y1), float(x2), float(y2)))
        done = min(i + a.batch, len(pngs))
        if (i // a.batch) % 10 == 0 or done == len(pngs):
            el = time.time() - t0
            print(f"  {done}/{len(pngs)}  {el:.0f}s  (con ~{el/done*(len(pngs)-done):.0f}s)",
                  flush=True)

    all_ids = [os.path.splitext(os.path.basename(p))[0] for p in pngs]
    have = {r[0] for r in rows}
    missing = [m for m in all_ids if m not in have]
    if missing:
        # Anh khong co du doan nao mat TRANG diem cua anh do -> quet lai that thap
        # truoc khi ket luan model that su khong thay gi.
        print(f"quet lai {len(missing)} anh chua co du doan voi conf=1e-7...", flush=True)
        id2p = {os.path.splitext(os.path.basename(p))[0]: p for p in pngs}
        for j in range(0, len(missing), a.batch):
            ch = [id2p[m] for m in missing[j:j + a.batch]]
            for f, r in zip(ch, model.predict([png_to_input(p) for p in ch], imgsz=imgsz,
                                              conf=1e-7, max_det=a.max_det, device="cpu",
                                              verbose=False, augment=a.tta)):
                iid = os.path.splitext(os.path.basename(f))[0]
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                for (x1, y1, x2, y2), c, s in zip(b.xyxy.cpu().numpy(),
                                                  b.cls.cpu().numpy().astype(int),
                                                  b.conf.cpu().numpy()):
                    if (x2 - x1) >= 1.0 and (y2 - y1) >= 1.0:
                        rows.append((iid, int(c), float(s),
                                     float(x1), float(y1), float(x2), float(y2)))

    rows.sort(key=lambda r: (r[0], -r[2]))
    df = pd.DataFrame(rows, columns=["image_id", "class_id", "confidence",
                                     "x1", "y1", "x2", "y2"])
    df.insert(0, "id", range(len(df)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"\nda ghi {a.out}: {len(df)} dong, {df.image_id.nunique()}/{len(all_ids)} anh")
    print(f"box suy bien da bo: {n_degen}")
    ok = True
    for name, cond in [
        ("moi anh deu co du doan", df.image_id.nunique() == len(all_ids)),
        ("class_id trong 0..17", df.class_id.between(0, 17).all()),
        ("confidence trong (0,1]", df.confidence.between(0, 1).all()),
        ("x2 > x1 va y2 > y1", ((df.x2 > df.x1) & (df.y2 > df.y1)).all()),
        ("toa do khong am", (df[["x1", "y1"]] >= -0.5).all().all()),
        ("khong co NaN", not df.isna().any().any()),
    ]:
        print(f"  [{'OK ' if cond else 'HONG'}] {name}")
        ok &= bool(cond)
    print("\n=> " + ("SAN SANG NOP" if ok else "CO KIEM TRA THAT BAI - dung lai"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
