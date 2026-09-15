# Nhat ky thi nghiem — HODC2026

| # | Ngay | Model | Kenh | imgsz | Epoch | mAP@0.5 | mAP@[.5:.95] | Thoi gian | LB | Ghi chu |
|---|------|-------|------|-------|-------|---------|--------------|-----------|----|---------|
| 1 | 15/09 | YOLO11s | 16 | 768 | 40 | 0.979 | **0.730** | 1.94h / 1×T4 | **0.604** | Baseline. COCO pretrain (492 tensor + thoi phong conv dau). Da hoi tu (4 epoch cuoi deu 0.730). |

## Lan 1 — chi tiet

- Tham so: **9,438,502**
- FLOPs: **32.4 G @ 768x768** (kich thuoc THUC TE khi train/inference)
  - 22.5 G @ 640x640 — day la so Ultralytics tu in ra, vi `get_flops()` mac dinh
    imgsz=640. **Khong dung so nay** trong bao cao: no khong khop cau hinh cua ta.
  - De doi chieu: YOLO11s goc (3 kenh, 80 lop) = 9,458,752 tham so, 21.8 G @640.
    Doi conv dau 3 -> 16 kenh chi them 3,744 tham so (864 -> 4,608).
- Chia: 2697 train / 300 val (ngau nhien, seed 0)
- Augmentation: mosaic 1.0, fliplr 0.5, scale 0.5, translate 0.1.
  **hsv_h/s/v = 0** (augmentation mau pha huy chu ky pho that/gia)
- Du lieu: TIFF 16 trang, GAIN 2.585 (mean anh -> 114 = gia tri nen cua Ultralytics)
- Da va loi mosaic dtype uint8 cua Ultralytics (2/2 cho)

### mAP@[.5:.95] theo lop

| Lop | mAP | | Lop | mAP |
|---|---|---|---|---|
| egg_plastic | 0.844 | | banana | 0.755 |
| egg | 0.816 | | charger_head | 0.749 |
| egg_wood | 0.808 | | car_toy | 0.721 |
| rubik | 0.800 | | table_tennis | 0.715 |
| orange_plastic | 0.786 | | stone_block | 0.731 |
| apple | 0.783 | | car | 0.567 |
| apple_plastic | 0.780 | | e-bike | 0.520 |
| orange | 0.777 | | people | **0.466** |
| banana_plastic | 0.773 | | | |
| badminton | 0.759 | | | |

### Nhan xet

1. **Cac cap that/gia deu ~0.78-0.84, khong lop nao sup do.** Chien luoc 16 bang
   hoat dong dung nhu phan tich pho du doan.
2. **mAP@0.5 = 0.979 nhung mAP@[.5:.95] = 0.730.** Khoang cach nay hoan toan la
   do DO CHINH XAC DINH VI, khong phai do phan loai. Model tim dung vat, dan sai
   vai pixel. Day la don bay lon nhat con lai -> tang do phan giai, them dau P2.
3. **3 lop yeu nhat deu la vat THE LON**: people 0.466, e-bike 0.520, car 0.567.
   Nguoc voi truc giac. Gia thuyet: bien cua chung mo ho (nguoi bi che khuat, xe
   co bong do) va so mau val qua it (car 19 anh, e-bike 17 anh).
4. **stone_block chi co 1 anh trong val** -> con so 0.731 cua no khong dang tin.
   Can chia val co phan tang theo lop.


---

## CANH BAO: tap val dang cho diem lac quan

| | mAP@[.5:.95] |
|---|---|
| val cua ta (300 anh) | 0.730 |
| public LB (50% test) | **0.604** |
| chenh lech | **-0.126** |

Khoang cach 0.126 la lon. Hai kha nang:

1. **Chia val bi ro ri.** Neu du lieu la cac khung hinh lien tiep cua cung mot canh,
   anh gan giong nhau roi vao ca train lan val -> val de hon that.
2. Phan phoi test khac train.

He qua thuc te: **khong duoc dung val de so sanh cac thi nghiem** cho den khi lam ro.
Neu val lac quan khong deu giua cac cau hinh, ta se chon nham huong toi uu.

Viec can lam truoc khi chay loat thi nghiem: kiem tra trung lap canh giua train/val,
va neu co thi chia lai theo nhom canh (group split) thay vi ngau nhien.

## Da va loi thu vien kaggle (scripts/submit.py)

`KaggleApi.upload_complete` boc file trong `TqdmBufferedReader` -> `requests` khong
biet do dai -> dung `Transfer-Encoding: chunked` -> endpoint resumable upload cua GCS
khong chap nhan -> **treo vo han** (file 341 byte cung treo).
Sua: truyen file tho kem header `Content-Length`. Luon dung `scripts/submit.py`,
khong dung `kaggle competitions submit` truc tiep.
