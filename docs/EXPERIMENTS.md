# Nhat ky thi nghiem — HODC2026

| # | Ngay | Model | Kenh | imgsz | Epoch | mAP@0.5 | mAP@[.5:.95] | Thoi gian | LB | Ghi chu |
|---|------|-------|------|-------|-------|---------|--------------|-----------|----|---------|
| 1 | 15/09 | YOLO11s | 16 | 768 | 40 | 0.979 | 0.730 (ro ri) | 1.94h | **0.6042** | Baseline. Chia ngau nhien -> val khong tin duoc.
| 2 | 15/09 | YOLO11s-P2 | 16 | 768 | 40 | 0.852 | 0.614 (lech) | 1.98h | **0.5848** | Them dau P2 + chia theo nhom canh. **Kem hon lan 1.**
| 3 | 16/09 | YOLO11s | 16 | 1024 | 40 | 0.940 | 0.6738 | 3.5h | 0.58875 | Bo chia CAN BANG (val dung duoc). Pretrain day du 493 tensor. Doi DUY NHAT imgsz vs lan 1.
| 4 | 16/09 | YOLO11s | 16 | 768 | 40 | 0.940 | **0.6744** | 1.94h | 0.59925 | Doi DUY NHAT imgsz vs lan 3.
| 5 | 16/09 | YOLO11s-P2 | 16 | 768 | 40 | 0.949 | 0.6695 | 1.92h | — | Doi DUY NHAT kien truc vs lan 4. **Kem hon 0.005 -> bo huong P2.**
| 6 | 16/09 | YOLO11**m** | 16 | 768 | 40 | 0.935 | 0.6697 | 2.05h | — | Doi DUY NHAT kich co model vs lan 4. Kem 0.005.
| 7 | 16/09 | YOLO11s | 16 | 768 | 40 | 0.941 | 0.6664 | 2.06h | — | Doi DUY NHAT augmentation (mixup .15, scale .9, translate .2). Kem 0.008. COCO pretrain (492 tensor + thoi phong conv dau). Da hoi tu (4 epoch cuoi deu 0.730). |

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


---

## Lan 2 — that bai, va vi sao

LB 0.5848 < 0.6042 cua lan 1. Nhung lan 2 doi **BON** thu cung luc nen khong tach
duoc nguyen nhan. Day la loi thiet ke thi nghiem:

| Yeu to | Lan 1 | Lan 2 |
|---|---|---|
| Kien truc | P3-P5 | P2-P5 |
| Chia val | ngau nhien (ro ri) | theo nhom canh |
| Batch | 16 | 12 |
| **Tensor COCO nap duoc** | **492** | **296** |

Nghi can chinh la dong cuoi. Them nhanh P2 lam **chi so cac lop trong head dich di**
(17->23, 19->25, 20->26, 22->28), nen Ultralytics khop trong so theo TEN bi truot.
Backbone van phu 100% (239/240) nhung phan neck mat gan het pretrain.
-> Lan 2 khong phai "P2 kem hon", ma la "P2 khoi dau voi it pretrain hon".

Cach sua neu quay lai P2: anh xa thu cong old->new {17:23, 19:25, 20:26, 22:28}
(da kiem tra: cac cap nay TRUNG shape).

## Bo chia theo nhom canh: het ro ri nhung MAT CAN BANG NANG

| Lop | train | val | |
|---|---|---|---|
| charger_head | 450 | **1** | vo nghia thong ke |
| car_toy | 353 | **2** | |
| rubik | 975 | **2** | |
| car | 548 | **4** | AP = 0.0 |
| stone_block | 267 | **5** | AP = 0.0 |
| orange | 108 | **76** | val chiem 41% ca lop |

Thuat toan chia uu tien nhom "chua nhieu lop nhat" roi lay tham lam -> gom lech.
mAP la trung binh theo lop nen vai lop 1-5 mau keo ca con so. **0.614 cung khong
tin duoc**, chi la nhieu theo kieu khac.

Can chia lai: vua theo nhom canh, vua can bang ti le moi lop (~10% so bbox cua
TUNG lop nam trong val).

## Bai hoc

1. **Moi lan chay chi doi MOT thu.** Quota 30h/tuan, moi lan ~2h -> khoang 14 lan.
   Doi nhieu thu cung luc la vut di mot lan do.
2. **Luon kiem tra so tensor pretrain nap duoc.** Doi kien truc la con so nay thay
   doi am tham.
3. LB la trong tai duy nhat cho den khi co bo val dang tin.


---

## Lan 3 (imgsz 1024, bo chia can bang)

val mAP@[.5:.95] = **0.6738** | mAP@0.5 = 0.9402 | hoi tu (4 epoch cuoi 0.673-0.674)
2769 anh train / 228 anh val | nap 493 tensor COCO (day du) | 3.5h tren 1 T4

### Lop manh / yeu

| Manh | mAP | | Yeu | mAP |
|---|---|---|---|---|
| egg_plastic | 0.808 | | **e-bike** | **0.298** |
| egg | 0.785 | | **people** | **0.328** |
| apple | 0.784 | | **stone_block** | 0.418 |
| apple_plastic | 0.782 | | car | 0.629 |
| banana_plastic | 0.780 | | charger_head | 0.659 |

**Cac cap that/gia van rat tot (0.75-0.81).** Diem yeu nam o vat the LON va bien
mo ho (e-bike, people) — nguoc voi truc giac ban dau la vat nho se kho.
Neu keo duoc 3 lop yeu nhat len ~0.6 thi mAP tong tang khoang +0.04.

## Sua trong infer.py

1. **Tu doc imgsz tu checkpoint.** Truoc do mac dinh 768; chay inference cho model
   train o 1024 se lech do phan giai, mAP tut ma khong bao loi gi.
2. **Quet lai anh trang.** Anh khong co du doan nao bi 0 diem. Sau luot chinh
   (conf=1e-4), quet lai rieng cac anh con trong voi conf=1e-7.
   Lan 3: cuu duoc 1 anh (2983) -> du 1000/1000.


---

## KET QUA AM QUAN TRONG: do phan giai da bao hoa

| Lan | imgsz | val mAP@[.5:.95] | thoi gian |
|---|---|---|---|
| 4 | 768 | **0.6744** | 1.94h |
| 3 | 1024 | 0.6738 | 3.5h |

Chenh 0.0006 = nhieu. **Tang do phan giai KHONG giup gi**, ma ton gap 1.8 lan GPU.

Ly do: anh sau demosaic chi ~490x250. Phong len 768 da la 1.57x, len 1024 la 2.09x
-> chi la noi suy, khong them thong tin. Tran thong tin nam o cam bien.
**Khong theo duoi huong tang imgsz nua.** Train o 768, danh phan tiet kiem cho
suc chua model.

## CHAN DOAN LAI: co HAI van de khac nhau, truoc do toi gop nham lam mot

Bang P/R/mAP cua lan 4:

| Lop | P | R | mAP50 | mAP50-95 | Van de |
|---|---|---|---|---|---|
| e-bike | 0.792 | **0.413** | 0.572 | 0.291 | KHONG PHAT HIEN RA |
| people | 0.712 | **0.569** | 0.666 | 0.316 | KHONG PHAT HIEN RA |
| stone_block | 0.636 | 0.749 | 0.837 | 0.385 | mot phan |
| egg | 0.987 | **1.000** | 0.995 | 0.769 | chi la DINH VI |
| orange | 0.945 | **1.000** | 0.995 | 0.817 | chi la DINH VI |
| rubik | 0.995 | **1.000** | 0.995 | 0.706 | chi la DINH VI |

1. **e-bike, people: loi PHAT HIEN.** Recall 0.41-0.57, mAP@0.5 cung thap.
   Model bo sot han, khong phai dan lech.
2. **Cac lop con lai: loi DINH VI.** Recall 1.000, mAP@0.5 = 0.995, nhung
   mAP@[.5:.95] chi 0.69-0.82. Toan bo phan mat la do bbox lech vai pixel.

Hai van de nay can hai cach chua khac nhau. Truoc do toi tuong tat ca la loi dinh vi
nen di tang do phan giai -> that bai.


---

## Lan 5: P2 co anh xa trong so -> van khong dang dung

| | mAP@0.5 | mAP@[.5:.95] | thoi gian | tensor COCO |
|---|---|---|---|---|
| Lan 4 (plain) | 0.9399 | **0.6744** | 1.94h | 493 |
| Lan 5 (P2) | **0.9488** | 0.6695 | 1.92h | 378 |

P2 PHAT HIEN tot hon (mAP@0.5 +0.009, e-bike 0.369 vs 0.291) nhung DINH VI kem hon
o cac lop nho (rubik 0.680 vs 0.706, table_tennis 0.666 vs 0.692). Tong: -0.005.
Cong them viec vinh vien mat 115 tensor pretrain do lech chi so -> **bo huong nay**.

## PHAT HIEN QUAN TRONG: train dang bi nghen o I/O, khong phai GPU

Lan 4 (plain, 22.5 GFLOPs): 1.94h
Lan 5 (P2, 29.9 GFLOPs, +33% phep tinh): 1.92h

Thoi gian **y het nhau** du khoi luong tinh toan chenh 33%. Nut co chai la doc va
giai ma 2769 file TIFF 4MB moi epoch (~16 anh/giay), khong phai GPU.

**He qua chien luoc:** tang suc chua model gan nhu MIEN PHI ve thoi gian. Day la
huong con lai dang gia nhat sau khi do phan giai bao hoa va P2 that bai.


---

## Doi chieu val <-> LB (quan trong cho cach lam viec)

| Lan | val (sach) | LB | chenh |
|---|---|---|---|
| 3 (1024) | 0.6738 | 0.58875 | -0.085 |
| 4 (768) | 0.6744 | 0.59925 | -0.075 |

**Val XEP HANG dung** (lan 4 > lan 3 o ca hai thang do) nhung **cao hon LB ~0.08**.
-> Tu gio xep hang cau hinh bang val, uoc luong **LB ~ val - 0.08**, khong can
ton luot nop cho moi thi nghiem.

## Tinh trang: 4 lan chay, CHUA cai thien duoc gi

LB tot nhat van la **lan 1 = 0.6042**, chinh la baseline dau tien.
Lan 4 = 0.59925, kem 0.005 — nam trong nhieu cua bang public (cham tren 500 anh),
nen khong ket luan lan 1 thuc su tot hon. Nhung ro rang la chua tien them buoc nao.

Khac biet lan 1 vs lan 4: batch 16 vs 8, va chia ngau nhien (2697 anh train) vs
chia theo nhom (2769 anh train). Lan 4 co NHIEU du lieu train hon ma diem khong cao hon.

## Gia thuyet cho huong tiep theo: OVERFITTING

Khoang cach val 0.674 -> test 0.599 la **0.08**. Mot phan do val con ro ri
(82% anh val van co anh tuong tu trong train o nguong 0.95), nhung phan con lai
nhiều kha nang la overfitting.

Neu dung, thi huong dang gia khong phai kien truc hay do phan giai ma la
**tang cuong augmentation / regularization**. Can thu:
- mixup, copy_paste (hien dang tat het)
- scale rong hon (hien 0.5)
- nhieu gain NHAN TOAN CUC (chung cho ca 16 bang) — an toan voi chu ky pho
- train tren TOAN BO 2997 anh cho lan cuoi (them ~8% du lieu)


---

# CAO NGUYEN: 5 huong, 5 lan khong an

| Huong | val mAP@[.5:.95] | vs baseline |
|---|---|---|
| **Baseline (lan 4): YOLO11s, 768, aug mac dinh** | **0.6744** | — |
| Do phan giai 1024 | 0.6738 | -0.001 |
| Dau P2 (stride 4) | 0.6695 | -0.005 |
| YOLO11m (2.1x tham so) | 0.6697 | -0.005 |
| Augmentation manh (mixup/scale/translate) | 0.6664 | -0.008 |
| Tinh chinh tham so suy luan (iou/conf/max_det) | 0.6682 | -0.006 |

Tat ca deu nam trong +-0.008 quanh baseline. Khong huong nao lam TE DI nhieu —
chung deu dung o cung mot cho. Day la dau hieu CHAM TRAN, khong phai chon sai tham so.

## Quet tham so suy luan (khong ton GPU dang ke)

| iou | conf | max_det | mAP@[.5:.95] |
|---|---|---|---|
| 0.5 | 0.001 | 300 | 0.6668 |
| 0.6 | 0.001 | 300 | 0.6674 |
| **0.7** | **0.0001** | **300** | **0.6682** (dang dung) |
| 0.8 | 0.001 | 300 | 0.6672 |
| 0.7 | 0.01 | 300 | 0.6646 |

Ca dai chi dao dong 0.0036. **Mac dinh da toi uu**, khong con gi de nhat o day.

## Du dia con lai (theo so lieu chan doan)

- Tran tren train (da hoc thuoc): **0.765** <- gioi han cua DO CHINH XAC NHAN
- Hien tai tren val: **0.671**
- Dư dia thuc: **0.094**, va toan bo la van de TONG QUAT HOA

Cac don bay CV tieu chuan da thu het. Con lai:
1. Train tren TOAN BO 2997 anh (thay vi 2769) — them 8% du lieu, gan nhu chac chan duong
2. Chuan hoa pho theo tung pixel — y tuong dac thu hyperspectral, chua thu
3. Lich hoc dai hon + cos_lr


---

# PHAN TICH LOI THEO LOP (16/09) — lat nguoc hai chan doan truoc

## Cap that/gia: DA GIAI XONG

| Cap | nham A->B | nham B->A | GT |
|---|---|---|---|
| apple / apple_plastic | 0 | 2 | 47 / 52 |
| banana / banana_plastic | 1 | 2 | 62 / 40 |
| orange / orange_plastic | 0 | 0 | 20 / 29 |
| egg / egg_plastic | 0 | 0 | 46 / 69 |
| car / car_toy | 0 | 0 | 55 / 35 |

Phan KHO NHAT cua cuoc thi — thu ban to chuc thiet ke rieng de danh do — gan nhu
khong con loi. Chien luoc 16 bang hoat dong dung nhu phan tich pho du doan tu dau.

## FP va bo sot deu KHONG phai van de

| nguong | FP tu nen | bo sot |
|---|---|---|
| conf >= 0.001 | 1029 | 21 |
| **conf >= 0.25** | **47** | 82 |

1029 FP o nguong thap chi la "duoi" tin cay thap, xep duoi cac du doan dung nen
gan nhu khong hai mAP (dung voi ket qua quet nguong: ca dai chi lech 0.0036).
O nguong that: precision ~95%, recall ~92%.

## KET LUAN: nut co chai la DO KHIT CUA BBOX

- mAP@0.5 = 0.94 (tim dung vat, phan dung lop)
- mAP@[.5:.95] = 0.67 (bbox lech vai pixel)
- Tran tren TRAIN (da hoc thuoc) = 0.765 <- mot phan la tran do chinh xac cua NHAN

Hai chan doan truoc cua toi deu SAI:
- "e-bike/people bi bo sot" -> khong, recall thap do do o nguong toi uu F1
- "qua nhieu bao dong gia" -> khong, chung deu tin cay thap va vo hai

## Kiem chung toa do bbox (theo cau hoi cua Quoc Anh)

Khu hoi VOC -> YOLO -> VOC tren 3428 bbox: sai so toi da **5.7e-14 pixel**.
Phep chuyen doi khong mat mat, khong co lech he thong. Loai tru gia thuyet nay.

## Con thieu: augmentation dac thu pho

`RandomHSV` cua Ultralytics **bo qua hoan toan** khi anh khong co dung 3 kenh
(`if img.shape[-1] != 3: return`). Nen viec tat hsv tu dau that ra khong thay doi gi —
no von da khong chay. Day la cho trong de cam nhieu DO LOI TOAN CUC (nhan ca 16 bang
voi cung mot he so ngau nhien) — phep duy nhat giu nguyen tuyet doi ti le giua cac bang.
CHUA THU.


---

# TONG KET 16/09: 9 huong, 9 lan khong vuot baseline

| Lan | Thay doi (so voi baseline lan 4) | val | thoi gian |
|---|---|---|---|
| **4** | **baseline: YOLO11s, 16 bang, 768, 40ep** | **0.6744** | 1.94h |
| 3 | imgsz 1024 | 0.6738 | 3.50h |
| 5 | dau P2 | 0.6695 | 1.92h |
| 6 | YOLO11m | 0.6697 | 2.05h |
| 7 | augmentation manh | 0.6664 | 2.06h |
| 8 | chuan hoa pho theo pixel | 0.6725 | 1.95h |
| 9 | 120 epoch | 0.6685 | 5.95h |
| 10 | **3 bang thay vi 16** | 0.6704 | **0.60h** |
| — | tinh chinh tham so suy luan | 0.6682 | 0.2h |

## Lan 10 lat lai gia dinh GOC cua ca du an

Toi chot "phai dung 16 bang" tu ngay dau, dua tren phan tich tach pho:
apple vs apple_plastic = 0.988 (16 bang) vs 0.654 (3 bang).
Roi dung 8 thi nghiem len tren ma khong bao gio kiem chung lai.

**Sai o dau:**
1. Phan tich do dung **bang 0,1,2 — ba bang LIEN KE, gan nhu trung nhau**.
   Lan 10 dung bang **0, 8, 15 trai deu pho** -> mang gan het thong tin cua 16 bang.
2. Phan tich chi dung **pho trung binh trong bbox** + hoi quy tuyen tinh.
   Detector con co hinh dang, ket cau, boi canh — nhieu hon nhieu.

Ket qua tren cac cap that/gia gan nhu y het nhau (apple 0.775 vs 0.777,
orange 0.817 vs 0.826, egg 0.769 vs 0.777).

**Bai hoc:** dung suy ket luan cho he thong phuc tap tu phep do tren mo hinh don gian,
va phai kiem chung gia dinh GOC som chu khong chi tinh chinh ben tren no.

## PHAT HIEN THUC DUNG: 3 bang nhanh gap 3.2 lan

| | 16 bang | 3 bang |
|---|---|---|
| thoi gian train 40ep | 1.94h | **0.60h** |
| dung luong dataset | 11.7 GB | **1.1 GB** |
| val mAP@[.5:.95] | 0.6744 | 0.6704 |

Chat luong tuong duong nhung **moi gio GPU mua duoc gap 3 lan so thi nghiem**.
Tu gio thu nghiem tren 3 bang, chi quay lai 16 bang cho ban cuoi neu no thuc su hon.

## Loi thu 7 cua Ultralytics: CLAHE

Khi so kenh == 3, Ultralytics BAT cac augmentation danh cho anh mau (CLAHE, HSV...)
von bi bo qua voi 16 kenh (`if img.shape[-1] != 3: return`). CLAHE doi CV_8UC1/CV_16UC1
nen vo ngay voi anh 3 kenh uint16. -> phai xuat uint8 (cung la cach script mau lam).

---

# 16/09 chieu — DE RUI RO PHASE 2 (khong ton quota GPU)

Bo xep hang ra 23/09 se khong nam trong ban sao cong khai nao. Neu khong dua duoc
no qua model thi toan bo cong viec khong sinh ra ket qua nao. Nen do truoc.

## Do 1: suy luan CPU tai may — GIAI QUYET XONG

| | |
|---|---|
| may | i7-1165G7, 8 luong, 19 GB RAM, torch CPU |
| toc do | **0.41 s/anh** o 768px (chi suy luan) |
| ca doi + suy luan | **~1.0 s/anh** (do thuc te tren 4 anh that) |
| 600 anh | **~10 phut** |

Hom qua toi coi day la rui ro lon nhat. Khong phai. Da viet `scripts/phase2_local.py`
chay MOT LUOT: PNG -> X2Cube -> 3 bang -> suy luan -> CSV, khong ghi TIFF trung gian.
Da chay thu tren 4 anh test THAT: 5/6 kiem tra dat, cai truot dung la `class_id 0..17`
vi dang dung trong so COCO 80 lop -> dung nhu mong doi.

Kiem chung kich thuoc: PNG (1012, 2004) -> cube (253, 501, 16) -> dau vao (253, 501, 3).
Dung 1/4 moi chieu.

## Do 2: API cuoc thi bi chan — DAY MOI LA RUI RO THAT

| endpoint | ket qua |
|---|---|
| `competitions download -f class.txt` (169 byte) | **429** |
| `competitions download` (tai gop ca bo) | **429** |
| `datasets download -f <anh>.png` | OK, **1.6 MB/s** |

Gioi han la theo SO LUOT GOI, khong phai bang thong — file 169 byte cung bi chan.
Van con hieu luc nhieu gio sau. Va `competition_sources` BI BO IM LANG khi day kernel:
kernel probe thay `/kaggle/input` RONG.

Nghia la: **khong co duong tu dong nao lay duoc bo xep hang**, ca tu may lan tu Kaggle.
Duong con lai la nho nguoi dung gan du lieu qua giao dien web mot lan.

## Phat hien phu: `GAIN` la ma chet trong nhanh 3 bang

`to_rgb8` lam min-max SAU khi nhan GAIN, ma min-max bat bien voi moi phep nhan duong.
Do thuc te: nhan toan cuc x1.7 truoc `to_rgb8` chi doi dau ra **toi da 1 don vi**.
GAIN chi co tac dung o nhanh 16 bang. Khong hai, nhung ly le ve GAIN chi ap cho 16 bang.

## Vi sao KHONG chay augmentation pho tren 3 bang

Da cai dat xong (`patch_spectral_gain`, cam vao cho trong ma `RandomHSV` de lai vi no
tu thoat khi so kenh != 3). Nhung do xong thi thay khong nen chay tren du lieu 3 bang:

`to_rgb8` chuan hoa min-max THEO TUNG ANH, tuc **moi bien thien do sang giua cac anh
da bi xoa sach** truoc khi model nhin thay. Them nhieu do sang luc train la day model
mot thu khong he ton tai luc test -> nhieu kha nang lam te di.

No chi co nghia o nhanh 16 bang (noi khong co chuan hoa theo anh). Giu ma lai, khong
tieu 0.6h quota vao day.

Ghi chu: `to_rgb8` chuan hoa TUNG BANG RIENG -> lam lech ti le giua cac bang, dung cai
dac trung dang le de phan biet that/gia. Do tren anh that: lech 2% (nho, vi 3 bang co
max gan nhau). Cung la mot bang chung nua cho ket luan lan 10: model khong dua nhieu
vao ti le pho, no dua vao hinh dang / ket cau / boi canh.

## Chot lai rui ro Phase 2 (do xong, khong con phong doan)

Ba phep thu, ket qua dut khoat:

| duong | ket qua |
|---|---|
| tai tu may | **429** (ke ca file 169 byte, ke ca tai gop) |
| `competition_sources` khi day kernel | **bi bo im lang**, `/kaggle/input` rong |
| tai tu BEN TRONG kernel Kaggle | **429** — cung loi y het |

-> Gioi han gan voi TAI KHOAN, khong phai dia chi mang. Doi sang may khac khong giup.

Nhung: `competition_list_files` CHAY BINH THUONG tu ben trong Kaggle (liet ke duoc
20 file), va **`competitions submissions` cung chay binh thuong**. Chi rieng
endpoint TAI du lieu bi chan.

**Nghia la duong NOP BAI — thu duy nhat quyet dinh diem — khong he bi anh huong.**
Cai bi chan chi la lay du lieu moi ve, va viec do chi can den 23/09, tuc con 7 ngay
de gioi han tu go. Da dat `scripts/watch_ratelimit.sh` thu 30 phut/lan va ghi nhat ky
vao `data/ratelimit_watch.log`, thoat ngay khi thong.

Neu den 22/09 van chua go thi phai nho nguoi dung gan du lieu cuoc thi bang tay qua
giao dien web Kaggle (nut "Add Input" -> tab Competitions) — day la viec chi nguoi dung
lam duoc, khong lam thay duoc qua API.

## Cong cu da dung san: gop hop co trong so (`src/utils/fusion.py`)

NMS CHON mot hop va vut phan con lai. WBF lay TRUNG BINH CO TRONG SO ca cum.
Do tren du lieu mo phong (nhieu 3px tren canh, 3 nguon):

    sai so canh: 1 model 2.43 px -> gop 3 nguon 1.36 px  (giam 44%)

Voi hop trung vi 24 px thi 1 px sai lech ~ 4% IoU. Day la don bay dung huong nhat
con lai: no khong co gang PHAT HIEN tot hon (mAP@0.5 da 0.94 roi) ma lam HOP KHIT hon.
Co ha diem cum it nguon dong y, neu khong ensemble se lam tang duong tinh gia.
5 kiem thu don vi deu dat.

---

# HIEU CHUAN NUT THAT (16/09 toi) — do quan trong nhat trong ngay

Tu kiem chung bo cham diem doc lap (pycocotools) bang du lieu mo phong giong HODC
(200 anh 501x253, hop 14-40 px), them nhieu Gauss vao canh hop:

| nhieu canh | mAP@[.5:.95] | mAP@0.5 |
|---|---|---|
| 0.0 px | **1.0000** | 1.0000 |
| 0.5 px | 0.8685 | 1.0000 |
| 1.0 px | 0.6982 | 1.0000 |
| 2.0 px | 0.4530 | 0.9386 |
| 3.0 px | 0.2585 | 0.7574 |
| 5.0 px | 0.0902 | 0.3463 |

Du doan hoan hao cho 1.000 -> bo cham diem DUNG.

## Doc ra duoc gi

Model cua ta: **mAP@[.5:.95] = 0.67, mAP@0.5 = 0.94**.
Hai con so nay KHONG nam tren cung mot diem cua duong cong:

* 0.67 ung voi nhieu canh ~1.0 px
* nhung nhieu 1.0 px le ra cho mAP@0.5 = **1.000**, ta chi co 0.94

-> Ta mat diem o HAI cho TACH BIET:
  1. **~1 px nhieu hop** (an ~0.30 diem)
  2. **~6% bo sot / nhan nham** (phan mAP@0.5 thieu so voi 1.0)

Truoc do toi goi chung la "hop chua khit". Bay gio tach duoc ra va biet ti trong.

## He qua truc tiep

**1 px nhieu canh ngon 0.30 diem mAP.** Ha nhieu tu 1.0 px xuong 0.5 px se dua
0.70 -> 0.87. KHONG mot huong nao trong 9 huong da thu cham vao duoc con so do:
kien truc, do phan giai, so epoch, augmentation, chuan hoa pho — khong cai nao
lam hop khit hon, chung chi doi cach model NHIN.

Phep duy nhat giam nhieu toa do la TRUNG BINH HOA nhieu du doan doc lap:
* TTA (lat + da ti le tren cung model)
* ensemble nhieu model + WBF

Mo phong cho thay gop 3 nguon giam sai so canh 44%. Neu ap duoc vao thuc te thi
1.0 px -> 0.56 px, tuc ~+0.15 mAP — lon hon MOI thu da thu cong lai.

**Canh bao tu chinh minh:** cac diem luu cua CUNG mot lan train co loi TUONG QUAN
manh, nen gop chung se duoc it hon nhieu so voi 3 lan train doc lap. Va tran nha
do do chinh xac cua nhan (train ceiling 0.765 ~ 0.85 px) van dung: khong the vuot
qua do chinh xac cua chinh nhan goc.

---

# LAN 11: dfl 1.5 -> 4.0  (3 bang, moi thu khac giu nguyen)

    mAP50-95 = 0.6754   (lan 10 cung cau hinh, dfl 1.5: 0.6704)
    mAP50    = 0.9377
    thoi gian = 0.626h

**+0.005.** Nho, nhung DUONG, va dung huong ma chan doan du bao (dfl la nhanh hoi quy
canh hop; tang trong so cua no = uu tien do khit hon do phan loai). Lan dau sau nhieu
lan co mot thay doi di dung huong. Cung vuot nhe moc 16 bang (0.6744).

# TTA khong an — nhung KHONG bac bo gia thuyet gop hop

    mAP50-95 co TTA = 0.6757   (khong TTA = 0.6754)   -> +0.0003, bang khong

**Vi sao day khong phai bang chung chong lai WBF:** TTA cua Ultralytics NOI cac du doan
tu cac ti le/lat khac nhau roi chay **NMS**. Ma NMS **CHON MOT hop va vut phan con lai**.
No khong he lay trung binh. Nen phep thu nay do "hop nhat cac phat hien", khong phai
"trung binh hoa toa do". Dung hai viec khac nhau.

## Nhung no goi ra mot kha nang toi CHUA XET

Neu cac goc nhin khac nhau (3 ti le x lat) cho hop gan y het nhau, thi sai so 1 px co
the la **THIEN LECH HE THONG** chu khong phai nhieu ngau nhien — vi du model deu dan
doan hop to hon (hoac nho hon) nhan that mot chut.

Dieu nay doi han cach chua:
* **nhieu ngau nhien** -> trung binh hoa (ensemble + WBF). Dat: can nhieu lan train.
* **thien lech he thong** -> HIEU CHINH mot he so. Gan nhu mien phi.

Da them phep do vao `kaggle_notebooks/ensemble/ens.py`: ghep du doan voi nhan that
(IoU>=0.5), do lech trung binh CO DAU cua chieu rong/cao/tam, roi tinh ti le
|thien lech| / do lech chuan. Neu ti le > 0.3 thi thien lech chiem uu the.
Kem theo quet thu he so phong to hop 0.94-1.06 va cham diem lai bang pycocotools.

Dung tieu 1.8h quota vao ensemble truoc khi biet cau tra loi nay.

## Meo ky thuat: tai output kernel khong bi chet giua chung

`kaggle kernels output` tai het moi file va chet o file trong so lon (IncompleteRead).
Dung `file_pattern` de chi lay thu can:

    api.kernels_output("user/kernel", path=..., file_pattern=r".*\.(log|json|yaml)$")

---

# HUONG DAU TIEN THAT SU AN: GOP HOP (16/09 toi)

Kernel `ensemble`, tren 228 anh val, cung mot bo trong so cho ca 4 phep do
(nen so sanh tuong doi la hop le du muc tuyet doi con dang tranh cai — xem duoi):

| cach | mAP@[.5:.95] | so voi moc |
|---|---|---|
| A. mot model, thuong | 0.5007 | — |
| B. mot model + TTA | 0.5265 | **+0.0258** |
| C. gop 4 diem luu bang WBF | 0.5375 | **+0.0368** |
| D. hieu chinh he so kich thuoc hop | 0.5021 | +0.0014 |

## Cau hoi "thien lech hay nhieu" da co cau tra loi

Ghep 744 cap (du doan vs nhan that, IoU>=0.5, conf>=0.25):

| | lech trung binh | do lech chuan |
|---|---|---|
| chieu rong | **-0.289 px** | 3.441 px |
| chieu cao | -0.258 px | 3.103 px |
| tam ngang | +0.046 px | 1.545 px |
| tam doc | +0.039 px | 1.557 px |

ti le |thien lech| / nhieu = **0.084** -> **nhieu ap dao tuyet doi**.

Nen phep hieu chinh he so gan nhu vo dung (+0.0014, dung nhu du bao), va
**lay trung binh nhieu du doan doc lap la duong dung**. Day khong con la suy luan
nua, no la so do duoc.

## Tai sao TTA trong `train.py` bao +0.0003 con o day bao +0.0258

Hai duong do KHAC NHAU:
* `model.val(augment=True)` — nhieu kha nang validator BO QUA tham so `augment`
* `model.predict(augment=True)` — chac chan co ap dung

Con so trong `ens.py` dang tin hon vi A va B dung Y HET mot duong, chi khac mot co.

Bai hoc lap lai lan thu ba trong du an nay: **do hai thu can so sanh bang CUNG mot
duong do**, dung so sanh so cua cong cu nay voi so cua cong cu khac.

## Mot cho CHUA GIAI THICH DUOC — khong duoc lo di

Ultralytics bao **0.6754**, pycocotools cua toi bao **0.5007**, tren cung trong so,
cung danh sach val. Chenh 26% tuong doi. Mot trong hai sai.

Dau vet: duong `predict()` chi cho 9.7 hop/anh. O nguong conf=0.001 thi con so do
la THAP — duoi cua phan bo (cac hop diem thap) la thu lam mAP tang, va neu no bi
cat thi mAP tut dung kieu nay. Nghi van: `predict()` co the dang bo qua tham so
`conf` va dung mac dinh 0.25.

Da them phep do vao `infer2.py`: in ra conf NHO NHAT trong ket qua predict.
* neu ~0.001 -> nguong co duoc ap dung, phai tim nguyen nhan khac
* neu ~0.25  -> tim ra thu pham

Chua dung con so 0.5007 de ket luan bat cu dieu gi ve muc tuyet doi.
Quan he THU TU (C > B > A) van dung vi ca ba chung duong do.

## Dang chay

* `train-2`, `train-3`: hai model DOC LAP (seed 2, 3; dfl=4.0) de gop hop —
  gop diem luu cua cung mot lan train da +0.037, model doc lap phai hon vi
  loi cua chung it tuong quan hon
* `prep-test-rgb`: bo test 3 bang. Truoc gio chi co ban 16 bang -> khong co no thi
  khong nop duoc model 3 bang. Phai chuyen doi Y HET prep_train_rgb.

---

# NOP BAI GOP HOP: LB 0.45993 — TUT MANH

    submissions/phase1_wbf3_tta.csv
    3 model doc lap (seed 0/2/3, dfl=4.0, 3 bang) + TTA + WBF
    val tung model: 0.6754 / 0.6736 / 0.6712
    LB: 0.45993   (moc tot nhat tu truoc: 0.60420)

## Toi lai mac DUNG cai loi da tu nhac minh sau lan 2

Ban nop nay doi HAI thu cung luc:
1. **3 bang** — chua he nop len leaderboard lan nao (moi chi do tren val)
2. **gop hop WBF + TTA** — cung chua he nop lan nao

Nen 0.460 KHONG cho biet cai nao gay ra. Phai tach ra moi biet.

## Manh moi so 1: do tin cay bi lech chuan

| | run4 (16 bang, don le, LB 0.599) | wbf3 (3 bang, gop, LB 0.460) |
|---|---|---|
| tong so hop | 27560 (27.6/anh) | 23603 (23.6/anh) |
| hop conf > 0.5 | **3231** | **2194** |
| hop conf > 0.25 | 3638 | 3527 |
| rong trung vi | 20.1 px | 20.1 px |
| cao trung vi | 27.1 px | 23.5 px |

**So hop tin cay cao tut 32%** trong khi tong so hop gan nhu khong doi.
Day dung la dau van tay cua phep phat "it nguon dong y" trong WBF:

    conf = mean(diem trong cum) * min(so_hop, n_models) / n_models

Hop nao chi 1/3 model thay thi bi CHIA BA. Ma mAP rat nhay voi THU TU xep hang
theo do tin cay. Tren val thi phep phat nay co loi (+0.037), nhung val chi co
228 anh va cung phan phoi voi train — tap test co the khac.

Nghi van phu: phep phat nay la PHAT KEP, vi lay `mean` cua cum VON DA phan anh
muc dong y roi (cum it thanh vien thi trung binh tren it diem).

## Manh moi so 2: chieu cao hop lech 13%

27.1 px -> 23.5 px. Co the la khac biet giua model 16 bang va 3 bang, cung co the
la dau hieu bo test 3 bang (`prep_test_rgb`) khong khop bo train 3 bang.
Chua ro. Phai kiem tra.

## Ke hoach tach nguyen nhan cho ngay mai (KHONG can train lai lan nao)

Da co san 3 model. Chi can chay lai suy luan voi cac bien the, moi lan mot thay doi:

| phep thu | tra loi cau hoi |
|---|---|
| 1 model, khong TTA, khong WBF, 3 bang | **3 bang co te hon 16 bang tren LB khong?** |
| gop 3 model, KHONG phat dong y (conf = max) | **phep phat co phai thu pham khong?** |
| 1 model + TTA | TTA don doc co an tren LB khong? |

Phep thu 1 la quan trong nhat: neu 3 bang mot minh cung ra ~0.46 thi van de la
o du lieu 3 bang chu khong phai o WBF, va toan bo huong "3 bang nhanh gap 3 lan"
phai xem lai — ke ca ket luan cua lan 10.

## Nhac lai cho chinh minh

Val da noi doi ve muc tuyet doi nhieu lan roi:
* run1: val 0.730 -> LB 0.604
* run4: val 0.674 -> LB 0.599
* wbf3: val ~0.67 (tung model) -> LB 0.460

Tu gio **moi thay doi lon phai nop rieng mot ban de do**, dung gop nhieu thay doi
vao mot ban nop de "tiet kiem luot".
