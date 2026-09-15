# Ke hoach thuc hien — HODC2026

Soan 15/09/2026. Han chot thuc te: **24/09/2026 16:00 UTC = 23:00 gio VN**
(Kaggle API bao, som hon mot ngay so voi tai lieu tom tat).

---

## 1. Rang buoc chi phoi moi quyet dinh

| Rang buoc | He qua |
|---|---|
| Con ~8 ngay | Khong co thoi gian thu nghiem kien truc moi. Dung do nghe da chin. |
| May local khong co GPU NVIDIA | Toan bo train chay tren Kaggle Notebooks. |
| Quota Kaggle 30h GPU/tuan, 12h/session | ~6-7 lan train. Moi lan chay phai co gia thuyet ro rang. |
| Chi mot model, cam ensemble | Khong WBF, khong multi-scale fusion, khong multi-model. |
| Diem = mAP@[0.5:0.95] | Dinh vi bbox chinh xac quan trong ngang phan loai. |
| 71% vat the < 32x32 px | Bai toan small-object. Do phan giai dau vao la don bay lon nhat. |
| Cap that/gia nam chung anh | Bat buoc dung du 16 bang pho. Pseudo-RGB la ngo cut. |

---

## 2. Kien truc de xuat

**Ultralytics YOLO11, sua dau vao thanh 16 kenh.**

Ly do chon: chin, augmentation san, huan luyen nhanh, xuat du doan de. Voi 8 ngay
thi mmdetection/detectron2 qua cham de dung. RT-DETR la phuong an du phong neu
YOLO11 vuong o phan da kenh.

### Cac quyet dinh thiet ke

1. **Giu nguyen 16 bang**, khong rut xuong 3. Day la ly do ton tai cua cuoc thi.

2. **Phong to anh dau vao len ~2x** (imgsz 1024, `rect=True` -> batch 1024x512).
   Anh goc chi ~492x247 nen phong 2x van re. Vat the trung vi 24px -> 48px,
   cai thien manh AP o nguong IoU cao.

3. **Nap trong so COCO roi "thoi phong" conv dau tu 3 -> 16 kenh**
   (lap lai / chia trung binh trong so RGB). Voi 3000 anh thi pretrain la thiet yeu.
   **Phai khai bao COCO pretrain** trong bao cao (luat muc 6).

4. **Them dau P2 (stride 4)** o giai doan cai thien. YOLO11 mac dinh bat dau tu P3
   (stride 8) — voi vat the 24px thi P2 dang gia.

5. **TAT augmentation mau (hsv_h, hsv_s, hsv_v = 0).** Day la diem de sai nhat:
   augmentation mau pha huy dung cai chu ky pho dung de phan biet that/gia.
   Chi giu bien doi hinh hoc (mosaic, flip, scale, translate) + co the them
   nhieu gain nhan toan cuc (chung cho ca 16 bang) de mo phong thay doi phoi sang
   — phep nay giu nguyen hinh dang pho tuong doi.

6. **Chuan hoa theo tung bang** bang mean/std tinh tren tap train (gia tri 16-bit,
   khong phai 0-255).

7. **Inference dat conf rat thap (0.001), max_det ~300.** Voi metric mAP thi cang
   nhieu detection duoc xep hang tot cang loi. Dung mac dinh conf=0.25 se mat vai diem.

8. **Khong dung TTA.** Ve ky thuat TTA la mot model, nhung no co "post-NMS fusion"
   — dung nguyen van thu bi cam o muc 6. Khong dang lieu.

### Viec can lam sach du lieu
- Loc bbox suy bien: da phat hien anh 1590 co `people` voi xmin==xmax==487 (rong 0px).
- Kep toa do ve trong [0, W-1] / [0, H-1].
- Khong hardcode kich thuoc anh (379x214 den 512x256, 330 kich thuoc khac nhau/513 anh).

---

## 3. Lich trinh

| Ngay | Viec | Tieu chi hoan thanh |
|---|---|---|
| **15/09** | Mo khoa du lieu tren Kaggle. EDA pho day du. Chuyen VOC -> YOLO. | Tra loi duoc: pho tach that/gia o bang nao |
| **16/09** | Baseline YOLO11s 16 kenh, ~30 epoch. Inference test set. **Nop thu Phase 1.** | CSV duoc Kaggle chap nhan, co diem tren LB |
| **17-19/09** | Cai thien: dau P2, imgsz, epoch, can bang lop. 1-2 thi nghiem/ngay. | Moi lan chay ghi vao EXPERIMENTS.md |
| **20-21/09** | Train ban cuoi voi cau hinh tot nhat. | Weight luu thanh Kaggle Dataset |
| **22/09** | **Dong bang.** Notebook inference chay thu tren test set. | CSV kiem tra du 8 muc checklist |
| **23/09** | Ranking set mo. Inference ca hai bo. Nop. | CSV co ca test + ranking |
| **24/09** | Du phong + nop lan cuoi truoc 23:00 VN. | Da nop trong Phase 2 |

**Nguyen tac:** moc 16/09 (co diem tren LB) quan trong hon moi thu khac. Mot model
tam duoc nop dung han thang mot model tot nop truot han.

---

## 4. Rui ro da biet

| Rui ro | Xu ly |
|---|---|
| Kaggle API khong gan duoc du lieu community competition vao kernel | Gan tay 1 lan tren web UI cho moi notebook |
| API chan tai file sau ~500 file | Khong tai du lieu ve local; lam viec truc tiep tren Kaggle |
| Ultralytics co the khong nhan 16 kenh | **Kiem chung truoc khi cam ket.** Du phong: torchvision FCOS/RetinaNet tu viet |
| Val split lac quan gia neu anh la khung hinh lien tiep cung canh | Kiem tra trung lap canh truoc khi chia val |
| Session Kaggle ngat sau 12h | Train co checkpoint + resume |
| Quen ranking set trong CSV Phase 2 | Checklist bat buoc truoc moi lan nop |
