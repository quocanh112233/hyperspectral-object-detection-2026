# Hyperspectral Object Detection Challenge 2026 — Bảng tra cứu

> Track 1 của 2nd Hyperspectral Remote Sensing Data Processing and Application Challenge
> Trang gốc: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026
> **Lưu ý: file này là bản tóm tắt. Trang Kaggle gốc luôn là nguồn chính xác nhất — kiểm tra lại trước mỗi mốc quan trọng.**

---

## 1. Bài toán

Object detection trên ảnh **hyperspectral 16 băng phổ (460–600 nm)**, phát hiện và phân loại **18 lớp**.

Điểm mấu chốt: bộ dữ liệu chứa các **cặp thật/giả (real-vs-counterfeit)** — apple vs apple_plastic, banana vs banana_plastic, egg vs egg_plastic vs egg_wood, car vs car_toy, orange vs orange_plastic. Ảnh RGB thường khó phân biệt các cặp này; thông tin phổ phản ánh **tính chất vật liệu** nên đây là lý do cuộc thi dùng hyperspectral. Nếu chỉ dùng pseudo-RGB 3 kênh, nhiều khả năng sẽ không đạt điểm tốt ở các lớp này.

---

## 2. Timeline

| Mốc | Ngày |
|---|---|
| Công bố cuộc thi | 25/06/2026 |
| Dữ liệu train mở | 25/07/2026 |
| **Phase 1** (test set) | 25/07 – 23/09/2026 |
| **Ranking set được mở** | 23/09/2026 |
| **Phase 2** (test + ranking) | 23/09 – 25/09/2026 |
| Hạn nộp gói code review (top 6) | 27/09/2026 |
| Công bố kết quả | 28/09/2026 |

**CỰC KỲ QUAN TRỌNG:** Submission nộp **trước 23/09 KHÔNG tính** vào xếp hạng cuối. Chỉ các submission trong Phase 2 (23–25/09) mới được dùng để tính điểm chung cuộc. Nếu file CSV Phase 2 chỉ có test set mà thiếu ranking set → **toàn bộ ảnh ranking bị 0 điểm**.

---

## 3. Dữ liệu

| File | Nội dung |
|---|---|
| `data_train.zip` | 11 GB — 3.000 ảnh PNG + 3.000 file VOC XML. Thư mục: `VIS/` (ảnh), `Annotations/VIS/` (nhãn) |
| `data_test.zip` | 3.7 GB — 1.000 ảnh PNG, thư mục `VIS/`, **không có nhãn** |
| `ranking_images.zip` | 1.000 ảnh PNG, mở ngày 23/09, **không có nhãn** |
| `class.txt` | Danh sách 18 lớp theo thứ tự từ điển, class_id = số dòng (bắt đầu từ 0) |
| `sample_submission.csv` | Mẫu file nộp |
| `pseudo_rgb_demo.py` | Script mẫu: demosaic + trích 3 băng thành ảnh pseudo-RGB |

Tổng dung lượng: **15.65 GB** — cần tính trước dung lượng ổ đĩa và thời gian tải.

### Định dạng ảnh
- PNG **1 kênh, 16-bit**, chụp bằng camera XIMEA MQ022HG-IM-SM4X4-VIS3
- Sensor dùng **spectral filter array 4×4**: mỗi khối 4×4 pixel mang 16 giá trị phổ tại cùng một vị trí không gian
- **Bắt buộc demosaic** thành cube `(H, W, 16)` bằng hàm chính thức `X2Cube(img, cellSize=4)` trước khi dùng

```python
import numpy as np
from PIL import Image

def X2Cube(img, cellSize=4):
    B = [cellSize, cellSize]
    skip = [cellSize, cellSize]
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    return out.reshape(M // cellSize, N // cellSize, cellSize * cellSize)

img = np.array(Image.open('path/to/image.png'))
cube = X2Cube(img)   # -> (H, W, 16)
```

### Định dạng nhãn — Pascal VOC XML
```xml
<annotation>
    <folder>train/VIS</folder>
    <filename>1000.png</filename>
    <size>
        <width>497</width>
        <height>251</height>
        <depth>16</depth>
    </size>
    <object>
        <name>badminton</name>
        <bndbox>
            <xmin>186</xmin><ymin>28</ymin>
            <xmax>201</xmax><ymax>50</ymax>
        </bndbox>
    </object>
</annotation>
```
- `<width>`, `<height>` là kích thước **sau khi demosaic** (bằng `cube.shape[:2]`), **không phải** kích thước file PNG gốc
- Kích thước có thể **chênh lệch vài pixel giữa các ảnh** → không được hardcode kích thước cố định
- `<depth>` luôn là 16
- Toạ độ bbox theo pixel, dạng góc trên-trái / góc dưới-phải

### Danh sách 18 lớp (class_id = chỉ số dòng, từ 0)
```
0  apple            9   egg
1  apple_plastic    10  egg_plastic
2  badminton        11  egg_wood
3  banana           12  orange
4  banana_plastic   13  orange_plastic
5  car              14  people
6  car_toy          15  rubik
7  charger_head     16  stone_block
8  e-bike           17  table_tennis
```

---

## 4. Đánh giá

- **Metric chính:** mAP@[0.5:0.95] — trung bình AP qua các ngưỡng IoU từ 0.50 đến 0.95, bước 0.05
- **Metric phụ:** mAP@0.5
- Điểm cuối cùng = mAP@[0.5:0.95] tổng hợp trên **cả test set và ranking set, trọng số bằng nhau**

**Leaderboard:**
- Phase 1: public LB hiển thị điểm trên **50% test set** (chọn ngẫu nhiên), 50% còn lại ẩn
- Phase 2: 100% test set được chấm công khai; ranking set hoàn toàn private, chỉ lộ sau khi kết thúc
- → Thứ hạng Phase 1 không phản ánh kết quả cuối, đừng dựa vào đó để đánh giá

---

## 5. Định dạng file nộp

```csv
id,image_id,class_id,confidence,x1,y1,x2,y2
0,1000,2,0.95,186,28,201,50
1,1000,9,0.72,366,68,383,86
2,1001,14,0.88,50,30,180,200
```

| Cột | Ý nghĩa |
|---|---|
| `id` | Số nguyên duy nhất mỗi dòng (0, 1, 2, …) — Kaggle yêu cầu |
| `image_id` | Mã ảnh (ví dụ `1000.png` → `1000`) |
| `class_id` | Chỉ số lớp 0–17 theo `class.txt` |
| `confidence` | Độ tin cậy 0–1 |
| `x1,y1,x2,y2` | Toạ độ bbox (góc trên-trái, góc dưới-phải) |

- Mỗi detection là một dòng; nhiều detection cùng ảnh → nhiều dòng riêng
- Phase 1: chỉ nộp dự đoán cho test set
- Phase 2: **gộp cả test set + ranking set vào CÙNG một file CSV**

---

## 6. Luật cần nhớ

**Bắt buộc tuân thủ — vi phạm sẽ bị loại:**
- **CHỈ ĐƯỢC DÙNG MỘT MODEL DUY NHẤT.** Cấm mọi hình thức ensemble: multi-model voting, weighted fusion, post-NMS fusion...
- Cấm dùng **dataset ngoài chưa khai báo** để pre-train
- Cấm tạo nhãn giả thủ công
- Cấm tấn công hệ thống chấm điểm hoặc gian lận dưới mọi hình thức
- Không chia sẻ code/dữ liệu riêng tư ra ngoài team (chia sẻ công khai trên forum của cuộc thi thì được)
- Một người chỉ dùng một tài khoản Kaggle, chỉ tham gia một team, chỉ chọn một track

**Giới hạn kỹ thuật:**
- Tối đa **3 submission/team/ngày**
- Team tối đa **5 người** (kể cả trưởng nhóm)
- Đủ 18 tuổi trở lên mới đủ điều kiện nhận giải

**Nếu lọt top 6 — phải nộp gói code review về `hottracking2025@gmail.com` trước 27/09:**
- Source code train + inference
- Trọng số model đã train
- Script inference tái lập đúng file CSV đã nộp
- README mô tả: môi trường, dependencies, lệnh chạy, **số tham số (parameter count) và FLOPs của model**

→ Nghĩa là phải giữ code sạch và ghi lại cấu hình ngay từ đầu, không để đến cuối mới dọn.

---

## 7. Giải thưởng

| Hạng | Phần thưởng |
|---|---|
| Nhất | 2.500 RMB + Certificate of Honor |
| Nhì | 1.500 RMB + Certificate of Honor |
| Ba | 1.000 RMB + Certificate of Honor |
| Excellence Award (3 đội) | Certificate of Honor |

Tổng: 5.000 RMB. Một thành viên của mỗi đội hạng Nhất/Nhì/Ba được miễn phí đăng ký và **được mời trình bày tại 10th National Symposium on Imaging Spectroscopy for Earth Observation**.

Giải pháp thắng cuộc sẽ được **open source**.

**Đơn vị tổ chức:** ISDE-CNC Imaging Spectroscopy for Earth Observation Committee, Nanjing University of Science and Technology, Shenzhen University, Wuhan University, University of Cambridge.

---

## 8. Checklist trước khi nộp Phase 2

- [ ] Đã chạy inference trên **cả** test set và ranking set
- [ ] File CSV chứa dự đoán của **cả hai bộ** trong cùng một file
- [ ] Cột `id` đánh số liên tục từ 0, không trùng lặp
- [ ] `class_id` nằm trong 0–17, khớp đúng thứ tự `class.txt`
- [ ] Toạ độ bbox theo hệ **sau demosaic**, không phải pixel của PNG gốc
- [ ] Chỉ dùng một model duy nhất, không ensemble
- [ ] Nộp trong khung 23–25/09
- [ ] Đã lưu sẵn code + weight + README (phòng khi lọt top 6)

---

## 9. Ghi chú thí nghiệm

| Lần | Ngày | Cấu hình (model, kênh, ảnh đầu vào, epoch…) | mAP@0.5 | mAP@[0.5:0.95] | Nhận xét |
|---|---|---|---|---|---|
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

**Việc cần làm tiếp:**
-
-
