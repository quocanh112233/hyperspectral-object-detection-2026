"""Gop hop co trong so (Weighted Box Fusion).

Vi sao dung cai nay chu khong phai NMS: NMS CHON MOT hop va vut phan con lai.
WBF lay TRUNG BINH CO TRONG SO toa do cua ca cum. Voi bai nay do la dung thu can:
chan doan cho thay mAP@0.5 = 0.94 nhung mAP@[.5:.95] = 0.67, tuc vat the da duoc tim
thay roi, chi la hop chua du khit. Trung binh nhieu hop doc lap lam giam nhieu toa do
theo 1/sqrt(n), con NMS thi khong giam gi ca.

Dung cho ca hai: gop ket qua nhieu model, hoac gop ket qua TTA cua mot model.
"""
import numpy as np


def _iou_matrix(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0, np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0]); yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2]); yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a1 + a2 - inter, 1e-9)


def wbf(boxes, scores, labels, iou_thr=0.55, n_models=1, conf_type="avg"):
    """boxes (N,4) xyxy, scores (N,), labels (N,) -> (M,4), (M,), (M,).

    n_models: so nguon dau vao. Dung de HA DIEM cua cum chi co it nguon dong y —
    mot hop ma chi 1/3 model nhin thay thi kem tin cay hon hop ca 3 deu thay.
    Khong co buoc nay thi ensemble lam TANG duong tinh gia thay vi giam.
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    out_b, out_s, out_l = [], [], []

    for lab in np.unique(labels):
        m = labels == lab
        b, s = boxes[m], scores[m]
        order = np.argsort(-s)
        b, s = b[order], s[order]

        clusters = []          # moi phan tu: [danh sach chi so]
        c_boxes = np.zeros((0, 4), np.float32)
        for i in range(len(b)):
            if len(clusters):
                ious = _iou_matrix(b[i], c_boxes)
                j = int(np.argmax(ious))
                if ious[j] >= iou_thr:
                    clusters[j].append(i)
                    idx = clusters[j]
                    w = s[idx][:, None]
                    c_boxes[j] = (b[idx] * w).sum(0) / max(w.sum(), 1e-9)
                    continue
            clusters.append([i])
            c_boxes = np.vstack([c_boxes, b[i][None]])

        for j, idx in enumerate(clusters):
            sc = s[idx]
            conf = float(sc.mean() if conf_type == "avg" else sc.max())
            # cum it nguon dong y -> ha diem tuyen tinh theo ti le dong y
            conf *= min(len(idx), n_models) / float(n_models)
            out_b.append(c_boxes[j]); out_s.append(conf); out_l.append(lab)

    if not out_b:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    o = np.argsort(-np.asarray(out_s))
    return (np.asarray(out_b, np.float32)[o], np.asarray(out_s, np.float32)[o],
            np.asarray(out_l)[o])
