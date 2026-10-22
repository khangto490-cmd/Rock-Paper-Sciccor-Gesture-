"""
build_frame_dataset_v2.py
--------------------------
Dựng dataset từng frame cho phương pháp M7:
  - đặc trưng: 11 số "từng ngón" (thay cho 63 tọa độ thô)
  - nhãn: gán theo ngưỡng độ duỗi, CỘNG THÊM "nhãn báo trước": LEAD frame ngay
          trước lúc vượt ngưỡng cũng mang nhãn kéo/bao
  - có thêm cột seq_id để train_static_v2.py chia kiểm tra theo CHUỖI

Chạy: python build_frame_dataset_v2.py
Output: frame_dataset_v2.csv
"""

import csv
import os

import numpy as np

from features import engineered_features_batch, ENG_DIM

DATA_DIR = "sequences"
OUT_CSV = "frame_dataset_v2.csv"

# SỬA CHO KHỚP với 2 ngưỡng Hi đang dùng ở build_frame_dataset.py
T_KEO = 1.30
T_BAO = 1.15

# Số frame "báo trước". Bảng so sánh cho thấy 3 là hợp lý: 6 không sớm thêm
# mà chỉ làm búa bị kích hoạt nhầm nhiều hơn.
LEAD = 2

NAMES = ["bua", "keo", "bao"]  # chỉ số 0, 1, 2


def ext_index_middle(P):
    return (np.linalg.norm(P[:, 8], axis=1) + np.linalg.norm(P[:, 12], axis=1)) / 2


def ext_all4(P):
    return np.mean([np.linalg.norm(P[:, i], axis=1) for i in (8, 12, 16, 20)], axis=0)


def frame_labels(seq, final):
    """Trả về mảng chỉ số nhãn (0=búa, 1=kéo, 2=bao), mỗi frame một nhãn."""
    P = seq.reshape(len(seq), 21, 3)
    lab = np.zeros(len(seq), dtype=int)           # mặc định: búa
    if final == "keo":
        mask, cls = ext_index_middle(P) >= T_KEO, 1
    elif final == "bao":
        mask, cls = ext_all4(P) >= T_BAO, 2
    else:
        return lab                                # chuỗi búa: luôn búa
    lab[mask] = cls
    if LEAD > 0 and mask.any():
        first = int(np.argmax(mask))              # frame đầu tiên vượt ngưỡng
        lab[max(0, first - LEAD):first] = cls     # nhãn báo trước
    return lab


def main():
    rows, counts, seq_id = [], {n: 0 for n in NAMES}, 0

    for final in ["keo", "bua", "bao"]:
        d = os.path.join(DATA_DIR, final)
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".npy"):
                continue
            seq = np.load(os.path.join(d, fname)).astype(np.float32)
            labels = frame_labels(seq, final)
            feats = engineered_features_batch(seq)            # (n, 11)
            for f, l in zip(feats, labels):
                rows.append([float(v) for v in f] + [NAMES[l], seq_id])
                counts[NAMES[l]] += 1
            seq_id += 1

    header = [f"e{i}" for i in range(ENG_DIM)] + ["label", "seq_id"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"Da ghi {len(rows)} frame tu {seq_id} chuoi vao {OUT_CSV}")
    print(f"Nguong: T_KEO={T_KEO}, T_BAO={T_BAO}, nhan bao truoc LEAD={LEAD} frame")
    for n in NAMES:
        print(f"  {n}: {counts[n]} frame")


if __name__ == "__main__":
    main()
